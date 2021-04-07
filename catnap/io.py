from __future__ import annotations
from typing import Optional, List, Dict, Tuple
import logging
import textwrap

import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

from .catmaid_interface import Catmaid, Bbox, ConnectorDetail
from .utils import CoordZYX

logger = logging.getLogger(__name__)


def trim_cols(df, required: List[str], name=None):
    try:
        return df[required]
    except KeyError:
        name = f"{name} " if name else ""
        msg = f"Invalid {name}dataframe columns.\n\tRequired: {required}\n\t     Got: {list(df.columns)}"
        raise ValueError(msg)


class TransformerMixin:
    _transformer_attr: Optional[str] = None

    def world_to_px(self, world_coords, as_int=False, round_z=False):
        if self._transformer_attr is None:
            out = (np.asarray(world_coords) - self.offset) / self.resolution
            if round_z:
                out[..., 0] = np.round(out[..., 0]).astype(out.dtype)
            if as_int:
                out = out.astype(np.uint64)
            return out
        else:
            return getattr(self, self._transformer_attr).world_to_px(
                world_coords, as_int, round_z
            )

    def px_to_world(self, px_coords):
        if self._transformer_attr is None:
            return (
                np.asarray(px_coords, dtype=np.float64) * self.resolution + self.offset
            )
        else:
            return getattr(self, self._transformer_attr).px_to_world(px_coords)


class Image(TransformerMixin):
    def __init__(
        self, array, resolution=(1, 1, 1), offset=(0, 0, 0), dims=("z", "y", "x")
    ):
        self.array = np.asarray(array)
        self.resolution = np.asarray(resolution, dtype=float)
        self.offset = np.asarray(offset, dtype=float)
        if list(dims) != ["z", "y", "x"]:
            raise NotImplementedError("Non-ZYX orientations are not supported")
        self.dims = dims

    def extents(self):
        """[[mins], [maxes]]"""
        return np.array([self.offset, self.offset + self.resolution * self.array.shape])

    def to_hdf5(self, f, name, mode="a", attrs=None):
        if not isinstance(f, h5py.Group):
            with h5py.File(f, mode) as f2:
                return self.to_hdf5(f2, name)
        ds = f.create_dataset(name, data=self.array, compression="gzip")
        ds.attrs["resolution"] = self.resolution
        ds.attrs["offset"] = self.offset
        ds.attrs["dims"] = self.dims
        if attrs is not None:
            ds.attrs.update(attrs)
        return ds

    def is_compatible(self, other: Image):
        try:
            self.raise_on_incompatible(other)
        except ValueError:
            return False
        return True

    def raise_on_incompatible(self, other: Image, names=("left", "right")):
        features = {}
        if not isinstance(self, Image) or not isinstance(other, Image):
            features["class"] = (type(self), type(other))
        if self.array.shape != other.array.shape:
            features["shape"] = (self.array.shape, other.array.shape)
        if tuple(self.resolution) != tuple(other.resolution):
            features["resolution"] = (tuple(self.resolution), tuple(other.resolution))
        if tuple(self.offset) != tuple(other.offset):
            features["offset"] = (tuple(self.offset), tuple(other.offset))
        if tuple(self.dims) != tuple(other.dims):
            features["dims"] = (tuple(self.dims), tuple(other.dims))

        if not features:
            return

        left_name, right_name = pad_strs(names)

        lines = []
        for k, (l_val, r_val) in features.items():
            lines.append(k)
            lines.append(f"  {left_name}: {l_val}")
            lines.append(f"  {right_name}: {r_val}")

        msg = textwrap.indent("\n".join(lines), "  ")
        raise ValueError("Images not compatible.\n" + msg)

    @classmethod
    def from_hdf5(cls, f, name=None):
        if isinstance(f, h5py.Dataset):
            return cls(f[:], f.attrs["resolution"], f.attrs["offset"], f.attrs["dims"])
        elif isinstance(f, h5py.Group):
            return cls.from_hdf5(f[name])
        else:
            with h5py.File(f, "r") as f2:
                return cls.from_hdf5(f2[name])

    def max_plus_one(self):
        if not issubclass(self.array.dtype, np.integer):
            raise TypeError("Array is not of integer subtype")
        return self.array.data.max() + 1

    def contains(self, coord: Tuple[float, float, float]) -> bool:
        """Whether a real-world coordinate tuple is inside the array"""
        diffs = self.extents - coord
        return np.all(diffs[0] <= 0) and np.all(diffs[1] >= 0)

    def sub_image_px(
        self, internal_offset: Tuple[int, int, int], shape: Tuple[int, int, int]
    ) -> Image:
        int_off = np.asarray(internal_offset, int)
        if np.any(int_off < 0):
            raise ValueError("internal_offset must be positive")
        if np.any(int_off + shape > self.array.shape):
            raise ValueError("sub-image extends beyond image")
        slicing = tuple(slice(o, o + s) for o, s in zip(int_off, shape))
        arr = self.array[slicing]
        return type(self)(
            arr, self.resolution, self.offset + int_off * self.resolution, self.dims
        )

    def sub_image(
        self,
        internal_offset: Tuple[float, float, float],
        shape: Tuple[float, float, float],
    ) -> Image:
        """Start and stop points are found in world coordinates; then rounded to pixels"""
        offset_px = np.round(self.offset + internal_offset).astype(int)
        stop_px = np.round(self.offset + internal_offset + shape).astype(int)
        return self.sub_image_px(offset_px, stop_px - offset_px)


def pad_strs(strs, prefix=True, pad=" "):
    if len(pad) != 1:
        raise ValueError("Pad string must be 1 character long")

    length = max(len(s) for s in strs)
    return [s + pad * (length - len(s)) for s in strs]


def serialize_treenodes(tns: pd.DataFrame):
    tns = tns.copy()
    tns["parent_id"] = np.array(tns["parent_id"].fillna(0), dtype=int)
    return tns


def deserialize_treenodes(tns: pd.DataFrame):
    tns = tns.copy()
    ids = pd.array(tns["parent_id"], dtype="UInt64")
    ids[ids == 0] = pd.NA
    tns["parent_id"] = ids
    return tns


def remove_single_nodes(treenodes: pd.DataFrame):
    """Remove all nodes belonging to skeletons with only 1 treenode in the dataframe"""
    skids, counts = np.unique(treenodes["skeleton_id"], return_counts=True)
    single_tns = skids[counts == 1]
    to_drop = np.zeros(len(treenodes), bool)
    for skid in single_tns:
        to_drop |= treenodes["skeleton_id"] == skid
    return treenodes.loc[~to_drop].copy()


class CatnapIO(TransformerMixin):
    _transformer_attr = "raw"

    def __init__(
        self,
        raw: Image,
        treenodes: pd.DataFrame,
        connectors: pd.DataFrame,
        partners: pd.DataFrame,
        labels: Optional[Image] = None,
    ):
        self.raw: Image = raw
        self.treenodes = remove_single_nodes(
            trim_cols(
                treenodes,
                ["treenode_id", "parent_id", "skeleton_id", "z", "y", "x"],
                "treenode",
            )
        )
        self.connectors = trim_cols(
            connectors, ["connector_id", "z", "y", "x"], "connector"
        )
        self.partners = trim_cols(
            partners,
            ["skeleton_id", "treenode_id", "connector_id", "is_presynaptic"],
            "partners",
        )
        self.labels: Optional[Image] = None
        self.set_labels(labels)

    def to_hdf5(self, fpath, gname=""):
        gname = gname.rstrip("/") if gname else ""
        prefix = f"{gname}/tables"
        with pd.HDFStore(fpath, "w") as f:
            serialize_treenodes(self.treenodes).to_hdf(f, f"{prefix}/treenodes")
            self.connectors.to_hdf(f, f"{prefix}/connectors")
            self.partners.to_hdf(f, f"{prefix}/partners")

        prefix = f"{gname}/volumes"
        with h5py.File(fpath, "a") as f:
            self.raw.to_hdf5(f, f"{prefix}/raw")
            if self.labels is not None:
                self.labels.to_hdf5(f, f"{prefix}/labels")

    @classmethod
    def from_hdf5(cls, fpath, gname="", ignore_labels=False):

        prefix = f"{gname}/tables"
        with pd.HDFStore(fpath, "r") as f:
            treenodes = deserialize_treenodes(pd.read_hdf(f, f"{prefix}/treenodes"))
            connectors = pd.read_hdf(f, f"{prefix}/connectors")
            partners = pd.read_hdf(f, f"{prefix}/partners")

        prefix = f"{gname}/volumes"
        with h5py.File(fpath, "r") as f:
            raw = Image.from_hdf5(f[f"{prefix}/raw"])
            labels = None
            if not ignore_labels:
                try:
                    labels = Image.from_hdf5(f[f"{prefix}/labels"])
                except KeyError:
                    pass

        return cls(raw, treenodes, connectors, partners, labels)

    @classmethod
    def from_catmaid(cls, catmaid: Catmaid, raw: Image, labels=None):
        dims = raw.dims

        extents = [CoordZYX({d: x for d, x in zip(dims, ext)}) for ext in raw.extents()]
        logger.info("Fetching annotations from CATMAID")
        raw_conns: pd.DataFrame
        treenodes, raw_conns = catmaid.nodes_in_bbox(Bbox.from_start_stop(*extents))
        connectors, partners = ConnectorDetail.to_connector_partners_df(
            tqdm(
                catmaid.connector_detail_many(raw_conns.connector_id),
                desc="connector details",
                total=len(raw_conns.connector_id),
            )
        )

        return cls(raw, treenodes, connectors, partners, labels)

    def set_labels(self, labels: Optional[Image]) -> bool:
        """Returns old labels"""
        if labels is not None:
            self.raw.raise_on_incompatible(labels, ("raw", "labels"))
        ret = self.labels is not None
        self.labels = labels
        return ret

    def coords_in_raw(self, zyx) -> np.ndarray:
        mins, maxes = self.raw.extents()
        zyx_arr = np.asarray(zyx)
        return np.all(np.logical_and(maxes > zyx_arr, zyx_arr >= mins), axis=1)

    def make_labels(self, treenode_radius=None, set_labels=False) -> Image:
        labels = Image(
            np.zeros_like(self.raw.array, np.uint64),
            self.raw.resolution,
            self.raw.offset,
            self.raw.dims,
        )
        if set_labels:
            self.labels = labels

        if not treenode_radius:
            return labels

        tns = self.treenodes.copy()
        zyx_world = tns[["z", "y", "x"]]
        idxs = self.coords_in_raw(zyx_world)
        zyx_px = ((zyx_world - labels.offset) / labels.resolution).astype(int)

        skels = tns["skeleton_id"][idxs]
        locs = zyx_px[idxs]

        skid_to_label: Dict[int, int] = {
            skid: label for label, skid in enumerate(np.unique(skels), 1)
        }
        for skid, (z, y, x) in zip(skels, locs.itertuples(index=False)):
            # +1 accounts for 0-based index
            slicing = (
                z,
                slice(max(0, y - treenode_radius + 1), y + treenode_radius + 1),
                slice(max(0, x - treenode_radius + 1), x + treenode_radius + 1),
            )
            labels.array[tuple(slicing)] = skid_to_label[skid]

        return labels

    def join_tables(self):
        merged = pd.merge(
            self.treenodes,
            self.partners,
            on="treenode_id",
            suffixes=("_t", "_tc"),
        )
        return pd.merge(
            merged, self.connectors, on="connector_id", suffixes=("_t", "_c")
        )
