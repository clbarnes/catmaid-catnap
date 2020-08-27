from __future__ import annotations
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
import h5py

from .catmaid_interface import Catmaid, Bbox, ConnectorDetail
from .utils import CoordZYX


def trim_cols(df, required: List[str], name=None):
    try:
        return df[required]
    except KeyError:
        name = f"{name} " if name else ""
        msg = f"Invalid {name}dataframe columns.\n\tRequired: {required}\n\t     Got: {list(df.columns)}"
        raise ValueError(msg)


class Image:
    def __init__(
        self, array, resolution=(1, 1, 1), offset=(0, 0, 0), dims=("z", "y", "x")
    ):
        self.array = np.asarray(array)
        self.resolution = np.asarray(resolution, dtype=float)
        self.offset = np.asarray(offset, dtype=float)
        self.dims = dims

    def extents(self):
        """[[mins], [maxes]]"""
        return np.array([self.offset, self.offset + self.resolution * self.array.shape])

    def to_hdf5(self, f, name, attrs=None):
        if not isinstance(f, h5py.Group):
            with h5py.File(f, "w") as f2:
                return self.to_hdf5(f2, name)
        ds = f.create_dataset(name, data=self.array, compression="gzip")
        ds.attrs["resolution"] = self.resolution
        ds.attrs["offset"] = self.offset
        ds.attrs["dims"] = self.dims
        if attrs is not None:
            ds.attrs.update(attrs)
        return ds

    def is_compatible(self, other: Image):
        return (
            isinstance(other, Image)
            and self.array.shape == other.array.shape
            and tuple(self.resolution) == tuple(other.resolution)
            and tuple(self.offset) == tuple(other.offset)
            and tuple(self.dims) == tuple(other.dims)
        )

    @classmethod
    def from_hdf5(cls, f, name=None):
        if isinstance(f, h5py.Dataset):
            return cls(f[:], f.attrs["resolution"], f.attrs["offset"], f.attrs["dims"])
        elif isinstance(f, h5py.Group):
            return cls.from_hdf5(f[name])
        else:
            with h5py.File(f, "r") as f2:
                return cls.from_hdf5(f2[name])


def serialize_treenodes(tns: pd.DataFrame):
    tns = tns.copy()
    tns["parent"] = np.array(tns["parent"].fillna(0), dtype=int)
    return tns


def deserialize_treenodes(tns: pd.DataFrame):
    tns = tns.copy()
    ids = pd.array(tns["parent"], dtype="UInt64")
    ids[ids == 0] = pd.NA
    tns["parent"] = ids
    return tns


class CatnapIO:
    def __init__(
        self,
        raw: Image,
        treenodes: pd.DataFrame,
        connectors: pd.DataFrame,
        partners: pd.DataFrame,
        labels: Optional[Image] = None,
    ):
        self.raw: Image = raw
        self.treenodes = trim_cols(
            treenodes, ["id", "parent", "skeleton", "z", "y", "x"], "treenode"
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
    def from_hdf5(cls, fpath, gname=""):
        prefix = f"{gname}/tables"
        with pd.HDFStore(fpath, "r") as f:
            treenodes = deserialize_treenodes(pd.read_hdf(f, f"{prefix}/treenodes"))
            connectors = pd.read_hdf(f, f"{prefix}/connectors")
            partners = pd.read_hdf(f, f"{prefix}/partners")

        prefix = f"{gname}/volumes"
        with h5py.File(fpath, "r") as f:
            raw = Image.from_hdf5(f[f"{prefix}/raw"])
            try:
                labels = Image.from_hdf5(f[f"{prefix}/labels"])
            except KeyError:
                labels = None

        return cls(raw, treenodes, connectors, partners, labels)

    @classmethod
    def from_catmaid(cls, catmaid: Catmaid, raw: Image, labels=None):
        dims = raw.dims

        extents = [CoordZYX({d: x for d, x in zip(dims, ext)}) for ext in raw.extents()]
        treenodes, raw_conns = catmaid.nodes_in_bbox(Bbox.from_start_stop(*extents))
        connectors, partners = ConnectorDetail.to_connector_partners_df(
            catmaid.connector_detail_many(raw_conns.id)
        )
        return cls(raw, treenodes, connectors, partners, labels)

    def set_labels(self, labels: Optional[Image]) -> bool:
        """Returns old labels"""
        if labels is not None:
            if not self.raw.is_compatible(labels):
                raise ValueError(
                    "Given labels incompatible with raw: must be an Image of the same shape, resolution, and offset"
                )
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

        skels = tns["skeleton"][idxs]
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
        tns = self.treenodes.rename(columns={"id": "treenode_id"})
        conns = self.connectors.rename(columns={"id": "connector_id"})
        merged = pd.merge(tns, self.partners, on="treenode_id", suffixes=("_t", "_tc"),)
        return pd.merge(merged, conns, on="connector_id", suffixes=("_t", "_c"))
