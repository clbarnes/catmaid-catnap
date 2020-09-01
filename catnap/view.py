from __future__ import annotations
from typing import NamedTuple, Iterable, Tuple, List, Dict
from copy import copy
import datetime as dt
import math

import numpy as np
import napari
import pandas as pd
from vispy.color import Color, ColorArray
from coordinates import MathDict

from .io import CatnapIO, Image, TransformerMixin
from .navigator import Navigator

DIMS = ["z", "y", "x"]


class NodeSpecArrays(NamedTuple):
    size: np.ndarray
    edge_color: ColorArray
    face_color: ColorArray

    @staticmethod
    def concatenate(arrs: Iterable[NodeSpecArrays]):
        sizes = []
        edge_colors = []
        face_colors = []

        for arr in arrs:
            sizes.append(arr.size)
            edge_colors.append(arr.edge_color.rgba)
            face_colors.append(arr.face_color.rgba)

        out = NodeSpecArrays(
            np.concatenate(sizes),
            ColorArray(np.concatenate(edge_colors, axis=0)),
            ColorArray(np.concatenate(face_colors, axis=0)),
        )
        return out

    def to_viewer_kwargs(self):
        return {
            "size": self.size,
            "edge_color": self.edge_color.rgba,
            "face_color": self.face_color.rgba,
        }


def get_cols(df: pd.DataFrame, names: List[str], rename=None) -> pd.DataFrame:
    subset = df[names]
    if rename:
        return subset.rename(columns=rename)
    return subset


class NodeSpec:
    default_size = 5
    default_edge_color = (0, 0, 0, 1)
    default_face_color = (1, 1, 1, 1)

    def __init__(self, size=None, edge_color=None, face_color=None):
        cls = type(self)
        self.size = cls.default_size if size is None else size
        self.edge_color = Color(
            cls.default_edge_color if edge_color is None else edge_color
        )
        self.face_color = Color(
            cls.default_face_color if face_color is None else face_color
        )

    def to_arrays(self, n: int):
        """size, edge_width, edge_color, face_color"""
        return NodeSpecArrays(
            np.asarray(n * [self.size]),
            ColorArray(n * [self.edge_color.rgba]),
            ColorArray(n * [self.face_color.rgba]),
        )


def timestamp():
    dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def remove_suffix(s: str, sep="_"):
    *most, last = s.split(sep)
    return sep.join(most)


class PreRenderer(TransformerMixin):
    _transformer_attr = "io"

    presynapse_spec = NodeSpec(face_color=(1, 0, 0, 1))
    postsynapse_spec = NodeSpec(face_color=(0, 1, 1, 1))
    connector_spec = NodeSpec(
        size=NodeSpec.default_size * 3, edge_color="y", face_color=(1, 1, 0, 0.3)
    )

    skel_edge_spec = {"edge_color": "y"}
    conn_edge_spec = {"edge_color": "cyan"}

    def __init__(
        self,
        catnap_io: CatnapIO,
        presynapse_spec=None,
        postsynapse_spec=None,
        connector_spec=None,
        skel_edge_spec=None,
        conn_edge_spec=None,
    ):
        self.io = catnap_io
        self.viewer = napari.Viewer("catnap", axis_labels=("z", "y", "x"))
        self.layers: Dict[str, napari.layers.Layer] = dict()
        self._inner_joined_tables = None
        self._navigator = Navigator(self.viewer)

        cls = type(self)

        self.presynapse_spec = presynapse_spec or cls.presynapse_spec
        self.postsynapse_spec = postsynapse_spec or cls.postsynapse_spec
        self.connector_spec = connector_spec or cls.connector_spec

        self.skel_edge_spec = skel_edge_spec or cls.skel_edge_spec
        self.conn_edge_spec = conn_edge_spec or cls.conn_edge_spec

    @property
    def _joined_tables(self):
        if self._inner_joined_tables is None:
            self._inner_joined_tables = self.io.join_tables()
        return self._inner_joined_tables

    def coords_to_px(self, zyx, as_int=False):
        arr = (zyx - self.io.raw.offset) / self.io.raw.resolution
        if as_int:
            arr = arr.astype(np.uint64)
        return arr

    def _filter_translate_points(self, df, as_int=False, dims=DIMS):
        dims = list(dims)
        idxs = self.io.coords_in_raw(df[dims])
        out = df[idxs].copy()
        out[dims] = self.coords_to_px(out[dims], as_int)
        return out

    def _partner_points(self) -> Tuple[pd.DataFrame, NodeSpecArrays]:
        tn_dims = [f"{d}_t" for d in DIMS]
        partners = self._filter_translate_points(self.joined_tables, dims=tn_dims)
        partners.sort_values("is_presynaptic", inplace=True)
        n_pre = np.sum(partners["is_presynaptic"])
        n_post = len(partners) - n_pre
        pre_arrays = self.presynapse_spec.to_arrays(n_pre)
        post_arrays = self.postsynapse_spec.to_arrays(n_post)
        node_specs = NodeSpecArrays.concatenate([post_arrays, pre_arrays])
        return partners[tn_dims].rename(columns=remove_suffix), node_specs

    def _connector_points(self) -> Tuple[pd.DataFrame, NodeSpecArrays]:
        connectors = self._filter_translate_points(self.io.connectors)
        return connectors[DIMS], self.connector_spec.to_arrays(len(connectors))

    def _points(self) -> Tuple[pd.DataFrame, NodeSpecArrays]:
        partner_locs, partner_specs = self._partner_points()
        conn_locs, conn_specs = self._connector_points()

        locs = partner_locs.append([conn_locs], ignore_index=True)
        specs = NodeSpecArrays.concatenate([partner_specs, conn_specs])
        return locs, specs

    def _world_startstop_to_px_vectors(self, starts, stops) -> np.ndarray:
        starts_px = self.world_to_px(starts).to_numpy()
        starts_px[:, 0] = np.round(starts_px[:, 0])
        stops_px = self.world_to_px(stops).to_numpy()
        stops_px[:, 0] = np.round(stops_px[:, 0])

        out = []
        zlim = len(self.io.raw.array)
        for start, stop in zip(starts_px, stops_px):
            proj = stop - start
            z_diff = proj[0]
            if z_diff == 0:
                if zlim > start[0] >= 0:
                    out.append([start, proj])
                continue
            abs_z_diff = abs(z_diff)
            new_proj = np.array([0, *(proj[1:] / abs_z_diff)])
            z_diff3 = proj * [1, 0, 0]
            for _ in range(int(abs_z_diff)):
                if 0 <= start[0] < zlim:
                    out.append([start, new_proj])
                start = start + z_diff3
                if 0 <= start[0] < zlim:
                    out.append([start, new_proj])
                start = start + new_proj
        return np.array(out, dtype=starts_px.dtype)

    def _connector_vectors(self) -> np.ndarray:
        merged = self.joined_tables

        start = get_cols(merged, ["z_c", "y_c", "x_c"], remove_suffix)
        stop = get_cols(merged, ["z_t", "y_t", "x_t"], remove_suffix)
        vecs = self._world_startstop_to_px_vectors(start, stop)
        return vecs

    def _skeleton_vectors(self) -> np.ndarray:
        tns = self.io.treenodes.rename(columns={"id": "child"})
        merged = pd.merge(
            tns,
            tns[~tns.parent.isna()],
            left_on="child",
            right_on="parent",
            suffixes=("_child", "_parent"),
        )
        child_points = merged[["z_child", "y_child", "x_child"]].rename(
            columns={f"{d}_child": d for d in DIMS}
        )
        parent_points = merged[["z_parent", "y_parent", "x_parent"]].rename(
            columns={f"{d}_parent": d for d in DIMS}
        )

        return self._world_startstop_to_px_vectors(child_points, parent_points)

    @property
    def joined_tables(self):
        if self._inner_joined_tables is None:
            self._inner_joined_tables = self.io.join_tables()
        return self._inner_joined_tables


class CatnapViewer(TransformerMixin):
    _transformer_attr = "prerenderer"

    _labels_name = "labels"
    _skel_edge_name = "skeleton edges"
    _raw_name = "raw"
    _conn_edge_name = "connector edges"
    _points_name = "locations"

    def __init__(self, prerenderer: PreRenderer):
        self.prerenderer = prerenderer

        self.viewer = napari.Viewer("catnap", axis_labels=("z", "y", "x"))
        self.layers: Dict[str, napari.layers.Layer] = dict()
        self._inner_joined_tables = None
        self._navigator = Navigator(self.viewer)

    @property
    def io(self):
        return self.prerenderer.io

    def _lowest_unused_label(self):
        """Also rejects special IDs"""
        existing = np.unique(self.layers[self._labels_name].data)
        if existing[0] == 0:
            existing = existing[1:]
        for expected, exist in enumerate(existing, 1):
            if expected != exist:
                return expected
        # alternative all-numpy implementation
        # expected = np.arange(len(existing), dtype=existing.dtype) + 1
        # diffs = existing - expected
        # return expected[(diffs != 0).argmax]

    @property
    def raw_layer(self) -> napari.layers.Image:
        return self.layers[self._raw_name]

    @property
    def labels_layer(self) -> napari.layers.Labels:
        return self.layers[self._labels_name]

    def next_id(self):
        this_id = self._lowest_unused_label()
        self.labels_layer.selected_label = this_id

    def show(self):
        self.layers[self.raw_name] = self.viewer.add_image(
            self.io.raw.array, name=self._raw_name
        )

        if self.io.labels is not None:
            self.layers[self.labels_name] = self.viewer.add_labels(
                self.io.labels.array, name=self._labels_name
            )

        skel_vecs = self.prerenderer._skeleton_vectors()
        self.layers[self.skel_edge_name] = self.viewer.add_vectors(
            skel_vecs, name=self._skel_edge_name, **self.prerenderer.skel_edge_spec
        )

        conn_vecs = self.prerenderer._connector_vectors()
        self.layers[self.conn_edge_name] = self.viewer.add_vectors(
            conn_vecs, name=self._conn_edge_name, **self.prerenderer.conn_edge_spec
        )
        points, point_specs = self.prerenderer._points()

        self.layers[self.points_name] = self.viewer.add_points(
            points, name=self._points_name, **point_specs.to_viewer_kwargs()
        )

        self.viewer.update_console({"cviewer": self})

    def export_labels(self, fpath, name="", with_source=False):
        arr = self.layers[self._labels_name].data
        if with_source:
            io = copy(self.io)
            io.labels = arr
            io.to_hdf5(fpath, name)
        else:
            img = Image(
                arr, self.io.raw.resolution, self.io.raw.offset, self.io.raw.dims
            )
            img.to_hdf5(fpath, name or "labels", {"date": timestamp()})

    def jump_to(self, z=None, y=None, x=None):
        dims = DIMS
        vals = MathDict({d: v for d, v in zip(dims, [z, y, x]) if v is not None})
        resolution = MathDict(
            {d: v for d, v in zip(dims, self.io.raw.resolution) if d in vals}
        )
        offset = MathDict({d: v for d, v in zip(dims, self.io.raw.offset) if d in vals})
        args = (vals - offset) / resolution
        return self.jump_to_px(**math.round(args))

    def jump_to_px(self, z=None, y=None, x=None):
        return self._navigator.move_to(x, y, z)
