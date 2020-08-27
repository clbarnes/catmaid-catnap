from __future__ import annotations
from typing import NamedTuple, Iterable, Tuple, List, Dict
from copy import copy
import datetime as dt

import numpy as np
import napari
import pandas as pd
from vispy.color import Color, ColorArray

from .io import CatnapIO, Image

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


class CatnapViewer:
    presynapse_spec = NodeSpec(face_color=(1, 0, 0, 1))
    postsynapse_spec = NodeSpec(face_color=(0, 1, 1, 1))
    connector_spec = NodeSpec(
        size=NodeSpec.default_size * 3, edge_color="y", face_color=(1, 1, 0, 0.3)
    )

    skel_edge_spec = {"edge_color": "y"}
    conn_edge_spec = {"edge_color": "cyan"}

    labels_name = "labels"
    skel_edge_name = "skeleton edges"
    raw_name = "raw"
    conn_edge_name = "connector edges"
    points_name = "locations"

    def __init__(self, catnap_io: CatnapIO):
        self.io = catnap_io
        self.viewer = napari.Viewer("catnap", axis_labels=("z", "y", "x"))
        self.layers: Dict[str, napari.layers.Layer] = dict()
        self._inner_joined_tables = None

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
        partners = self._filter_translate_points(self._joined_tables, dims=tn_dims)
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
        starts_px = self.coords_to_px(starts).to_numpy()
        starts_px[:, 0] = np.round(starts_px[:, 0])
        stops_px = self.coords_to_px(stops).to_numpy()
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
        merged = self._joined_tables

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

    def show(self):
        self.layers[self.raw_name] = self.viewer.add_image(
            self.io.raw.array, name=self.raw_name
        )

        if self.io.labels is not None:
            self.layers[self.labels_name] = self.viewer.add_labels(
                self.io.labels.array, name=self.labels_name
            )

        skel_vecs = self._skeleton_vectors()
        self.layers[self.skel_edge_name] = self.viewer.add_vectors(
            skel_vecs, name=self.skel_edge_name, **self.skel_edge_spec
        )

        conn_vecs = self._connector_vectors()
        self.layers[self.conn_edge_name] = self.viewer.add_vectors(
            conn_vecs, name=self.conn_edge_name, **self.conn_edge_spec
        )
        points, point_specs = self._points()

        self.layers[self.points_name] = self.viewer.add_points(
            points, name=self.points_name, **point_specs.to_viewer_kwargs()
        )

        self.viewer.update_console({"catnap": self})

    def export_labels(self, fpath, name="", with_source=False):
        arr = self.layers[self.labels_name].data
        if with_source:
            io = copy(self.io)
            io.labels = arr
            io.to_hdf5(fpath, name)
        else:
            img = Image(
                arr, self.io.raw.resolution, self.io.raw.offset, self.io.raw.dims
            )
            img.to_hdf5(fpath, name or "labels", {"date": timestamp()})
