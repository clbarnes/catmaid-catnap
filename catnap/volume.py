from __future__ import annotations
from typing import Iterable, Iterator, Tuple, List, NamedTuple
from random import Random
from copy import deepcopy
import logging

import numpy as np
import pandas as pd
from napari import Viewer
import networkx as nx
from tqdm import tqdm
from vispy.color import Color, ColorArray

from .utils import CoordZYX
from .catmaid_interface import ConnectorDetail
from .navigator import Navigator

logger = logging.getLogger(__name__)


class Image:
    def __init__(
        self,
        data: np.ndarray,
        translate=(0, 0, 0),
        scale=(1, 1, 1),
        name=None,
        metadata=None,
    ):
        self.data = data
        self.translate = np.array(translate, dtype=float)
        self.scale = np.array(scale, dtype=float)
        self.name = name
        self.metadata = metadata

    @property
    def end(self):
        return self.scale * self.data.shape + self.translate

    def add_to_viewer(self, viewer: Viewer):
        viewer.add_image(
            self.data,
            translate=self.translate,
            scale=self.scale,
            name=self.name,
            metadata=self.metadata,
        )
        nav = Navigator(viewer)
        nav.move_to(x=self.translate[2], y=self.translate[1])

    def to_label(self, name=None, metadata=None) -> Image:
        return Image(
            np.zeros_like(self.data, np.uint64),
            self.translate,
            self.scale,
            name,
            metadata,
        )


def points_to_px(points, resolution=(1, 1, 1), offset=(0, 0, 0), round_z=True):
    if points is None:
        return None
    p = (points - offset) / resolution
    if round_z:
        p[:, 0] = np.round(p[:, 0])
    return p


class Paths:
    def __init__(self, paths: List[np.ndarray], colours=None):
        self.paths = paths
        self.colours = colours

    def add_to_viewer(self, viewer: Viewer, **kwargs):
        viewer.add_shapes(
            self.paths, ndim=3, shape_type="path", edge_color=self.colours, **kwargs
        )

    def _px_paths(self, resolution, offset):
        out = []
        for path in self.paths:
            p = (path - offset) / resolution
            p[:, 0] = np.round(p[:, 0])
            out.append(p)
        return out


RGB = Tuple[float, float, float]
LocTuple = Tuple[float, float, float]


def nx_to_lines(g: nx.DiGraph) -> Iterator[List[LocTuple]]:
    # if None, a new branch has started: yield the last one
    to_visit: List[List[LocTuple]] = sorted(
        [node] for node, deg in g.in_degree if deg == 0
    )
    logger.debug("to_visit: %s", to_visit)
    visited_children = set()

    while to_visit:
        this_branch = to_visit.pop()
        # logger.debug("this_branch: %s", this_branch)

        while True:
            parent = this_branch[-1]
            children = sorted(set(g.successors(parent)) - visited_children)
            if len(children) == 0:
                if len(this_branch) > 1:
                    yield this_branch
                break
            elif len(children) == 1:
                this_branch.extend(children)
                visited_children.update(children)
            else:
                this_branch.append(children.pop())
                visited_children.add(this_branch[-1])
                to_visit.extend([parent, c] for c in children)


class Skeletons(Paths):
    @classmethod
    def from_treenode_table(cls, treenodes) -> Skeletons:
        logger.debug("Generating skeletons from treenode table")
        tns = treenodes.rename(columns={"id": "child"})
        merged = pd.merge(
            tns,
            tns[~tns.parent.isna()],
            left_on="child",
            right_on="parent",
            suffixes=("_child", "_parent"),
        )
        paths = []
        colours = []
        uniques = merged.skeleton_child.unique()
        for skel_id in tqdm(uniques):
            col = idx_to_colour(skel_id)
            tns = merged[merged.skeleton_child == skel_id]
            g = nx.DiGraph()
            for row in tns.itertuples():
                g.add_edge(
                    (row.z_parent, row.y_parent, row.x_parent),
                    (row.z_child, row.y_child, row.x_child),
                )

            for path in nx_to_lines(g):
                paths.append(np.array(path))
                colours.append(col)

        return cls(paths, colours)

    def to_px(self, resolution=(1, 1, 1), offset=(0, 0, 0)) -> Skeletons:
        return type(self)(self._px_paths(resolution, offset), deepcopy(self.colours),)


def idx_to_colour(idx, seed=0):
    r = Random(idx + seed)
    return (r.random(), r.random(), r.random())


class Vectors:
    def __init__(self, vectors):
        if vectors.shape[1:] != (2, 3):
            raise ValueError(
                f"vectors array must have shape (count, 2, n_dims), where this is {vectors.shape}"
            )
        self.vectors = vectors

    def add_to_viewer(self, viewer: Viewer, **kwargs):
        viewer.add_vectors(self.vectors, **kwargs)

    def _px_vectors(self, resolution, offset):
        v = self.vectors / resolution
        # Nx2xD
        v[:, 0, :] -= np.array(offset) / resolution
        v[:, :, 0] = np.round(v[:, :, 0])
        out = []
        # iterate over N, unpack over 2
        for pt, proj in v:
            z_diff = proj[0]
            if z_diff == 0:
                out.append([pt, proj])
                continue
            abs_z_diff = abs(z_diff)
            new_proj = np.array([0, *(proj[1:] / abs_z_diff)])
            z_diff3 = proj * [1, 0, 0]
            for _ in range(int(abs_z_diff)):
                out.append([pt, new_proj])
                pt = pt + z_diff3
                out.append([pt, new_proj])
                pt = pt + new_proj
        return np.array(out)


class SkeletonVectors(Vectors):
    def __init__(self, vectors, treenodes=None, skeleton_ids=None):
        """
        Vectors array must have shape ``(count, 2, n_dims)``,
        where ``vectors[:, 0, :]`` is the initial point locations (child nodes),
        and ``vectors[:, 1, :]`` is the projection towards the parent node.
        """
        super().__init__(vectors)
        if treenodes is None:
            treenodes = np.unique(
                np.concatenate(
                    [self.vectors[:, 0, :], self.vectors.sum(axis=1)], axis=0
                ),
                axis=0,
            )
        self.treenodes = treenodes
        self.skeleton_ids = skeleton_ids

    @classmethod
    def from_treenode_table(cls, treenodes) -> SkeletonVectors:
        tns = treenodes.rename(columns={"id": "child"})
        merged = pd.merge(
            tns,
            tns[~tns.parent.isna()],
            left_on="child",
            right_on="parent",
            suffixes=("_child", "_parent"),
        )
        child_points = merged[["z_child", "y_child", "x_child"]].to_numpy()
        parent_points = merged[["z_parent", "y_parent", "x_parent"]].to_numpy()
        treenodes = tns[["z", "y", "x"]].to_numpy()

        directions = parent_points - child_points
        return cls(
            np.stack([child_points, directions], axis=1),
            treenodes=treenodes,
            skeleton_ids=tns["skeleton"].to_numpy(),
        )

    def to_px(self, resolution=(1, 1, 1), offset=(0, 0, 0)) -> SkeletonVectors:
        return SkeletonVectors(
            self._px_vectors(resolution, offset),
            points_to_px(self.treenodes, resolution, offset),
            self.skeleton_ids,
        )

    def node_spec_arrays(self) -> Tuple[np.ndarray, NodeSpecArrays]:
        if self.skeleton_ids is None:
            return self.treenodes, NodeSpec().to_arrays(len(self.treenodes))

        treenodes = []
        arrs = []
        for skid in np.unique(self.skeleton_ids):
            tns = self.treenodes[self.skeleton_ids == skid]
            treenodes.append(tns)
            arrs.append(NodeSpec.from_skeleton_id(skid).to_arrays(len(tns)))
        return np.concatenate(treenodes, axis=0), NodeSpecArrays.concatenate(arrs)


class Connectors(Vectors):
    def __init__(
        self,
        vectors,
        nodes=None,
        presynapses=None,
        postsynapses=None,
        is_presynapse=None,
    ):
        """
        Vectors array must have shape ``(count, 2, n_dims)``,
        where ``vectors[:, 0, :]`` is the initial point locations (connector nodes),
        and ``vectors[:, 1, :]`` is the projection towards associated treenodes.
        """
        super().__init__(vectors)
        if is_presynapse is not None:
            if len(is_presynapse) != vectors.shape[0]:
                raise ValueError("is_presynapse is not the same length as vectors")
            nodes = np.unique(self.vectors[:, 0, :], axis=0)
            presynapses = self.vectors[is_presynapse, :, :].sum(axis=1)
            postsynapses = self.vectors[~is_presynapse, :, :].sum(axis=1)
        self.nodes = nodes
        self.presynapses = presynapses
        self.postsynapses = postsynapses

    @classmethod
    def from_responses(cls, treenodes, connector_details: Iterable[ConnectorDetail]):
        treenode_locs = {
            row.id: CoordZYX(row.z, row.y, row.x)
            for row in treenodes.itertuples(index=False)
        }
        locs = []
        dirs = []
        is_presyn = []
        for detail in connector_details:
            if detail is None:
                continue
            conn_zyx = np.array(detail.location.to_list())
            for partner in detail.partners:
                try:
                    tn_zyx = np.array(treenode_locs[partner.partner_id].to_list())
                except KeyError:
                    continue

                locs.append(conn_zyx)
                dirs.append(tn_zyx - conn_zyx)
                is_presyn.append("pre" in partner.relation_name)
        return cls(
            np.stack([locs, dirs], axis=1),
            is_presynapse=np.array(is_presyn, dtype=bool),
        )

    def to_px(self, resolution=(1, 1, 1), offset=(0, 0, 0)) -> Connectors:
        vecs = self._px_vectors(resolution, offset)
        return type(self)(
            vecs,
            *[
                points_to_px(p, resolution, offset)
                for p in (self.nodes, self.presynapses, self.postsynapses)
            ],
        )

    def node_spec_arrays(self) -> Tuple[np.ndarray, NodeSpecArrays]:
        nodes = []
        arrs = []
        # connector nodes
        nodes.append(self.nodes)
        arrs.append(DEFAULT_CONNECTOR.to_arrays(len(self.nodes)))
        # presynaptics
        nodes.append(self.presynapses)
        arrs.append(DEFAULT_PRESYNAPSE.to_arrays(len(self.presynapses)))
        # postsynaptics
        nodes.append(self.postsynapses)
        arrs.append(DEFAULT_POSTSYNAPSE.to_arrays(len(self.postsynapses)))
        return np.concatenate(nodes, axis=0), NodeSpecArrays.concatenate(arrs)


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

    @classmethod
    def from_skeleton_id(cls, skeleton_id, size=10, edge_color=(0, 0, 0, 1)):
        return cls(size, edge_color, idx_to_colour(skeleton_id))


DEFAULT_PRESYNAPSE = NodeSpec(face_color=(1, 0, 0, 1))
DEFAULT_POSTSYNAPSE = NodeSpec(face_color=(0, 1, 1, 1))
DEFAULT_CONNECTOR = NodeSpec(
    size=NodeSpec.default_size * 3, edge_color="y", face_color=(1, 1, 0, 0.3)
)
