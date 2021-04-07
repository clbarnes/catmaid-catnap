from __future__ import annotations
from typing import Iterable, Iterator, Tuple, DefaultDict
from itertools import combinations
import logging
from collections import defaultdict
import warnings

import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from skimage import measure

from .io import (
    CatnapIO,
    Image,
    TransformerMixin,
    serialize_treenodes,
    deserialize_treenodes,
)
from .utils import LocationOfInterest

logger = logging.getLogger(__name__)


class FalseMerge(LocationOfInterest):
    _headers = ("label", "skeleton1", "skeleton2") + LocationOfInterest._headers

    def __init__(self, label: int, skel1: int, skel2: int, location):
        self.label = label
        self.skel1 = skel1
        self.skel2 = skel2
        super().__init__(location)

    @classmethod
    def from_skeletons(
        cls, label: int, skel1: int, skel2: int, treenodes
    ) -> FalseMerge:
        dims = ["z", "y", "x"]
        tns1 = treenodes[treenodes["skeleton_id"] == skel1][dims]
        tns2 = treenodes[treenodes["skeleton_id"] == skel2][dims]
        sq = cdist(tns1, tns2)
        idx1, idx2 = np.unravel_index(np.argmin(sq), sq.shape)
        loc = (tns1.iloc[idx1][dims] + tns2.iloc[idx2][dims]) / 2
        return cls(label, skel1, skel2, loc)

    @classmethod
    def from_n_skeletons(
        cls, label: int, skels: Iterable[int], treenodes: pd.DataFrame
    ) -> Iterator[FalseMerge]:
        for skel1, skel2 in combinations(skels, 2):
            if skel1 == skel2:
                continue
            yield cls.from_skeletons(label, skel1, skel2, treenodes)

    @classmethod
    def to_dataframe(cls, false_merges: Iterable[FalseMerge]):
        rows = [
            [m.label, m.skel1, m.skel2, m.location[0], m.location[1], m.location[2]]
            for m in false_merges
        ]
        return pd.DataFrame(rows, columns=cls._headers)

    def as_row(self, sep=","):
        return sep.join(
            [str(self.label), str(self.skel1), str(self.skel2), super().as_row(sep)]
        )


class FalseSplit(LocationOfInterest):
    _headers = (
        "skeleton",
        "node1",
        "node2",
        "label1",
        "label2",
    ) + LocationOfInterest._headers

    def __init__(self, skel, node1, node2, label1, label2, location):
        self.skel: int = skel
        self.node1: int = node1
        self.node2: int = node2
        self.label1: int = label1
        self.label2: int = label2
        super().__init__(location)

    @classmethod
    def from_edges(cls, edges) -> Iterator[FalseSplit]:
        splits = edges[edges["label"] != edges["label_parent"]]
        locs = (
            splits[["z", "y", "x"]].to_numpy()
            + splits[["z_parent", "y_parent", "x_parent"]].to_numpy()
        ) / 2
        for edge_row, zyx in zip(
            splits.itertuples(index=False),
            locs,
        ):
            yield cls(
                edge_row.skeleton_id,
                edge_row.treenode_id,
                edge_row.parent_id,
                edge_row.label,
                edge_row.label_parent,
                zyx,
            )

    @classmethod
    def to_dataframe(cls, false_splits: Iterable[FalseSplit]):
        rows = [
            [
                m.skel,
                m.node1,
                m.node2,
                m.label1,
                m.label2,
                m.location[0],
                m.location[1],
                m.location[2],
            ]
            for m in false_splits
        ]
        return pd.DataFrame(rows, columns=cls._headers)

    def as_row(self, sep=","):
        return sep.join(
            [
                str(self.skel),
                str(self.node1),
                str(self.node2),
                str(self.label1),
                str(self.label2),
                super().as_row(sep),
            ]
        )


def incrementor(start=0):
    while True:
        yield start
        start += 1


def id_map_factory(start=0):
    incr = incrementor(start)

    def new_val():
        return next(incr)

    return new_val


class Assessor(TransformerMixin):
    """
    Attributes
    ----------
    treenodes:
      treenode_id, parent_id, skeleton_id, z, y, x, z_px, y_px, x_px, in_raw, treenode_id_parent, skeleton_id_parent, z_parent, y_parent, x_parent, z_px_parent, y_px_parent, x_px_parent, in_raw_parent, label, label_parent

    """

    _transformer_attr = "io"

    def __init__(self, catnap_io: CatnapIO):
        super().__init__()
        if catnap_io.labels is None:
            raise ValueError("Given CatnapIO must have a label volume")
        self.io = catnap_io

        self.treenodes = self._prepare_treenodes()

    @property
    def internal_treenodes(self) -> pd.DataFrame:
        return self.treenodes[self.treenodes["in_raw"]]

    @property
    def internal_edges(self) -> pd.DataFrame:
        in_raw_parent = np.asarray(self.treenodes["in_raw_parent"].fillna(False), bool)
        return self.treenodes[self.treenodes["in_raw"] & in_raw_parent]

    def false_splits(self) -> Iterator[FalseSplit]:
        yield from FalseSplit.from_edges(self.internal_edges)

    def false_merges(self) -> Iterator[FalseMerge]:
        tns = self.internal_treenodes
        for label in np.unique(tns["label"]):
            these = tns[tns["label"] == label]
            skels = np.unique(these["skeleton_id"])
            yield from FalseMerge.from_n_skeletons(label, skels, these)

    def untraced(self) -> Iterator[int]:
        """Report labelled segments with no treenodes in them; ignores 0"""
        lbl = self.treenodes["label"]
        node_labels = np.unique(lbl[~pd.isna(lbl)])
        all_labels = np.unique(self.io.labels.array)
        diff = np.setdiff1d(all_labels, node_labels, True)
        it = iter(diff)
        first = next(it)
        if first != 0:
            yield first
        yield from it

    def _prepare_treenodes(self) -> pd.DataFrame:
        tns = self._treenodes_px()
        tns["label"] = self._child_labels(tns)
        tns["label_parent"] = self._parent_labels(tns)
        return tns

    def _treenodes_px(self) -> pd.DataFrame:
        tns = serialize_treenodes(self.io.treenodes)
        world = tns[["z", "y", "x"]].to_numpy()
        px = self.world_to_px(world, True, True)
        tns[["z_px", "y_px", "x_px"]] = px
        tns["in_raw"] = self.io.coords_in_raw(world)

        merged = pd.merge(
            tns,
            tns,
            how="left",
            left_on="parent_id",
            right_on="treenode_id",
            suffixes=(None, "_parent"),
        )
        merged.drop(columns=["parent_id_parent"], inplace=True)
        return deserialize_treenodes(merged)

    def _child_labels(self, merged_treenodes) -> pd.array:
        in_raw = merged_treenodes["in_raw"]
        px_locs = np.asarray(merged_treenodes[in_raw][["z_px", "y_px", "x_px"]], int)
        child_labels = self.io.labels.array[tuple(px_locs.T)]
        all_child = pd.array(np.full(len(merged_treenodes), np.nan), dtype="UInt64")
        all_child[in_raw] = child_labels
        return all_child

    def _parent_labels(self, merged_treenodes) -> pd.array:
        in_raw = np.asarray(merged_treenodes["in_raw_parent"].fillna(False), bool)
        px_loc = np.asarray(
            merged_treenodes[in_raw][["z_px_parent", "y_px_parent", "x_px_parent"]],
            int,
        )
        parent_labels = self.io.labels.array[tuple(px_loc.T)]

        all_parent = pd.array(np.full(len(merged_treenodes), np.nan), dtype="UInt64")

        all_parent[in_raw] = parent_labels
        return all_parent

    def relabel(self) -> Assessor:
        """Create new labels for every connected component.

        Useful to distinguish between disconnected fragments with the same label.
        """
        if self.io.labels is not None:
            logger.info("Relabelling volume")
            lbl = self.io.labels
            first_zero = (lbl.array == 0).argmax()
            relabelled = measure.label(lbl.array) + 1
            to_zero = relabelled.flatten()[first_zero]
            relabelled[relabelled == to_zero] = 0
            labels = Image(
                relabelled,
                lbl.resolution,
                lbl.offset,
                lbl.dims,
            )
        else:
            labels = None

        return type(self)(
            CatnapIO(
                self.io.raw,
                self.io.treenodes,
                self.io.connectors,
                self.io.partners,
                labels,
            )
        )

    def _merge_skeleton_mappings(self) -> Iterator[Tuple[int, int]]:
        """Get best-effort mappings required for skeletons to share a label."""
        tns = self.treenodes
        new_id = self.io.labels.max_plus_one()
        seg_skels = defaultdict(set)
        for row in tns.itertuples(index=False):
            seg_skels[row.label].add(row.skeleton_id)

        skel_label: DefaultDict[int, int] = defaultdict(id_map_factory(new_id))

        for current_label, skels in sorted(seg_skels.items()):
            if len(skels) != 1:
                warnings.warn(
                    f"Skipping merge: multiple skeletons found in fragment with label {current_label} ({sorted(skels)})"
                )
                continue
            skel = list(skels).pop()
            new_label = skel_label[skel]
            yield current_label, new_label

    def merge_skeleton_labels(self):
        """Relabel segments so that skeletons share a label.

        For every skeleton, flood fill any fragment containing only treenodes from that skeleton.
        An unused new label is chosen for each skeleton.

        Segments with treenodes from more than one skeleton will be skipped.
        """
        label_arr = self.io.labels.array
        for current_label, new_label in self._merge_skeleton_mappings():
            label_arr[label_arr == current_label] = new_label
        label_img = Image(
            label_arr,
            self.io.labels.resolution,
            self.io.labels.resolution,
            self.io.labels.dims,
        )
        return type(self)(
            CatnapIO(
                self.io.raw,
                self.io.treenodes,
                self.io.connectors,
                self.io.partners,
                label_img,
            )
        )
