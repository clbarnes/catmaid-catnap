from __future__ import annotations
from typing import Iterable, Iterator
from itertools import combinations

import pandas as pd
import numpy as np
from scipy.spatial import cdist
from skimage import measure

from .io import (
    CatnapIO,
    Image,
    TransformerMixin,
    serialize_treenodes,
    deserialize_treenodes,
)
from .utils import LocationOfInterest


class FalseMerge(LocationOfInterest):
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
        tns1 = treenodes[treenodes["skeleton"] == skel1, dims]
        tns2 = treenodes[treenodes["skeleton"] == skel2, dims]
        sq = cdist(tns1, tns2)
        idx1, idx2 = np.unravel_index(np.argmin(sq), sq.shape)
        loc = (tns1.iloc[idx1][dims] + tns2.iloc[idx2][dims]) / 2
        return cls(label, skel1, skel2, loc)

    @classmethod
    def from_n_skeletons(
        cls, label: int, skels: Iterable[int], treenodes: pd.DataFrame
    ) -> Iterator[FalseMerge]:
        for skel1, skel2 in combinations(skels, 2):
            yield cls.from_skeletons(label, skel1, skel2, treenodes)


class FalseSplit(LocationOfInterest):
    def __init__(self, skel, node1, node2, label1, label2, location):
        self.skel: int = skel
        self.node1: int = node1
        self.node2: int = node1
        self.label1: int = label1
        self.label2: int = label2
        super().__init__(location)

    @classmethod
    def from_edges(cls, edges) -> Iterator[FalseSplit]:
        splits = edges[edges["label"] != edges["label_parent"]]
        locs = (
            splits[["z", "y", "x"]] + splits[["z_parent", "y_parent", "x_parent"]]
        ) / 2
        for edge_row, loc_row in zip(
            splits.itertuples(index=False), locs.itertuples(index=False),
        ):
            yield cls(
                edge_row.skeleton,
                edge_row.id,
                edge_row.parent,
                edge_row.label,
                edge_row.label_parent,
                np.array(loc_row),
            )


class Assessor(TransformerMixin):
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
        in_raw_parent = (
            np.asarray(self.treenodes["in_raw_parent"].fillna(False), bool),
        )
        return self.treenodes[self.treenodes["in_raw"] & in_raw_parent]

    def false_splits(self):
        yield from FalseSplit.from_edges(self.internal_edges)

    def false_merges(self):
        tns = self.internal_treenodes
        for label in np.unique(tns["label"]):
            these = tns[tns["label"] == label]
            skels = np.unique(these["skeleton"])
            yield from FalseMerge.from_n_skeletons(skels)

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
            left_on="parent",
            right_on="id",
            suffixes=(None, "_parent"),
        )
        merged.drop("parent_parent", inplace=True)
        return deserialize_treenodes(merged)

    def _child_labels(self, merged_treenodes) -> pd.array:
        idxs = merged_treenodes[
            self.treenodes["in_raw"], ["z_px", "y_px", "x_px"],
        ].to_numpy()
        child_labels = self.io.labels[idxs]
        all_child = pd.array(np.full(len(self.treenodes), np.nan), dtype="UInt64")
        all_child[merged_treenodes["in_raw"]] = child_labels
        return all_child

    def _parent_labels(self, merged_treenodes) -> pd.array:
        in_raw = (np.asarray(merged_treenodes["in_raw_parent"].fillna(False), bool),)
        idxs = np.asarray(
            merged_treenodes[in_raw, ["z_px_parent", "y_px_parent", "x_px_parent"]],
            int,
        )
        parent_labels = self.io.labels[idxs]

        all_parent = pd.array(np.full(len(self.treenodes), np.nan), dtype="UInt64")

        all_parent[in_raw] = parent_labels
        return all_parent

    def relabel(self) -> Assessor:
        if self.io.labels is not None:
            lbl = self.io.labels
            relabelled = Image(
                measure.label(lbl.array), lbl.resolution, lbl.offset, lbl.dims,
            )
        else:
            relabelled = None

        return type(self)(
            CatnapIO(
                self.io.raw,
                self.io.treenodes,
                self.io.connectors,
                self.io.partners,
                relabelled,
            )
        )
