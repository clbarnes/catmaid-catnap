from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional, List, Dict, Any, Iterable
import logging
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor

from requests import HTTPError
from catpy.applications import CatmaidClientApplication
from catpy.applications.morphology import lol_to_df
import pandas as pd
import numpy as np

from .bbox import Bbox
from .utils import CoordZYX

logger = logging.getLogger(__name__)


DEFAULT_WORKERS = 10


def treenode_table(response):
    edit_time_dtype = None
    df = lol_to_df(
        response,
        [
            "treenode_id",
            "parent_id",
            "x",
            "y",
            "z",
            "confidence",
            "radius",
            "skeleton_id",
            "edit_time",
            "user_id",
        ],
        [
            np.uint64,
            pd.UInt64Dtype(),
            np.float64,
            np.float64,
            np.float64,
            np.int8,
            np.float64,
            np.uint64,
            edit_time_dtype,
            np.uint64,
        ],
    )
    # df.index = df.treenode_id
    return df


def connector_node_table(response):
    edit_time_dtype = None
    df = lol_to_df(
        response,
        ["connector_id", "x", "y", "z", "confidence", "edit_time", "user_id"],
        [
            np.uint64,
            np.float64,
            np.float64,
            np.float64,
            np.int8,
            edit_time_dtype,
            np.uint64,
        ],
    )
    return df


def merge_node_tables(dfs: Sequence[pd.DataFrame], drop_subset=None):
    merged = pd.concat(dfs, ignore_index=True)
    deduped = merged.drop_duplicates(subset=drop_subset)
    return deduped


def merge_treenode_tables(dfs: Sequence[pd.DataFrame]):
    df = merge_node_tables(dfs, ["treenode_id", "skeleton_id"])
    # if len(df.treenode_id) == len(np.unique(df.treenode_id)):
    #     df.index = df.treenode_id
    # else:
    #     raise ValueError("Resulting treenode table does not have unique rows")
    return df


def merge_connector_tables(dfs: Sequence[pd.DataFrame]):
    return merge_node_tables(dfs, ["connector_id"])


@dataclass
class ConnectorPartner:
    link_id: int
    partner_id: int
    confidence: int
    skeleton_id: int
    relation_id: int
    relation_name: str


@dataclass
class ConnectorDetail:
    # 'connector_id': detail[0],
    # 'x': detail[1],
    # 'y': detail[2],
    # 'z': detail[3],
    # 'confidence': detail[4],
    # 'partners': [p for p in detail[5]]
    connector_id: int
    location: CoordZYX
    confidence: int
    partners: List[ConnectorPartner]

    @classmethod
    def from_response(cls, response):
        return cls(
            response["connector_id"],
            CoordZYX(response["z"], response["y"], response["x"]),
            response["confidence"],
            [ConnectorPartner(**p) for p in response["partners"]],
        )

    @staticmethod
    def to_connector_partners_df(details: Iterable[ConnectorDetail]):
        dims = ["x", "y", "z"]
        conn_ids = []
        locs = []
        partners_dfs = []
        for det in details:
            conn_ids.append(det.connector_id)
            locs.append([det.location[d] for d in dims])
            partners_dfs.append(det.to_partners_df())

        connectors = pd.DataFrame(
            np.array(conn_ids, dtype=np.uint64), columns=["connector_id"]
        )
        connectors[dims] = pd.DataFrame(np.array(locs), columns=dims)
        first, *others = partners_dfs
        partners = first.append(list(others))
        return connectors, partners

    def to_partners_df(self):
        headers = ("skeleton_id", "treenode_id", "connector_id", "is_presynaptic")
        is_presyn = []
        ids = []
        for p in self.partners:
            is_presyn.append(p.relation_name.startswith("pre"))
            ids.append([p.skeleton_id, p.partner_id, self.connector_id])

        df = pd.DataFrame(np.array(ids, dtype=np.uint64), columns=headers[:-1])
        df[headers[-1]] = np.array(is_presyn, bool)
        return df


class Catmaid(CatmaidClientApplication):
    def nodes_in_bbox(
        self,
        bbox: Bbox,
        treenodes=True,
        connectors=True,
        splits: Sequence[int] = (2, 2, 2),
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        logger.debug("Getting nodes in bbox %s", bbox)
        data = bbox.to_catmaid()
        try:
            response = self.post((self.project_id, "/node/list"), data)
        except HTTPError as e:
            if e.errno == 504:
                logger.warning("Server timeout; splitting Bbox")
                response = {3: True}
            else:
                raise e

        if not response[3]:
            tn_df = treenode_table(response[0]) if treenodes else None
            conn_df = connector_node_table(response[1]) if connectors else None
            logger.debug("Got %s treenodes, %s connectors", len(tn_df), len(conn_df))
            return tn_df, conn_df

        # node limit reached
        logger.info("Splitting bbox into %s", splits)
        tn_dfs = []
        conn_dfs: List[pd.DataFrame] = []
        for sub_bb in bbox.split(*splits):
            tn_df, conn_df = self.nodes_in_bbox(sub_bb, treenodes, connectors, splits)
            if treenodes and tn_df is not None:
                tn_dfs.append(tn_df)
            if connectors and conn_df is not None:
                conn_dfs.append(conn_df)

        return (
            merge_treenode_tables(tn_dfs) if treenodes else None,
            merge_connector_tables(conn_dfs) if connectors else None,
        )

    def connector_detail(self, conn_id: int):
        return ConnectorDetail.from_response(
            self.get(f"{self.project_id}/connectors/{conn_id}")
        )

    def connector_detail_many(self, conn_ids, threads=DEFAULT_WORKERS):
        yield from batch(
            self.connector_detail, [to_args_kwargs(c) for c in conn_ids], threads
        )


ArgsKwargs = Tuple[Sequence, Dict[str, Any]]


def to_args_kwargs(*args, **kwargs):
    return args, kwargs


def batch(fn, args_kwargs: Iterable[ArgsKwargs], workers=DEFAULT_WORKERS):
    with ThreadPoolExecutor(workers) as exe:
        futs = [exe.submit(fn, *args, **kwargs) for args, kwargs in args_kwargs]
        for f in futs:
            yield f.result()


def get_creds() -> Path:
    try:
        return Path(os.environ["CATMAID_CREDENTIALS"])
    except KeyError:
        raise RuntimeError(
            "Use CATMAID_CREDENTIALS env var to give location of catmaid credentials file"
        )


def get_catmaid() -> Catmaid:
    return Catmaid.from_json(get_creds())
