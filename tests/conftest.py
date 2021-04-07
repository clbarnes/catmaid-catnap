from itertools import combinations_with_replacement
from typing import List, Tuple, Optional

import pytest
import numpy as np
import pandas as pd

from catnap import Image, CatnapIO

SHAPE = (2, 10, 10)


@pytest.fixture
def raw_img():
    r = np.random.RandomState(1991)
    arr = r.randint(0, 256, SHAPE).astype(np.uint8)
    return Image(arr)


@pytest.fixture
def label_img(raw_img: Image):
    arr = np.zeros(raw_img.array.shape, dtype=np.uint64)
    for idx, (y, x) in enumerate(
        combinations_with_replacement([(None, 5), (5, None)], 2), 1
    ):
        arr[:, slice(*y), slice(*x)] = idx
    return Image(arr, raw_img.resolution, raw_img.offset, raw_img.dims)


def paths_to_tn_table(paths: List[List[Tuple[float, float, float]]]) -> pd.DataFrame:
    """
    1 outer list per skeleton, 1 inner list per path
    """
    prev_tnid = 0
    tnids: List[int] = []
    parent_ids: List[Optional[int]] = []
    skids = []
    xs = []
    ys = []
    zs = []
    for skel_id, path in enumerate(paths, 101):
        parent_ids.append(None)
        for tn_id, zyx in enumerate(path, prev_tnid + 1):
            if len(parent_ids) == len(tnids):
                parent_ids.append(prev_tnid)
            tnids.append(tn_id)
            skids.append(skel_id)
            zs.append(zyx[0])
            ys.append(zyx[1])
            xs.append(zyx[2])

            prev_tnid = tn_id
    return pd.DataFrame.from_dict(
        {
            "treenode_id": np.array(tnids, np.uint64),
            "parent_id": pd.array(parent_ids, "UInt64"),
            "skeleton_id": np.array(skids, np.uint64),
            "z": np.array(zs),
            "y": np.array(ys),
            "x": np.array(xs),
        }
    )


@pytest.fixture
def catnap_io(raw_img: Image, label_img: Image):
    treenodes = paths_to_tn_table(
        [
            [(0, 2, 2), (0, 2, 3)],
            [(0, 2, 6), (0, 6, 6)],
            [(0, 6, 2), (0, 6, 3)],
            [(0, 8, 2), (0, 8, 3)],
        ]
    )
    connectors = pd.DataFrame([], columns=["connector_id", "z", "y", "x"])
    partners = pd.DataFrame(
        [],
        columns=["skeleton_id", "treenode_id", "connector_id", "is_presynaptic"],
    )
    return CatnapIO(raw_img, treenodes, connectors, partners, label_img)
