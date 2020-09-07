from typing import Optional, Tuple
from abc import abstractmethod

from strenum import StrEnum
from enum import auto
from coordinates import spaced_coordinate
import pandas as pd
import numpy as np

CoordZYX = spaced_coordinate("CoordZYX", "zyx")

DEFAULT_OFFSET = (0, 0, 0)
DEFAULT_RESOLUTION = (1, 1, 1)


def treenodes_to_vecs(treenodes: pd.DataFrame):
    merged = pd.merge(
        treenodes, treenodes, left_on="id", right_on="parent", suffixes=("", "_parent")
    )
    child_points = merged[["z", "y", "x"]].to_numpy()
    parent_points = merged[["z_parent", "y_parent", "x_parent"]].to_numpy()
    directions = parent_points - child_points
    return np.stack([child_points, directions], axis=1)


def default(arg, default):
    if arg is None:
        return default
    return arg


class CsvRow:
    _headers: Optional[Tuple[str, ...]] = None

    @classmethod
    def header(cls, sep=","):
        if cls._headers is None:
            return None
        return sep.join(cls._headers)

    @abstractmethod
    def as_row(self, sep=","):
        pass


class LocationOfInterest(CsvRow):
    _headers = ("z", "y", "x")

    def __init__(self, location: np.ndarray):
        self.location = location

    def as_row(self, sep=","):
        return sep.join(str(i) for i in self.location)


class Viewable(StrEnum):
    TREENODE = auto()
    CONNECTOR = auto()
    SKELETON = auto()
