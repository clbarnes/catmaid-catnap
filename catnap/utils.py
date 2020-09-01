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
