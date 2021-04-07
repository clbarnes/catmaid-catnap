from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import itertools

import numpy as np
from coordinates import Coordinate

from .utils import CoordZYX


@dataclass
class Bbox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    @classmethod
    def empty(cls):
        return cls(
            float("inf"),
            -float("inf"),
            float("inf"),
            -float("inf"),
            float("inf"),
            -float("inf"),
        )

    def update(self, x, y, z, r=0):
        self.xmin = min(self.xmin, x - r)
        self.ymin = min(self.ymin, y - r)
        self.zmin = min(self.zmin, z - r)

        self.xmax = max(self.xmax, x + r)
        self.ymax = max(self.ymax, y + r)
        self.zmax = max(self.zmax, z + r)

    def __add__(self, other):
        if isinstance(other, Bbox):
            start, stop = other.to_start_stop()
            new = self.copy()
            new.update(**start)
            new.update(**stop)
            return new
        elif isinstance(other, Coordinate):
            new = self.copy()
            new.update(**other)
            return new

        return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, Bbox):
            start, stop = other.to_start_stop()
            self.update(**start)
            self.update(**stop)
            return self
        elif isinstance(other, Coordinate):
            self.update(**other)
            return self

        return NotImplemented

    def copy(self, **kwargs):
        as_d = self.as_dict()
        as_d.update(kwargs)
        return type(self)(**as_d)

    def as_dict(self):
        return {
            dim + side: getattr(self, dim + side)
            for dim in "xyz"
            for side in ("min", "max")
        }

    @classmethod
    def from_treenodes(cls, tn_table):
        return cls(
            tn_table.x.min(),
            tn_table.x.max(),
            tn_table.y.min(),
            tn_table.y.max(),
            tn_table.z.min(),
            tn_table.z.max(),
        )

    @classmethod
    def from_offset_shape(cls, offset: CoordZYX, shape: CoordZYX):
        stop = offset + shape
        return cls.from_start_stop(offset, stop)

    @classmethod
    def from_start_stop(cls, start: CoordZYX, stop: CoordZYX):
        bbox = cls.empty()
        bbox.update(**start)
        bbox.update(**stop)
        return bbox

    def to_catmaid(self):
        return {
            "left": self.xmin,
            "top": self.ymin,
            "z1": self.zmin,
            "right": self.xmax,
            "bottom": self.ymax,
            "z2": self.zmax,
        }

    def to_start_stop(self) -> Tuple[CoordZYX, CoordZYX]:
        return (
            CoordZYX(self.xmin, self.ymin, self.zmin),
            CoordZYX(self.xmax, self.ymax, self.zmax),
        )

    def to_offset_shape(self) -> Tuple[CoordZYX, CoordZYX]:
        start, stop = self.to_start_stop()
        return start, stop - start

    def split(self, x=1, y=1, z=1):
        start, stop = self.to_start_stop()
        shape = stop - start
        split_factors = CoordZYX(x=x, y=y, z=z)
        new_shape = shape / split_factors
        start_points = [
            np.linspace(start[dim], stop[dim], split_factors[dim], endpoint=False)
            for dim in "xyz"
        ]
        for xmin, ymin, zmin in itertools.product(*start_points):
            yield Bbox.from_offset_shape(CoordZYX(xmin, ymin, zmin), new_shape)

    def __contains__(self, xyz):
        if isinstance(xyz, Bbox):
            start, stop = xyz.to_start_stop()
            return start.to_list() in self and stop.to_list() in self

        x, y, z = xyz
        return (
            self.xmin <= x < self.xmax
            and self.ymin <= y < self.ymax
            and self.zmin <= z < self.zmax
        )

    def contains(self, arr):
        bools = np.logical_and(self.xmin <= arr[:, 0], arr[:, 0] < self.xmax)
        bools = np.logical_and(
            bools,
            np.logical_and(self.ymin <= arr[:, 1], arr[:, 1] < self.ymax),
        )
        return np.logical_and(
            bools,
            np.logical_and(self.zmin <= arr[:, 2], arr[:, 2] < self.zmax),
        )

    def enlarge(self, by: float) -> Bbox:
        return Bbox(
            self.xmin - by,
            self.xmax + by,
            self.ymin - by,
            self.ymax + by,
            self.zmin - by,
            self.zmax + by,
        )

    def __str__(self):
        return f"Bbox({self.__dict__})"
