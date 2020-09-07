from __future__ import annotations
from contextlib import contextmanager

from vispy.geometry import Rect
from napari import Viewer

from .utils import default


class Navigator:
    def __init__(self, viewer: Viewer):
        self.viewer = viewer

    @contextmanager
    def camera_state(self):
        camera = self.viewer.window.qt_viewer.view.camera
        d = camera.get_state()
        yield d
        camera.set_state(d)

    def location(self):
        z = self.viewer.dims.point[0]
        x, y = self.viewer.window.qt_viewer.view.camera.get_state()["rect"].center
        return z, y, x

    def move_to(self, x=None, y=None, z=None, scale=1) -> Navigator:
        if z is not None:
            self.viewer.dims.set_point(0, z)
        with self.camera_state() as state:
            rect = state["rect"]
            size = (rect.width * scale, rect.height * scale)
            center = (default(x, rect.center[0]), default(y, rect.center[1]))
            pos = (
                center[0] - size[0] / 2,
                center[1] - size[1] / 2,
            )
            state["rect"] = Rect(pos, size)
        return self

    def move_by(self, x=0, y=0, z=0, scale=1) -> Navigator:
        if z:
            new_z = self.viewer.dims.point + z
            self.viewer.dims.set_point(new_z)
        with self.camera_state() as state:
            rect = state["rect"]
            size = (rect.width * scale, rect.height * scale)
            center = (
                rect.center[0] + x,
                rect.center[1] + y,
            )
            pos = (
                center[0] - size[0] / 2,
                center[1] - size[1] / 2,
            )
            state["rect"] = Rect(pos, size)
        return self
