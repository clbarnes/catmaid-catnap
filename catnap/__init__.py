from .version import __version__
from .catmaid_interface import Catmaid
from .io import CatnapIO, Image
from .view import CatnapViewer, PreRenderer
from napari import gui_qt

__version_info__ = tuple(int(v) for v in __version__.split("."))

__all__ = ["Catmaid", "CatnapIO", "Image", "CatnapViewer", "PreRenderer", "gui_qt"]
