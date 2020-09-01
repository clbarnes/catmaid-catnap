from argparse import ArgumentParser

from .utils import parse_hdf5_path, setup_logging
from .. import CatnapIO, CatnapViewer, gui_qt


def main():
    setup_logging()
    parser = ArgumentParser()
    parser.add_argument("input", type=parse_hdf5_path, help="Path to HDF5 group containing catnap-formatted data, in the form'{file_path}:{group_path}'. If the group path is not given, it will default to the file's root.")
    parser.add_argument("-v", "--verbose", help="Increase logging verbosity")

    args = parser.parse_args()

    io = CatnapIO.from_hdf5(args.input[0], args.input[1] or "")

    with gui_qt():
        cviewer = CatnapViewer(io)
        cviewer.show()
