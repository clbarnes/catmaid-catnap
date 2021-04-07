from argparse import ArgumentParser

from .utils import (
    setup_logging_argv,
    add_verbosity,
    DataAddress,
    hdf5_to_image,
)
from .. import CatnapIO, CatnapViewer, gui_qt


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "input",
        help="Path to HDF5 group containing catnap-formatted data, in the form '{file_path}:{group_path}'. If the group path is not given, it will default to the file's root.",
    )
    parser.add_argument(
        "-l",
        "--label",
        help="Path to HDF5 dataset containing label data (if it's not in the expected place in the input HDF5), in the form '{file_path}:{group_path}'. If the file path is not given, uses the 'input' file.",
    )


def main():
    setup_logging_argv()
    parser = ArgumentParser()
    add_verbosity(parser)
    add_arguments(parser)
    args = parser.parse_args()

    inp_add = DataAddress.from_str(args.input, no_slice=True, object_name="/")
    label_given = bool(args.label)
    io = CatnapIO.from_hdf5(inp_add.file_path, inp_add.object_name, label_given)
    if label_given:
        lab_add = DataAddress.from_str(args.label, slicing=...)
        io.set_labels(hdf5_to_image(lab_add))

    with gui_qt():
        cviewer = CatnapViewer(io)
        cviewer.show()
