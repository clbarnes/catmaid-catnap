from argparse import ArgumentParser
from pathlib import Path
import json
import logging

from catpy import CatmaidClient

from .utils import (
    parse_tuple,
    setup_logging_argv,
    add_verbosity,
    DataAddress,
    read_image,
)
from .. import Catmaid, CatnapIO

logger = logging.getLogger(__name__)


def add_catmaid_args(parser: ArgumentParser):
    grp = parser.add_argument_group("catmaid connection details")
    grp.add_argument("--base-url", help="Base CATMAID URL to make requests to")
    grp.add_argument("--project-id", type=int)
    grp.add_argument("--token", help="CATMAID user auth token")
    grp.add_argument("--auth-name", help="Username for HTTP auth, if necessary")
    grp.add_argument("--auth-pass", help="Password for HTTP auth, if necessary")
    grp.add_argument(
        "-c",
        "--credentials",
        type=Path,
        help="Path to JSON file containing credentials (command line arguments will take precedence)",
    )


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "input",
        help="Path to HDF5 dataset containing raw data, in the form '{file_path}:{dataset_path}'",
    )
    parser.add_argument(
        "output",
        help="Path to HDF5 group to write raw, annotation, and label data, in the form'{file_path}:{group_path}'. If the group path is not given, it will default to the file's root.",
    )
    parser.add_argument(
        "-o",
        "--offset",
        type=parse_tuple,
        help="Offset, in world units, of the raw data's (0, 0, 0) from the CATMAID project's (0, 0, 0), in the form 'z,y,x'. Will default to the raw dataset's 'offset' attribute if applicable, or '0,0,0' otherwise",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        type=parse_tuple,
        help="Size, in word units, of voxels in the raw data, in the form 'z,y,x'. Will default to the raw dataset's 'resolution' attribute if applicable, or '1,1,1' otherwise",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force usage of the given offset and arguments, even if the dataset has its own which do not match",
    )
    parser.add_argument(
        "-t",
        "--transpose-attrs",
        action="store_true",
        help="Reverse offset and resolution attributes read from the source (may be necessary in some N5 datasets)",
    )
    parser.add_argument(
        "--label",
        "-l",
        help="If there is existing label data, give it here in the same format as for 'input'. Offset and resolution are assumed to be identical to the raw (conflicting attributes will raise an error).",
    )
    parser.add_argument(
        "-s",
        "--seed-radius",
        type=int,
        help="Radius of the label seed squares placed at each treenode, in px",
    )
    add_catmaid_args(parser)
    return parser


def parse_args(args=None):
    parser = ArgumentParser()
    add_arguments(parser)
    add_verbosity(parser)
    return parser.parse_args(args)


def main():
    setup_logging_argv()
    args = parse_args()

    inp_add = DataAddress.from_str(args.input, slicing=...)
    raw = read_image(
        inp_add,
        args.offset,
        args.resolution,
        args.force,
        args.transpose_attrs,
    )

    if args.label is not None:
        lab_add = DataAddress.from_str(
            args.label, file_path=inp_add.file_path, slicing=...
        )
        label = read_image(
            lab_add,
            raw.offset,
            raw.resolution,
            args.force,
            args.transpose_attrs,
        )
    else:
        label = None

    if args.credentials:
        with open(args.credentials) as f:
            creds = json.load(f)
    else:
        creds = dict()

        for key in ["base_url", "project_id", "token", "auth_name", "auth_pass"]:
            val = getattr(args, key, None)
            if val is not None:
                creds[key] = val

    catmaid = Catmaid(CatmaidClient(**creds))

    io = CatnapIO.from_catmaid(catmaid, raw, label)
    if label is None:
        io.make_labels(args.seed_radius, True)

    out_addr = DataAddress.from_str(args.output, no_slice=True, object_name="/")

    io.to_hdf5(out_addr.file_path, out_addr.object_name)
