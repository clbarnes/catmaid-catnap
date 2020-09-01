from argparse import ArgumentParser
from pathlib import Path
from functools import partial
import json
from typing import Iterable

import numpy as np
import h5py
from catpy import CatmaidClient

from .utils import parse_hdf5_path, parse_tuple, setup_logging
from .. import Catmaid, CatnapIO, Image
from ..utils import DEFAULT_OFFSET, DEFAULT_RESOLUTION


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


def parse_args(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        "input",
        type=parse_hdf5_path,
        help="Path to HDF5 dataset containing raw data, in the form '{file_path}:{dataset_path}'",
    )
    parser.add_argument(
        "output",
        type=partial(parse_hdf5_path, internal_default="/"),
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
        "--label",
        "-l",
        type=parse_hdf5_path,
        help="If there is existing label data, give it here in the same format as for 'input'. Offset and resolution are assumed to be identical to the raw (conflicting attributes will raise an error).",
    )
    parser.add_argument(
        "-s",
        "--seed-radius",
        type=int,
        help="Radius of the label seed squares placed at each treenode, in px",
    )
    add_catmaid_args(parser)
    parser.add_argument("-v", "--verbose", action="count", help="Increase logging verbosity")
    return parser.parse_args(args)


def run(
    input_fpath,
    input_dataset,
    output_fpath,
    output_group=None,
    offset=None,
    resolution=None,
    label_fpath=None,
    label_dataset=None,
    seed_radius=None,
    base_url=None,
    project_id=None,
    token=None,
    auth_name=None,
    auth_pass=None,
    credentials=None,
):
    pass


def same_arrs(it: Iterable, default=None):
    last = None
    for item in it:
        if item is None:
            continue
        transformed = np.asarray(item)
        if last is not None:
            if not np.allclose(last, transformed):
                raise ValueError("Items are not the same")
        last = transformed

    if last is None:
        return default
    else:
        return last


def hdf5_to_image(fpath, ds, offset=None, resolution=None):
    with h5py.File(fpath, "r") as f:
        ds = f[ds]
        this_off = ds.attrs.get("offset")
        this_res = ds.attrs.get("resolution")
        arr = ds[:]

    try:
        new_off = same_arrs([this_off, offset], DEFAULT_OFFSET)
        new_res = same_arrs([this_res, resolution], DEFAULT_RESOLUTION)
    except ValueError:
        raise ValueError(
            "Mismatch between resolution/ offset in file and explicitly given "
        )

    return Image(arr, new_res, new_off)


def main():
    setup_logging()
    args = parse_args()

    raw_fpath, raw_name = args.input
    raw = hdf5_to_image(raw_fpath, raw_name, args.offset, args.resolution)

    if args.label is not None:
        lab_fpath, lab_name = args.label
        if lab_fpath is None:
            lab_fpath = raw_fpath
        label = hdf5_to_image(lab_fpath, lab_name, raw.offset, raw.resolution)
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

    io.to_hdf5(args.output[0], args.output[1])
