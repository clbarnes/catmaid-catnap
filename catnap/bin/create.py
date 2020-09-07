from argparse import ArgumentParser
from pathlib import Path
from functools import partial
import json
from typing import Iterable
import logging

import numpy as np
import h5py
from catpy import CatmaidClient

from .utils import parse_hdf5_path, parse_tuple, setup_logging_argv, add_verbosity
from .. import Catmaid, CatnapIO, Image
from ..utils import DEFAULT_OFFSET, DEFAULT_RESOLUTION

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
    parser.add_argument("-f", "--force", action="store_true", help="Force usage of the given offset and arguments, even if the dataset has its own which do not match")
    parser.add_argument("-t", "--transpose-attrs", action="store_true", help="Reverse offset and resolution attributes read from the source (may be necessary in some N5 datasets")
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
    return parser


def parse_args(args=None):
    parser = ArgumentParser()
    add_arguments(parser)
    add_verbosity(parser)
    return parser.parse_args(args)


def same_arrs(it: Iterable, default=None, force=False):
    last = None
    for item in it:
        if item is None:
            continue
        transformed = np.asarray(item)
        if last is not None:
            if not np.allclose(last, transformed):
                msg = "Items are not the same"
                if force:
                    logger.warn(msg)
                else:
                    raise ValueError("Items are not the same")
            if force:
                return last
        last = transformed

    if last is None:
        return default
    else:
        return last


def zarr_to_image(fpath, ds=None, offset=None, resolution=None, force=False, transpose=False):
    import zarr

    arr_or_group = zarr.open(fpath, "r")
    if not ds:
        arr = arr_or_group[:]
    else:
        arr = arr_or_group[ds][:]
    this_off = arr.attrs.get("offset")
    this_res = arr.attrs.get("resolution")

    if transpose:
        this_off = rev(this_off)
        this_res = rev(this_res)

    try:
        new_off = same_arrs([this_off, offset], DEFAULT_OFFSET, force)
        new_res = same_arrs([this_res, resolution], DEFAULT_RESOLUTION, force)
    except ValueError:
        raise ValueError(
            "Mismatch between resolution/ offset in file and explicitly given "
        )

    return Image(arr, new_res, new_off)


def n5_to_image(fpath, ds, offset=None, resolution=None, force=False, transpose=False):
    import pyn5

    f = pyn5.File(fpath, pyn5.Mode.READ_ONLY)
    arr = f[ds][:]
    this_off = arr.attrs.get("offset")
    this_res = arr.attrs.get("resolution")

    if transpose:
        this_off = rev(this_off)
        this_res = rev(this_res)

    try:
        new_off = same_arrs([this_off, offset], DEFAULT_OFFSET, force)
        new_res = same_arrs([this_res, resolution], DEFAULT_RESOLUTION, force)
    except ValueError:
        raise ValueError(
            "Mismatch between resolution/ offset in file and explicitly given "
        )

    return Image(arr, new_res, new_off)


def rev(arr):
    """Reverse array-like if not None"""
    if arr is None:
        return None
    else:
        return arr[::-1]


def hdf5_to_image(fpath, ds, offset=None, resolution=None, force=False, transpose=False):
    with h5py.File(fpath, "r") as f:
        ds = f[ds]
        this_off = ds.attrs.get("offset")
        this_res = ds.attrs.get("resolution")
        arr = ds[:]

    if transpose:
        this_off = rev(this_off)
        this_res = rev(this_res)

    try:
        new_off = same_arrs([this_off, offset], DEFAULT_OFFSET, force)
        new_res = same_arrs([this_res, resolution], DEFAULT_RESOLUTION, force)
    except ValueError:
        raise ValueError(
            "Mismatch between resolution/ offset in file and explicitly given "
        )

    return Image(arr, new_res, new_off)


def pick_reader(fpath, ds):
    p = Path(fpath)
    if p.is_file():
        logger.debug("Path is a file, assuming HDF5: %s", fpath)
        return hdf5_to_image
    elif not p.is_dir():
        raise FileNotFoundError("No file or directory found at %s", p)

    # is dir - does this handle symlinks?
    ext = p.suffix.lower()
    if ext == ".n5":
        return n5_to_image
    if ext in (".zr", ".zarr") or not ds:
        return zarr_to_image

    raise ValueError("Could not infer dataset type from path")


def read_image(fpath, ds, offset=None, resolution=None, force=False, transpose=False):
    reader = pick_reader(fpath, ds)
    return reader(fpath, ds, offset, resolution, force, transpose)


def main():
    setup_logging_argv()
    args = parse_args()

    raw_fpath, raw_name = args.input
    raw = read_image(raw_fpath, raw_name, args.offset, args.resolution, args.force, args.transpose_attrs)

    if args.label is not None:
        lab_fpath, lab_name = args.label
        if lab_fpath is None:
            lab_fpath = raw_fpath
        label = read_image(lab_fpath, lab_name, raw.offset, raw.resolution, args.force, args.transpose_attrs)
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
