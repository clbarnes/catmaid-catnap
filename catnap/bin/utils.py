import sys
import logging
from typing import Tuple, Optional, Union, Iterable, Any, Mapping
import re
from pathlib import Path
from argparse import ArgumentParser
from enum import auto

from strenum import StrEnum
import numpy as np
import zarr

from ..io import Image
from ..constants import DEFAULT_OFFSET, DEFAULT_RESOLUTION

logger = logging.getLogger(__name__)


def parse_tuple_or_str(s, fn=float, dims=3) -> Union[Tuple, str]:
    try:
        return parse_tuple(s, fn, dims)
    except ValueError as e:
        if "xpected length" in str(e):
            raise
        return s


def parse_tuple(s, fn=float, dims=3) -> Tuple:
    logger.debug("Parsing tuple from %s", s)
    t = tuple(fn(d) for d in s.split(","))
    if dims is not None and len(t) != dims:
        raise ValueError(f"Expected length {dims} tuple but got {len(t)}: '{s}'")
    return t


class StoreFormat(StrEnum):
    HDF5 = auto()
    ZARR = auto()
    N5 = auto()

    @classmethod
    def from_path_name(cls, fpath: Path, name: str):
        """Get type of existing store"""
        p = Path(fpath)
        if p.is_file():
            logger.debug("Path is a file, assuming HDF5: %s", fpath)
            return cls.HDF5
        elif not p.is_dir():
            raise FileNotFoundError("No file or directory found at %s", p)

        # is dir - does this handle symlinks?
        # TODO: check metadata files
        ext = p.suffix.lower()
        if ext == ".n5":
            return cls.N5
        if ext in (".zr", ".zarr") or not name:
            return cls.ZARR

        raise ValueError("Could not infer dataset type from path")


SliceItem = Union[slice, int]
Slicing = Union[SliceItem, Tuple[SliceItem, ...]]


def parse_slicing_part(s: str) -> SliceItem:
    if not s:
        return slice(None)
    if s == "...":
        return Ellipsis  # type: ignore
    try:
        return int(s)
    except ValueError:
        pass

    if "+" in s:
        start, length = [int(i) if i else None for i in s.split("+")]
        if length is not None and length < 1:
            raise ValueError("Size must be at least 1")
        if start is None or length is None:
            stop = length
        else:
            stop = start + length
    elif ">" in s:
        start, stop = [int(i) if i else None for i in s.split(">")]
    else:
        raise ValueError("Don't know how to parse into slicing: '%s'", s)
    return slice(start, stop)


def parse_slicing(s: str) -> Slicing:
    return tuple(parse_slicing_part(p) for p in strip_whitespace(s).split(","))


def strip_whitespace(s):
    return "".join(s.split())


class DataAddress:
    slice_parse_desc = (
        "Dimensions are separated with a comma, and indices are in pixel space. "
        "For each dimension, you can use '{value}' for a single index, "
        "or '{start}>{stop}', or '{start}+{size}', or leave empty to select entire dimension. "
        "{start} and {stop} can be negative. "
        "Empty {start} starts at 0. Empty {stop} or {size} continues to the end. "
        "Ellipses can be used to fill out any number of dimensions. "
        "If there are fewer indices than there are dimensions, "
        "there is an implicit trailing ellipsis. "
        "For example, '1>-5' maps to `slice(1, -5)`; '5+10' maps to `slice(5, 15)`; "
        "'5,,3>-10,...,20+30' is equivalent to `array[5, :, 3:-10, ..., 20:50]`. "
        "Command line arguments including slices should be quoted. "
        "Some storage backends may not support all numpy-like slicing operations."
    )

    def __init__(self, file_path=None, object_name=None, slicing=None):
        self.file_path: Optional[Path] = (
            Path(file_path) if file_path is not None else None
        )
        self.object_name: Optional[str] = object_name
        self.slicing: Optional[Slicing] = slicing

    @classmethod
    def from_str(
        cls,
        s: str,
        sep=":",
        no_slice=False,
        file_path=None,
        object_name=None,
        slicing=None,
    ):
        """file_path, object_name, and slicing are defaults if not given in str"""
        logger.debug("Parsing HDF5 file path and internal path from %s", s)
        parts = s.split(sep)
        fpath = None
        oname = None
        sl = None

        if len(parts) >= 1 and parts[0]:
            fpath = Path(parts[0])
        if len(parts) >= 2 and parts[1]:
            oname = parts[1]
        if len(parts) == 3 and parts[2]:
            sl = parse_slicing(parts[2])
        if len(parts) > (2 if no_slice else 3):
            raise ValueError("Too many components in data address '%s'", s)

        return cls(fpath, oname, sl).defaults(file_path, object_name, slicing)

    def defaults(self, file_path=None, object_name=None, slicing=None):
        fp = file_path if self.file_path is None else self.file_path
        on = object_name if self.object_name is None else self.object_name
        sl = slicing if self.slicing is None else self.slicing
        return type(self)(fp, on, sl)

    def get_format(self) -> StoreFormat:
        if not self.file_path:
            raise ValueError("Requires file path")
        return StoreFormat.from_path_name(self.file_path, self.object_name)  # type: ignore


def parse_hdf5_path(
    s, sep=":", internal_default=None
) -> Tuple[Optional[Path], Optional[str]]:
    logger.debug("Parsing HDF5 file path and internal path from %s", s)
    first, *others = s.split(sep)
    fpath = Path(first) if first else None
    if len(others) == 1:
        internal = others[0]
    elif not others:
        internal = internal_default
    else:
        raise ValueError(f"Found >1 separator ({sep}) in argument: {s}")
    return fpath, internal


def setup_logging_argv(args=None, strip=False):
    if args is None:
        args = sys.argv[1:]

    out = []
    regex = re.compile("-v+")
    counter = 0
    for arg in args[1:]:
        if arg == "--verbose":
            counter += 1
        else:
            match = regex.fullmatch(arg)
            if match is None:
                out.append(arg)
            else:
                counter += len(arg) - 1

    root_lvl, dep_level = {
        0: (logging.WARNING, logging.WARNING),
        1: (logging.INFO, logging.WARNING),
        2: (logging.DEBUG, logging.WARNING),
        3: (logging.DEBUG, logging.INFO),
        4: (logging.DEBUG, logging.DEBUG),
    }.get(counter, (logging.DEBUG, logging.DEBUG))

    setup_logging(root_lvl, dep_level)
    logger.debug("Set verbosity to %s", counter)

    if strip:
        return out
    else:
        return args


def setup_logging(root_lvl, dep_lvl=None):
    logging.basicConfig(level=root_lvl)
    if dep_lvl is not None and dep_lvl != root_lvl:
        for name in [
            "requests",
            "urllib3",
            "numexpr",
            "ipykernel",
            "asyncio",
            "traitlets",
            "parso",
        ]:
            logging.getLogger(name).setLevel(dep_lvl)


def add_verbosity(parser: ArgumentParser):
    parser.add_argument(
        "-v", "--verbose", action="count", help="Increase logging verbosity"
    )


def slicing_offset(slicing, shape):
    if slicing is None:
        slicing = Ellipsis
    slicing = np.index_exp[slicing]
    ellipsis_count = sum(sl == Ellipsis for sl in slicing)

    if ellipsis_count > 1:
        raise ValueError("More than one ellipsis")
    elif ellipsis_count == 1:
        this_slicing = []
        for sl in slicing:
            if sl == Ellipsis:
                to_add = max(0, len(shape) - len(slicing) + 1)
                this_slicing.extend(slice(None) for _ in range(to_add))
            else:
                this_slicing.append(sl)
        slicing = tuple(this_slicing)
    elif ellipsis_count == 0:
        slicing += tuple(slice(None) for _ in range(len(shape) - len(slicing)))

    if len(slicing) != len(shape):
        raise ValueError(
            "Could not rectify sizes of slicing <%s> and shape <%s>", slicing, shape
        )

    offset = []
    for sl, sh in zip(slicing, shape):
        if isinstance(sl, int):
            raise ValueError("Integer slice not accepted")
        start, stop, step = sl.indices(sh)
        if step != 1:
            raise ValueError("Non-trivial steps not accepted")
        if start >= stop:
            raise ValueError("Empty slices and negative steps not allowed")
        offset.append(start)

    return tuple(offset)


def rectify_res_offset(ds_res, ds_offset, res, offset, slicing, shape, force):
    """
    Make sure that if a resolution/offset is given AND exists in the data, they are the same.
    """
    try:
        new_res = same_arrs([ds_res, res], DEFAULT_RESOLUTION, force)
        int_offset = (
            ds_offset + np.array(DEFAULT_RESOLUTION) * slicing_offset(slicing, shape)
        ).astype(int)
        new_off = same_arrs([int_offset, offset], DEFAULT_OFFSET, force)
    except ValueError:
        raise ValueError(
            "Mismatch between resolution/ offset in file and explicitly given "
        )
    return new_res, new_off


def get_res_off(
    d: Mapping,
    resolution: Union[str, Tuple[float, ...]],
    offset: Union[str, Tuple[float, ...]],
) -> Tuple[Any, Any, Any, Any]:
    """
    Account for whether the resolution/offset given is a string key, or something parseable as a tuple of numbers.
    """
    if isinstance(resolution, str):
        res_key = resolution
        out_res = None
    else:
        res_key = "resolution"
        out_res = resolution

    this_res = d.get(res_key)

    if isinstance(offset, str):
        off_key = offset
        out_off = None
    else:
        off_key = "offset"
        out_off = offset

    this_off = d.get(off_key)

    return this_res, this_off, out_res, out_off


def hdf5_to_image(
    data_address: DataAddress,
    resolution=None,
    offset=None,
    force=False,
    transpose=False,
):
    import h5py

    with h5py.File(data_address.file_path, "r") as f:
        ds = f[data_address.object_name]
        shape = ds.shape
        this_res, this_off, resolution, offset = get_res_off(
            ds.attrs, resolution, offset
        )
        arr = ds[data_address.slicing]

    if transpose:
        this_off = rev(this_off)
        this_res = rev(this_res)

    new_res, new_off = rectify_res_offset(
        this_res, this_off, resolution, offset, data_address.slicing, shape, force
    )

    return Image(arr, new_res, new_off)


def zarr_to_image(
    data_address: DataAddress,
    resolution=None,
    offset=None,
    force=False,
    transpose=False,
):
    arr_or_group = zarr.open(data_address.file_path, "r")
    if data_address.object_name is None:
        shape = arr_or_group.shape
        arr = arr_or_group[data_address.slicing]
    else:
        ds = arr_or_group[data_address.object_name]
        shape = ds.shape
        arr = ds[data_address.slicing]
    this_res, this_off, resolution, offset = get_res_off(ds.attrs, resolution, offset)

    if transpose:
        this_off = rev(this_off)
        this_res = rev(this_res)

    new_res, new_off = rectify_res_offset(
        this_res, this_off, resolution, offset, data_address.slicing, shape, force
    )

    return Image(arr, new_res, new_off)


def read_image(
    address: DataAddress, offset=None, resolution=None, force=False, transpose=False
):
    reader = {
        StoreFormat.HDF5: hdf5_to_image,
        StoreFormat.ZARR: zarr_to_image,
        StoreFormat.N5: zarr_to_image,
    }[
        StoreFormat.from_path_name(address.file_path, address.object_name)
    ]  # type: ignore
    return reader(address, offset, resolution, force, transpose)


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


def rev(arr):
    """Reverse array-like if not None"""
    if arr is None:
        return None
    else:
        return arr[::-1]
