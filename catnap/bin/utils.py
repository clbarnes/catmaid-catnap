import sys
import logging
from typing import Tuple, Optional
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_tuple(s, fn=float, dims=3) -> Tuple:
    logger.debug("Parsing tuple from %s", s)
    t = tuple(fn(d) for d in s.split(","))
    if dims is not None and len(t) != dims:
        raise ValueError(f"Expected length {dims} tuple but got {len(t)}: '{s}'")
    return t


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


def setup_logging(args=None, strip=False):
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

    logging.basicConfig(level=root_lvl)

    for name in [
        "requests",
        "urllib3",
        "numexpr",
        "ipykernel",
        "asyncio",
        "traitlets",
        "parso",
    ]:
        logging.getLogger(name).setLevel(dep_level)

    logger.debug("Set verbosity to %s", counter)

    if strip:
        return out
    else:
        return args
