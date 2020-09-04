from argparse import ArgumentParser
from contextlib import contextmanager
import sys
import logging

from .. import CatnapIO, Assessor
from ..assess import FalseMerge, FalseSplit
from .utils import setup_logging_argv, add_verbosity, parse_hdf5_path

logger = logging.getLogger(__name__)


@contextmanager
def file_or_stdout(p):
    if hasattr(p, "write"):
        yield p
    elif p == "-":
        yield sys.stdout
    else:
        with open(p, "w") as f:
            yield f


def add_arguments(parser: ArgumentParser):
    parser.add_argument(
        "input",
        type=parse_hdf5_path,
        help="Path to HDF5 group containing catnap-formatted data, in the form'{file_path}:{group_path}'. If the group path is not given, it will default to the file's root.",
    )
    msg = "Assess false {}  and write to CSV file. If '-' is given, write to stdout."
    parser.add_argument("-m", "--false-merge", help=msg.format("merges"))
    parser.add_argument("-s", "--false-split", help=msg.format("splits"))
    parser.add_argument(
        "-r",
        "--relabel",
        action="store_true",
        help="Assign each connected component a new label. Useful to assess whether there are skeletons which correctly share labels around their treenodes, but those labelled regions are not contiguous.",
    )
    return parser


def main():
    setup_logging_argv()
    parser = ArgumentParser(
        description="Merges are assessed before splits regardless of argument order."
    )
    add_verbosity(parser)
    add_arguments(parser)
    args = parser.parse_args()
    io = CatnapIO.from_hdf5(args.input[0], args.input[1] or "")
    assessor = Assessor(io)
    if args.relabel:
        assessor = assessor.relabel()

    if args.false_merge:
        logger.info("Assessing false merges")
        with file_or_stdout(args.false_merge) as f:
            print(FalseMerge.header(), file=f)
            for m in assessor.false_merges():
                print(m.as_row(), file=f)

    if args.false_split:
        logger.info("Assessing false splits")
        with file_or_stdout(args.false_split) as f:
            print(FalseSplit.header(), file=f)
            for m in assessor.false_splits():
                print(m.as_row(), file=f)
