"""Command line interface: ``flashder -r ref.rttm -s sys.rttm [-u all.uem]``."""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .api import score
from .io import load
from .metrics import STYLES


def _load_list(paths, scp, fmt):
    if scp:
        with open(scp, encoding="utf-8") as f:
            paths = list(paths or []) + [line.strip() for line in f if line.strip()]
    return load(paths, fmt=fmt)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="flashder", description="Fast, exact DER / JER for speaker diarization.")
    p.add_argument("-r", "--ref", nargs="+", default=[], help="reference file(s): rttm, seglst (.json), stm, lab")
    p.add_argument("-R", "--ref-scp", help="file listing reference files, one per line")
    p.add_argument("-s", "--sys", nargs="+", default=[], help="system file(s)")
    p.add_argument("-S", "--sys-scp", help="file listing system files, one per line")
    p.add_argument("-u", "--uem", help="UEM file")
    p.add_argument("--ref-format", choices=["rttm", "seglst", "stm", "lab"], help="default: from extension")
    p.add_argument("--sys-format", choices=["rttm", "seglst", "stm", "lab"], help="default: from extension")
    p.add_argument("-c", "--collar", type=float, default=0.0, help="collar in seconds on EACH side (md-eval -c)")
    p.add_argument("-1", "--skip-overlap", "--ignore-overlaps", action="store_true", help="exclude reference overlap")
    p.add_argument("--style", choices=sorted(STYLES), default="md-eval", help="conventions to reproduce")
    p.add_argument("--resolution", type=float, help="score on frames of this size (s) instead of exactly")
    p.add_argument("--no-jer", action="store_true", help="skip JER")
    p.add_argument("--jer-min-ref-dur", type=float, default=0.0)
    p.add_argument("--count-overlapping-turns", action="store_true", help="pyannote's counting of same-speaker overlap")
    p.add_argument(
        "--average",
        choices=["micro", "macro"],
        default="micro",
        help="micro: accumulate durations over all files (md-eval/dscore/pyannote); macro: mean of per-file scores",
    )
    p.add_argument("--per-file", action="store_true", help="print one row per recording")
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    p.add_argument("--digits", type=int, default=2)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = p.parse_args(argv)

    if not (args.ref or args.ref_scp) or not (args.sys or args.sys_scp):
        p.error("need reference (-r/-R) and system (-s/-S) files")

    report = score(
        _load_list(args.ref, args.ref_scp, args.ref_format),
        _load_list(args.sys, args.sys_scp, args.sys_format),
        args.uem,
        collar=args.collar,
        skip_overlap=args.skip_overlap,
        style=args.style,
        resolution=args.resolution,
        jer=not args.no_jer,
        jer_min_ref_dur=args.jer_min_ref_dur,
        count_overlapping_turns=args.count_overlapping_turns,
        average=args.average,
    )
    if args.json:
        json.dump(report.to_dict(), sys.stdout, indent=1)
        sys.stdout.write("\n")
    else:
        print(report.table(digits=args.digits, per_file=args.per_file))
    return 0


if __name__ == "__main__":
    sys.exit(main())
