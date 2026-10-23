"""Speed (and agreement) of flashder vs other diarization scorers.

    python benchmarks/bench.py --dscore-dir /path/to/dscore [--quick]

Two synthetic datasets with realistic turn structure, three evaluations:

  der         DER, no collar
  der-c-1     DER, 0.25 s collar (each side), reference overlap excluded
  der+jer     DER and JER, no collar

Python libraries are timed on scoring of in-memory data (each in its native
input format, conversion excluded). md-eval / dscore only exist as scripts, so
they are timed as whole processes, next to the flashder CLI doing the same.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

import flashder
from flashder import Segments, io

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------ data
def speaker_turns(rng, duration, n_spk, talk=0.35, mean_turn=3.0):
    """Alternating on/off renewal process per speaker -> overlapping turns."""
    out = []
    for k in range(n_spk):
        t = rng.exponential(mean_turn)
        while t < duration:
            d = rng.exponential(mean_turn) + 0.2
            out.append((t, min(t + d, duration), f"spk{k}"))
            t += d + rng.exponential(mean_turn * (1 - talk) / talk * n_spk / 2)
    return out


def corrupt(rng, turns, n_spk):
    out = []
    swap = {f"spk{k}": f"h{(k + 1) % n_spk}" for k in range(n_spk)}  # labels differ from reference
    for s, e, spk in turns:
        r = rng.random()
        if r < 0.05:  # dropped turn
            continue
        lab = swap[spk] if rng.random() > 0.08 else f"h{rng.integers(0, n_spk + 1)}"  # confusion
        s2, e2 = s + rng.normal(0, 0.2), e + rng.normal(0, 0.2)
        if e2 > s2 + 0.05:
            out.append((max(0.0, s2), e2, lab))
    for _ in range(len(turns) // 20):  # false alarms
        s = rng.uniform(0, max(e for _, e, _ in turns))
        out.append((s, s + rng.exponential(1.0) + 0.1, f"h{rng.integers(0, n_spk)}"))
    return out


def make_dataset(seed, n_files, duration, spk_range):
    rng = np.random.default_rng(seed)
    ref, hyp = {}, {}
    for i in range(n_files):
        n_spk = int(rng.integers(spk_range[0], spk_range[1] + 1))
        r = speaker_turns(rng, duration, n_spk)
        # round to ms so RTTM round trips are lossless for every tool
        ref[f"rec{i:04d}"] = Segments.from_tuples((round(s, 3), round(e, 3), k) for s, e, k in r)
        hyp[f"rec{i:04d}"] = Segments.from_tuples((round(s, 3), round(e, 3), k) for s, e, k in corrupt(rng, r, n_spk))
    uem = {f: np.array([[0.0, duration]]) for f in ref}
    return ref, hyp, uem


# ----------------------------------------------------------------- tools
def timeit(fn, repeat):
    best, out = float("inf"), None
    for _ in range(repeat):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return best, out


def bench_flashder(ref, hyp, uem, collar, skip, jer):
    rep = flashder.score(ref, hyp, uem, collar=collar, skip_overlap=skip, jer=jer)
    return rep.der, rep.jer


def prep_pyannote(ref, hyp, uem):
    from pyannote.core import Segment, Timeline

    return (
        {f: s.to_pyannote(uri=f) for f, s in ref.items()},
        {f: s.to_pyannote(uri=f) for f, s in hyp.items()},
        {f: Timeline([Segment(*u) for u in uem[f]]) for f in uem},
    )


def bench_pyannote(data, collar, skip, jer):
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    ref, hyp, uem = data
    m = DiarizationErrorRate(collar=2 * collar, skip_overlap=skip)
    j = JaccardErrorRate(collar=2 * collar, skip_overlap=skip) if jer else None
    for f in ref:
        m(ref[f], hyp[f], uem=uem[f])
        if j is not None:
            j(ref[f], hyp[f], uem=uem[f])
    return abs(m), (abs(j) if j is not None else None)


def prep_spyder(ref, hyp, uem):
    return (
        {f: [(spk, s, e) for s, e, spk in segs] for f, segs in ref.items()},
        {f: [(spk, s, e) for s, e, spk in segs] for f, segs in hyp.items()},
        {f: [tuple(u) for u in uem[f]] for f in uem},
    )


def bench_spyder(data, collar, skip, jer):
    import spyder

    ref, hyp, uem = data
    res = spyder.DER(ref, hyp, uem=uem, collar=collar, regions="nonoverlap" if skip else "all")
    return res.der if hasattr(res, "der") else list(res.values())[0].der, None


def prep_simpleder(ref, hyp, uem):
    return prep_spyder(ref, hyp, uem)[:2]


def bench_simpleder(data, collar, skip, jer):
    import simpleder

    ref, hyp = data
    # simpleder has no UEM and no multi-file accumulation: per-file, then micro-average by hand
    num = den = 0.0
    for f in ref:
        r = ref[f]
        tot = sum(e - s for _, s, e in r)
        num += simpleder.DER(r, hyp[f], collar=collar, skip_overlap=skip) * tot
        den += tot
    return num / den, None


def run_mdeval(mdeval, td, collar, skip):
    cmd = ["perl", str(mdeval), "-r", f"{td}/ref.rttm", "-s", f"{td}/hyp.rttm", "-u", f"{td}/all.uem", "-c", str(collar)]
    out = subprocess.run(cmd + (["-1"] if skip else []), capture_output=True, text=True).stdout
    return float(re.search(r"OVERALL SPEAKER DIARIZATION ERROR = ([\d.]+)", out).group(1)) / 100, None


def run_dscore(dscore, td, collar, skip):
    cmd = [sys.executable, f"{dscore}/score.py", "-r", f"{td}/ref.rttm", "-s", f"{td}/hyp.rttm", "-u", f"{td}/all.uem"]
    cmd += ["--collar", str(collar)] + (["--ignore_overlaps"] if skip else [])
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    ov = next(line for line in out.splitlines() if "OVERALL" in line).split()
    return float(ov[3]) / 100, float(ov[4]) / 100


def run_flashder_cli(td, collar, skip, jer):
    exe = shutil.which("flashder") or sys.executable
    cmd = [exe] if exe != sys.executable else [exe, "-m", "flashder.cli"]
    cmd += ["-r", f"{td}/ref.rttm", "-s", f"{td}/hyp.rttm", "-u", f"{td}/all.uem", "-c", str(collar), "--json"]
    cmd += (["-1"] if skip else []) + ([] if jer else ["--no-jer"])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    import json

    ov = json.loads(out)["*** OVERALL ***"]
    return ov["der"], ov.get("jer")


# ------------------------------------------------------------------ main
EVALS = {
    "der": dict(collar=0.0, skip=False, jer=False),
    "der-c-1": dict(collar=0.25, skip=True, jer=False),
    "der+jer": dict(collar=0.0, skip=False, jer=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dscore-dir", type=Path, help="dscore checkout (also provides md-eval-22.pl)")
    ap.add_argument("--quick", action="store_true", help="smaller datasets")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--skip", nargs="*", default=[], help="tools to skip")
    args = ap.parse_args()

    scale = 0.2 if args.quick else 1.0
    n_ch, n_ami = int(500 * scale), max(2, int(24 * scale))
    datasets = {
        f"CALLHOME-like ({n_ch} x 5 min, 2-4 spk)": make_dataset(0, n_ch, 300.0, (2, 4)),
        f"AMI-like ({n_ami} x 30 min, 4-5 spk)": make_dataset(1, n_ami, 1800.0, (4, 5)),
    }
    mdeval = args.dscore_dir / "scorelib" / "md-eval-22.pl" if args.dscore_dir else None

    rows = []
    for dname, (ref, hyp, uem) in datasets.items():
        n_turns = sum(len(s) for s in ref.values())
        print(f"\n## {dname}: {n_turns} reference turns, {sum(len(s) for s in hyp.values())} hypothesis turns")
        td = tempfile.mkdtemp()
        io.save_rttm(ref, f"{td}/ref.rttm")
        io.save_rttm(hyp, f"{td}/hyp.rttm")
        io.save_uem(uem, f"{td}/all.uem")
        prepared = {
            "pyannote.metrics": prep_pyannote(ref, hyp, uem),
            "spyder": prep_spyder(ref, hyp, uem),
            "simpleder": prep_simpleder(ref, hyp, uem),
        }

        for ename, ev in EVALS.items():
            c, sk, jer = ev["collar"], ev["skip"], ev["jer"]
            results = {}
            results["flashder"] = timeit(lambda: bench_flashder(ref, hyp, uem, c, sk, jer), args.repeat)
            if "pyannote" not in args.skip:
                results["pyannote.metrics"] = timeit(
                    lambda: bench_pyannote(prepared["pyannote.metrics"], c, sk, jer), 1
                )
            if "spyder" not in args.skip and not jer:
                results["spyder"] = timeit(lambda: bench_spyder(prepared["spyder"], c, sk, jer), args.repeat)
            if "simpleder" not in args.skip and not jer and ename == "der":
                results["simpleder"] = timeit(lambda: bench_simpleder(prepared["simpleder"], c, sk, jer), 1)
            # whole-process timings
            results["flashder CLI (process)"] = timeit(lambda: run_flashder_cli(td, c, sk, jer), 1)
            if mdeval and not jer:
                results["md-eval-22.pl (process)"] = timeit(lambda: run_mdeval(mdeval, td, c, sk), 1)
            if args.dscore_dir and (jer or ename == "der-c-1") and "dscore" not in args.skip:
                results["dscore (process)"] = timeit(lambda: run_dscore(args.dscore_dir, td, c, sk), 1)

            base = results["flashder"][0]
            print(f"\n### {ename}  (collar={c}, skip_overlap={sk}, jer={jer})")
            print(f"{'tool':26s} {'time':>10s} {'vs flashder':>12s} {'DER %':>8s} {'JER %':>8s}")
            for tool, (t, (d, j)) in results.items():
                jtxt = f"{100 * j:8.2f}" if j is not None else f"{'-':>8s}"
                print(f"{tool:26s} {t * 1000:8.1f}ms {t / base:11.1f}x {100 * d:8.2f} {jtxt}")
                rows.append((dname, ename, tool, t, d, j))

    if "frames" not in args.skip:
        bench_frames(args.repeat, quick=args.quick)
    return rows


def bench_frames(repeat, quick=False):
    """EEND-style validation: a batch of (frames, speakers) activities."""
    import torch

    from flashder import der_frames
    from flashder.torch import der_frames as der_frames_torch

    B, T, S = (16 if quick else 64), 5000, 4  # 64 chunks of 50 s at 10 ms
    rng = np.random.default_rng(0)
    flips = rng.random((B, T, S)) < 0.01
    ref = (rng.random((B, 1, S)) < 0.3) ^ (np.cumsum(flips, 1) % 2 == 1)
    noise = rng.random((B, T, S)) < 0.002
    hyp = (ref ^ (np.cumsum(noise, 1) % 2 == 1))[..., ::-1].copy()
    print(f"\n## Frame-level batch: {B} x {T} frames x {S} speakers (10 ms), collar 25 frames, DER")

    results = {}
    results["flashder numpy (loop)"] = timeit(
        lambda: sum(der_frames(ref[b], hyp[b], 0.01, collar=25).errors for b in range(B)), repeat
    )
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) + (
        ["mps"] if torch.backends.mps.is_available() else []
    )
    for dev in devices:
        r, h = torch.tensor(ref, device=dev), torch.tensor(hyp, device=dev)
        der_frames_torch(r, h, collar=25)  # warm-up

        def run():
            out = der_frames_torch(r, h, collar=25)
            return out["errors"].sum().item() * 0.01

        results[f"flashder.torch ({dev})"] = timeit(run, repeat)

    # other tools only take turns: convert outside the timed region
    segs = [(Segments.from_frames(ref[b], 0.01), Segments.from_frames(hyp[b], 0.01)) for b in range(B)]
    sp = [([(str(k), s, e) for s, e, k in r], [(str(k), s, e) for s, e, k in h]) for r, h in segs]

    def run_spyder():
        import spyder

        return sum(
            spyder.DER(r, h, uem=[(0.0, T * 0.01)], collar=0.25).der * sum(e - s for _, s, e in r) for r, h in sp
        )

    results["spyder (loop)"] = timeit(run_spyder, repeat)
    pa = [(r.to_pyannote(), h.to_pyannote()) for r, h in segs]

    def run_pyannote():
        from pyannote.core import Segment, Timeline
        from pyannote.metrics.diarization import DiarizationErrorRate

        m = DiarizationErrorRate(collar=0.5)
        for r, h in pa:
            m(r, h, uem=Timeline([Segment(0, T * 0.01)]))
        return abs(m)

    results["pyannote.metrics (loop)"] = timeit(run_pyannote, 1)

    base = results["flashder numpy (loop)"][0]
    print(f"{'tool':26s} {'time':>10s} {'vs numpy':>10s}")
    for tool, (t, _) in results.items():
        print(f"{tool:26s} {t * 1000:8.1f}ms {t / base:9.2f}x")


if __name__ == "__main__":
    main()
