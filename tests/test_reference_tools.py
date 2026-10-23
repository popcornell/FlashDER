"""Agreement with pyannote.metrics, NIST md-eval-22.pl and dscore."""

import re
import subprocess
import sys
import warnings

import numpy as np
import pytest

from flashder import io, score, score_file

from conftest import random_segments

CONFIGS = [(c, so) for c in (0.0, 0.25, 0.5) for so in (False, True)]


# ---------------------------------------------------------------- pyannote
pyannote = pytest.importorskip("pyannote.metrics.diarization")


def _pyannote(metric, ref, hyp, uem, collar, skip_overlap):
    from pyannote.core import Segment, Timeline

    m = metric(collar=2 * collar, skip_overlap=skip_overlap)  # pyannote collar = total width
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m(ref.to_pyannote(), hyp.to_pyannote(), uem=Timeline([Segment(*u) for u in uem]), detailed=True)


@pytest.mark.parametrize("collar,skip_overlap", CONFIGS)
@pytest.mark.parametrize("seed", range(5))
def test_pyannote_der_jer(seed, collar, skip_overlap):
    rng = np.random.default_rng(seed)
    # pyannote counts overlapping turns of one speaker twice, merge them here
    ref = random_segments(rng, 40, 4, merge=True)
    hyp = random_segments(rng, 45, 5, prefix="h", merge=True)
    uem = [[1.0, 30.0], [32.5, 62.0]]
    p = _pyannote(pyannote.DiarizationErrorRate, ref, hyp, uem, collar, skip_overlap)
    pj = _pyannote(pyannote.JaccardErrorRate, ref, hyp, uem, collar, skip_overlap)
    d, j = score_file(ref, hyp, uem, collar=collar, skip_overlap=skip_overlap, style="pyannote")
    assert d.total == pytest.approx(p["total"], abs=1e-9)
    assert d.miss == pytest.approx(p["missed detection"], abs=1e-9)
    assert d.false_alarm == pytest.approx(p["false alarm"], abs=1e-9)
    assert d.confusion == pytest.approx(p["confusion"], abs=1e-9)
    assert j.jer == pytest.approx(pj["jaccard error rate"], abs=1e-9)


@pytest.mark.parametrize("collar,skip_overlap", CONFIGS)
@pytest.mark.parametrize("seed", range(5))
def test_pyannote_count_overlapping_turns(seed, collar, skip_overlap):
    rng = np.random.default_rng(100 + seed)
    ref = random_segments(rng, 40, 4)  # unmerged: same-speaker overlaps present
    hyp = random_segments(rng, 45, 5, prefix="h")
    uem = [[0.0, 65.0]]
    p = _pyannote(pyannote.DiarizationErrorRate, ref, hyp, uem, collar, skip_overlap)
    d, _ = score_file(
        ref, hyp, uem, collar=collar, skip_overlap=skip_overlap, style="pyannote", count_overlapping_turns=True
    )
    assert d.der == pytest.approx(p["diarization error rate"], abs=1e-9)


# ----------------------------------------------------------------- md-eval
def _write_case(tmp_path, seed, n_files=4):
    rng = np.random.default_rng(seed)
    # unmerged turns on purpose: md-eval counts a speaker once
    ref = {f"f{i}": random_segments(rng, 40, 4) for i in range(n_files)}
    hyp = {f"f{i}": random_segments(rng, 45, 5, prefix="h") for i in range(n_files)}
    uem = {f"f{i}": np.array([[1.5, 30.0], [33.0, 58.2]]) for i in range(n_files)}
    io.save_rttm(ref, tmp_path / "ref.rttm")
    io.save_rttm(hyp, tmp_path / "hyp.rttm")
    io.save_uem(uem, tmp_path / "all.uem")
    return tmp_path / "ref.rttm", tmp_path / "hyp.rttm", tmp_path / "all.uem"


def _mdeval_number(out, key):
    return float(re.search(re.escape(key) + r"\s*=\s*([\d.]+)", out).group(1))


@pytest.mark.parametrize("use_uem", [True, False])
@pytest.mark.parametrize("collar,skip_overlap", CONFIGS)
@pytest.mark.parametrize("seed", range(3))
def test_md_eval(mdeval, tmp_path, seed, collar, skip_overlap, use_uem):
    ref, hyp, uem = _write_case(tmp_path, seed)
    cmd = ["perl", str(mdeval), "-r", str(ref), "-s", str(hyp), "-c", str(collar)]
    cmd += ["-u", str(uem)] if use_uem else []
    cmd += ["-1"] if skip_overlap else []
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

    rep = score(ref, hyp, uem if use_uem else None, collar=collar, skip_overlap=skip_overlap, jer=False)
    # md-eval prints times with 2-3 decimals and DER with 2
    assert rep.total.total == pytest.approx(_mdeval_number(out, "SCORED SPEAKER TIME"), abs=6e-3)
    assert rep.total.miss == pytest.approx(_mdeval_number(out, "MISSED SPEAKER TIME"), abs=6e-3)
    assert rep.total.false_alarm == pytest.approx(_mdeval_number(out, "FALARM SPEAKER TIME"), abs=6e-3)
    assert rep.total.confusion == pytest.approx(_mdeval_number(out, "SPEAKER ERROR TIME"), abs=6e-3)
    assert 100 * rep.der == pytest.approx(_mdeval_number(out, "OVERALL SPEAKER DIARIZATION ERROR"), abs=6e-3)


# ------------------------------------------------------------------ dscore
@pytest.mark.parametrize("use_uem", [True, False])
@pytest.mark.parametrize("collar,skip_overlap", CONFIGS)
@pytest.mark.parametrize("seed", range(3))
def test_dscore(dscore_dir, tmp_path, seed, collar, skip_overlap, use_uem):
    pytest.importorskip("intervaltree")
    pytest.importorskip("tabulate")
    ref, hyp, uem = _write_case(tmp_path, 1000 + seed)
    cmd = [sys.executable, str(dscore_dir / "score.py"), "-r", str(ref), "-s", str(hyp)]
    cmd += ["--collar", str(collar), "--n_digits", "4"]
    cmd += ["-u", str(uem)] if use_uem else []
    cmd += ["--ignore_overlaps"] if skip_overlap else []
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    rows = {line.split()[0]: line.split() for line in out.splitlines()[2:] if line.strip() and "OVERALL" not in line}
    overall = next(line for line in out.splitlines() if "OVERALL" in line).split()

    rep = score(ref, hyp, uem if use_uem else None, collar=collar, skip_overlap=skip_overlap, style="dscore")
    assert 100 * rep.der == pytest.approx(float(overall[3]), abs=1e-4)
    assert 100 * rep.jer == pytest.approx(float(overall[4]), abs=1e-4)
    for fid, row in rows.items():
        assert 100 * rep.files[fid].der == pytest.approx(float(row[1]), abs=1e-4)
        assert 100 * rep.jer_files[fid].jer == pytest.approx(float(row[2]), abs=1e-4)
