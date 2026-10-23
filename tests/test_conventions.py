"""The examples of docs/conventions.md.

Each example lists the value reported by every tool. flashder must reproduce
each value with the matching style; when the tools are available, the tools
themselves are run to check that the documented values are still correct.
"""

import re
import subprocess
import sys
import warnings

import pytest

from flashder import Segments, io, score


def S(*turns):
    return Segments.from_tuples(turns)


# name: (ref, hyp, uem, collar, skip_overlap, expected)
# expected: {tool: (DER, JER)}, JER None where the tool has none
EXAMPLES = {
    "1-collar-width": (
        {"r": S((0, 10, "A"))},
        {"r": S((0.2, 10, "x"))},
        {"r": [[0, 10]]},
        0.25,
        False,
        {"md-eval": (0.0, None), "dscore": (0.0, 2.0), "pyannote": (0.0, 0.0), "flashder": (0.0, 0.0)},
    ),
    "2-default-uem": (
        {"r": S((0, 10, "A"))},
        {"r": S((0, 12, "x"))},
        None,
        0.0,
        False,
        {"md-eval": (0.0, None), "dscore": (20.0, 16.67), "pyannote": (20.0, 16.67), "flashder": (20.0, 16.67)},
    ),
    "3-mapping-region": (
        {"r": S((0, 1, "A"), (2, 3, "A"), (4, 5, "A"), (6, 7, "A"), (8, 10, "B"))},
        {"r": S((0, 1, "x"), (2, 3, "x"), (4, 5, "x"), (6, 7, "x"), (8, 10, "x"))},
        {"r": [[0, 10]]},
        0.5,
        False,
        {"md-eval": (100.0, None), "dscore": (100.0, 66.67), "pyannote": (0.0, 0.0), "flashder": (0.0, 0.0)},
    ),
    "4-uem-cut": (
        {"r": S((0, 10, "A"))},
        {"r": S((0, 4.8, "x"))},
        {"r": [[0, 5]]},
        0.25,
        False,
        {"md-eval": (4.21, None), "dscore": (0.0, 4.0), "pyannote": (4.21, 4.21), "flashder": (4.21, 4.21)},
    ),
    "5-same-speaker-overlap": (
        {"r": S((0, 2, "A"), (1, 3, "A"))},
        {"r": S((0, 2.5, "x"))},
        {"r": [[0, 3]]},
        0.0,
        False,
        {"md-eval": (16.67, None), "dscore": (16.67, 16.67), "pyannote-count": (37.5, 16.67), "flashder": (16.67, 16.67)},
    ),
    "5-same-speaker-overlap-excluded": (
        {"r": S((0, 2, "A"), (1, 3, "A"))},
        {"r": S((0, 2.5, "x"))},
        {"r": [[0, 3]]},
        0.0,
        True,
        {"md-eval": (25.0, None), "dscore": (16.67, 16.67), "pyannote-count": (25.0, 25.0), "flashder": (16.67, 16.67)},
    ),
    "6-recording-without-reference": (
        {"r1": S((0, 10, "A"))},
        {"r1": S((0, 10, "x")), "r2": S((0, 5, "y"))},
        {"r1": [[0, 10]], "r2": [[0, 5]]},
        0.0,
        False,
        {"md-eval": (0.0, None), "dscore": (0.0, 0.0), "pyannote": (50.0, None), "flashder": (50.0, 0.0)},
    ),
    "7-jer-mapping": (
        {"r": S((0, 1, "A"), (0, 3, "B"))},
        {"r": S((0, 8, "x"), (2, 3, "y"))},
        {"r": [[0, 8]]},
        0.0,
        False,
        {"md-eval": (175.0, None), "dscore": (175.0, 77.08), "pyannote": (175.0, 81.25), "flashder": (175.0, 77.08)},
    ),
    "8-jer-frames": (
        {"r": S((0, 1.004, "A"))},
        {"r": S((0, 1.0, "x"))},
        {"r": [[0, 2]]},
        0.0,
        False,
        {"md-eval": (0.4, None), "dscore": (0.4, 0.99), "pyannote": (0.4, 0.4), "flashder": (0.4, 0.4)},
    ),
    "5-touching-turns": (  # point 5: splitting a turn leaves the score unchanged
        {"r": S((0, 2, "A"), (2, 4, "A"))},
        {"r": S((0, 1.9, "x"), (2.1, 4, "x"))},
        {"r": [[0, 4]]},
        0.25,
        False,
        {"md-eval": (0.0, None), "dscore": (0.0, 5.0), "pyannote": (0.0, 0.0), "flashder": (5.71, 5.71)},
    ),
}

CASES = [(name, tool) for name, ex in EXAMPLES.items() for tool in ex[5]]


@pytest.mark.parametrize("name,tool", CASES)
def test_flashder_reproduces(name, tool):
    ref, hyp, uem, collar, skip, expected = EXAMPLES[name]
    style = tool.replace("-count", "")
    rep = score(
        ref, hyp, uem, collar=collar, skip_overlap=skip, style=style, count_overlapping_turns=tool.endswith("-count")
    )
    der, jer = expected[tool]
    assert 100 * rep.der == pytest.approx(der, abs=5e-3)
    if jer is not None:
        assert 100 * rep.jer == pytest.approx(jer, abs=5e-3)


def test_collar_width_pyannote_misuse():
    pytest.importorskip("pyannote.metrics")
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate

    v = DiarizationErrorRate(collar=0.25)(
        S((0, 10, "A")).to_pyannote(), S((0.2, 10, "x")).to_pyannote(), uem=Timeline([Segment(0, 10)])
    )
    assert 100 * v == pytest.approx(0.77, abs=5e-3)
    # same value in flashder with a per-side collar of 0.125
    assert 100 * score(S((0, 10, "A")), S((0.2, 10, "x")), [[0, 10]], collar=0.125).der == pytest.approx(
        0.77, abs=5e-3
    )


def test_recording_without_reference_jer():
    ref, hyp, uem, *_ = EXAMPLES["6-recording-without-reference"]
    rep = score(ref, hyp, uem, style="dscore")
    assert rep.jer_files["r2"].jer == 1.0 and rep.files["r2"].der == 1.0
    assert rep.jer == 0.0  # r2 has no reference speaker and does not enter the overall JER


def test_subframe_turn_counts_for_dscore_jer():
    ref = {"r": S((0, 1, "A"), (1.503, 1.508, "B"))}  # B covers no 10 ms frame
    hyp = {"r": S((0, 1, "x"))}
    assert 100 * score(ref, hyp, {"r": [[0, 2]]}, style="dscore").jer == pytest.approx(50.0)


# ------------------------------------------------------- original tools
def _write(tmp_path, ref, hyp, uem):
    io.save_rttm(ref, tmp_path / "ref.rttm")
    io.save_rttm(hyp, tmp_path / "hyp.rttm")
    if uem is None:
        return []
    io.save_uem(uem, tmp_path / "all.uem")
    return ["-u", str(tmp_path / "all.uem")]


@pytest.mark.parametrize("name", [n for n, t in CASES if t == "md-eval"])
def test_md_eval_values(mdeval, tmp_path, name):
    ref, hyp, uem, collar, skip, expected = EXAMPLES[name]
    u = _write(tmp_path, ref, hyp, uem)
    cmd = ["perl", str(mdeval), "-r", str(tmp_path / "ref.rttm"), "-s", str(tmp_path / "hyp.rttm"), "-c", str(collar)]
    out = subprocess.run(cmd + u + (["-1"] if skip else []), capture_output=True, text=True).stdout
    der = float(re.search(r"OVERALL SPEAKER DIARIZATION ERROR = ([\d.]+)", out).group(1))
    assert der == pytest.approx(expected["md-eval"][0], abs=5e-3)


@pytest.mark.parametrize("name", [n for n, t in CASES if t == "dscore"])
def test_dscore_values(dscore_dir, tmp_path, name):
    pytest.importorskip("intervaltree")
    pytest.importorskip("tabulate")
    ref, hyp, uem, collar, skip, expected = EXAMPLES[name]
    u = _write(tmp_path, ref, hyp, uem)
    cmd = [sys.executable, str(dscore_dir / "score.py"), "-r", str(tmp_path / "ref.rttm")]
    cmd += ["-s", str(tmp_path / "hyp.rttm"), "--collar", str(collar)] + u + (["--ignore_overlaps"] if skip else [])
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    ov = next(line for line in out.splitlines() if "OVERALL" in line).split()
    assert float(ov[3]) == pytest.approx(expected["dscore"][0], abs=5e-3)
    assert float(ov[4]) == pytest.approx(expected["dscore"][1], abs=5e-3)
    if name == "6-recording-without-reference":
        rows = {line.split()[0]: line.split() for line in out.splitlines()[2:] if line.strip()}
        assert float(rows["r2"][1]) == 100.0 and float(rows["r2"][2]) == 100.0


@pytest.mark.parametrize("name", [n for n, t in CASES if t.startswith("pyannote")])
def test_pyannote_values(name):
    pytest.importorskip("pyannote.metrics")
    from pyannote.core import Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    ref, hyp, uem, collar, skip, expected = EXAMPLES[name]
    der, jer = next(v for k, v in expected.items() if k.startswith("pyannote"))
    m = DiarizationErrorRate(collar=2 * collar, skip_overlap=skip)
    j = JaccardErrorRate(collar=2 * collar, skip_overlap=skip)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in sorted(set(ref) | set(hyp)):
            r = ref.get(f, Segments.empty()).to_pyannote(uri=f)
            h = hyp.get(f, Segments.empty()).to_pyannote(uri=f)
            tl = Timeline([Segment(*x) for x in uem[f]]) if uem else None
            m(r, h, uem=tl)
            if jer is not None:
                j(r, h, uem=tl)
            elif f not in ref:
                # pyannote.metrics 4.0 raises, later versions report 100 percent
                try:
                    assert j(r, h, uem=tl) == pytest.approx(1.0)
                except ZeroDivisionError:
                    pass
    assert 100 * abs(m) == pytest.approx(der, abs=5e-3)
    if jer is not None:
        assert 100 * abs(j) == pytest.approx(jer, abs=5e-3)


def test_subframe_turn_dscore(dscore_dir, tmp_path):
    pytest.importorskip("intervaltree")
    pytest.importorskip("tabulate")
    u = _write(tmp_path, {"r": S((0, 1, "A"), (1.503, 1.508, "B"))}, {"r": S((0, 1, "x"))}, {"r": [[0, 2]]})
    cmd = [sys.executable, str(dscore_dir / "score.py"), "-r", str(tmp_path / "ref.rttm")]
    out = subprocess.run(cmd + ["-s", str(tmp_path / "hyp.rttm")] + u, capture_output=True, text=True).stdout
    assert float(next(line for line in out.splitlines() if "OVERALL" in line).split()[4]) == pytest.approx(50.0)
