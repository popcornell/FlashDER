import json

import numpy as np
import pytest

from flashder import Segments, io, score

from conftest import random_segments


@pytest.fixture
def diar(rng):
    return {f"rec{i}": random_segments(rng, 20, 3, merge=True).sorted() for i in range(3)}


def same(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        x, y = a[k].sorted(), b[k].sorted()
        np.testing.assert_allclose(x.starts, y.starts, atol=1e-9)
        np.testing.assert_allclose(x.ends, y.ends, atol=1e-9)
        assert list(x.speakers) == list(y.speakers)


@pytest.mark.parametrize("fmt", ["rttm", "seglst", "stm"])
def test_roundtrip(diar, tmp_path, fmt):
    ext = {"rttm": "rttm", "seglst": "json", "stm": "stm"}[fmt]
    path = tmp_path / f"x.{ext}"
    io.save(diar, path)
    assert io.detect_format(path) == fmt
    same(diar, io.load(path))


def test_detect_without_extension(diar, tmp_path):
    for fmt in ["rttm", "seglst", "stm"]:
        p = tmp_path / f"noext_{fmt}"
        io.save(diar, p, fmt=fmt)
        assert io.detect_format(p) == fmt


def test_lab(diar, tmp_path):
    p = tmp_path / "rec0.lab"
    io.save_lab(diar["rec0"], p)
    same({"rec0": diar["rec0"]}, io.load(p))


def test_rttm_skips_non_speaker_lines(tmp_path):
    p = tmp_path / "a.rttm"
    p.write_text(
        ";; comment\n"
        "SPKR-INFO f 1 <NA> <NA> <NA> unknown A <NA> <NA>\n"
        "SPEAKER f 1 0.50 1.25 <NA> <NA> A <NA> <NA>\n"
        "LEXEME f 1 0.6 0.2 hello lex A <NA> <NA>\n"
    )
    d = io.load(p)
    assert list(d["f"]) == [(0.5, 1.75, "A")]


def test_stm_ignores_excluded_regions(tmp_path):
    p = tmp_path / "a.stm"
    p.write_text(
        ";; header\n"
        "f 1 A 0.0 2.0 <o,f0,male> hello there\n"
        "f 1 inter_segment_gap 2.0 3.0\n"
        "f 1 B 3.0 4.0 IGNORE_TIME_SEGMENT_IN_SCORING\n"
        "f 1 B 4.0 5.0 hi\n"
    )
    assert list(io.load(p)["f"]) == [(0.0, 2.0, "A"), (4.0, 5.0, "B")]


def test_seglst_string_times(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps([{"session_id": "s", "start_time": "1.5", "end_time": "2", "speaker": "A", "words": "x"}]))
    assert list(io.load(p)["s"]) == [(1.5, 2.0, "A")]


def test_kaldi(tmp_path):
    (tmp_path / "segments").write_text("u1 rec 0.0 1.0\nu2 rec 1.0 2.5\n")
    (tmp_path / "utt2spk").write_text("u1 A\nu2 B\n")
    d = io.load_kaldi(tmp_path / "segments", tmp_path / "utt2spk")
    assert list(d["rec"]) == [(0.0, 1.0, "A"), (1.0, 2.5, "B")]


def test_uem(tmp_path):
    p = tmp_path / "a.uem"
    io.save_uem({"f": np.array([[0, 1], [2, 3.5]])}, p)
    np.testing.assert_allclose(io.load_uem(p)["f"], [[0, 1], [2, 3.5]])


def test_score_across_formats(diar, tmp_path):
    io.save(diar, tmp_path / "ref.rttm")
    io.save(diar, tmp_path / "hyp.json")
    assert score(tmp_path / "ref.rttm", tmp_path / "hyp.json").der == pytest.approx(0, abs=1e-12)


def test_segments_frames_roundtrip(rng):
    act = rng.random((500, 3)) < 0.3
    segs = Segments.from_frames(act, 0.01)
    back, labels = segs.to_frames(0.01, 500, labels=[0, 1, 2])
    np.testing.assert_array_equal(back, act)


def test_merge_overlaps_strict():
    s = Segments.from_tuples([(0, 1, "A"), (1, 2, "A"), (1.5, 3, "A"), (0, 1, "B")])
    assert list(s.merge_overlaps(strict=True)) == [(0, 1, "A"), (0, 1, "B"), (1, 3, "A")]
    assert list(s.merge_overlaps(strict=False)) == [(0, 1, "B"), (0, 3, "A")]
