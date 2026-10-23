"""Hand-computed cases: every number here can be checked with pen and paper."""

import numpy as np
import pytest

from flashder import Segments, der_frames, score, score_file


def S(*turns):
    return Segments.from_tuples(turns)


def test_perfect_with_relabelled_speakers():
    ref = S((0, 5, "A"), (5, 10, "B"))
    hyp = S((0, 5, "x"), (5, 10, "y"))
    d, j = score_file(ref, hyp)
    assert d.der == 0 and j.jer == 0
    assert d.mapping == {"A": "x", "B": "y"}


def test_miss_false_alarm_confusion():
    ref = S((0, 10, "A"), (10, 20, "B"))
    hyp = S((0, 8, "x"), (10, 20, "x"), (20, 22, "y"))
    d, _ = score_file(ref, hyp, [[0, 22]])
    assert d.total == pytest.approx(20)
    assert d.miss == pytest.approx(2)  # 8-10
    assert d.false_alarm == pytest.approx(2)  # 20-22
    assert d.confusion == pytest.approx(8)  # x mapped to B (10 s) beats A (8 s)
    assert d.der == pytest.approx(12 / 20)


def test_overlap_counts_both_speakers():
    ref = S((0, 10, "A"), (5, 10, "B"))
    hyp = S((0, 10, "x"))
    d, _ = score_file(ref, hyp)
    assert d.total == pytest.approx(15)
    assert d.miss == pytest.approx(5)
    assert d.der == pytest.approx(5 / 15)


def test_collar_is_per_side():
    ref = S((0, 10, "A"))
    hyp = S((1, 10, "x"))
    d, _ = score_file(ref, hyp, collar=1.0)
    # collars [-1, 1] and [9, 11] leave [1, 9] scored: no error
    assert d.total == pytest.approx(8)
    assert d.der == 0


def test_skip_overlap():
    ref = S((0, 10, "A"), (5, 10, "B"))
    hyp = S((0, 10, "x"))
    d, _ = score_file(ref, hyp, skip_overlap=True)
    assert d.total == pytest.approx(5)
    assert d.der == 0


def test_default_uem_md_eval_vs_union():
    ref = S((0, 10, "A"))
    hyp = S((0, 12, "x"))
    assert score_file(ref, hyp, style="md-eval")[0].false_alarm == 0  # UEM = reference extent
    assert score_file(ref, hyp, style="pyannote")[0].false_alarm == pytest.approx(2)  # UEM = union


def test_same_speaker_overlapping_turns():
    ref = S((0, 2, "A"), (1, 3, "A"))
    hyp = S((0, 3, "x"))
    assert score_file(ref, hyp)[0].der == 0
    d, _ = score_file(ref, hyp, style="pyannote", count_overlapping_turns=True)
    assert d.der == pytest.approx(0.25)  # pyannote: A counted twice on [1, 2]


def test_empty_cases():
    ref, hyp = S((0, 5, "A")), Segments.empty()
    d, j = score_file(ref, hyp)
    assert d.der == 1.0 and j.jer == 1.0
    d, j = score_file(Segments.empty(), S((0, 5, "x")), [[0, 5]])
    assert d.total == 0 and d.der == 1.0 and j.jer == 1.0
    d, j = score_file(Segments.empty(), Segments.empty(), [[0, 5]])
    assert d.der == 0 and j.jer == 0


def test_jer_unmapped_speaker():
    ref = S((0, 10, "A"), (10, 20, "B"))
    hyp = S((0, 20, "x"))
    _, j = score_file(ref, hyp)
    # one speaker maps with jaccard 10/20, the other is unmapped
    assert j.jer == pytest.approx((0.5 + 1.0) / 2)


def test_multi_file_files_without_reference():
    ref = {"a": S((0, 10, "A"))}
    hyp = {"a": S((0, 10, "x")), "b": S((0, 5, "y"))}
    assert score(ref, hyp).der == 0  # md-eval ignores files absent from the reference
    assert score(ref, hyp, style="pyannote").total.false_alarm == pytest.approx(5)


def test_resolution_frames_close_to_exact(rng):
    from conftest import random_segments

    ref = random_segments(rng, 30, 3)
    hyp = random_segments(rng, 30, 4, prefix="h")
    exact, _ = score_file(ref, hyp)
    framed, _ = score_file(ref, hyp, resolution=0.001)
    # i * 0.001 is not exactly representable, so a few boundary frames flip
    # (same as dscore's turns_to_frames)
    assert framed.der == pytest.approx(exact.der, abs=1e-3)


def test_der_frames_numpy():
    ref = np.zeros((100, 2), bool)
    ref[:50, 0] = ref[50:, 1] = True
    hyp = ref[:, ::-1].copy()
    hyp[90:] = False
    d = der_frames(ref, hyp, frame_shift=0.01)
    assert d.total == pytest.approx(1.0)
    assert d.miss == pytest.approx(0.1)
    assert d.confusion == 0


def test_frame_collar_matches_segment_collar():
    ref = np.zeros((100, 1), bool)
    ref[20:60, 0] = True
    hyp = np.zeros((100, 1), bool)
    hyp[25:60, 0] = True
    d = der_frames(ref, hyp, collar=5)
    assert d.total == 30 and d.der == 0  # frames 15-24 and 55-64 removed


def test_micro_vs_macro_average():
    # long recording: 90 s, 9 s missed (10%); short one: 10 s, 5 s missed (50%)
    ref = {"long": S((0, 90, "A")), "short": S((0, 10, "A"))}
    hyp = {"long": S((0, 81, "x")), "short": S((0, 5, "x"))}
    micro = score(ref, hyp)
    assert micro.der == pytest.approx(14 / 100)  # accumulated durations
    macro = score(ref, hyp, average="macro")
    assert macro.der == pytest.approx((0.1 + 0.5) / 2)  # mean of per-file DERs
    assert micro.overall("macro") == macro.overall()  # both always available
    # JER: micro pools reference speakers across files, macro averages files
    assert micro.jer == pytest.approx((0.1 + 0.5) / 2)  # one speaker per file
    with pytest.raises(ValueError):
        score(ref, hyp, average="weighted")
