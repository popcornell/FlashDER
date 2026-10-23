"""Diarization error rate (DER) and Jaccard error rate (JER)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import Grid, build_grid
from .segments import Segments, as_uem, union_extent

__all__ = [
    "Style",
    "STYLES",
    "DERComponents",
    "JERComponents",
    "der_from_grid",
    "jer_from_grid",
    "score_file",
]


@dataclass(frozen=True)
class Style:
    """Conventions that differ between reference scoring tools.

    default_uem
        UEM used when none is given: ``"ref"`` = extent of the reference
        (md-eval), ``"union"`` = extent of reference and hypothesis (dscore,
        pyannote).
    mapping_region
        Where the optimal speaker mapping is computed: ``"uem"`` = whole UEM,
        ignoring collars and overlap exclusion (md-eval), ``"scored"`` = only
        the scored region (pyannote).
    trim_and_merge
        Crop turns to the UEM and merge overlapping turns of the same speaker
        *before* computing collars (dscore). Turns cut by the UEM then get a
        collar at the cut point.
    jer_mapping
        ``"jaccard"`` minimises the per-speaker JER (dscore); ``"overlap"``
        maximises the overlap duration (pyannote).
    jer_region
        ``"uem"`` (dscore: no collar, no overlap exclusion) or ``"scored"``
        (pyannote: same region as DER).
    jer_resolution
        Frame size used for JER when no explicit ``resolution`` is requested.
        dscore computes DER exactly (through md-eval) but JER on 10 ms frames.
    """

    name: str
    default_uem: Literal["ref", "union"] = "ref"
    mapping_region: Literal["uem", "scored"] = "uem"
    trim_and_merge: bool = False
    jer_mapping: Literal["jaccard", "overlap"] = "jaccard"
    jer_region: Literal["uem", "scored"] = "uem"
    jer_resolution: float | None = None


STYLES = {
    "md-eval": Style("md-eval"),
    "dscore": Style("dscore", default_uem="union", trim_and_merge=True, jer_resolution=0.01),
    "pyannote": Style(
        "pyannote", default_uem="union", mapping_region="scored", jer_mapping="overlap", jer_region="scored"
    ),
}


# --------------------------------------------------------------------- DER
@dataclass
class DERComponents:
    """Durations (seconds) summed over the scored region."""

    total: float = 0.0  # scored reference speaker time
    miss: float = 0.0
    false_alarm: float = 0.0
    confusion: float = 0.0
    scored_time: float = 0.0  # duration of the scored region
    scored_speech: float = 0.0  # scored time with >= 1 reference speaker
    mapping: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def correct(self) -> float:
        return self.total - self.miss - self.confusion

    @property
    def errors(self) -> float:
        return self.miss + self.false_alarm + self.confusion

    @property
    def der(self) -> float:
        return _ratio(self.errors, self.total)

    @property
    def miss_rate(self) -> float:
        return _ratio(self.miss, self.total)

    @property
    def false_alarm_rate(self) -> float:
        return _ratio(self.false_alarm, self.total)

    @property
    def confusion_rate(self) -> float:
        return _ratio(self.confusion, self.total)

    def __add__(self, other: "DERComponents") -> "DERComponents":
        return DERComponents(
            self.total + other.total,
            self.miss + other.miss,
            self.false_alarm + other.false_alarm,
            self.confusion + other.confusion,
            self.scored_time + other.scored_time,
            self.scored_speech + other.scored_speech,
        )

    def as_dict(self) -> dict:
        return {
            "der": self.der,
            "miss": self.miss_rate,
            "false_alarm": self.false_alarm_rate,
            "confusion": self.confusion_rate,
            "total": self.total,
            "miss_time": self.miss,
            "false_alarm_time": self.false_alarm,
            "confusion_time": self.confusion,
            "scored_time": self.scored_time,
        }


def _ratio(num: float, den: float) -> float:
    if den > 0:
        return num / den
    return 0.0 if num == 0 else 1.0


def _scored_mask(grid: Grid, skip_overlap: bool) -> np.ndarray:
    m = grid.uem & ~grid.collar
    if skip_overlap:
        m &= ~grid.ref_overlap
    return m


def optimal_mapping(ref: np.ndarray, hyp: np.ndarray, weights: np.ndarray):
    """Hungarian mapping maximising co-occurrence time.

    Returns aligned index arrays ``(ref_idx, hyp_idx)``; pairs that never
    co-occur are dropped (they cannot affect any metric).
    """
    if len(ref) == 0 or len(hyp) == 0:
        return np.empty(0, int), np.empty(0, int)
    overlap = (ref * weights) @ hyp.T.astype(np.float64)
    ri, hj = linear_sum_assignment(overlap, maximize=True)
    keep = overlap[ri, hj] > 0
    return ri[keep], hj[keep]


def der_from_grid(
    grid: Grid,
    skip_overlap: bool = False,
    mapping_region: str = "uem",
    count_overlapping_turns: bool = False,
) -> DERComponents:
    """DER components of one recording.

    With ``count_overlapping_turns=True`` a speaker with ``n`` overlapping
    turns counts as ``n`` speakers (pyannote.metrics behaviour; needs a grid
    built with ``keep_turn_counts=True``).  By default a speaker counts once,
    as in md-eval and dscore.
    """
    scored = _scored_mask(grid, skip_overlap)
    map_mask = scored if mapping_region == "scored" else grid.uem

    # only cells that matter for the mapping (a superset of the scored ones)
    cols = np.flatnonzero(map_mask & (grid.weights > 0))
    if count_overlapping_turns:
        if grid.ref_turns is None:
            raise ValueError("count_overlapping_turns needs a grid built with keep_turn_counts=True")
        ref, hyp = grid.ref_turns[:, cols], grid.hyp_turns[:, cols]
    else:
        ref, hyp = grid.ref[:, cols], grid.hyp[:, cols]
    w = grid.weights[cols]
    ri, hj = optimal_mapping(ref, hyp, w)

    ws = w * scored[cols]
    n_ref = ref.sum(0, dtype=np.int32)
    n_hyp = hyp.sum(0, dtype=np.int32)
    if not len(ri):
        n_corr = np.zeros_like(n_ref)
    elif count_overlapping_turns:
        n_corr = np.minimum(ref[ri], hyp[hj]).sum(0, dtype=np.int32)
    else:
        n_corr = (ref[ri] & hyp[hj]).sum(0, dtype=np.int32)

    return DERComponents(
        total=float(ws @ n_ref),
        miss=float(ws @ np.maximum(n_ref - n_hyp, 0)),
        false_alarm=float(ws @ np.maximum(n_hyp - n_ref, 0)),
        confusion=float(ws @ (np.minimum(n_ref, n_hyp) - n_corr)),
        scored_time=float(ws.sum()),
        scored_speech=float(ws @ (n_ref > 0)),
        mapping={grid.ref_labels[i]: grid.hyp_labels[j] for i, j in zip(ri.tolist(), hj.tolist())},
    )


# --------------------------------------------------------------------- JER
@dataclass
class JERComponents:
    """Per-reference-speaker Jaccard errors (each in ``[0, 1]``).

    ``num_hyp_speakers`` is only needed for dscore's edge cases where there is
    no reference speech at all.
    """

    errors: np.ndarray = field(default_factory=lambda: np.zeros(0))
    labels: list = field(default_factory=list)
    num_hyp_speakers: int = 0
    mapping: dict = field(default_factory=dict, compare=False, repr=False)

    @property
    def num_ref_speakers(self) -> int:
        return len(self.errors)

    @property
    def jer(self) -> float:
        if self.num_ref_speakers == 0:
            return 1.0 if self.num_hyp_speakers > 0 else 0.0
        return float(np.mean(self.errors))

    def __add__(self, other: "JERComponents") -> "JERComponents":
        return JERComponents(
            np.concatenate([self.errors, other.errors]),
            self.labels + other.labels,
            self.num_hyp_speakers + other.num_hyp_speakers,
        )


def jer_from_grid(
    grid: Grid,
    region: np.ndarray | None = None,
    mapping: str = "jaccard",
    min_ref_dur: float = 0.0,
) -> JERComponents:
    """Jaccard error rate as defined for DIHARD (dscore).

    ``region`` defaults to the UEM.  ``mapping="jaccard"`` pairs speakers by
    minimising the Jaccard error (dscore), ``"overlap"`` by maximising their
    co-occurrence (pyannote).
    """
    full_uem = region is None
    region = grid.uem if full_uem else region
    cols = np.flatnonzero(region & (grid.weights > 0))
    ref, hyp, w = grid.ref[:, cols], grid.hyp[:, cols], grid.weights[cols]
    ref_dur = ref.astype(np.float64) @ w
    hyp_dur = hyp.astype(np.float64) @ w

    if full_uem and grid.ref_present is not None:
        # frame mode: like dscore, a speaker whose turns fall inside the UEM
        # counts even if they are too short to cover a single frame
        ref_keep, hyp_keep = grid.ref_present.copy(), grid.hyp_present.copy()
    else:
        ref_keep, hyp_keep = ref_dur > 0, hyp_dur > 0
    ref_keep &= ref_dur >= min_ref_dur
    r_idx, h_idx = np.flatnonzero(ref_keep), np.flatnonzero(hyp_keep)
    ref, hyp, ref_dur, hyp_dur = ref[r_idx], hyp[h_idx], ref_dur[r_idx], hyp_dur[h_idx]
    labels = [grid.ref_labels[i] for i in r_idx]

    if len(r_idx) == 0 or len(h_idx) == 0:
        return JERComponents(np.ones(len(r_idx)), labels, len(h_idx))

    inter = (ref * w) @ hyp.T.astype(np.float64)
    union = ref_dur[:, None] + hyp_dur[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        cost = 1.0 - inter / union
    cost[~np.isfinite(cost)] = 1.0

    if mapping == "jaccard":
        ri, hj = linear_sum_assignment(cost)
    elif mapping == "overlap":
        ri, hj = linear_sum_assignment(inter, maximize=True)
        keep = inter[ri, hj] > 0
        ri, hj = ri[keep], hj[keep]
    else:
        raise ValueError(f"unknown JER mapping {mapping!r}")

    errors = np.ones(len(labels))
    errors[ri] = cost[ri, hj]
    return JERComponents(
        errors,
        labels,
        len(h_idx),
        mapping={labels[i]: grid.hyp_labels[h_idx[j]] for i, j in zip(ri.tolist(), hj.tolist())},
    )


# ---------------------------------------------------------------- one file
def resolve_style(style) -> Style:
    if isinstance(style, Style):
        return style
    try:
        return STYLES[style]
    except KeyError:
        raise ValueError(f"unknown style {style!r}, choose from {sorted(STYLES)}") from None


def score_file(
    ref: Segments,
    hyp: Segments,
    uem=None,
    *,
    collar: float = 0.0,
    skip_overlap: bool = False,
    style="md-eval",
    resolution: float | None = None,
    jer: bool = True,
    jer_min_ref_dur: float = 0.0,
    jer_resolution="auto",
    count_overlapping_turns: bool = False,
) -> tuple[DERComponents, JERComponents | None]:
    """Score a single recording; see :func:`flashder.score` for parameters."""
    st = resolve_style(style)
    if uem is None:
        uem = union_extent(ref, hyp) if st.default_uem == "union" else union_extent(ref)
    uem = as_uem(uem)
    if st.trim_and_merge:
        ref = ref.crop(uem).merge_overlaps()
        hyp = hyp.crop(uem).merge_overlaps()

    grid = build_grid(ref, hyp, uem, collar=collar, resolution=resolution, keep_turn_counts=count_overlapping_turns)
    der = der_from_grid(
        grid,
        skip_overlap=skip_overlap,
        mapping_region=st.mapping_region,
        count_overlapping_turns=count_overlapping_turns,
    )
    if not jer:
        return der, None
    if jer_resolution == "auto":
        jer_resolution = resolution if resolution is not None else st.jer_resolution
    if jer_resolution != resolution:
        grid = build_grid(ref, hyp, uem, collar=collar, resolution=jer_resolution)
    region = _scored_mask(grid, skip_overlap) if st.jer_region == "scored" else None
    return der, jer_from_grid(grid, region, mapping=st.jer_mapping, min_ref_dur=jer_min_ref_dur)
