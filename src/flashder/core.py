"""Turn segments into a common grid of boolean activity matrices.

Every scorer in this package works on a :class:`Grid`: a set of ``K`` cells
(either the elementary intervals delimited by *all* boundaries in the
problem, which makes scoring exact, or fixed-size frames), each with a
duration, plus one boolean row per speaker telling whether it is active in
that cell.  All metrics then reduce to bitwise ops, counts and one small
matrix product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segments import Segments, as_uem

__all__ = ["Grid", "build_grid", "grid_from_frames"]


@dataclass
class Grid:
    weights: np.ndarray  # (K,) float64, duration of each cell
    ref: np.ndarray  # (R, K) bool
    hyp: np.ndarray  # (H, K) bool
    uem: np.ndarray  # (K,) bool, inside the evaluation map
    collar: np.ndarray  # (K,) bool, inside a no-score collar
    ref_overlap: np.ndarray  # (K,) bool, >= 2 reference segments active
    ref_labels: list
    hyp_labels: list
    # frame mode only: speakers with positive (exact) duration inside the UEM,
    # which may differ from ``ref.any(1)`` when short turns cover no frame
    ref_present: np.ndarray | None = None
    hyp_present: np.ndarray | None = None
    # number of active turns per (speaker, cell); only kept on request, for
    # pyannote's counting of overlapping turns of the same speaker
    ref_turns: np.ndarray | None = None
    hyp_turns: np.ndarray | None = None

    @property
    def num_cells(self) -> int:
        return len(self.weights)


def _paint(rows: np.ndarray, i0: np.ndarray, i1: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    """Count, for every (row, cell), how many intervals ``[i0, i1)`` cover it."""
    width = n_cols + 1
    flat = np.bincount(rows * width + i0, minlength=n_rows * width)
    flat -= np.bincount(rows * width + i1, minlength=n_rows * width)
    return np.cumsum(flat.reshape(n_rows, width)[:, :-1], axis=1)


def _present(segs: Segments, idx: np.ndarray, n: int, uem: np.ndarray) -> np.ndarray:
    if len(segs) == 0 or len(uem) == 0:
        return np.zeros(n, dtype=bool)
    ov = np.minimum(segs.ends[:, None], uem[None, :, 1]) - np.maximum(segs.starts[:, None], uem[None, :, 0])
    pos = (ov > 0).any(1)
    out = np.zeros(n, dtype=bool)
    out[idx[pos]] = True
    return out


def build_grid(
    ref: Segments,
    hyp: Segments,
    uem,
    collar: float = 0.0,
    resolution: float | None = None,
    keep_turn_counts: bool = False,
) -> Grid:
    """Build the scoring grid for one recording.

    Parameters
    ----------
    ref, hyp
        Reference and hypothesis segments.
    uem
        ``(n, 2)`` array of evaluated regions.
    collar
        No-score collar, in seconds, on *each side* of every reference
        segment boundary (md-eval / dscore convention).
    resolution
        ``None`` for exact, boundary-driven cells.  A float gives fixed frames
        of that size where frame ``i`` covers time ``i * resolution``
        (dscore's ``turns_to_frames`` semantics).
    keep_turn_counts
        Also store how many turns of each speaker cover each cell
        (``Grid.ref_turns`` / ``Grid.hyp_turns``).
    """
    uem = as_uem(uem)
    ref = ref.select(ref.ends > ref.starts)
    hyp = hyp.select(hyp.ends > hyp.starts)

    ref_labels, ref_idx = np.unique(ref.speakers.astype(str), return_inverse=True) if len(ref) else ([], np.empty(0, int))
    hyp_labels, hyp_idx = np.unique(hyp.speakers.astype(str), return_inverse=True) if len(hyp) else ([], np.empty(0, int))
    # keep original label objects (np.unique on str only fixes the order)
    ref_labels = _original_labels(ref, ref_idx, len(ref_labels))
    hyp_labels = _original_labels(hyp, hyp_idx, len(hyp_labels))
    R, H = len(ref_labels), len(hyp_labels)

    if collar > 0 and len(ref):
        bounds = np.concatenate([ref.starts, ref.ends])
        c_start, c_end = bounds - collar, bounds + collar
    else:
        c_start = c_end = np.empty(0)

    if len(uem) == 0:
        K = 0
        edges = np.empty(0)
    elif resolution is None:
        lo, hi = uem[:, 0].min(), uem[:, 1].max()
        edges = np.unique(np.concatenate([uem.ravel(), ref.starts, ref.ends, hyp.starts, hyp.ends, c_start, c_end]))
        edges = edges[(edges >= lo) & (edges <= hi)]
        K = len(edges) - 1
    else:
        K = int(uem[:, 1].max() / resolution)
        edges = resolution * np.arange(K)

    if K <= 0:
        z = np.zeros(0, dtype=bool)
        return Grid(
            np.zeros(0), np.zeros((R, 0), bool), np.zeros((H, 0), bool), z, z, z,
            list(ref_labels), list(hyp_labels), np.zeros(R, bool), np.zeros(H, bool),
        )

    if resolution is None:
        weights = np.diff(edges)

        def index(t):
            return np.searchsorted(edges, np.clip(t, lo, hi))
    else:
        weights = np.full(K, float(resolution))

        def index(t):
            return np.searchsorted(edges, t)

    ref_count = _paint(ref_idx, index(ref.starts), index(ref.ends), R, K)
    hyp_count = _paint(hyp_idx, index(hyp.starts), index(hyp.ends), H, K)
    zeros = np.zeros(len(uem), dtype=int)
    uem_mask = _paint(zeros, index(uem[:, 0]), index(uem[:, 1]), 1, K)[0] > 0
    if len(c_start):
        z = np.zeros(len(c_start), dtype=int)
        collar_mask = _paint(z, index(c_start), index(c_end), 1, K)[0] > 0
    else:
        collar_mask = np.zeros(K, dtype=bool)

    return Grid(
        weights=weights,
        ref=ref_count > 0,
        hyp=hyp_count > 0,
        uem=uem_mask,
        collar=collar_mask,
        # md-eval and pyannote count overlapping *segments*, so overlapping
        # turns of one speaker also make an overlap region
        ref_overlap=ref_count.sum(0) >= 2,
        ref_labels=list(ref_labels),
        hyp_labels=list(hyp_labels),
        ref_present=None if resolution is None else _present(ref, ref_idx, R, uem),
        hyp_present=None if resolution is None else _present(hyp, hyp_idx, H, uem),
        ref_turns=ref_count.astype(np.int32) if keep_turn_counts else None,
        hyp_turns=hyp_count.astype(np.int32) if keep_turn_counts else None,
    )


def _original_labels(segs: Segments, idx: np.ndarray, n: int) -> list:
    out = [None] * n
    for i, lab in zip(idx.tolist(), segs.speakers.tolist()):
        if out[i] is None:
            out[i] = lab
    return out


def _frame_collar(ref: np.ndarray, valid: np.ndarray, collar: int) -> np.ndarray:
    """Frames within ``collar`` frames of a reference change point.

    ``ref`` is ``(R, T)``.  Edge ``b`` sits between frame ``b - 1`` and ``b``;
    frame ``i`` is excluded when an edge lies in ``[i - collar + 1, i + collar]``,
    i.e. ``collar`` frames are removed on each side of a boundary.
    """
    R, T = ref.shape
    if collar <= 0 or R == 0:
        return np.zeros(T, dtype=bool)
    padded = np.zeros((R, T + 2), dtype=bool)
    padded[:, 1:-1] = ref & valid
    edges = (padded[:, 1:] != padded[:, :-1]).any(0)  # (T + 1,)
    cs = np.concatenate([[0], np.cumsum(edges)])
    i = np.arange(T)
    lo = np.clip(i - collar + 1, 0, T + 1)
    hi = np.clip(i + collar + 1, 0, T + 1)
    return cs[hi] - cs[lo] > 0


def grid_from_frames(
    ref: np.ndarray,
    hyp: np.ndarray,
    frame_shift: float = 1.0,
    mask: np.ndarray | None = None,
    collar: int = 0,
) -> Grid:
    """Grid from frame-level activities of shape ``(num_frames, num_speakers)``.

    ``collar`` is given in frames, per side.  ``mask`` (``(num_frames,)``)
    plays the role of the UEM.
    """
    ref = np.asarray(ref).astype(bool).T
    hyp = np.asarray(hyp).astype(bool).T
    T = max(ref.shape[1], hyp.shape[1])
    if ref.shape[1] != hyp.shape[1]:
        # pad the shorter one, hypotheses are often a few frames off
        ref = np.pad(ref, ((0, 0), (0, T - ref.shape[1])))
        hyp = np.pad(hyp, ((0, 0), (0, T - hyp.shape[1])))
    valid = np.ones(T, dtype=bool) if mask is None else np.asarray(mask).astype(bool)
    ref = ref & valid
    return Grid(
        weights=np.full(T, float(frame_shift)),
        ref=ref,
        hyp=hyp,
        uem=valid,
        collar=_frame_collar(ref, valid, collar),
        ref_overlap=ref.sum(0) >= 2,
        ref_labels=list(range(ref.shape[0])),
        hyp_labels=list(range(hyp.shape[0])),
    )
