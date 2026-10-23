"""Batched frame-level DER / JER in PyTorch (CPU or GPU).

Typical use is validation of end-to-end diarization models, where reference
and hypothesis already are ``(batch, frames, speakers)`` activity tensors::

    from flashder.torch import der_frames
    out = der_frames(ref, hyp_probs > 0.5, mask=valid, collar=25)  # 25 frames each side
    der = out["errors"].sum() / out["total"].sum()

The optimal speaker mapping is solved on-device by exhaustive search over
permutations when ``max(R, H) <= max_perm_speakers`` (fast for the usual
2-8 speakers), otherwise with scipy's Hungarian solver on CPU.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

__all__ = ["der_frames", "jer_frames", "frame_collar", "optimal_permutation"]


@lru_cache(maxsize=16)
def _perms(n: int, device: str) -> torch.Tensor:
    return torch.tensor(list(itertools.permutations(range(n))), dtype=torch.long, device=device)


def _pad_speakers(x: torch.Tensor, n: int) -> torch.Tensor:
    return F.pad(x, (0, n - x.shape[-1])) if x.shape[-1] < n else x


def optimal_permutation(score: torch.Tensor, max_perm_speakers: int = 8, maximize: bool = True) -> torch.Tensor:
    """Best assignment for a batch of ``(B, n, n)`` score matrices.

    Returns ``perm`` of shape ``(B, n)``: row ``i`` is matched to column
    ``perm[:, i]``.
    """
    B, n, _ = score.shape
    if n == 0:
        return torch.zeros(B, 0, dtype=torch.long, device=score.device)
    if n <= max_perm_speakers:
        perms = _perms(n, str(score.device))  # (P, n)
        rows = torch.arange(n, device=score.device)
        total = score[:, rows, perms].sum(-1)  # (B, P)
        best = total.argmax(-1) if maximize else total.argmin(-1)
        return perms[best]
    out = []
    for s in score.detach().cpu().double().numpy():  # MPS has no float64
        _, cols = linear_sum_assignment(s, maximize=maximize)
        out.append(cols)
    return torch.as_tensor(np.stack(out), dtype=torch.long, device=score.device)


def frame_collar(ref: torch.Tensor, mask: torch.Tensor, collar: int) -> torch.Tensor:
    """``(B, T)`` bool: frames within ``collar`` frames of a reference boundary.

    Boundaries include segment starts/ends at the edges of the valid region,
    matching md-eval which puts a collar around every reference boundary.
    """
    B, T, _ = ref.shape
    if collar <= 0:
        return torch.zeros(B, T, dtype=torch.bool, device=ref.device)
    r = F.pad((ref & mask[..., None]).to(torch.int8), (0, 0, 1, 1))  # (B, T + 2, R)
    edges = (r[:, 1:] != r[:, :-1]).any(-1)  # (B, T + 1), edge b between frames b-1 and b
    # frame i excluded iff an edge lies in [i - collar + 1, i + collar]
    e = F.pad(edges.float()[:, None], (collar - 1, collar))
    return F.max_pool1d(e, kernel_size=2 * collar, stride=1)[:, 0, :T] > 0


def _prepare(ref, hyp, mask, threshold):
    ref = torch.as_tensor(ref)
    hyp = torch.as_tensor(hyp, device=ref.device)
    if ref.dim() == 2:
        ref, hyp = ref[None], hyp[None]
        mask = None if mask is None else torch.as_tensor(mask)[None]
    ref = ref > 0.5 if ref.is_floating_point() else ref.bool()
    hyp = hyp > threshold if hyp.is_floating_point() else hyp.bool()
    B, T = ref.shape[:2]
    if hyp.shape[1] != T:  # tolerate a few frames of length mismatch
        hyp = F.pad(hyp, (0, 0, 0, T - hyp.shape[1])) if hyp.shape[1] < T else hyp[:, :T]
    if mask is None:
        mask = torch.ones(B, T, dtype=torch.bool, device=ref.device)
    else:
        mask = torch.as_tensor(mask, device=ref.device).bool()
    return ref & mask[..., None], hyp, mask


@torch.no_grad()
def der_frames(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    collar: int = 0,
    skip_overlap: bool = False,
    mapping_region: str = "uem",
    threshold: float = 0.5,
    max_perm_speakers: int = 8,
) -> dict[str, torch.Tensor]:
    """DER components for a batch of frame-level activities.

    Parameters
    ----------
    ref, hyp
        ``(B, T, R)`` and ``(B, T, H)`` (or unbatched ``(T, S)``) tensors.
        Bool/integer tensors are used as is; floating tensors are thresholded
        (``ref`` at 0.5, ``hyp`` at ``threshold``).
    mask
        ``(B, T)`` valid frames (padding / UEM).
    collar
        No-score collar in frames on each side of reference boundaries.
    skip_overlap
        Ignore frames where the reference has >= 2 active speakers.
    mapping_region
        ``"uem"``: speaker mapping on all valid frames (md-eval);
        ``"scored"``: only on scored frames (pyannote).

    Returns
    -------
    dict of ``(B,)`` tensors counted in frames: ``total``, ``miss``,
    ``false_alarm``, ``confusion``, ``errors``, ``scored`` and the mapping
    ``perm`` ``(B, n)`` (reference ``i`` -> hypothesis ``perm[:, i]``,
    indices ``>= H`` mean unmapped).
    """
    ref, hyp, mask = _prepare(ref, hyp, mask, threshold)
    R, H = ref.shape[-1], hyp.shape[-1]
    n = max(R, H)

    n_ref = ref.sum(-1, dtype=torch.int32)
    n_hyp = (hyp & mask[..., None]).sum(-1, dtype=torch.int32)
    scored = mask & ~frame_collar(ref, mask, collar)
    if skip_overlap:
        scored &= n_ref < 2
    map_w = (scored if mapping_region == "scored" else mask).float()

    ref_p, hyp_p = _pad_speakers(ref, n), _pad_speakers(hyp, n)
    # co-occurrence counts; float32 is exact up to 2**24 frames
    overlap = torch.einsum("btr,bt,bth->brh", ref_p.float(), map_w, hyp_p.float())
    perm = optimal_permutation(overlap, max_perm_speakers)

    hyp_mapped = torch.gather(hyp_p, 2, perm[:, None, :].expand(-1, hyp_p.shape[1], -1))
    n_corr = (ref_p & hyp_mapped).sum(-1, dtype=torch.int32)

    w = scored.to(torch.int64)
    out = {
        "total": (w * n_ref).sum(-1),
        "miss": (w * (n_ref - n_hyp).clamp(min=0)).sum(-1),
        "false_alarm": (w * (n_hyp - n_ref).clamp(min=0)).sum(-1),
        "confusion": (w * (torch.minimum(n_ref, n_hyp) - n_corr)).sum(-1),
        "scored": w.sum(-1),
    }
    out["errors"] = out["miss"] + out["false_alarm"] + out["confusion"]
    out["perm"] = perm
    return out


@torch.no_grad()
def jer_frames(
    ref: torch.Tensor,
    hyp: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    threshold: float = 0.5,
    max_perm_speakers: int = 8,
) -> dict[str, torch.Tensor]:
    """Per-recording JER (dscore definition) for a batch of frame activities.

    Returns ``jer`` ``(B,)`` (mean over active reference speakers),
    ``speaker_errors`` ``(B, R)``, ``num_ref_speakers`` ``(B,)``.  The
    corpus-level JER is ``speaker_errors.sum() / num_ref_speakers.sum()``.
    """
    ref, hyp, mask = _prepare(ref, hyp, mask, threshold)
    hyp = hyp & mask[..., None]
    R, H = ref.shape[-1], hyp.shape[-1]
    n = max(R, H)
    ref_p, hyp_p = _pad_speakers(ref, n).float(), _pad_speakers(hyp, n).float()

    inter = torch.einsum("btr,bth->brh", ref_p, hyp_p)
    ref_dur, hyp_dur = ref_p.sum(1), hyp_p.sum(1)
    union = ref_dur[:, :, None] + hyp_dur[:, None, :] - inter
    cost = torch.where(union > 0, 1 - inter / union.clamp(min=1), torch.ones_like(union))
    perm = optimal_permutation(cost, max_perm_speakers, maximize=False)

    errors = torch.gather(cost, 2, perm[:, :, None])[..., 0][:, :R]  # (B, R)
    # padded (inactive) hypothesis columns have cost 1, i.e. unmapped speakers
    active = ref_dur[:, :R] > 0
    errors = torch.where(active, errors, torch.zeros_like(errors))
    num = active.sum(-1)
    jer = torch.where(
        num > 0,
        errors.sum(-1) / num.clamp(min=1),
        (hyp_dur.sum(-1) > 0).float(),
    )
    return {"jer": jer, "speaker_errors": errors, "num_ref_speakers": num, "perm": perm}
