"""High level, multi-recording scoring."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import io
from .metrics import DERComponents, JERComponents, resolve_style, score_file
from .segments import Segments

__all__ = ["score", "der", "jer", "Report", "OVERALL"]


OVERALL = "*** OVERALL ***"


@dataclass
class Report:
    """Per-recording results plus the dataset-level summary.

    ``average="micro"`` (default, as md-eval / dscore / pyannote) sums the
    error and reference durations over all recordings before dividing, so long
    recordings weigh more; JER is averaged over all reference speakers of the
    dataset.  ``average="macro"`` is the unweighted mean of per-recording
    values.
    """

    files: dict[str, DERComponents] = field(default_factory=dict)
    jer_files: dict[str, JERComponents] = field(default_factory=dict)
    total: DERComponents = field(default_factory=DERComponents)
    jer_total: JERComponents | None = None
    average: str = "micro"
    # recordings included in the dataset-level numbers (md-eval skips files
    # absent from the reference)
    scored_files: list = field(default_factory=list)

    def overall(self, average: str | None = None) -> dict[str, float]:
        """Dataset-level ``der``, ``miss``, ``false_alarm``, ``confusion``
        (and ``jer``) as fractions."""
        average = average or self.average
        if average == "micro":
            t = self.total
            out = {"der": t.der, "miss": t.miss_rate, "false_alarm": t.false_alarm_rate, "confusion": t.confusion_rate}
            if self.jer_total is not None:
                out["jer"] = self.jer_total.jer
        elif average == "macro":
            fs = [self.files[f] for f in self.scored_files]
            mean = lambda vals: float(np.mean(vals)) if len(vals) else 0.0  # noqa: E731
            out = {
                "der": mean([c.der for c in fs]),
                "miss": mean([c.miss_rate for c in fs]),
                "false_alarm": mean([c.false_alarm_rate for c in fs]),
                "confusion": mean([c.confusion_rate for c in fs]),
            }
            if self.jer_total is not None:
                out["jer"] = mean([self.jer_files[f].jer for f in self.scored_files])
        else:
            raise ValueError(f"average must be 'micro' or 'macro', got {average!r}")
        return out

    @property
    def der(self) -> float:
        return self.overall()["der"]

    @property
    def jer(self) -> float | None:
        return self.overall().get("jer")

    def to_dict(self) -> dict:
        rows = {}
        for fid, c in self.files.items():
            row = c.as_dict()
            if fid in self.jer_files:
                row["jer"] = self.jer_files[fid].jer
            rows[fid] = row
        rows[OVERALL] = {**self.overall(), "average": self.average, "total": self.total.total}
        return rows

    def table(self, digits: int = 2, per_file: bool = True) -> str:
        has_jer = self.jer_total is not None
        cols = ["DER", "Miss", "FA", "Conf"] + (["JER"] if has_jer else []) + ["Scored(s)"]
        keys = ["der", "miss", "false_alarm", "confusion"] + (["jer"] if has_jer else [])
        rows = []
        if per_file:
            for fid, c in self.files.items():
                vals = [c.der, c.miss_rate, c.false_alarm_rate, c.confusion_rate]
                if has_jer:
                    vals.append(self.jer_files[fid].jer if fid in self.jer_files else float("nan"))
                rows.append([fid] + [f"{100 * v:.{digits}f}" for v in vals] + [f"{c.total:.{digits}f}"])
        ov = self.overall()
        label = OVERALL if self.average == "micro" else "*** OVERALL (macro) ***"
        rows.append([label] + [f"{100 * ov[k]:.{digits}f}" for k in keys] + [f"{self.total.total:.{digits}f}"])

        header = ["File"] + cols
        widths = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
        fmt = lambda r: "  ".join(  # noqa: E731
            str(x).ljust(w) if i == 0 else str(x).rjust(w) for i, (x, w) in enumerate(zip(r, widths))
        )
        sep = "-" * len(fmt(header))
        out = [fmt(header), sep] + [fmt(r) for r in rows[:-1]] + ([sep] if len(rows) > 1 else [])
        out.append(fmt(rows[-1]))
        return "\n".join(out)

    def __str__(self) -> str:
        return self.table(per_file=False)


# ------------------------------------------------------------ input coercion
def _as_diarization(x, name: str) -> dict[str, Segments]:
    if isinstance(x, (str, os.PathLike)) or (isinstance(x, (list, tuple)) and x and isinstance(x[0], (str, os.PathLike))):
        return io.load(x)
    if isinstance(x, Segments):
        return {"": x}
    if isinstance(x, dict):
        return {k: v if isinstance(v, Segments) else Segments.from_pyannote(v) for k, v in x.items()}
    if hasattr(x, "itertracks"):  # pyannote.core.Annotation
        return {x.uri or "": Segments.from_pyannote(x)}
    raise TypeError(f"cannot interpret {name} of type {type(x).__name__}")


def _as_uem(x, file_ids) -> dict[str, np.ndarray] | None:
    if x is None:
        return None
    if isinstance(x, (str, os.PathLike)):
        return io.load_uem(x)
    if isinstance(x, dict):
        return {k: _timeline_to_array(v) for k, v in x.items()}
    arr = _timeline_to_array(x)
    return {fid: arr for fid in file_ids}


def _timeline_to_array(x) -> np.ndarray:
    if hasattr(x, "segments_list_") or hasattr(x, "support"):  # pyannote Timeline / Segment
        segs = [x] if hasattr(x, "start") and not hasattr(x, "support") else list(x)
        return np.array([[s.start, s.end] for s in segs], dtype=np.float64).reshape(-1, 2)
    return np.asarray(x, dtype=np.float64).reshape(-1, 2)


# ---------------------------------------------------------------------- API
def score(
    reference,
    hypothesis,
    uem=None,
    *,
    collar: float = 0.0,
    skip_overlap: bool = False,
    style: str = "flashder",
    resolution: float | None = None,
    jer: bool = True,
    jer_min_ref_dur: float = 0.0,
    jer_resolution="auto",
    count_overlapping_turns: bool = False,
    average: str = "micro",
) -> Report:
    """Compute DER (and JER) over one or many recordings.

    Parameters
    ----------
    reference, hypothesis
        Path(s) to RTTM / SegLST / STM / lab files, ``{file_id: Segments}``,
        a single :class:`Segments`, or pyannote ``Annotation`` objects.
    uem
        Path to a UEM file, ``{file_id: (n, 2) array}``, or a single array
        applied to every file. Missing files use the style's default UEM.
    collar
        No-score collar in seconds on **each side** of every reference
        boundary (md-eval ``-c``, dscore ``--collar``). pyannote's ``collar``
        is the total width, i.e. pyannote ``collar=0.5`` == ``collar=0.25``
        here.
    skip_overlap
        Exclude regions where the reference has overlapping segments
        (md-eval ``-1``, dscore ``--ignore_overlaps``).
    style
        ``"flashder"`` (default) uses the FlashDER conventions;
        ``"md-eval"``, ``"dscore"`` or ``"pyannote"`` reproduce that tool. See
        :class:`flashder.metrics.Style` and ``docs/conventions.md``.
    resolution
        ``None`` (default) scores exactly in continuous time. A frame size in
        seconds (e.g. ``0.01``) discretises like dscore does for JER.
    jer
        Also compute the Jaccard error rate.
    jer_min_ref_dur
        Ignore reference speakers shorter than this (seconds) for JER.
    jer_resolution
        ``"auto"`` (default) uses ``resolution``, or the style's own JER frame
        size (10 ms for dscore). ``None`` forces exact JER, a float forces
        frames of that size.
    count_overlapping_turns
        Count a speaker ``n`` times where ``n`` of their turns overlap, as
        pyannote.metrics does for DER. Off by default: a speaker is either
        speaking or not (md-eval, dscore). Only needed to reproduce pyannote
        numbers on annotations with overlapping same-speaker turns.
    average
        ``"micro"`` (default, what md-eval / dscore / pyannote report):
        accumulate durations over the whole dataset, then divide.
        ``"macro"``: unweighted mean of per-recording DER / JER. Both are
        available afterwards through ``Report.overall("micro" | "macro")``.
    """
    if average not in ("micro", "macro"):
        raise ValueError(f"average must be 'micro' or 'macro', got {average!r}")
    st = resolve_style(style)
    ref = _as_diarization(reference, "reference")
    hyp = _as_diarization(hypothesis, "hypothesis")
    file_ids = list(dict.fromkeys(list(ref) + list(hyp)))
    uems = _as_uem(uem, file_ids)
    if uems is not None:
        missing = [f for f in file_ids if f not in uems]
        if missing:
            warnings.warn(f"{len(missing)} file(s) missing from the UEM use the default UEM, e.g. {missing[0]!r}")

    report = Report(jer_total=JERComponents() if jer else None, average=average)
    empty = Segments.empty()
    for fid in file_ids:
        d, j = score_file(
            ref.get(fid, empty),
            hyp.get(fid, empty),
            None if uems is None else uems.get(fid),
            collar=collar,
            skip_overlap=skip_overlap,
            style=st,
            resolution=resolution,
            jer=jer,
            jer_min_ref_dur=jer_min_ref_dur,
            jer_resolution=jer_resolution,
            count_overlapping_turns=count_overlapping_turns,
        )
        report.files[fid] = d
        # md-eval (and so dscore) silently ignores recordings absent from the
        # reference; pyannote counts their speech as false alarm
        if fid in ref or st.count_files_without_ref:
            report.total = report.total + d
            report.scored_files.append(fid)
        if j is not None:
            report.jer_files[fid] = j
            report.jer_total = report.jer_total + j
    return report


def der(reference, hypothesis, uem=None, **kwargs) -> float:
    """Shortcut returning the overall DER as a fraction."""
    return score(reference, hypothesis, uem, jer=False, **kwargs).der


def jer(reference, hypothesis, uem=None, **kwargs) -> float:
    """Shortcut returning the overall JER as a fraction."""
    return score(reference, hypothesis, uem, jer=True, **kwargs).jer
