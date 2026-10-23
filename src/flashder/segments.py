"""Light-weight containers for speaker segments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, Union

import numpy as np

__all__ = ["Segments", "Diarization", "as_uem", "union_extent"]


@dataclass
class Segments:
    """Speaker segments of a single recording.

    Stored as three parallel arrays (``starts``, ``ends``, ``speakers``) so
    that every operation downstream is vectorised.
    """

    starts: np.ndarray
    ends: np.ndarray
    speakers: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=object))

    def __post_init__(self):
        self.starts = np.asarray(self.starts, dtype=np.float64).reshape(-1)
        self.ends = np.asarray(self.ends, dtype=np.float64).reshape(-1)
        self.speakers = np.asarray(self.speakers, dtype=object).reshape(-1)
        if not (len(self.starts) == len(self.ends) == len(self.speakers)):
            raise ValueError("starts, ends and speakers must have the same length")

    # ------------------------------------------------------------------ build
    @classmethod
    def empty(cls) -> "Segments":
        return cls(np.empty(0), np.empty(0), np.empty(0, dtype=object))

    @classmethod
    def from_tuples(cls, tuples: Iterable[Sequence]) -> "Segments":
        """Build from ``(start, end, speaker)`` tuples."""
        tuples = list(tuples)
        if not tuples:
            return cls.empty()
        starts, ends, speakers = zip(*tuples)
        return cls(np.array(starts, float), np.array(ends, float), np.array(speakers, dtype=object))

    @classmethod
    def from_pyannote(cls, annotation) -> "Segments":
        """Build from a ``pyannote.core.Annotation``."""
        return cls.from_tuples(
            (seg.start, seg.end, label) for seg, _, label in annotation.itertracks(yield_label=True)
        )

    @classmethod
    def from_frames(cls, activity: np.ndarray, frame_shift: float, labels=None, offset: float = 0.0) -> "Segments":
        """Build from a ``(num_frames, num_speakers)`` binary activity matrix."""
        activity = np.asarray(activity).astype(bool)
        n_frames, n_spk = activity.shape
        labels = list(range(n_spk)) if labels is None else list(labels)
        padded = np.zeros((n_frames + 2, n_spk), dtype=np.int8)
        padded[1:-1] = activity
        d = np.diff(padded, axis=0)
        # nonzero on the transposed array yields speaker-major order, so onsets
        # and offsets pair up one-to-one
        spk, on = np.nonzero(d.T == 1)
        _, off = np.nonzero(d.T == -1)
        return cls(
            offset + on * frame_shift,
            offset + off * frame_shift,
            np.array([labels[i] for i in spk], dtype=object),
        )

    # ------------------------------------------------------------------ views
    def __len__(self) -> int:
        return len(self.starts)

    def __iter__(self):
        return zip(self.starts.tolist(), self.ends.tolist(), self.speakers.tolist())

    @property
    def labels(self) -> list:
        return sorted(set(self.speakers.tolist()), key=str)

    def to_pyannote(self, uri: str | None = None):
        from pyannote.core import Annotation, Segment

        ann = Annotation(uri=uri)
        for i, (s, e, spk) in enumerate(self):
            ann[Segment(s, e), i] = spk
        return ann

    def to_frames(self, frame_shift: float, num_frames: int | None = None, labels=None):
        """Return ``(activity, labels)`` with frame ``i`` active iff
        ``start <= i * frame_shift < end``."""
        labels = self.labels if labels is None else list(labels)
        index = {lab: i for i, lab in enumerate(labels)}
        if num_frames is None:
            num_frames = int(np.ceil(self.ends.max() / frame_shift)) if len(self) else 0
        times = frame_shift * np.arange(num_frames)
        act = np.zeros((num_frames, len(labels)), dtype=bool)
        b = np.searchsorted(times, self.starts)
        e = np.searchsorted(times, self.ends)
        for bi, ei, spk in zip(b, e, self.speakers):
            act[bi:ei, index[spk]] = True
        return act, labels

    # ------------------------------------------------------------- transforms
    def select(self, mask) -> "Segments":
        return Segments(self.starts[mask], self.ends[mask], self.speakers[mask])

    def sorted(self) -> "Segments":
        order = np.lexsort((self.ends, self.starts))
        return self.select(order)

    def crop(self, uem: np.ndarray) -> "Segments":
        """Intersect every segment with the UEM regions (``(n, 2)`` array)."""
        uem = as_uem(uem)
        if len(self) == 0 or len(uem) == 0:
            return Segments.empty()
        s = np.maximum(self.starts[:, None], uem[None, :, 0])
        e = np.minimum(self.ends[:, None], uem[None, :, 1])
        keep = e > s
        seg_idx, _ = np.nonzero(keep)
        return Segments(s[keep], e[keep], self.speakers[seg_idx])

    def merge_overlaps(self, strict: bool = True) -> "Segments":
        """Merge overlapping segments of the same speaker.

        With ``strict=True`` (dscore/intervaltree behaviour) touching segments
        are kept separate; with ``strict=False`` they are merged too.
        """
        if len(self) == 0:
            return self
        out_s, out_e, out_k = [], [], []
        for spk in self.labels:
            m = self.speakers == spk
            s, e = self.starts[m], self.ends[m]
            order = np.lexsort((e, s))
            s, e = s[order], e[order]
            run_end = np.maximum.accumulate(e)
            # a new run starts where the segment begins after everything before it
            new = np.ones(len(s), dtype=bool)
            if strict:
                new[1:] = s[1:] >= run_end[:-1]
            else:
                new[1:] = s[1:] > run_end[:-1]
            starts = s[new]
            ends = np.maximum.reduceat(e, np.flatnonzero(new))
            out_s.append(starts)
            out_e.append(ends)
            out_k.append(np.full(len(starts), spk, dtype=object))
        return Segments(np.concatenate(out_s), np.concatenate(out_e), np.concatenate(out_k)).sorted()

    def extent(self) -> tuple[float, float] | None:
        if len(self) == 0:
            return None
        return float(self.starts.min()), float(self.ends.max())


Diarization = Mapping[str, Segments]
UEMLike = Union[np.ndarray, Sequence[Sequence[float]], None]


def as_uem(uem: UEMLike) -> np.ndarray:
    if uem is None:
        return np.empty((0, 2))
    uem = np.asarray(uem, dtype=np.float64).reshape(-1, 2)
    return uem[uem[:, 1] > uem[:, 0]]


def union_extent(*segments: Segments) -> np.ndarray:
    """``[[min start, max end]]`` over all given segment sets (dscore/pyannote default UEM)."""
    ext = [s.extent() for s in segments if len(s)]
    if not ext:
        return np.empty((0, 2))
    return np.array([[min(e[0] for e in ext), max(e[1] for e in ext)]])
