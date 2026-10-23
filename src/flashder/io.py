"""Readers and writers for common speaker-attributed segment formats.

Every reader returns ``dict[file_id, Segments]``; UEM readers return
``dict[file_id, (n, 2) ndarray]``.

Supported formats
-----------------
``rttm``    NIST RTTM, ``SPEAKER`` lines (RT-09 eval plan, appendix A).
``seglst``  CHiME/MeetEval Segment-wise Long-form Speech Transcription (JSON).
``stm``     NIST/sclite STM (``inter_segment_gap`` and
            ``IGNORE_TIME_SEGMENT_IN_SCORING`` lines are skipped).
``lab``     Audacity labels / ``start end speaker`` per line, one recording
            per file (the file stem is the recording id).
``kaldi``   Kaldi ``segments`` + ``utt2spk``.
``uem``     NIST UEM (``file chnl tbeg tend``).
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Union

import numpy as np

from .segments import Segments

PathLike = Union[str, os.PathLike]

__all__ = [
    "load",
    "save",
    "load_rttm",
    "save_rttm",
    "load_seglst",
    "save_seglst",
    "load_stm",
    "save_stm",
    "load_lab",
    "save_lab",
    "load_kaldi",
    "load_uem",
    "save_uem",
    "detect_format",
]


def _lines(path: PathLike) -> Iterable[list[str]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;") or line.startswith("#"):
                continue
            yield line.split()


def _group(records) -> dict[str, Segments]:
    acc = defaultdict(list)
    for file_id, start, end, speaker in records:
        acc[file_id].append((start, end, speaker))
    return {k: Segments.from_tuples(v) for k, v in acc.items()}


def _fmt(t: float, digits: int) -> str:
    return f"{t:.{digits}f}"


# ------------------------------------------------------------------- RTTM
def load_rttm(path: PathLike) -> dict[str, Segments]:
    def records():
        for f in _lines(path):
            if f[0] != "SPEAKER":
                continue
            start, dur = float(f[3]), float(f[4])
            yield f[1], start, start + dur, f[7]

    return _group(records())


def save_rttm(diarization: dict[str, Segments], path: PathLike, digits: int = 3, channel: str = "1") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for file_id, segs in diarization.items():
            for s, e, spk in segs.sorted():
                f.write(
                    f"SPEAKER {file_id} {channel} {_fmt(s, digits)} {_fmt(e - s, digits)} <NA> <NA> {spk} <NA> <NA>\n"
                )


# ----------------------------------------------------------------- SegLST
def load_seglst(path: PathLike) -> dict[str, Segments]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):  # tolerate {"segments": [...]}
        data = data.get("segments", data)
    return _group(
        (str(d["session_id"]), float(d["start_time"]), float(d["end_time"]), str(d["speaker"])) for d in data
    )


def save_seglst(diarization: dict[str, Segments], path: PathLike, digits: int = 3) -> None:
    out = [
        {
            "session_id": file_id,
            "start_time": round(s, digits),
            "end_time": round(e, digits),
            "speaker": str(spk),
            "words": "",
        }
        for file_id, segs in diarization.items()
        for s, e, spk in segs.sorted()
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


# -------------------------------------------------------------------- STM
_STM_SKIP_SPEAKERS = {"inter_segment_gap"}
_STM_SKIP_TEXT = {"ignore_time_segment_in_scoring"}


def load_stm(path: PathLike) -> dict[str, Segments]:
    def records():
        for f in _lines(path):
            if len(f) < 5 or f[2].lower() in _STM_SKIP_SPEAKERS:
                continue
            text = [w for w in f[5:] if not (w.startswith("<") and w.endswith(">"))]
            if len(text) == 1 and text[0].lower() in _STM_SKIP_TEXT:
                continue
            yield f[0], float(f[3]), float(f[4]), f[2]

    return _group(records())


def save_stm(diarization: dict[str, Segments], path: PathLike, digits: int = 3, channel: str = "1") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for file_id, segs in diarization.items():
            for s, e, spk in segs.sorted():
                f.write(f"{file_id} {channel} {spk} {_fmt(s, digits)} {_fmt(e, digits)}\n")


# --------------------------------------------------------- Audacity / lab
def load_lab(path: PathLike, file_id: str | None = None) -> dict[str, Segments]:
    file_id = Path(path).stem if file_id is None else file_id
    return _group((file_id, float(f[0]), float(f[1]), " ".join(f[2:])) for f in _lines(path) if len(f) >= 3)


def save_lab(segments: Segments, path: PathLike, digits: int = 3) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for s, e, spk in segments.sorted():
            f.write(f"{_fmt(s, digits)}\t{_fmt(e, digits)}\t{spk}\n")


# ------------------------------------------------------------------ Kaldi
def load_kaldi(segments: PathLike, utt2spk: PathLike) -> dict[str, Segments]:
    spk = {f[0]: f[1] for f in _lines(utt2spk)}
    return _group((f[1], float(f[2]), float(f[3]), spk[f[0]]) for f in _lines(segments))


# -------------------------------------------------------------------- UEM
def load_uem(path: PathLike) -> dict[str, np.ndarray]:
    acc = defaultdict(list)
    for f in _lines(path):
        if len(f) == 4:
            acc[f[0]].append((float(f[2]), float(f[3])))
        elif len(f) == 3:  # some tools drop the channel
            acc[f[0]].append((float(f[1]), float(f[2])))
        else:
            raise ValueError(f"malformed UEM line in {path}: {' '.join(f)}")
    return {k: np.array(v, dtype=np.float64) for k, v in acc.items()}


def save_uem(uem: dict[str, np.ndarray], path: PathLike, digits: int = 3, channel: str = "1") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for file_id, regions in uem.items():
            for s, e in np.asarray(regions).reshape(-1, 2):
                f.write(f"{file_id} {channel} {_fmt(s, digits)} {_fmt(e, digits)}\n")


# --------------------------------------------------------------- dispatch
_EXT = {
    ".rttm": "rttm",
    ".json": "seglst",
    ".seglst": "seglst",
    ".stm": "stm",
    ".lab": "lab",
    ".txt": "lab",
    ".uem": "uem",
}


def detect_format(path: PathLike) -> str:
    ext = Path(path).suffix.lower()
    if ext in _EXT:
        return _EXT[ext]
    with open(path, encoding="utf-8") as f:
        head = f.read(4096).lstrip()
    if head.startswith("[") or head.startswith("{"):
        return "seglst"
    for line in head.splitlines():
        if not line.strip() or line.startswith(";;"):
            continue
        fields = line.split()
        if fields[0] in ("SPEAKER", "SPKR-INFO", "LEXEME", "SEGMENT", "NON-SPEECH", "NON-LEX"):
            return "rttm"
        if len(fields) >= 5:
            return "stm"
        if len(fields) == 3:
            return "lab"
        break
    raise ValueError(f"cannot detect the format of {path}; pass it explicitly")


_LOADERS = {"rttm": load_rttm, "seglst": load_seglst, "stm": load_stm, "lab": load_lab}
_SAVERS = {"rttm": save_rttm, "seglst": save_seglst, "stm": save_stm}


def load(paths: PathLike | Iterable[PathLike], fmt: str | None = None) -> dict[str, Segments]:
    """Load one or more files into ``{file_id: Segments}``."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out: dict[str, list[Segments]] = defaultdict(list)
    for p in paths:
        loader = _LOADERS[fmt or detect_format(p)]
        for k, v in loader(p).items():
            out[k].append(v)
    return {
        k: v[0]
        if len(v) == 1
        else Segments(
            np.concatenate([s.starts for s in v]),
            np.concatenate([s.ends for s in v]),
            np.concatenate([s.speakers for s in v]),
        )
        for k, v in out.items()
    }


def save(diarization: dict[str, Segments], path: PathLike, fmt: str | None = None, **kwargs) -> None:
    fmt = fmt or _EXT.get(Path(path).suffix.lower(), "rttm")
    _SAVERS[fmt](diarization, path, **kwargs)
