"""flashder: fast, exact diarization error rate (DER) and Jaccard error rate (JER)."""

from .api import Report, der, jer, score
from .core import Grid, build_grid, grid_from_frames
from .io import load, load_uem, save
from .metrics import STYLES, DERComponents, JERComponents, Style, der_from_grid, jer_from_grid, score_file
from .segments import Segments

__version__ = "0.1.0"

__all__ = [
    "score",
    "der",
    "jer",
    "Report",
    "Segments",
    "load",
    "load_uem",
    "save",
    "Style",
    "STYLES",
    "DERComponents",
    "JERComponents",
    "Grid",
    "build_grid",
    "grid_from_frames",
    "der_from_grid",
    "jer_from_grid",
    "score_file",
    "der_frames",
]


def der_frames(ref, hyp, frame_shift: float = 1.0, mask=None, collar: int = 0, skip_overlap: bool = False):
    """DER components from frame-level ``(num_frames, num_speakers)`` activities (numpy)."""
    return der_from_grid(grid_from_frames(ref, hyp, frame_shift, mask, collar), skip_overlap=skip_overlap)
