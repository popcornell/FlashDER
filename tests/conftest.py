import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from flashder import Segments

CACHE = Path(__file__).parent / ".cache"
DSCORE_URL = "https://github.com/nryant/dscore.git"


def random_segments(rng, n, n_spk, dur=60.0, prefix="s", digits=3, merge=False):
    """Random turns on a ``10**-digits`` grid (so RTTM round trips are lossless)."""
    s = np.round(rng.uniform(0, dur, n), digits)
    d = np.round(rng.uniform(0.1, 5.0, n), digits)
    spk = np.array([f"{prefix}{i}" for i in rng.integers(0, n_spk, n)], dtype=object)
    segs = Segments(s, s + d, spk)
    return segs.merge_overlaps(strict=False) if merge else segs


@pytest.fixture
def rng(request):
    return np.random.default_rng(abs(hash(request.node.name)) % 2**32)


@pytest.fixture(scope="session")
def dscore_dir():
    """Path to a dscore checkout (bundles NIST md-eval-22.pl).

    Uses ``$DSCORE_DIR`` if set, otherwise clones into ``tests/.cache``.
    """
    env = os.environ.get("DSCORE_DIR")
    if env:
        return Path(env)
    target = CACHE / "dscore"
    if not (target / "score.py").exists():
        if shutil.which("git") is None:
            pytest.skip("git not available to fetch dscore")
        CACHE.mkdir(exist_ok=True)
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", DSCORE_URL, str(target)], capture_output=True)
        if r.returncode != 0:
            pytest.skip(f"could not clone dscore: {r.stderr.decode()[:200]}")
    return target


@pytest.fixture(scope="session")
def mdeval(dscore_dir):
    if shutil.which("perl") is None:
        pytest.skip("perl not available")
    return dscore_dir / "scorelib" / "md-eval-22.pl"
