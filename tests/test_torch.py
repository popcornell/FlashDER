import numpy as np
import pytest

torch = pytest.importorskip("torch")

from flashder import der_from_grid, grid_from_frames, jer_from_grid  # noqa: E402
from flashder.torch import der_frames, jer_frames  # noqa: E402

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")
if torch.backends.mps.is_available():
    DEVICES.append("mps")


def _runs(rng, B, T, S, p_on, switch=0.03):
    """Activities made of random-length runs, so overlap counts rarely tie.

    With tied optimal mappings scipy and the permutation search may pick
    different (equally optimal) speaker maps and the DER can legitimately differ.
    """
    flips = rng.random((B, T, S)) < switch
    start = rng.random((B, 1, S)) < p_on
    return start ^ (np.cumsum(flips, axis=1) % 2 == 1)


def _batch(rng, B, T, R, H):
    ref = _runs(rng, B, T, R, 0.25)
    hyp = _runs(rng, B, T, H, 0.3)
    lens = rng.integers(T // 2, T + 1, B)
    mask = np.arange(T)[None] < lens[:, None]
    return ref, hyp, mask


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("R,H", [(3, 4), (4, 2), (2, 2), (10, 9)])  # 10 > 8: Hungarian fallback
@pytest.mark.parametrize("collar", [0, 5])
@pytest.mark.parametrize("skip_overlap", [False, True])
def test_der_frames_matches_numpy(device, R, H, collar, skip_overlap):
    rng = np.random.default_rng(R * 100 + H)
    ref, hyp, mask = _batch(rng, 6, 2000, R, H)
    out = der_frames(
        torch.tensor(ref, device=device),
        torch.tensor(hyp, device=device),
        torch.tensor(mask, device=device),
        collar=collar,
        skip_overlap=skip_overlap,
    )
    for b in range(len(ref)):
        exp = der_from_grid(grid_from_frames(ref[b], hyp[b], 1.0, mask[b], collar), skip_overlap=skip_overlap)
        assert out["total"][b].item() == exp.total
        assert out["miss"][b].item() == exp.miss
        assert out["false_alarm"][b].item() == exp.false_alarm
        assert out["confusion"][b].item() == exp.confusion


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("R,H", [(3, 4), (4, 2), (10, 9)])
def test_jer_frames_matches_numpy(device, R, H):
    rng = np.random.default_rng(R * 10 + H)
    ref, hyp, mask = _batch(rng, 5, 2000, R, H)
    out = jer_frames(
        torch.tensor(ref, device=device), torch.tensor(hyp, device=device), torch.tensor(mask, device=device)
    )
    exp = [jer_from_grid(grid_from_frames(ref[b], hyp[b], 1.0, mask[b])).jer for b in range(len(ref))]
    np.testing.assert_allclose(out["jer"].cpu().numpy(), exp, atol=1e-5)


def test_probabilities_are_thresholded():
    ref = torch.zeros(1, 10, 2)
    ref[0, :5, 0] = ref[0, 5:, 1] = 1
    hyp = ref.flip(-1) * 0.9 + 0.05
    out = der_frames(ref, hyp)
    assert out["errors"].item() == 0


def test_unbatched_input():
    ref = torch.zeros(10, 2, dtype=torch.bool)
    ref[:5, 0] = True
    out = der_frames(ref, ref)
    assert out["total"].shape == (1,) and out["errors"].item() == 0
