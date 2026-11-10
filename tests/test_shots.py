import math

import numpy as np
import pytest
import torch

from qmletd.shots import add_shot_noise, moments_batched
from qmletd.vqc import VQC


@pytest.mark.parametrize("readout", ["local", "local_mean", "global"])
def test_parameter_shift_equals_autograd(readout):
    torch.manual_seed(1)
    d, L = 4, 2
    m = VQC(d, L, "linear", readout, affine=False, complex_dtype=torch.complex128)
    m.theta.data = m.theta.data.double()
    x = torch.rand(5, d, dtype=torch.float64)
    m.expval(x).sum().backward()
    P = m.theta.numel()
    sh = torch.zeros(2 * P + 1, P, dtype=torch.float64)
    sh[1 + torch.arange(P), torch.arange(P)] = math.pi / 2
    sh[1 + P + torch.arange(P), torch.arange(P)] = -math.pi / 2
    th = (m.theta.detach().flatten()[None] + sh).repeat_interleave(5, 0).reshape(-1, *m.theta.shape)
    E, _ = moments_batched(m, x.repeat(2 * P + 1, 1), th)
    E = E.reshape(2 * P + 1, 5)
    ps = ((E[1:P + 1] - E[P + 1:]) / 2).sum(dim=1)
    assert torch.allclose(ps, m.theta.grad.flatten(), atol=1e-10)


def test_shot_noise_statistics():
    g = torch.Generator().manual_seed(0)
    E = torch.full((20000,), 0.3, dtype=torch.float64)
    est = add_shot_noise(E, torch.ones_like(E), 1000, True, g)
    assert est.mean().item() == pytest.approx(0.3, abs=0.005)
    assert est.std().item() == pytest.approx(math.sqrt((1 - 0.09) / 1000), rel=0.05)
    E2 = torch.full_like(E, 0.5)
    est2 = add_shot_noise(E, E2, 400, False, g)
    assert est2.std().item() == pytest.approx(math.sqrt((0.5 - 0.09) / 400), rel=0.05)


def test_train_returns_all_keys():
    from qmletd.shots import train_vqc_shots
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(80, 3))
    y = (X[:, 0] > 0).astype(int)
    _, info = train_vqc_shots(X, y, X, y, d=3, depth=1, shots=100, steps=4, batch=8, eval_every=2)
    for k in ("cos_mean_all", "cos_first25", "rel_err_median_all", "best_val_pr_auc", "shots_total"):
        assert k in info
