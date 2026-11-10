"""The custom PyTorch simulator must agree with Qiskit to numerical precision."""
import numpy as np
import pytest
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import SparsePauliOp, Statevector

from qmletd.sim import entangling_pairs, pauli_expectations, zz_feature_states
from qmletd.vqc import VQC

C128 = torch.complex128


@pytest.mark.parametrize("d,reps,ent", [(2, 2, "full"), (3, 2, "full"), (4, 1, "linear"),
                                        (4, 3, "full"), (5, 2, "linear")])
def test_zz_feature_map(d, reps, ent):
    rng = np.random.default_rng(d * 10 + reps)
    X = rng.uniform(-1.5, 1.5, size=(4, d))
    ours = zz_feature_states(X, reps=reps, entanglement=ent, complex_dtype=C128).numpy()
    for i, x in enumerate(X):
        qc = zz_feature_map(feature_dimension=d, reps=reps, entanglement=ent)
        ref = Statevector(qc.assign_parameters(x)).data
        assert abs(np.vdot(ref, ours[i])) == pytest.approx(1.0, abs=1e-10)


def test_pauli_expectations():
    d = 3
    X = np.random.default_rng(1).uniform(-1, 1, size=(3, d))
    psi = zz_feature_states(X, complex_dtype=C128)
    ours = pauli_expectations(psi, d).numpy()
    for i in range(len(X)):
        sv = Statevector(psi[i].numpy())
        for q in range(d):
            for k, P in enumerate("XYZ"):
                label = ["I"] * d
                label[d - 1 - q] = P          # Qiskit labels are big-endian strings
                ref = sv.expectation_value(SparsePauliOp("".join(label))).real
                assert ours[i, 3 * q + k] == pytest.approx(ref, abs=1e-10)


def _qiskit_vqc(x, theta, d, depth, ent, reupload):
    qc = QuantumCircuit(d)
    for q in range(d):
        qc.ry(x[q], q)
    for q in range(d - 1):
        qc.cx(q, q + 1)
    for l in range(depth):
        if reupload and l > 0:
            for q in range(d):
                qc.ry(x[q], q)
        for q in range(d):
            qc.rz(theta[l, q, 0], q)
            qc.ry(theta[l, q, 1], q)
        for i, j in entangling_pairs(d, ent):
            qc.cz(i, j)
    for q in range(d):
        qc.rz(theta[depth, q, 0], q)
        qc.ry(theta[depth, q, 1], q)
    return Statevector(qc)


@pytest.mark.parametrize("d,depth,ent,reupload", [(2, 1, "linear", False), (3, 2, "ring", False),
                                                  (4, 2, "full", True), (5, 3, "linear", True)])
@pytest.mark.parametrize("readout", ["local", "local_mean", "global"])
def test_vqc_expectation(d, depth, ent, reupload, readout):
    torch.manual_seed(0)
    m = VQC(d, depth, ent, readout, reupload, affine=False, complex_dtype=C128)
    m.theta.data = m.theta.data.double()
    X = np.random.default_rng(2).uniform(-1.5, 1.5, size=(3, d))
    ours = m.expval(torch.as_tensor(X, dtype=torch.float64)).detach().numpy()
    th = m.theta.detach().numpy()
    if readout == "local":
        op = SparsePauliOp("I" * (d - 1) + "Z")
    elif readout == "global":
        op = SparsePauliOp("Z" * d)
    else:
        op = SparsePauliOp(["I" * (d - 1 - q) + "Z" + "I" * q for q in range(d)], coeffs=[1 / d] * d)
    for i, x in enumerate(X):
        ref = _qiskit_vqc(x, th, d, depth, ent, reupload).expectation_value(op).real
        assert ours[i] == pytest.approx(ref, abs=1e-10)


def test_local_readout_light_cone():
    """<Z_0> depends only on features 0..L+1: documents the light-cone limitation."""
    torch.manual_seed(0)
    d = 8
    for L in (1, 2, 3):
        m = VQC(d, L, "linear", "local", affine=False, complex_dtype=C128)
        x = torch.rand(1, d, dtype=torch.float64)
        e0 = m.expval(x).item()
        dep = [j for j in range(d) if abs(m.expval(x + 0.7 * (torch.arange(d) == j)).item() - e0) > 1e-9]
        assert dep == list(range(L + 2))
        m2 = VQC(d, L, "linear", "local_mean", affine=False, complex_dtype=C128)
        m2.theta.data = m.theta.data.clone()
        e1 = m2.expval(x).item()
        assert all(abs(m2.expval(x + 0.7 * (torch.arange(d) == j)).item() - e1) > 1e-9 for j in range(d))
