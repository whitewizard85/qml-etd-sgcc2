"""E5 - Fidelity kernel under realistic noise (fake IBM backend) or on real IBM hardware.

Compute-uncompute estimator: K(x, y) = P(0...0) after U(x) U(y)^dagger, estimated with shots.
Compares on the SAME small balanced subsets: ideal kernel (exact), noisy/hardware kernel,
classical RBF. Reports kernel error and SVM performance.

Local noisy simulation (no account needed):
  python scripts/06_noisy_kernel.py --d 4 --n-train 60 --n-test 120 --backend FakeTorino
Real hardware (requires `QiskitRuntimeService.save_account(...)` done once; consumes QPU time!):
  python scripts/06_noisy_kernel.py --d 4 --n-train 30 --n-test 40 --ibm-backend ibm_xxx --shots 1000

NOTE: the real-hardware path could not be tested by the author of this script (no account);
check the first run with a very small --n-train/--n-test. The SamplerV2 import of
qiskit-ibm-runtime is deprecated in recent releases and may need updating.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qmletd import common, prep, store  # noqa: E402
from qmletd.kernels import fidelity_gram, quantum_states  # noqa: E402

EXP = "E5_noisy_kernel"


def get_backend(a):
    if a.ibm_backend:
        from qiskit_ibm_runtime import QiskitRuntimeService
        return QiskitRuntimeService().backend(a.ibm_backend), True
    import qiskit_ibm_runtime.fake_provider as fp
    return getattr(fp, a.backend)(), False


def template(d, reps, ent):
    from qiskit.circuit.library import zz_feature_map
    u = zz_feature_map(d, reps=reps, entanglement=ent, parameter_prefix="x")
    v = zz_feature_map(d, reps=reps, entanglement=ent, parameter_prefix="y")
    qc = u.compose(v.inverse())
    qc.measure_all()
    return qc


def estimate(isa, backend, pairs, shots, chunk):
    from qiskit_ibm_runtime import SamplerV2
    sampler = SamplerV2(mode=backend)
    out = np.empty(len(pairs))
    for s in range(0, len(pairs), chunk):
        res = sampler.run([(isa, pairs[s:s + chunk])], shots=shots).result()
        arr = res[0].data.meas.array                     # (n, shots, n_bytes)
        out[s:s + chunk] = np.all(arr == 0, axis=-1).mean(axis=-1)
        print(f"  circuits {min(s + chunk, len(pairs))}/{len(pairs)}")
    return out


def svm_eval(Ktr, ytr, Kte, yte, C):
    w, V = np.linalg.eigh((Ktr + Ktr.T) / 2)          # project noisy Gram onto PSD cone
    Ktr = (V * np.clip(w, 0, None)) @ V.T
    s = SVC(C=C, kernel="precomputed").fit(Ktr, ytr).decision_function(Kte)
    return {"pr_auc": float(average_precision_score(yte, s)), "roc_auc": float(roc_auc_score(yte, s))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--bandwidth", type=float, default=0.5, help="use the best value found in E2")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--entanglement", default="linear", help="linear keeps circuits shallow on hardware")
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--backend", default="FakeTorino")
    ap.add_argument("--ibm-backend", default=None)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--opt-level", type=int, default=3)
    a = ap.parse_args()

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    cache = common.DataCache()
    P = cache.get(a.split, a.d)
    Xtr, ytr = prep.stratified_subsample(P.X_tr, P.y_tr, a.n_train, seed=a.split + 101)
    # balanced test subset drawn from the real (unbalanced) test partition
    Xte, yte = prep.stratified_subsample(P.X_te, P.y_te, a.n_test, seed=a.split + 202)
    Ztr, Zte = Xtr * a.bandwidth, Xte * a.bandwidth     # bandwidth applied to circuit inputs

    backend, real = get_backend(a)
    qc = template(a.d, a.reps, a.entanglement)
    isa = generate_preset_pass_manager(backend=backend, optimization_level=a.opt_level).run(qc)
    ops = isa.count_ops()
    print(f"[E5] backend={backend.name} depth={isa.depth()} 2q-gates="
          f"{sum(v for k, v in ops.items() if k in ('cx', 'ecr', 'cz'))}")

    iu = np.triu_indices(len(Xtr), k=1)
    pairs_tr = np.hstack([Ztr[iu[0]], Ztr[iu[1]]])
    ii, jj = np.meshgrid(np.arange(len(Xte)), np.arange(len(Xtr)), indexing="ij")
    pairs_te = np.hstack([Zte[ii.ravel()], Ztr[jj.ravel()]])
    t0 = time.time()
    k_tr = estimate(isa, backend, pairs_tr, a.shots, a.chunk)
    k_te = estimate(isa, backend, pairs_te, a.shots, a.chunk)
    secs = time.time() - t0

    Kn_tr = np.eye(len(Xtr))
    Kn_tr[iu] = k_tr
    Kn_tr[(iu[1], iu[0])] = k_tr
    Kn_te = k_te.reshape(len(Xte), len(Xtr))
    Ki_tr = fidelity_gram(quantum_states(Xtr, a.bandwidth, a.reps, a.entanglement))
    Ki_te = fidelity_gram(quantum_states(Xte, a.bandwidth, a.reps, a.entanglement),
                          quantum_states(Xtr, a.bandwidth, a.reps, a.entanglement))
    g0 = 1.0 / (a.d * Xtr.var())
    Kc_tr, Kc_te = rbf_kernel(Xtr, Xtr, gamma=g0), rbf_kernel(Xte, Xtr, gamma=g0)

    out = {
        "backend": backend.name, "real_hardware": real, "d": a.d, "bandwidth": a.bandwidth,
        "reps": a.reps, "entanglement": a.entanglement, "n_train": len(ytr), "n_test": len(yte),
        "shots": a.shots, "transpiled_depth": isa.depth(), "ops": dict(ops), "seconds": secs,
        "kernel_mae_train": float(np.abs(Kn_tr - Ki_tr)[iu].mean()),
        "kernel_mae_test": float(np.abs(Kn_te - Ki_te).mean()),
        "kernel_rel_fro_test": float(np.linalg.norm(Kn_te - Ki_te) / np.linalg.norm(Ki_te)),
        "ideal": svm_eval(Ki_tr, ytr, Ki_te, yte, a.C),
        "noisy": svm_eval(Kn_tr, ytr, Kn_te, yte, a.C),
        "rbf": svm_eval(Kc_tr, ytr, Kc_te, yte, a.C),
        "note": "balanced test subset: PR-AUC is NOT comparable with full-test PR-AUC of E1/E2",
    }
    name = f"noisykernel__{backend.name}__d{a.d}__bw{a.bandwidth}__n{a.n_train}x{a.n_test}__s{a.split}"
    store.save(EXP, name, out, None, K_noisy_train=Kn_tr, K_noisy_test=Kn_te,
               K_ideal_train=Ki_tr, K_ideal_test=Ki_te, y_train=ytr, y_test=yte)
    print(f"[E5] MAE(test)={out['kernel_mae_test']:.4f}  ideal={out['ideal']}  "
          f"noisy={out['noisy']}  rbf={out['rbf']}  ({secs:.0f}s)")
