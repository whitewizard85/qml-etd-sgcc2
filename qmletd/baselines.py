"""Classical baselines on the same tabular features, and a 1D CNN on raw daily series."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from .config import seed_everything


def _score(m, X):
    if hasattr(m, "decision_function"):
        return m.decision_function(X)
    return m.predict_proba(X)[:, 1]


def grids(seed):
    import lightgbm as lgb

    return {
        "logreg": [LogisticRegression(C=c, max_iter=2000) for c in (0.01, 0.1, 1, 10)],
        "svm_rbf": [SVC(C=c, kernel="rbf", gamma=g) for c in (1, 10, 100) for g in ("scale", 0.1, 1.0)],
        "rf": [RandomForestClassifier(n_estimators=400, max_depth=md, min_samples_leaf=5,
                                      n_jobs=-1, random_state=seed) for md in (None, 12)],
        "lgbm": [lgb.LGBMClassifier(n_estimators=n, learning_rate=0.05, num_leaves=nl,
                                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                    random_state=seed, verbose=-1)
                 for n in (300, 800) for nl in (15, 63)],
        "mlp": [MLPClassifier(hidden_layer_sizes=h, alpha=1e-3, max_iter=300, early_stopping=True,
                              random_state=seed) for h in ((32,), (64, 32))],
    }


def fit_select(name, X_tr, y_tr, X_va, y_va, seed, X_tune=None, y_tune=None):
    """Grid search with selection by validation PR-AUC; optional smaller tuning set,
    then refit of the selected configuration on the full training set."""
    cands = grids(seed)[name]
    Xt, yt = (X_tune, y_tune) if X_tune is not None else (X_tr, y_tr)
    best, best_ap, t0 = None, -1, time.time()
    for m in cands:
        m.fit(Xt, yt)
        ap = average_precision_score(y_va, _score(m, X_va))
        if ap > best_ap:
            best, best_ap = m, ap
    if X_tune is not None:
        best = copy.deepcopy(best).fit(X_tr, y_tr)
    tune_s = time.time() - t0
    return best, {"val_pr_auc_selection": float(best_ap), "params": str(best.get_params()),
                  "search_seconds": tune_s}


score = _score


# --------------------------------------------------------------------- 1D CNN on raw data

class CNN1D(nn.Module):
    def __init__(self, in_ch=2, width=32):
        super().__init__()
        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(),
                                 nn.MaxPool1d(2))
        self.net = nn.Sequential(block(in_ch, width, 7), block(width, width * 2, 5),
                                 block(width * 2, width * 4, 5), block(width * 4, width * 4, 3))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(width * 8, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        h = self.net(x)
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
        return self.head(h).squeeze(1)


def series_tensor(series, mask, idx, mu, sd):
    x = np.log1p(np.clip(np.asarray(series[idx], dtype=np.float32), 0, None))
    x = (x - mu) / sd
    m = np.asarray(mask[idx], dtype=np.float32)
    return torch.from_numpy(np.stack([x, m], axis=1))


def train_cnn(series, mask, y, split, device, epochs=40, batch=256, lr=1e-3, seed=0, patience=6):
    seed_everything(seed)
    tr, va, te = split.train, split.val, split.test
    lx = np.log1p(np.clip(np.asarray(series[tr], dtype=np.float32), 0, None))
    mu, sd = float(lx.mean()), float(lx.std() + 1e-6)
    Xtr, Xva, Xte = (series_tensor(series, mask, i, mu, sd) for i in (tr, va, te))
    ytr = torch.as_tensor(y[tr], dtype=torch.float32)
    model = CNN1D().to(device)
    pos_w = torch.tensor((1 - y[tr].mean()) / y[tr].mean(), device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    @torch.no_grad()
    def pred(X):
        model.eval()
        return np.concatenate([torch.sigmoid(model(X[s:s + 1024].to(device))).cpu().numpy()
                               for s in range(0, len(X), 1024)])

    g = torch.Generator().manual_seed(seed)
    best, bad, t0 = (-1, None, 0), 0, time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            l = lossf(model(Xtr[idx].to(device)), ytr[idx].to(device))
            l.backward()
            opt.step()
        ap = average_precision_score(y[va], pred(Xva))
        if ap > best[0] + 1e-5:
            best, bad = (ap, copy.deepcopy(model.state_dict()), ep), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best[1])
    return pred(Xva), pred(Xte), {"best_val_pr_auc": float(best[0]), "best_epoch": best[2],
                                  "train_seconds": time.time() - t0}
