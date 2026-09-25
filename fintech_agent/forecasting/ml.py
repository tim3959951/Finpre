"""Models trained locally on the M2 (MPS for torch, CPU for LightGBM).

DLinearForecaster : global DLinear (trend/remainder linear heads) with 9-quantile pinball loss, trained on MPS.
LGBMForecaster    : LightGBM quantile regression on engineered return/volatility features (direct h-step).

Both are scale-free (inputs normalised by recent volatility) so one model is trained across many tickers.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

from .base import QUANTILES, Forecaster, ForecastResult, clean_context
from .foundation import resolve_device

log = logging.getLogger(__name__)


# =============================================================================== DLinear
def _dl_windows(series: list[np.ndarray], L: int, H: int, stride: int = 1):
    X, Y = [], []
    for s in series:
        lx = np.log(np.maximum(clean_context(s), 1e-9))
        for t in range(L, len(lx) - H + 1, stride):
            ctx, fut = lx[t - L:t], lx[t:t + H]
            sd = np.std(np.diff(ctx)) + 1e-6
            X.append((ctx - ctx[-1]) / sd)
            Y.append((fut - ctx[-1]) / sd)
    return np.asarray(X, np.float32), np.asarray(Y, np.float32)


def _build_dlinear(L: int, H: int, nq: int, kernel: int = 25):
    import torch
    from torch import nn

    class DLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel = kernel
            self.pool = nn.AvgPool1d(kernel, stride=1)
            self.trend = nn.Linear(L, H * nq)
            self.remainder = nn.Linear(L, H * nq)

        def forward(self, x):                      # x: (B, L)
            pad = (self.kernel - 1) // 2
            xp = torch.cat([x[:, :1].repeat(1, pad), x, x[:, -1:].repeat(1, self.kernel - 1 - pad)], dim=1)
            trend = self.pool(xp.unsqueeze(1)).squeeze(1)
            out = self.trend(trend) + self.remainder(x - trend)
            return out.view(-1, H, nq)

    return DLinear()


class DLinearForecaster(Forecaster):
    name, family, trainable = "dlinear", "ml", True

    def __init__(self, context: int = 128, max_horizon: int = 20, epochs: int = 15, lr: float = 1e-3,
                 batch_size: int = 512, device: str = "auto", checkpoint: str | Path | None = None, seed: int = 0):
        self.L, self.H, self.epochs, self.lr, self.bs = context, max_horizon, epochs, lr, batch_size
        self.device, self.checkpoint, self.seed = resolve_device(device), checkpoint, seed
        self.net = None
        self.spread_scale = 1.0
        self.train_info: dict = {}

    def fit(self, series):
        import torch
        torch.manual_seed(self.seed)
        # time-ordered split *per series*: the last 15% of every ticker's windows is the validation set
        tr, va = [], []
        for s in series:
            Xs, Ys = _dl_windows([s], self.L, self.H)
            if len(Xs) < 20:
                continue
            k = max(1, int(len(Xs) * 0.15))
            tr.append((Xs[:-k - self.H], Ys[:-k - self.H]))    # gap of H windows: no target overlap
            va.append((Xs[-k:], Ys[-k:]))
        if not tr or sum(len(x) for x, _ in tr) < 100:
            raise ValueError("dlinear: not enough training windows; need longer history")
        Xtr, Ytr = np.concatenate([x for x, _ in tr]), np.concatenate([y for _, y in tr])
        Xva, Yva = np.concatenate([x for x, _ in va]), np.concatenate([y for _, y in va])
        X = Xtr
        dev = torch.device(self.device)
        net = _build_dlinear(self.L, self.H, len(QUANTILES)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=1e-4)
        qs = torch.tensor(QUANTILES, device=dev).view(1, 1, -1)

        def pinball(pred, y):
            e = y.unsqueeze(-1) - pred
            return torch.maximum(qs * e, (qs - 1) * e).mean()

        xtr, ytr = torch.from_numpy(Xtr).to(dev), torch.from_numpy(Ytr).to(dev)
        xva, yva = torch.from_numpy(Xva).to(dev), torch.from_numpy(Yva).to(dev)
        best, best_state, t0 = float("inf"), None, time.time()
        for ep in range(self.epochs):
            net.train()
            perm = torch.randperm(len(xtr), device=dev)
            for i in range(0, len(xtr), self.bs):
                idx = perm[i:i + self.bs]
                opt.zero_grad()
                loss = pinball(net(xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(pinball(net(xva), yva))
            if vl < best:
                best, best_state = vl, {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        self.net = net.eval()
        # post-hoc recalibration on the (time-ordered) validation windows: rescale the quantile fan around
        # the median to minimise pinball loss (a proper scoring rule), fixing over/under-confidence
        with torch.no_grad():
            pv = np.sort(net(xva).cpu().numpy(), axis=-1)             # (N, H, Q)
        med = pv[..., QUANTILES.index(0.5)][..., None]
        qv = np.asarray(QUANTILES).reshape(1, 1, -1)
        best_s, best_loss = 1.0, np.inf
        for s_ in np.linspace(0.5, 3.0, 51):
            e = Yva[..., None] - (med + s_ * (pv - med))
            loss = float(np.maximum(qv * e, (qv - 1) * e).mean())
            if loss < best_loss:
                best_s, best_loss = float(s_), loss
        self.spread_scale = best_s
        self.train_info = {"windows": int(len(X)), "val_windows": int(len(Xva)), "val_pinball": round(best, 5),
                           "spread_scale": round(self.spread_scale, 3), "device": self.device,
                           "seconds": round(time.time() - t0, 1), "epochs": self.epochs}
        log.info("dlinear trained: %s", self.train_info)
        if self.checkpoint:
            self.save(self.checkpoint)
        return self

    def save(self, path):
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "L": self.L, "H": self.H, "info": self.train_info,
                    "spread_scale": self.spread_scale}, path)

    def load(self, path):
        import torch
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.L, self.H, self.train_info = ck["L"], ck["H"], ck.get("info", {})
        self.spread_scale = float(ck.get("spread_scale", 1.0))
        self.net = _build_dlinear(self.L, self.H, len(QUANTILES)).to(self.device)
        self.net.load_state_dict(ck["state"])
        self.net.eval()
        return self

    def predict(self, contexts, horizon):
        import torch
        if self.net is None:
            if self.checkpoint and Path(self.checkpoint).exists():
                self.load(self.checkpoint)
            else:
                raise RuntimeError("dlinear is not trained: run `python scripts/train_models.py` or call fit()")
        if horizon > self.H:
            raise ValueError(f"dlinear trained for horizon<={self.H}, asked {horizon}")
        X, lasts, sds = [], [], []
        for c in contexts:
            lx = np.log(np.maximum(clean_context(c, self.L), 1e-9))
            if len(lx) < self.L:  # left-pad short contexts with the first value
                lx = np.r_[np.full(self.L - len(lx), lx[0]), lx]
            sd = np.std(np.diff(lx)) + 1e-6
            X.append((lx - lx[-1]) / sd)
            lasts.append(lx[-1])
            sds.append(sd)
        with torch.no_grad():
            out = self.net(torch.tensor(np.asarray(X), dtype=torch.float32, device=self.device)).cpu().numpy()
        out = np.sort(out, axis=-1)                                  # no quantile crossing
        med = out[..., QUANTILES.index(0.5)][..., None]
        out = med + (out - med) * self.spread_scale                  # conformal recalibration
        res = []
        for i in range(len(contexts)):
            lp = lasts[i] + out[i, :horizon, :] * sds[i]
            qs = {q: np.exp(lp[:, j]) for j, q in enumerate(QUANTILES)}
            res.append(ForecastResult(self.name, qs[0.5].copy(), qs))
        return res


# =============================================================================== LightGBM
def _lgbm_features(lx: np.ndarray) -> np.ndarray | None:
    """Features at the last point of a log-price array."""
    if len(lx) < 61:
        return None
    r = np.diff(lx)
    v5, v20, v60 = r[-5:].std(), r[-20:].std() + 1e-9, r[-60:].std() + 1e-9
    # vol-normalised momentum and distance-to-moving-average (scale free across tickers)
    f = [(lx[-1] - lx[-1 - k]) / (v20 * np.sqrt(k)) for k in (1, 2, 3, 5, 10, 20, 60)]
    f += [(lx[-1] - lx[-20:].mean()) / v20, (lx[-1] - lx[-60:].mean()) / v60]
    f += [v5 / v60, v20 / v60, v20 * np.sqrt(252)]
    lo, hi = lx[-60:].min(), lx[-60:].max()
    f.append((lx[-1] - lo) / (hi - lo + 1e-9))                    # position in 60d range
    up, dn = np.clip(r[-14:], 0, None).mean(), -np.clip(r[-14:], None, 0).mean()
    f.append(up / (up + dn + 1e-12))                              # RSI-like
    return np.asarray(f, np.float64)


class LGBMForecaster(Forecaster):
    name, family, trainable = "lgbm", "ml", True

    def __init__(self, horizon: int = 5, n_estimators: int = 200, stride: int = 2, seed: int = 0,
                 n_jobs: int | None = None):
        self.h, self.n_est, self.stride, self.seed = horizon, n_estimators, stride, seed
        # macOS: LightGBM's libomp and PyTorch's bundled libomp segfault when both run threaded in one process
        self.n_jobs = n_jobs if n_jobs is not None else (1 if sys.platform == "darwin" else -1)
        self.models: dict[float, object] = {}

    def fit(self, series):
        import lightgbm as lgb
        X, Y = [], []
        for s in series:
            lx = np.log(np.maximum(clean_context(s), 1e-9))
            for t in range(61, len(lx) - self.h, self.stride):
                f = _lgbm_features(lx[:t + 1])
                v20 = np.diff(lx[t - 20:t + 1]).std() + 1e-9
                X.append(f)
                Y.append((lx[t + self.h] - lx[t]) / (v20 * np.sqrt(self.h)))
        X, Y = np.asarray(X), np.asarray(Y)
        if len(X) < 200:
            raise ValueError(f"lgbm: only {len(X)} training rows")
        for q in QUANTILES:
            m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=self.n_est, learning_rate=0.03,
                                  num_leaves=15, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=self.seed, verbose=-1,
                                  n_jobs=self.n_jobs)
            m.fit(X, Y)
            self.models[q] = m
        return self

    def predict(self, contexts, horizon):
        if not self.models:
            raise RuntimeError("lgbm is not trained: call fit() (the backtester does this automatically)")
        feats, lasts, vols = [], [], []
        for c in contexts:
            lx = np.log(np.maximum(clean_context(c), 1e-9))
            feats.append(_lgbm_features(lx))
            lasts.append(lx[-1])
            vols.append(np.diff(lx[-21:]).std() + 1e-9)
        ok = [f is not None for f in feats]
        F = np.asarray([f for f in feats if f is not None])
        preds = {q: m.predict(F) for q, m in self.models.items()} if len(F) else {}
        res, j = [], 0
        steps = np.arange(1, horizon + 1) / horizon
        for i in range(len(contexts)):
            if not ok[i]:
                res.append(ForecastResult(self.name, np.full(horizon, np.exp(lasts[i]))))
                continue
            scale = vols[i] * np.sqrt(self.h) * np.sqrt(horizon / self.h)
            qv = np.sort([preds[q][j] for q in QUANTILES]) * scale
            med = qv[QUANTILES.index(0.5)]
            # direct h-step model -> path: drift linear in time, spread grows with sqrt(time)
            qs = {q: np.exp(lasts[i] + med * steps + (qv[k] - med) * np.sqrt(steps)) for k, q in enumerate(QUANTILES)}
            res.append(ForecastResult(self.name, qs[0.5].copy(), qs))
            j += 1
        return res
