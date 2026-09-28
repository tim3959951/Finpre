"""Stock-ranking scorers: a LightGBM cross-sectional ranker and single-factor baselines.

The ranker is trained to predict each stock's cross-sectional percentile of forward return (pointwise
regression on ranks — robust to outliers and to the market's overall direction, which ranking cannot use).
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .panel import feature_columns


class Scorer:
    """Returns a score per (date, ticker) row: higher = expected to outperform its peers."""
    name = "scorer"
    trainable = False

    def score(self, rows: pd.DataFrame) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class FactorScorer(Scorer):
    """Single-factor baseline, e.g. momentum or 20-day foreign net buying (percentile features)."""

    def __init__(self, name: str, column: str, sign: float = 1.0):
        self.name, self.column, self.sign = name, column, sign

    def available(self, panel: pd.DataFrame) -> bool:
        return self.column in panel and panel[self.column].notna().mean() > 0.5

    def score(self, rows):
        return self.sign * rows[self.column].to_numpy(dtype=float)


FACTORS = {
    "momentum": ("mom_120_20", 1.0),          # 6-month momentum skipping the last month
    "reversal": ("r5", -1.0),                 # 1-week short-term reversal
    "low_vol": ("vol60", -1.0),               # low volatility anomaly
    "foreign_flow": ("foreign_20", 1.0),      # 20-day foreign net buying (TW only)
    "trust_flow": ("trust_20", 1.0),          # 20-day investment-trust net buying (TW only)
}


def factor_scorers(panel: pd.DataFrame) -> list[FactorScorer]:
    out = [FactorScorer(n, c, s) for n, (c, s) in FACTORS.items()]
    return [f for f in out if f.available(panel)]


class LGBMRanker(Scorer):
    """LightGBM on cross-sectional percentile features -> predicted forward-return percentile."""
    trainable = True

    def __init__(self, chips: bool = False, n_estimators: int = 300, learning_rate: float = 0.03,
                 num_leaves: int = 15, min_child_samples: int = 200, seed: int = 0, n_jobs: int | None = None,
                 name: str | None = None):
        self.chips = chips
        self.name = name or ("xs-lgbm-chips" if chips else "xs-lgbm")
        self.params = dict(n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves,
                           min_child_samples=min_child_samples, subsample=0.7, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=1.0, random_state=seed, verbose=-1)
        env_jobs = os.environ.get("FA_LGBM_JOBS")
        self.n_jobs = n_jobs if n_jobs is not None else int(env_jobs) if env_jobs else (1 if sys.platform == "darwin" else -1)
        self.model = None
        self.columns: list[str] = []

    def fit(self, rows: pd.DataFrame, columns: list[str] | None = None) -> "LGBMRanker":
        import lightgbm as lgb
        self.columns = columns or feature_columns(rows, self.chips)
        d = rows.dropna(subset=["fwd_rank"])
        if len(d) < 1000:
            raise ValueError(f"{self.name}: only {len(d)} labelled rows")
        self.model = lgb.LGBMRegressor(n_jobs=self.n_jobs, **self.params)
        self.model.fit(d[self.columns].to_numpy(dtype=np.float32), d["fwd_rank"].to_numpy(dtype=np.float32))
        return self

    def score(self, rows):
        if self.model is None:
            raise RuntimeError(f"{self.name} is not trained")
        return self.model.predict(rows[self.columns].to_numpy(dtype=np.float32))

    def feature_importance(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        imp = self.model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp / (imp.sum() + 1e-12), index=self.columns).sort_values(ascending=False)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"name": self.name, "chips": self.chips, "model": self.model, "columns": self.columns}, f)

    @classmethod
    def load(cls, path) -> "LGBMRanker":
        with open(path, "rb") as f:
            d = pickle.load(f)
        r = cls(chips=d["chips"], name=d["name"])
        r.model, r.columns = d["model"], d["columns"]
        return r
