"""Rolling-origin (walk-forward) backtest across tickers and models.

Leakage rules
* Each forecast only sees prices strictly before its origin.
* Trainable models (dlinear, lgbm) are fit once on data before the *earliest* origin across all tickers.
* Foundation models are zero-shot, but their pretraining corpora have cutoffs: for a clean test prefer
  windows after the model's release (TimesFM 2.5: 2025-09, Chronos-2: 2025-10) — see `min_origin`.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..forecasting.base import Forecaster
from ..forecasting.registry import get_forecaster
from .metrics import leaderboard, window_metrics

log = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    horizon: int = 5
    context_length: int = 512
    n_windows: int = 60
    step: int = 5
    min_context: int = 128
    min_origin: str | None = None      # e.g. "2025-10-01" to evaluate only after model release
    cost_bps: float = 0.0
    threshold: float = 0.0


def make_origins(n: int, cfg: BacktestConfig, index: pd.DatetimeIndex | None = None) -> list[int]:
    last = n - cfg.horizon
    origins = [last - i * cfg.step for i in range(cfg.n_windows)]
    origins = [t for t in origins if t >= cfg.min_context]
    if cfg.min_origin is not None and index is not None:
        cutoff = pd.Timestamp(cfg.min_origin)
        origins = [t for t in origins if index[t] >= cutoff]
    return sorted(origins)


def run_backtest(prices: dict[str, pd.Series], models: list[str] | dict[str, Forecaster], cfg: BacktestConfig,
                 settings=None, progress=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-window rows, leaderboard)."""
    series = {t: s.dropna().astype(float) for t, s in prices.items() if len(s.dropna()) > cfg.min_context + cfg.horizon}
    plan: list[tuple[str, int]] = []
    for t, s in series.items():
        plan += [(t, o) for o in make_origins(len(s), cfg, s.index)]
    if not plan:
        raise ValueError("no backtest windows — not enough history")
    earliest = min(series[t].index[o] for t, o in plan)
    contexts = [series[t].to_numpy()[max(0, o - cfg.context_length):o] for t, o in plan]
    actuals = [series[t].to_numpy()[o:o + cfg.horizon] for t, o in plan]

    if isinstance(models, dict):
        forecasters = models
    else:
        forecasters = {m: get_forecaster(m, cfg.horizon, settings) for m in models}

    rows = []
    for k, (name, fc) in enumerate(forecasters.items()):
        if progress:
            progress(k, len(forecasters), name)
        t0 = time.time()
        try:
            if getattr(fc, "trainable", False):
                train = [s[s.index < earliest].to_numpy() for s in series.values()]
                fc.fit([x for x in train if len(x) > 200])
            preds = fc.predict(contexts, cfg.horizon)
        except Exception as e:
            log.exception("model %s failed: %s", name, e)
            continue
        secs = time.time() - t0
        for (t, o), ctx, y, f in zip(plan, contexts, actuals, preds):
            m = window_metrics(y, f, ctx)
            m.update(model=name, ticker=t, origin=series[t].index[o], seconds=secs)
            rows.append(m)
    windows = pd.DataFrame(rows)
    if windows.empty:
        raise RuntimeError("all models failed")
    lb = leaderboard(windows, cfg.horizon, cfg.cost_bps, cfg.step, cfg.threshold)
    lb.attrs["config"] = asdict(cfg)
    lb.attrs["train_cutoff"] = str(earliest.date())
    return windows, lb
