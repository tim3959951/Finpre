"""Statistical baselines. For asset prices the random walk (`naive`) is the benchmark that matters:
a model that can't beat it on MASE / directional accuracy has no forecasting edge."""
from __future__ import annotations

import logging

import numpy as np

from .base import QUANTILES, Forecaster, ForecastResult, clean_context

log = logging.getLogger(__name__)


def _empirical_return_quantiles(x: np.ndarray, horizon: int, levels=QUANTILES, lookback: int = 750,
                                demean: bool = True) -> dict[float, np.ndarray]:
    """Quantiles of k-step cumulative log returns (k=1..h) from the context itself (fat tails preserved)."""
    lx = np.log(np.maximum(x[-(lookback + 1):], 1e-9))
    r = np.diff(lx)
    out = {q: np.zeros(horizon) for q in levels}
    if len(r) < 20:
        sd = 0.02
        from statistics import NormalDist
        for q in levels:
            z = NormalDist().inv_cdf(q)
            out[q] = z * sd * np.sqrt(np.arange(1, horizon + 1))
        return out
    mu = r.mean() if demean else 0.0
    rc = r - mu
    cs = np.concatenate([[0.0], np.cumsum(rc)])
    for k in range(1, horizon + 1):
        if len(r) >= k + 20:
            ksum = cs[k:] - cs[:-k]
            qs = np.quantile(ksum, levels)
        else:  # not enough overlapping windows: sqrt-time scaling of 1-step quantiles
            qs = np.quantile(rc, levels) * np.sqrt(k)
        for q, v in zip(levels, qs):
            out[q][k - 1] = v
    return out


class NaiveForecaster(Forecaster):
    """Random walk: tomorrow = today. Quantiles from empirical k-step return distribution."""
    name, family = "naive", "baseline"

    def predict(self, contexts, horizon):
        res = []
        for c in contexts:
            x = clean_context(c)
            last = x[-1]
            rq = _empirical_return_quantiles(x, horizon)
            res.append(ForecastResult(self.name, np.full(horizon, last),
                                      {q: last * np.exp(v) for q, v in rq.items()}))
        return res


class DriftForecaster(Forecaster):
    """Random walk with drift (mean log return of the last year)."""
    name, family = "drift", "baseline"

    def __init__(self, lookback: int = 252):
        self.lookback = lookback

    def predict(self, contexts, horizon):
        res = []
        for c in contexts:
            x = clean_context(c)
            last = x[-1]
            lr = np.diff(np.log(x[-(self.lookback + 1):]))
            mu = lr.mean() if len(lr) else 0.0
            steps = np.arange(1, horizon + 1)
            rq = _empirical_return_quantiles(x, horizon)
            res.append(ForecastResult(self.name, last * np.exp(mu * steps),
                                      {q: last * np.exp(mu * steps + v) for q, v in rq.items()}))
        return res


class ArimaForecaster(Forecaster):
    """AutoARIMA on log prices via statsforecast (falls back to naive on failure)."""
    name, family = "arima", "statistical"

    def __init__(self, max_context: int = 500):
        self.max_context = max_context
        self._fallback = NaiveForecaster()

    def predict(self, contexts, horizon):
        try:
            from statsforecast.models import AutoARIMA
        except ImportError:
            log.warning("statsforecast not installed; arima -> naive")
            return [ForecastResult(self.name, r.point, r.quantiles) for r in self._fallback.predict(contexts, horizon)]
        pairs = {0.1: 80, 0.2: 60, 0.3: 40, 0.4: 20}
        res = []
        for c in contexts:
            x = clean_context(c, self.max_context)
            try:
                f = AutoARIMA(season_length=1).forecast(y=np.log(x), h=horizon, level=[20, 40, 60, 80])
                qs = {0.5: np.exp(f["mean"])}
                for lo, lvl in pairs.items():
                    qs[lo] = np.exp(f[f"lo-{lvl}"])
                    qs[round(1 - lo, 1)] = np.exp(f[f"hi-{lvl}"])
                res.append(ForecastResult(self.name, np.exp(f["mean"]), dict(sorted(qs.items()))))
            except Exception as e:  # pragma: no cover - numerical edge cases
                log.debug("arima failed: %s", e)
                r = self._fallback.predict([x], horizon)[0]
                res.append(ForecastResult(self.name, r.point, r.quantiles))
        return res


class EnsembleForecaster(Forecaster):
    """Median of member point forecasts; quantiles averaged across members (Vincentization)."""
    name, family = "ensemble", "ensemble"

    def __init__(self, members: list[Forecaster]):
        self.members = members
        self.commercial_ok = all(m.commercial_ok for m in members)

    def predict(self, contexts, horizon):
        per_model = [m.predict(contexts, horizon) for m in self.members]
        out = []
        for i in range(len(contexts)):
            rs = [pm[i] for pm in per_model]
            point = np.median(np.stack([r.point for r in rs]), axis=0)
            levels = set.intersection(*[set(r.quantiles) for r in rs]) if all(r.quantiles for r in rs) else set()
            qs = {q: np.mean(np.stack([r.quantiles[q] for r in rs]), axis=0) for q in sorted(levels)}
            out.append(ForecastResult(self.name, point, qs))
        return out
