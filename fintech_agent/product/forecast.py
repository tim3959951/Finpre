"""One fast, consistent way for product features to get a forecast for a ticker.

Uses the market x horizon champion (runs/champion.json -> settings) when it can be served cheaply (a trained
checkpoint or a baseline); otherwise falls back to the empirical random-walk quantiles, and says so. Every result
carries the model name so reports can cite it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..data.provider import DataProvider
from ..data.symbols import Symbol
from ..evaluation.experiments import ExperimentStore
from ..forecasting.base import QUANTILES, ForecastResult, run_predict
from ..forecasting.registry import SPECS, get_forecaster, load_trained

log = logging.getLogger(__name__)


@dataclass
class TickerForecast:
    symbol: Symbol
    name: str
    last: float
    as_of: str
    horizon: int
    model: str
    quantile_returns: dict[float, float]     # level -> simple return at the horizon
    p_up: float
    hv20_ann: float

    def band_pct(self, lo: float = 0.1, hi: float = 0.9) -> tuple[float, float]:
        return self.quantile_returns[lo] * 100, self.quantile_returns[hi] * 100

    def to_dict(self) -> dict:
        lo, hi = self.band_pct()
        return {"ticker": self.symbol.code, "name": self.name, "market": self.symbol.market, "last": round(self.last, 4),
                "as_of": self.as_of, "horizon": self.horizon, "model": self.model,
                "median_pct": round(self.quantile_returns[0.5] * 100, 2), "p10_pct": round(lo, 2),
                "p90_pct": round(hi, 2), "p_up": round(self.p_up, 3), "hv20_ann_pct": round(self.hv20_ann * 100, 1)}


def _serveable(name: str, market: str, horizon: int, settings: Settings):
    if name in SPECS and SPECS[name].trainable:
        return load_trained(name, market, horizon, settings)
    if name in ("naive", "drift"):
        return get_forecaster(name, horizon, settings)
    return None                                   # foundation models are too slow for batch product calls


def forecast_result(provider: DataProvider, sym: Symbol, px: pd.DataFrame, horizon: int, settings: Settings,
                    model_name: str | None = None, allow_foundation: bool = False) -> tuple[ForecastResult, str]:
    """(ForecastResult, model actually used) for the champion — or `model_name` — with a naive fallback."""
    s = settings
    ctx_len = int(s.get_path("forecasting.context_length", 512))
    series = px["close"].astype(float).to_numpy()[-ctx_len:]
    champion = model_name or ExperimentStore(s).champion(sym.market, horizon)
    model = _serveable(champion, sym.market, horizon, s)
    name = champion
    if model is None and allow_foundation:
        try:
            model = get_forecaster(champion, horizon, s)
        except Exception as e:  # pragma: no cover - optional heavy deps
            log.warning("champion %s unavailable: %s", champion, e)
    cov = None
    if model is not None and getattr(model, "uses_covariates", False):
        try:
            c = provider.covariates(sym, px)
            cov = [c.iloc[-(ctx_len + horizon):]] if len(c) == len(px) else None
        except Exception:
            cov = None
        if cov is None:
            model = None
    if model is None:
        model, name = get_forecaster("naive", horizon, s), "naive"
    fc: ForecastResult = run_predict(model, [series], horizon, cov)[0]
    fc.model = name
    return fc, name


def forecast_ticker(provider: DataProvider, ticker: str | Symbol, horizon: int = 5, settings: Settings | None = None,
                    prices: pd.DataFrame | None = None, allow_foundation: bool = False) -> TickerForecast:
    s = settings or get_settings()
    sym = ticker if isinstance(ticker, Symbol) else provider.symbol(ticker)
    px = prices if prices is not None else provider.prices(sym)
    if px is None or len(px) < 80:
        raise ValueError(f"not enough price history for {sym.code}")
    close = px["close"].astype(float)
    fc, name = forecast_result(provider, sym, px, horizon, s, allow_foundation=allow_foundation)
    series = close.to_numpy()
    last = float(series[-1])
    qr = {q: float(fc.q(q)[-1] / last - 1) for q in QUANTILES}
    lr = np.log(close).diff().dropna()
    prof = provider.profile(sym) or {}
    return TickerForecast(sym, str(prof.get("name") or sym.code), last, str(px.index[-1].date()), horizon, name, qr,
                          float(fc.prob_above(last)), float(lr.tail(20).std() * np.sqrt(252)))
