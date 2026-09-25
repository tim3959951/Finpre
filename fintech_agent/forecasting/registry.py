"""Model registry: name -> forecaster factory, with metadata used by the UI, the quant agent and A/B tests."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from ..config import Settings, get_settings
from .base import Forecaster
from .baselines import ArimaForecaster, DriftForecaster, EnsembleForecaster, NaiveForecaster
from .foundation import ChronosForecaster, TiRexForecaster, TimesFM25Forecaster, TimesFM3Forecaster
from .ml import DLinearForecaster, LGBMForecaster

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    description: str
    license: str
    commercial_ok: bool
    trainable: bool
    factory: Callable[[Settings, int], Forecaster]


def _dev(s: Settings) -> str:
    return s.get_path("forecasting.inference_device", "cpu")


SPECS: dict[str, ModelSpec] = {s.name: s for s in [
    ModelSpec("timesfm-2.5", "foundation", "Google TimesFM 2.5 · 200M · 16k context · quantile head (v1 champion)",
              "Apache-2.0", True, False, lambda s, h: TimesFM25Forecaster(_dev(s))),
    ModelSpec("timesfm-3.0", "foundation", "Google TimesFM 3.0 · multivariate · research benchmark only",
              "TimesFM-NC-1.0 (non-commercial)", False, False, lambda s, h: TimesFM3Forecaster(_dev(s))),
    ModelSpec("chronos-2", "foundation", "Amazon Chronos-2 · 120M encoder · covariate-aware",
              "Apache-2.0", True, False, lambda s, h: ChronosForecaster("amazon/chronos-2", "chronos-2", _dev(s))),
    ModelSpec("chronos-bolt-small", "foundation", "Amazon Chronos-Bolt small · 48M · very fast",
              "Apache-2.0", True, False, lambda s, h: ChronosForecaster("amazon/chronos-bolt-small", "chronos-bolt-small", _dev(s))),
    ModelSpec("chronos-bolt-base", "foundation", "Amazon Chronos-Bolt base · 205M",
              "Apache-2.0", True, False, lambda s, h: ChronosForecaster("amazon/chronos-bolt-base", "chronos-bolt-base", _dev(s))),
    ModelSpec("tirex", "foundation", "NX-AI TiRex · xLSTM 35M (optional: pip install tirex-ts)",
              "NXAI Community License (check before commercial use)", False, False, lambda s, h: TiRexForecaster(_dev(s))),
    ModelSpec("naive", "baseline", "Random walk (last price) — the benchmark to beat",
              "-", True, False, lambda s, h: NaiveForecaster()),
    ModelSpec("drift", "baseline", "Random walk with 1-year drift", "-", True, False, lambda s, h: DriftForecaster()),
    ModelSpec("arima", "statistical", "AutoARIMA on log price (statsforecast)", "-", True, False,
              lambda s, h: ArimaForecaster()),
    ModelSpec("dlinear", "ml", "DLinear quantile model trained locally on M2 (MPS)", "-", True, True,
              lambda s, h: DLinearForecaster(max_horizon=max(h, 20), device=s.get_path("forecasting.training_device", "auto"))),
    ModelSpec("lgbm", "ml", "LightGBM quantile regression on vol-normalised features", "-", True, True,
              lambda s, h: LGBMForecaster(horizon=h)),
]}

_CACHE: dict[tuple[str, int], Forecaster] = {}


def list_models(settings: Settings | None = None, include_noncommercial: bool | None = None) -> list[ModelSpec]:
    s = settings or get_settings()
    allow_nc = s.get_path("forecasting.allow_noncommercial_models", False) if include_noncommercial is None else include_noncommercial
    out = [m for m in SPECS.values() if m.commercial_ok or allow_nc]
    out.append(ModelSpec("ensemble", "ensemble", "Median/Vincentized ensemble of " + ", ".join(
        s.get_path("forecasting.ensemble_members", [])), "-", True, False, lambda *_: None))
    return out


def get_forecaster(name: str, horizon: int = 5, settings: Settings | None = None, fresh: bool = False) -> Forecaster:
    """Return a forecaster. Zero-shot models are cached (weights load once); trainable ones are fresh by default."""
    s = settings or get_settings()
    if name == "ensemble":
        members = [get_forecaster(m, horizon, s) for m in s.get_path("forecasting.ensemble_members", [])]
        return EnsembleForecaster(members)
    if name not in SPECS:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(SPECS) + ['ensemble']}")
    spec = SPECS[name]
    if spec.trainable or fresh:
        return spec.factory(s, horizon)
    key = (name, 0)
    if key not in _CACHE:
        _CACHE[key] = spec.factory(s, horizon)
    return _CACHE[key]
