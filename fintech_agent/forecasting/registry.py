"""Model registry: name -> forecaster factory, with metadata used by the UI, the quant agent and A/B tests."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from ..config import Settings, get_settings
from .base import Forecaster
from .baselines import ArimaForecaster, DriftForecaster, EnsembleForecaster, NaiveForecaster
from .foundation import (ChronosCovForecaster, ChronosForecaster, TiRexForecaster, TimesFM3CovForecaster,
                         TimesFM3Forecaster, TimesFM25Forecaster, TimesFM25XRegForecaster)
from .ml import DLinearForecaster, LGBMCovForecaster, LGBMForecaster

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
    covariates: bool = False


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
    # ---- covariate-aware (籌碼 / market / FX): see data/covariates.py
    ModelSpec("chronos-2-cov", "foundation", "Chronos-2 + past covariates (法人/融資/大盤/費半/匯率/量能)",
              "Apache-2.0", True, False,
              lambda s, h: ChronosCovForecaster("amazon/chronos-2", "chronos-2-cov", _dev(s)), covariates=True),
    ModelSpec("timesfm-2.5-xreg", "foundation", "TimesFM 2.5 + XReg in-context ridge on h-day-lagged covariates",
              "Apache-2.0", True, False, lambda s, h: TimesFM25XRegForecaster(_dev(s)), covariates=True),
    ModelSpec("timesfm-3.0-cov", "foundation", "TimesFM 3.0 + past-only covariates · research only",
              "TimesFM-NC-1.0 (non-commercial)", False, False, lambda s, h: TimesFM3CovForecaster(_dev(s)),
              covariates=True),
    ModelSpec("lgbm-cov", "ml", "LightGBM quantile + covariate features (flows, index, FX changes)", "-", True, True,
              lambda s, h: LGBMCovForecaster(horizon=h), covariates=True),
]}

_CACHE: dict[tuple[str, int], Forecaster] = {}


def checkpoint_path(settings: Settings, name: str, market: str, horizon: int):
    """Where train_models.py stores a trained model for live use."""
    return settings.resolve_path("forecasting.checkpoints_dir") / f"{name}_{market}_h{horizon}.pkl"


def load_trained(name: str, market: str, horizon: int, settings: Settings | None = None) -> Forecaster | None:
    """A trainable model restored from its checkpoint, or None if it has not been trained for this market/horizon."""
    s = settings or get_settings()
    spec = SPECS.get(name)
    if spec is None or not spec.trainable:
        return None
    key = (f"{name}:{market}", horizon)
    if key in _CACHE:
        return _CACHE[key]
    p = checkpoint_path(s, name, market, horizon)
    if name == "dlinear":
        p = p.with_suffix(".pt")
    if not p.exists():
        return None
    fc = spec.factory(s, horizon)
    fc.load(p)
    _CACHE[key] = fc
    return fc


def list_models(settings: Settings | None = None, include_noncommercial: bool | None = None) -> list[ModelSpec]:
    s = settings or get_settings()
    allow_nc = s.get_path("forecasting.allow_noncommercial_models", False) if include_noncommercial is None else include_noncommercial
    out = [m for m in SPECS.values() if m.commercial_ok or allow_nc]
    out.append(ModelSpec("ensemble", "ensemble", "Median/Vincentized ensemble of " + ", ".join(
        s.get_path("forecasting.ensemble_members", [])), "-", True, False, lambda *_: None))
    out.append(ModelSpec("ensemble-cov", "ensemble", "Ensemble of " + ", ".join(
        s.get_path("forecasting.ensemble_cov_members", [])), "-", True, False, lambda *_: None, covariates=True))
    return out


def get_forecaster(name: str, horizon: int = 5, settings: Settings | None = None, fresh: bool = False) -> Forecaster:
    """Return a forecaster. Zero-shot models are cached (weights load once); trainable ones are fresh by default."""
    s = settings or get_settings()
    if name in ("ensemble", "ensemble-cov"):
        key = "forecasting.ensemble_members" if name == "ensemble" else "forecasting.ensemble_cov_members"
        members = [get_forecaster(m, horizon, s) for m in s.get_path(key, [])]
        return EnsembleForecaster(members, name=name)
    if name not in SPECS:
        raise KeyError(f"unknown model '{name}'. Available: {sorted(SPECS) + ['ensemble', 'ensemble-cov']}")
    spec = SPECS[name]
    if spec.trainable or fresh:
        return spec.factory(s, horizon)
    key = (name, 0)
    if key not in _CACHE:
        _CACHE[key] = spec.factory(s, horizon)
    return _CACHE[key]
