"""Zero-shot time-series foundation models: Google TimesFM (2.5 / 3.0), Amazon Chronos-2 / Chronos-Bolt, NX-AI TiRex.

All heavy imports are lazy so the rest of the system works without torch installed.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

from .base import QUANTILES, Forecaster, ForecastResult, clean_context

log = logging.getLogger(__name__)
_LOCK = threading.Lock()


def resolve_device(pref: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if pref and pref != "auto":
        if pref == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        if pref.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TimesFM25Forecaster(Forecaster):
    """google/timesfm-2.5-200m-pytorch — v1 champion. Apache-2.0."""
    name, family = "timesfm-2.5", "foundation"
    repo = "google/timesfm-2.5-200m-pytorch"

    def __init__(self, device: str = "cpu", max_context: int = 1024, max_horizon: int = 256, batch_size: int = 32):
        self.device = resolve_device(device)
        self.max_context, self.max_horizon, self.batch_size = max_context, max_horizon, batch_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        with _LOCK:
            if self._model is None:
                import timesfm
                import torch
                m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(self.repo, torch_compile=False)
                if self.device != "cpu":  # the package only auto-selects cuda/cpu; route to MPS manually
                    m.model.device = torch.device(self.device)
                    m.model.to(self.device)
                m.compile(timesfm.ForecastConfig(
                    max_context=self.max_context, max_horizon=self.max_horizon, normalize_inputs=True,
                    use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
                    fix_quantile_crossing=True, per_core_batch_size=self.batch_size))
                self._model = m
        return self._model

    def predict(self, contexts, horizon):
        m = self._load()
        inputs = [clean_context(c, self.max_context).astype(np.float32) for c in contexts]
        # NB: timesfm pads the *input list in place* up to the batch size -> always pass a fresh list
        point, quant = m.forecast(horizon=horizon, inputs=list(inputs))
        n = len(inputs)
        point, quant = np.asarray(point)[:n, :horizon], np.asarray(quant)[:n, :horizon]
        # quant[..., 0] is the mean, [..., 1:10] are the 10th..90th percentiles
        return [ForecastResult(self.name, point[i].astype(float),
                               {q: quant[i, :, j + 1].astype(float) for j, q in enumerate(QUANTILES)})
                for i in range(n)]


class TimesFM3Forecaster(Forecaster):
    """google/timesfm-3.0-pytorch — research benchmark only (TimesFM Non-Commercial License v1.0)."""
    name, family = "timesfm-3.0", "foundation"
    license, commercial_ok = "TimesFM-NC-1.0", False
    repo = "google/timesfm-3.0-pytorch"

    def __init__(self, device: str = "cpu", max_context: int = 2048):
        self.device, self.max_context = resolve_device(device), max_context
        self._model = None

    def _load(self):
        if self._model is None:
            with _LOCK:
                if self._model is None:
                    import timesfm
                    self._model = timesfm.TimesFM3Forecaster.from_pretrained(self.repo, device=self.device)
        return self._model

    def predict(self, contexts, horizon):
        m = self._load()
        inputs = [clean_context(c, self.max_context).astype(np.float32) for c in contexts]
        outs = list(m.predict_batch(inputs, horizon=horizon, return_quantiles=True, make_positive=True))
        res = []
        for o in outs:
            point = np.asarray(o.forecast, dtype=float).reshape(-1)[:horizon]
            qarr = np.asarray(o.quantiles, dtype=float).reshape(horizon, -1) if o.quantiles is not None else None
            qs = {q: qarr[:, j] for j, q in enumerate(QUANTILES)} if qarr is not None and qarr.shape[1] == 9 else {}
            res.append(ForecastResult(self.name, point, qs))
        return res


class ChronosForecaster(Forecaster):
    """amazon/chronos-2 (120M) or amazon/chronos-bolt-{tiny,mini,small,base}. Apache-2.0."""
    family = "foundation"

    def __init__(self, repo: str, name: str, device: str = "cpu", max_context: int = 2048, batch_size: int = 64):
        self.repo, self.name = repo, name
        self.device, self.max_context, self.batch_size = resolve_device(device), max_context, batch_size
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            with _LOCK:
                if self._pipe is None:
                    from chronos import BaseChronosPipeline
                    self._pipe = BaseChronosPipeline.from_pretrained(self.repo, device_map=self.device)
        return self._pipe

    def predict(self, contexts, horizon):
        import torch
        pipe = self._load()
        tensors = [torch.tensor(clean_context(c, self.max_context), dtype=torch.float32) for c in contexts]
        res: list[ForecastResult] = []
        for s in range(0, len(tensors), self.batch_size):
            chunk = tensors[s:s + self.batch_size]
            qs, mean = pipe.predict_quantiles(chunk, prediction_length=horizon, quantile_levels=list(QUANTILES))
            for i in range(len(chunk)):
                q_i = qs[i] if isinstance(qs, list) else qs[i]
                m_i = mean[i] if isinstance(mean, list) else mean[i]
                q_np = q_i.detach().float().cpu().numpy().reshape(-1, horizon, len(QUANTILES))[0]
                m_np = m_i.detach().float().cpu().numpy().reshape(-1, horizon)[0]
                # median as point forecast (more robust than the mean for prices)
                res.append(ForecastResult(self.name, q_np[:, QUANTILES.index(0.5)].astype(float),
                                          {q: q_np[:, j].astype(float) for j, q in enumerate(QUANTILES)}))
        return res


class TiRexForecaster(Forecaster):
    """NX-AI/TiRex (xLSTM, 35M). Optional: `pip install tirex-ts`."""
    name, family = "tirex", "foundation"
    license, commercial_ok = "NXAI Community License", False

    def __init__(self, device: str = "cpu", max_context: int = 2048):
        self.device, self.max_context = resolve_device(device), max_context
        self._model = None

    def predict(self, contexts, horizon):
        import torch
        if self._model is None:
            from tirex import load_model
            self._model = load_model("NX-AI/TiRex", device=self.device)
        data = [torch.tensor(clean_context(c, self.max_context), dtype=torch.float32) for c in contexts]
        quantiles, mean = self._model.forecast(context=data, prediction_length=horizon)
        q = quantiles.detach().cpu().numpy()        # (N, h, 9) at 0.1..0.9
        return [ForecastResult(self.name, q[i, :, 4].astype(float), {lv: q[i, :, j].astype(float) for j, lv in enumerate(QUANTILES)})
                for i in range(len(data))]
