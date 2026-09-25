"""Common forecaster interface. Every model — foundation, statistical or trained locally — speaks this API
so the backtester, A/B tests and agents can swap them freely."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class ForecastResult:
    model: str
    point: np.ndarray                                   # (horizon,)
    quantiles: dict[float, np.ndarray] = field(default_factory=dict)   # level -> (horizon,)

    def __post_init__(self):
        self.point = np.asarray(self.point, dtype=float).reshape(-1)
        self.quantiles = {float(k): np.asarray(v, dtype=float).reshape(-1) for k, v in self.quantiles.items()}

    @property
    def horizon(self) -> int:
        return len(self.point)

    def q(self, level: float) -> np.ndarray:
        """Quantile path; linear interpolation between available levels, point forecast if none."""
        if not self.quantiles:
            return self.point
        if level in self.quantiles:
            return self.quantiles[level]
        lv = sorted(self.quantiles)
        lo = max([x for x in lv if x <= level], default=lv[0])
        hi = min([x for x in lv if x >= level], default=lv[-1])
        if lo == hi:
            return self.quantiles[lo]
        w = (level - lo) / (hi - lo)
        return (1 - w) * self.quantiles[lo] + w * self.quantiles[hi]

    def prob_above(self, threshold: float, step: int = -1) -> float:
        """P(value at `step` > threshold) from the quantile curve (piecewise-linear CDF)."""
        if not self.quantiles:
            return float(self.point[step] > threshold)
        lv = sorted(self.quantiles)
        vals = np.array([self.quantiles[x][step] for x in lv])
        vals = np.maximum.accumulate(vals)                         # guard against crossing
        if threshold <= vals[0]:
            # extrapolate tail linearly using the first segment's slope
            return float(1 - max(0.0, lv[0] - (vals[0] - threshold) * (lv[1] - lv[0]) / max(vals[1] - vals[0], 1e-12)))
        if threshold >= vals[-1]:
            return float(max(0.0, 1 - lv[-1] - (threshold - vals[-1]) * (lv[-1] - lv[-2]) / max(vals[-1] - vals[-2], 1e-12)))
        cdf = float(np.interp(threshold, vals, lv))
        return 1 - cdf

    def to_dict(self) -> dict:
        return {"model": self.model, "point": self.point.round(4).tolist(),
                "quantiles": {str(k): v.round(4).tolist() for k, v in self.quantiles.items()}}


class Forecaster:
    """Base class. Subclasses implement `predict`; trainable ones also `fit`."""

    name: str = "base"
    family: str = "baseline"            # foundation | statistical | ml | baseline | ensemble
    license: str = "Apache-2.0"
    commercial_ok: bool = True
    trainable: bool = False

    def fit(self, series: list[np.ndarray]) -> "Forecaster":
        return self

    def predict(self, contexts: list[np.ndarray], horizon: int) -> list[ForecastResult]:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} {self.name}>"


def clean_context(x: np.ndarray, max_len: int | None = None) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    return a[-max_len:] if max_len else a
