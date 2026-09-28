#!/usr/bin/env python
"""Train the locally-trained models on a ticker universe and save checkpoints for the live quant agent.

  python scripts/train_models.py --universe tw50 --models lgbm --horizons 5 20 --years 13
  python scripts/train_models.py --universe us50 --models lgbm lgbm-cov dlinear --horizons 5 20

Checkpoints: checkpoints/{model}_{market}_h{H}.pkl (LightGBM) / .pt (DLinear, trained on MPS).
The quant agent loads them automatically when the model is the champion or in `forecasting.panel`.
Benchmarks never use these files — they retrain on pre-cutoff data to avoid look-ahead bias.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fintech_agent.config import get_settings  # noqa: E402
from fintech_agent.data import LiveDataProvider  # noqa: E402
from fintech_agent.data.universe import UNIVERSES  # noqa: E402
from fintech_agent.forecasting.ml import DLinearForecaster  # noqa: E402
from fintech_agent.forecasting.registry import SPECS, checkpoint_path  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", choices=sorted(UNIVERSES), default="tw50")
    ap.add_argument("--models", nargs="*", default=["lgbm", "lgbm-cov"])
    ap.add_argument("--horizons", nargs="*", type=int, default=[5, 20])
    ap.add_argument("--epochs", type=int, default=20, help="DLinear epochs")
    ap.add_argument("--years", type=float, default=None, help="history to train on (default data.history_years)")
    args = ap.parse_args()
    s = get_settings()
    prov = LiveDataProvider(s)
    prov.fm.max_wait_s = 3600
    if args.years:
        prov.years = args.years
    market, tickers = UNIVERSES[args.universe]
    need_cov = any(SPECS[m].covariates for m in args.models if m in SPECS)
    series, covs = [], []
    for t in tickers:
        sym = prov.symbol(t)
        px = prov.prices(sym)
        if len(px) <= 300:
            continue
        c = prov.covariates(sym, px) if need_cov else None
        if need_cov and (c is None or not len(c)):
            continue
        series.append(px["close"].to_numpy())
        covs.append(c)
    print(f"{market}: {len(series)} tickers")
    for h in args.horizons:
        for name in args.models:
            spec = SPECS[name]
            ck = checkpoint_path(s, name, market, h)
            t0 = time.time()
            if name == "dlinear":
                m = DLinearForecaster(max_horizon=h, epochs=args.epochs, checkpoint=ck.with_suffix(".pt"),
                                      device=s.get_path("forecasting.training_device", "auto"))
                m.fit(series)
            else:
                m = spec.factory(s, h)
                m.fit(series, covs) if spec.covariates else m.fit(series)
                m.save(ck)
            print(f"  {name} h={h}: trained in {time.time() - t0:.1f}s → {ck.name}")


if __name__ == "__main__":
    main()
