#!/usr/bin/env python
"""Train the local ML models on the M2 (DLinear on MPS GPU) across a ticker universe and save a checkpoint.

  python scripts/train_models.py --market TW --epochs 20
  python scripts/train_models.py --tickers 2330 2317 AAPL NVDA --device mps --horizon 20

The checkpoint (checkpoints/dlinear_h{H}.pt) is what the quant agent loads when `dlinear` is in its panel.
Benchmarks do NOT use it — they retrain on pre-cutoff data to avoid look-ahead bias.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fintech_agent.config import get_settings  # noqa: E402
from fintech_agent.data import LiveDataProvider  # noqa: E402
from fintech_agent.forecasting.ml import DLinearForecaster  # noqa: E402

UNIVERSE = {"TW": ["2330", "2317", "2454", "2308", "2881", "2882", "2412", "1301", "2002", "3008", "2603", "0050", "0056"],
            "US": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "JNJ", "WMT", "SPY", "QQQ", "TSM"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="TW", choices=["TW", "US", "ALL"])
    ap.add_argument("--tickers", nargs="*")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--context", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default=None, help="mps | cpu | cuda (default: settings.forecasting.training_device)")
    args = ap.parse_args()
    s = get_settings()
    prov = LiveDataProvider(s)
    tickers = args.tickers or (UNIVERSE["TW"] + UNIVERSE["US"] if args.market == "ALL" else UNIVERSE[args.market])
    series = []
    for t in tickers:
        px = prov.prices(prov.symbol(t))
        if len(px) > 300:
            series.append(px["close"].to_numpy())
            print(f"  {t}: {len(px)} bars")
    ck = s.resolve_path("forecasting.checkpoints_dir") / f"dlinear_h{args.horizon}.pt"
    model = DLinearForecaster(context=args.context, max_horizon=args.horizon, epochs=args.epochs,
                              device=args.device or s.get_path("forecasting.training_device", "auto"), checkpoint=ck)
    t0 = time.time()
    model.fit(series)
    print(f"trained in {time.time() - t0:.1f}s on {model.device}: {model.train_info}\ncheckpoint → {ck}")


if __name__ == "__main__":
    main()
