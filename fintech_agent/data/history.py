"""Incremental on-disk history for daily datasets (FinMind 籌碼 etc.).

The first call downloads the full range; later calls only fetch the days after the last stored date (with a small
overlap for late revisions). This keeps the daily job inside FinMind's hourly quota instead of re-downloading years
of history for hundreds of stocks every day. Older date-keyed cache files are adopted on first use.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import pandas as pd

log = logging.getLogger(__name__)
_LOCK = threading.Lock()


class IncrementalHistory:
    def __init__(self, root: Path, legacy_dir: Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.legacy_dir = Path(legacy_dir) if legacy_dir else None

    def _path(self, dataset: str, code: str) -> Path:
        return self.root / f"{dataset}_{code}.parquet"

    def _load(self, dataset: str, code: str) -> pd.DataFrame:
        p = self._path(dataset, code)
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception as e:  # pragma: no cover - corrupt file: refetch
                log.warning("history %s unreadable (%s); refetching", p.name, e)
                return pd.DataFrame()
        if self.legacy_dir is not None:          # adopt the largest old date-keyed cache file, if any
            olds = sorted(self.legacy_dir.glob(f"{dataset}_{code}_*.parquet"), key=lambda f: f.stat().st_size)
            if olds:
                try:
                    return pd.read_parquet(olds[-1])
                except Exception:
                    pass
        return pd.DataFrame()

    def get(self, dataset: str, code: str, start: str, end: str, fetch: Callable[[str, str], pd.DataFrame],
            keys: tuple[str, ...] = ("date",), overlap_days: int = 5, stale_ok_days: int = 0) -> pd.DataFrame:
        """stale_ok_days: accept stored history that ends up to this many days before `end` (backtests don't need
        today's rows, and skipping the refresh saves one API call per stock)."""
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        with _LOCK:
            df = self._load(dataset, code)
        parts = [df] if len(df) else []
        if len(df):
            dates = pd.to_datetime(df["date"])
            lo, hi = dates.min(), dates.max()
            if start_ts < lo - pd.Timedelta(days=10):
                parts.append(fetch(start, (lo - pd.Timedelta(days=1)).date().isoformat()))
            if end_ts > hi + pd.Timedelta(days=stale_ok_days):
                parts.append(fetch((hi - pd.Timedelta(days=overlap_days)).date().isoformat(), end))
        else:
            parts.append(fetch(start, end))
        parts = [p for p in parts if p is not None and len(p)]
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"]).drop_duplicates(list(keys), keep="last").sort_values(list(keys))
        if len(parts) > 1 or not self._path(dataset, code).exists():
            with _LOCK:
                out.to_parquet(self._path(dataset, code))
        m = (out["date"] >= start_ts) & (out["date"] <= end_ts)
        return out[m].reset_index(drop=True)
