"""Tiny on-disk cache (parquet for DataFrames, JSON for dicts/lists) with TTL."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd


class DiskCache:
    def __init__(self, root: Path, ttl_hours: float = 12.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_hours * 3600

    def _path(self, key: str, ext: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)[:80]
        digest = hashlib.md5(key.encode()).hexdigest()[:8]
        return self.root / f"{safe}_{digest}.{ext}"

    def _fresh(self, p: Path) -> bool:
        return p.exists() and (time.time() - p.stat().st_mtime) < self.ttl

    def frame(self, key: str, loader: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        p = self._path(key, "parquet")
        if self._fresh(p):
            try:
                return pd.read_parquet(p)
            except Exception:
                pass
        df = loader()
        if df is not None and len(df):
            try:
                df.to_parquet(p)
            except Exception:
                pass
        return df

    def obj(self, key: str, loader: Callable[[], Any]) -> Any:
        p = self._path(key, "json")
        if self._fresh(p):
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        val = loader()
        if val:
            try:
                p.write_text(json.dumps(val, ensure_ascii=False, default=str), encoding="utf-8")
            except Exception:
                pass
        return val

    def clear(self) -> int:
        n = 0
        for p in self.root.glob("*"):
            if p.is_file():
                p.unlink()
                n += 1
        return n
