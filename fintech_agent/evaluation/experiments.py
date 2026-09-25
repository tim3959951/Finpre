"""Experiment tracking, champion registry and shadow-mode (online A/B) prediction log. SQLite + JSON, no server."""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..forecasting.base import ForecastResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, created_at TEXT, kind TEXT, market TEXT, horizon INTEGER,
  tickers TEXT, models TEXT, config TEXT, leaderboard TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, model TEXT, ticker TEXT, market TEXT,
  origin_date TEXT, horizon INTEGER, last_price REAL, point_h REAL, q10_h REAL, q50_h REAL, q90_h REAL,
  p_up REAL, role TEXT, actual_h REAL, resolved_at TEXT);
CREATE INDEX IF NOT EXISTS ix_pred_unresolved ON predictions(resolved_at);
"""


class ExperimentStore:
    def __init__(self, settings: Settings | None = None, path: Path | None = None):
        s = settings or get_settings()
        self.dir = s.resolve_path("evaluation.runs_dir")
        self.path = path or self.dir / "experiments.sqlite"
        self.champion_file = self.dir / "champion.json"
        self.default_champion = s.get_path("forecasting.champion", "timesfm-2.5")
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    # ------------------------------------------------------------------ offline runs
    def log_run(self, kind: str, market: str, horizon: int, tickers: list[str], models: list[str], config: dict,
                lb: pd.DataFrame, notes: str = "") -> str:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self._conn() as c:
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (run_id, datetime.now().isoformat(timespec="seconds"), kind, market, horizon,
                       json.dumps(tickers), json.dumps(models), json.dumps(config, default=str),
                       lb.to_json(orient="records"), notes))
        return run_id

    def runs(self, limit: int = 50) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql("SELECT run_id, created_at, kind, market, horizon, tickers, models, notes "
                               "FROM runs ORDER BY created_at DESC LIMIT ?", c, params=(limit,))

    def run_leaderboard(self, run_id: str) -> pd.DataFrame:
        with self._conn() as c:
            row = c.execute("SELECT leaderboard FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return pd.read_json(row[0]) if row else pd.DataFrame()

    # ------------------------------------------------------------------ champion registry
    def champions(self) -> dict:
        if self.champion_file.exists():
            return json.loads(self.champion_file.read_text())
        return {}

    def champion(self, market: str, horizon: int) -> str:
        return self.champions().get(market, {}).get(str(horizon), {}).get("model", self.default_champion)

    def set_champion(self, market: str, horizon: int, model: str, reason: str, run_id: str | None = None) -> None:
        ch = self.champions()
        ch.setdefault(market, {})[str(horizon)] = {"model": model, "since": datetime.now().isoformat(timespec="seconds"),
                                                   "reason": reason, "run_id": run_id}
        self.champion_file.write_text(json.dumps(ch, ensure_ascii=False, indent=2))

    # ------------------------------------------------------------------ shadow mode (online A/B)
    def log_prediction(self, fc: ForecastResult, ticker: str, market: str, origin_date: str, last_price: float,
                       role: str = "shadow") -> None:
        h = fc.horizon
        with self._conn() as c:
            c.execute("INSERT INTO predictions (created_at, model, ticker, market, origin_date, horizon, last_price, "
                      "point_h, q10_h, q50_h, q90_h, p_up, role) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (datetime.now().isoformat(timespec="seconds"), fc.model, ticker, market, origin_date, h,
                       last_price, float(fc.point[-1]), float(fc.q(0.1)[-1]), float(fc.q(0.5)[-1]),
                       float(fc.q(0.9)[-1]), float(fc.prob_above(last_price)), role))

    def resolve(self, price_lookup) -> int:
        """Fill actuals for predictions whose horizon has elapsed. price_lookup(ticker) -> pd.Series of closes."""
        with self._conn() as c:
            pending = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NULL", c)
        n = 0
        for ticker, g in pending.groupby("ticker"):
            s = price_lookup(ticker)
            if s is None or len(s) == 0:
                continue
            for _, r in g.iterrows():
                after = s[s.index > pd.Timestamp(r["origin_date"])]
                if len(after) >= r["horizon"]:
                    with self._conn() as c:
                        c.execute("UPDATE predictions SET actual_h=?, resolved_at=? WHERE id=?",
                                  (float(after.iloc[int(r["horizon"]) - 1]), datetime.now().isoformat(timespec="seconds"),
                                   int(r["id"])))
                    n += 1
        return n

    def online_scores(self) -> pd.DataFrame:
        with self._conn() as c:
            df = pd.read_sql("SELECT * FROM predictions WHERE resolved_at IS NOT NULL", c)
        if df.empty:
            return df
        df["abs_err_pct"] = (df["point_h"] - df["actual_h"]).abs() / df["actual_h"] * 100
        df["dir_hit"] = np.sign(df["point_h"] - df["last_price"]) == np.sign(df["actual_h"] - df["last_price"])
        df["in80"] = (df["actual_h"] >= df["q10_h"]) & (df["actual_h"] <= df["q90_h"])
        return df.groupby(["model", "horizon"]).agg(n=("id", "count"), mape_h=("abs_err_pct", "mean"),
                                                   dir_acc=("dir_hit", "mean"), cov80=("in80", "mean")).reset_index()
