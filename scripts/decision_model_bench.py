#!/usr/bin/env python
"""Decision-model benchmark: can a System One model (TypeSafe Jev, or the open Strands Decider) judge
(a) the 5-day up/down of a stock from a market snapshot, and (b) the price impact of a headline?

  # the open model serves the same POST /v1/systemone API as Jev
  strands-decider serve StrandsAgents/strands-decider-2B-hobson-v19 --port 8100 --device mps
  python scripts/decision_model_bench.py updown --universe tw50 --backend strands
  python scripts/decision_model_bench.py updown --universe us50 --backend typesafe        # needs TYPESAFE_API_KEY
  python scripts/decision_model_bench.py news --markets US TW --backend strands

Up/down protocol
  * The (ticker, origin) windows and labels come from the forecasting backtest (logs/b50_{universe}_h5.csv), so
    the decision model and every time-series / ML model are scored on identical rows with identical labels.
  * The state uses only data up to the close before the origin and is anonymized (no ticker, name or date):
    a language model cannot look up what happened next.
  * Metrics per model: Brier, log loss, AUC, accuracy, ECE, per-origin rank IC, top-minus-bottom quintile
    return, and Brier after walk-forward Platt calibration (fit only on earlier origins).

News protocol
  * Headlines from Yahoo Finance (US) and FinMind (TW) accumulate in logs/news_{market}.jsonl across runs.
  * Labels: abnormal return (stock minus index) from the last close before publication to the next close
    ("reaction"), and over the session after that ("drift", the tradable part).
  * Scorers: the decision model (impact score + relevance), the keyword lexicon (features/sentiment.py) and
    FinBERT (ProsusAI/finbert, English only) when transformers is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fintech_agent.features import indicators as I  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
JEV_COST_PER_M_INPUT = 0.042          # USD per 1M input tokens (TypeSafe pricing, output tokens free)
MARKET_INDEX = {"TW": "^TWII", "US": "^GSPC"}
CLOSE_TIME = {"TW": ("Asia/Taipei", 13, 30), "US": ("America/New_York", 16, 0)}


# =========================================================================================== backend
class SystemOne:
    """Client for POST /v1/systemone (TypeSafe Jev or a local strands-decider server), with a JSONL cache."""

    def __init__(self, backend: str, url: str | None = None, model: str | None = None,
                 cache: Path | None = None, concurrency: int = 1, timeout: float = 120.0):
        self.backend, self.timeout, self.concurrency = backend, timeout, max(1, concurrency)
        self.headers = {"content-type": "application/json"}
        if backend == "typesafe":
            key = os.environ.get("TYPESAFE_API_KEY")
            if not key:
                raise SystemExit("TYPESAFE_API_KEY is not set (get a key at typesafe.ai, then export it)")
            self.url = url or "https://api.typesafe.ai/v1/systemone"
            self.headers["authorization"] = f"Bearer {key}"
            self.model = model or "jev-latest"
        elif backend == "strands":
            self.url = url or "http://127.0.0.1:8100/v1/systemone"
            self.model = model or self._served_model() or "strands-decider"
        elif backend == "mock":                    # deterministic fake answers, for tests
            self.url, self.model = "mock", "mock"
        else:
            raise SystemExit(f"unknown backend {backend}")
        self.ident = f"{backend}:{self.model}"
        self.cache_path = cache
        self._cache: dict[str, dict] = {}
        self._lock = threading.Lock()
        if cache and cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                    self._cache[r["key"]] = r["response"]
                except Exception:
                    continue

    def _served_model(self) -> str | None:
        import requests
        try:
            return requests.get(self.url.rsplit("/v1/", 1)[0] + "/health", timeout=10).json().get("model")
        except Exception:
            return None

    def _key(self, payload: dict) -> str:
        blob = json.dumps({"ident": self.ident, **payload}, sort_keys=False, ensure_ascii=False)
        return hashlib.sha1(blob.encode()).hexdigest()

    def _post(self, payload: dict) -> dict:
        if self.backend == "mock":
            return _mock_answer(payload)
        import requests
        delay = 2.0
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                r = requests.post(self.url, json=payload, headers=self.headers, timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == 5:
                    raise
                logging.warning("request failed (%s); retrying in %.0fs", e, delay)
                time.sleep(delay)
                delay *= 2
                continue
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(float(r.headers.get("retry-after", delay)))
                delay *= 2
                continue
            r.raise_for_status()
            out = r.json()
            out.setdefault("latency_ms", round((time.perf_counter() - t0) * 1000, 1))
            return out
        raise RuntimeError("unreachable")

    def ask(self, state, questions: dict) -> dict:
        payload = {"model": self.model, "state": state, "questions": questions}
        key = self._key(payload)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return {**hit, "cached": True}
        out = self._post(payload)
        with self._lock:
            self._cache[key] = out
            if self.cache_path:
                with self.cache_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "response": out}, ensure_ascii=False) + "\n")
        return out

    def ask_many(self, items: list[tuple], progress_every: int = 100) -> list[dict]:
        out: list[dict | None] = [None] * len(items)
        t0, done = time.time(), [0]

        def run(i):
            out[i] = self.ask(*items[i])
            with self._lock:
                done[0] += 1
                if done[0] % progress_every == 0 or done[0] == len(items):
                    rate = done[0] / max(time.time() - t0, 1e-9)
                    eta = (len(items) - done[0]) / max(rate, 1e-9)
                    print(f"  {done[0]}/{len(items)} · {rate:.2f}/s · eta {eta / 60:.1f} min", flush=True)
        if self.concurrency == 1:
            for i in range(len(items)):
                run(i)
        else:
            with ThreadPoolExecutor(self.concurrency) as ex:
                list(ex.map(run, range(len(items))))
        return out  # type: ignore[return-value]


def _mock_answer(payload: dict) -> dict:
    h = int(hashlib.md5(json.dumps(payload["state"], ensure_ascii=False).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)
    ans = {}
    for name, q in payload["questions"].items():
        if q["type"] == "noul":
            ans[name] = {"type": "noul", "noul": float(rng.uniform(0.3, 0.7))}
        else:
            opts = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
            p = rng.dirichlet(np.ones(len(opts)))
            probs = {o: float(x) for o, x in zip(opts, p)}
            if q["type"] == "choice":
                ans[name] = {"type": "choice", "choice": opts[int(p.argmax())], "probabilities": probs}
            else:
                ans[name] = {"type": "score", "score": float(np.dot(p, np.arange(len(p)))), "probabilities": probs}
    return {"model": "mock", "answers": ans, "usage": {"input_tokens": 400, "output_tokens": 3}, "latency_ms": 0.0}


def noul_p(a: dict) -> float:
    v = a.get("noul")
    if isinstance(v, bool) or v is None:
        probs = a.get("probabilities") or {}
        return float(probs.get("true", 1.0 if v else 0.0))
    return float(v)


def choice_p(a: dict, option: str) -> float:
    probs = a.get("probabilities") or {}
    if probs:
        return float(probs.get(option, 0.0))
    return 1.0 if a.get("choice") == option else 0.0


def score_unit(a: dict, levels: int) -> float:
    """Expected level in [0, 1] (0 = first level, 1 = last)."""
    probs = a.get("probabilities") or {}
    if probs:
        p = np.array([float(probs.get(str(i), 0.0)) for i in range(levels)])
        if p.sum() > 0:
            return float(np.dot(p / p.sum(), np.arange(levels)) / (levels - 1))
    return float(a.get("score", (levels - 1) / 2)) / (levels - 1)


# =========================================================================================== up/down
def updown_questions(reverse: bool = False) -> dict:
    move = {"rise": "The price rises by more than 2%.",
            "flat": "The price changes by 2% or less in either direction.",
            "fall": "The price falls by more than 2%."}
    up = {"true": "The close 5 trading days later is above the latest close.",
          "false": "The close 5 trading days later is at or below the latest close."}
    if reverse:
        move, up = dict(reversed(list(move.items()))), dict(reversed(list(up.items())))
    return {
        "up": {"type": "noul", "instructions": "Will this stock's closing price 5 trading days from now be higher "
                                               "than its latest close?", "criteria": up},
        "move": {"type": "choice", "instructions": "How will this stock's price change over the next 5 trading days?",
                 "criteria": move},
        "outlook": {"type": "score", "instructions": "Rate the price outlook for this stock over the next 5 trading days.",
                    "criteria": ["clearly bearish", "somewhat bearish", "neutral", "somewhat bullish",
                                 "clearly bullish"]},
    }


def feature_frame(px: pd.DataFrame, mkt_close: pd.Series | None, cov: pd.DataFrame | None) -> pd.DataFrame:
    """Causal (past-only) features for every row: row t uses data up to and including t."""
    c, v = px["close"].astype(float), px["volume"].astype(float)
    f = pd.DataFrame(index=px.index)
    for k in (1, 5, 20, 60, 250):
        f[f"r{k}"] = c / c.shift(k) - 1
    for n in (20, 60, 240):
        f[f"ma{n}"] = c / c.rolling(n, min_periods=int(n * 0.8)).mean() - 1
    f["rsi"] = I.rsi(c, 14)
    kd = I.kd(px["high"], px["low"], c)
    f["k"], f["d"] = kd["k"], kd["d"]
    osc = I.macd(c)["osc"]
    f["macd"], f["macd_chg3"] = osc, osc - osc.shift(3)
    f["vol_ratio"] = v / v.rolling(20, min_periods=10).mean()
    lr = np.log(c).diff()
    f["hv20"] = lr.rolling(20, min_periods=15).std() * math.sqrt(252)
    lo, hi = c.rolling(252, min_periods=200).min(), c.rolling(252, min_periods=200).max()
    f["pos52"] = (c - lo) / (hi - lo).replace(0, np.nan)
    if mkt_close is not None and len(mkt_close):
        m = mkt_close.astype(float).sort_index()
        m = m[~m.index.duplicated(keep="last")]
        m = m.reindex(m.index.union(px.index)).ffill().reindex(px.index)
        f["m_r5"], f["m_r20"] = m / m.shift(5) - 1, m / m.shift(20) - 1
        f["m_ma200"] = m / m.rolling(200, min_periods=150).mean() - 1
        f["m_hv20"] = np.log(m).diff().rolling(20, min_periods=15).std() * math.sqrt(252)
    if cov is not None and len(cov):
        cv = cov.reindex(px.index)
        for g in ("foreign", "trust"):
            col = f"{g}_flow_lvl"
            if col in cv:
                lvl = cv[col]
                f[f"{g}5"] = lvl - lvl.shift(5)
                f[f"{g}_days5"] = (lvl.diff() > 0).astype(float).rolling(5).sum()
        if "margin_z" in cv:
            f["margin"] = cv["margin_z"]
    return f


def _pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:+.{digits}f}%"


def _rel(x: float, what: str) -> str:
    return f"{abs(x) * 100:.1f}% {'above' if x >= 0 else 'below'} its {what}"


def render_state(r: pd.Series, market: str) -> str:
    ok = lambda *ks: all(k in r and pd.notna(r[k]) for k in ks)  # noqa: E731
    region = "Taiwan" if market == "TW" else "US"
    lines = [f"Daily snapshot of a large-cap {region} stock at its latest close. No information after this close."]
    rets = [f"{lab} {_pct(r[k])}" for k, lab in (("r1", "1-day"), ("r5", "5-day"), ("r20", "20-day"),
                                                 ("r60", "60-day"), ("r250", "250-day")) if ok(k)]
    if rets:
        lines.append("Returns: " + ", ".join(rets) + ".")
    trend = [_rel(r[k], lab) for k, lab in (("ma20", "20-day average"), ("ma60", "60-day average"),
                                             ("ma240", "240-day average")) if ok(k)]
    if trend:
        lines.append("Trend: close is " + "; ".join(trend) + ".")
    osc = []
    if ok("rsi"):
        osc.append(f"RSI(14) {r['rsi']:.0f}")
    if ok("k", "d"):
        osc.append(f"stochastic K {r['k']:.0f}, D {r['d']:.0f}")
    if ok("macd", "macd_chg3"):
        osc.append(f"MACD histogram {'positive' if r['macd'] >= 0 else 'negative'} and "
                   f"{'rising' if r['macd_chg3'] >= 0 else 'falling'} over the last 3 days")
    if osc:
        lines.append("Oscillators: " + "; ".join(osc) + ".")
    vol = []
    if ok("vol_ratio"):
        vol.append(f"volume {r['vol_ratio']:.1f}x its 20-day average")
    if ok("hv20"):
        vol.append(f"20-day volatility {r['hv20'] * 100:.0f}% annualized")
    if ok("pos52"):
        vol.append(f"close at {r['pos52'] * 100:.0f}% of its 52-week range (0% = low, 100% = high)")
    if vol:
        lines.append("Activity: " + "; ".join(vol) + ".")
    if ok("m_r5", "m_r20"):
        s = f"Market index: 5-day {_pct(r['m_r5'])}, 20-day {_pct(r['m_r20'])}"
        if ok("m_ma200"):
            s += f"; index {_rel(r['m_ma200'], '200-day average')}"
        if ok("m_hv20"):
            s += f"; index volatility {r['m_hv20'] * 100:.0f}% annualized"
        lines.append(s + ".")
    flows = []
    if ok("foreign5", "foreign_days5"):
        flows.append(f"foreign investors were net buyers on {int(r['foreign_days5'])} of the last 5 days, "
                     f"5-day net {r['foreign5']:+.1f}x average daily volume")
    if ok("trust5"):
        flows.append(f"investment trusts 5-day net {r['trust5']:+.1f}x average daily volume")
    if ok("margin"):
        flows.append(f"margin balance {_rel(r['margin'], '60-day average')}")
    if flows:
        lines.append("Institutional flows: " + "; ".join(flows) + ".")
    return "\n".join(lines)


def load_grid(path: Path, min_origin: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unique (ticker, origin) windows with the backtest label, and the models' p_up wide table."""
    w = pd.read_csv(path, dtype={"ticker": str})
    w["origin"] = pd.to_datetime(w["origin"])
    if min_origin:
        w = w[w["origin"] >= pd.Timestamp(min_origin)]
    grid = w.drop_duplicates(["ticker", "origin"])[["ticker", "origin", "ret_true", "last"]].reset_index(drop=True)
    wide = w.pivot_table(index=["ticker", "origin"], columns="model", values="p_up")
    return grid, wide


def build_states(prov, grid: pd.DataFrame, market: str, chips: bool) -> pd.DataFrame:
    from fintech_agent.data import parse_symbol
    mk = prov.prices(parse_symbol(MARKET_INDEX[market]), years=getattr(prov, "years", 6) + 1)
    mkt_close = mk["close"] if len(mk) else None
    rows = []
    for t, g in grid.groupby("ticker"):
        sym = parse_symbol(t, default_market=market)
        px = prov.prices(sym)
        if len(px) < 300:
            print(f"  {t}: only {len(px)} bars, skipped")
            continue
        cov = None
        if chips and market == "TW":
            try:
                cov = prov.covariates(sym, px)
            except Exception as e:                    # quota or network: run without chips for this stock
                print(f"  {t}: covariates unavailable ({e})")
        f = feature_frame(px, mkt_close, cov)
        pos = px.index.get_indexer(pd.DatetimeIndex(g["origin"]))
        for (_, r), o in zip(g.iterrows(), pos):
            if o <= 0:
                continue
            row = f.iloc[o - 1]
            rows.append({"ticker": t, "origin": r["origin"], "ret_true": r["ret_true"],
                         "asof": px.index[o - 1], "last_ratio": float(px["close"].iloc[o - 1] / r["last"]),
                         "state": render_state(row, market), **{k: row[k] for k in f.columns}})
        print(f"  {t}: {len(g)} windows", flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- metrics
def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _fit_platt(x: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> tuple[float, float]:
    """1-D logistic regression y ~ sigmoid(a + b x) by Newton's method with a small ridge."""
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(50):
        p = 1 / (1 + np.exp(-(X @ w)))
        g = X.T @ (p - y) + ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    a, b = w
    return float(a), float(b)


def walk_forward_calibrate(df: pd.DataFrame, col: str, min_origins: int = 8) -> pd.Series:
    """Platt-calibrate `col` using only windows whose origin is earlier (their 5-day outcome is known by then,
    since origins are 5 trading days apart)."""
    out = pd.Series(np.nan, index=df.index)
    origins = np.sort(df["origin"].unique())
    y = (df["ret_true"] > 0).to_numpy(float)
    x = _logit(df[col].to_numpy())
    od = df["origin"].to_numpy()
    for i, o in enumerate(origins):
        if i < min_origins:
            continue
        tr, te = od < o, od == o
        a, b = _fit_platt(x[tr], y[tr])
        out[te] = 1 / (1 + np.exp(-(a + b * x[te])))
    return out


def auc(p: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import rankdata
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(tot)


def per_origin(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for o, g in df.groupby("origin"):
        y = (g["ret_true"] > 0).to_numpy(float)
        ic = spread = np.nan
        if len(g) >= 10 and g[col].nunique() >= 2:             # a constant forecast has no ranking
            ic = g[col].rank().corr(g["ret_true"].rank())
            q = pd.qcut(g[col].rank(method="first"), 5, labels=False)
            spread = g["ret_true"][q == 4].mean() - g["ret_true"][q == 0].mean()
        rows.append({"origin": o, "ic": ic, "spread": spread, "brier": np.mean((g[col].to_numpy() - y) ** 2)})
    return pd.DataFrame(rows, columns=["origin", "ic", "spread", "brier"])


def boot_ci(x: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def score_models(df: pd.DataFrame, models: list[str], ref: str = "coin") -> pd.DataFrame:
    """One row per model on the rows where every model has a prediction."""
    df = df.dropna(subset=models).copy()
    y = (df["ret_true"] > 0).to_numpy(float)
    cal = {m: walk_forward_calibrate(df, m) for m in models}
    has_cal = np.all([cal[m].notna().to_numpy() for m in models], axis=0)
    per = {m: per_origin(df, m) for m in models}
    ref_brier = per_origin(df, ref).set_index("origin")["brier"] if ref in models else None
    rows = []
    for m in models:
        p = np.clip(df[m].to_numpy(float), 1e-4, 1 - 1e-4)
        po = per[m]
        r = {"model": m, "n": len(df), "brier": np.mean((p - y) ** 2),
             "logloss": -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)),
             "auc": auc(p, y), "acc": np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1))),
             "ece": ece(p, y), "mean_p": p.mean()}
        r["bss_coin"] = 1 - r["brier"] / 0.25
        ics = po["ic"].dropna()
        if len(ics) > 2:
            r["ic"] = ics.mean()
            r["ic_t"] = ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics)))
            r["ic_lo"], r["ic_hi"] = boot_ci(ics.to_numpy())
            r["ic_pos_share"] = (ics > 0).mean()
            r["spread_pct"] = po["spread"].mean() * 100
            r["spread_lo"], r["spread_hi"] = (v * 100 for v in boot_ci(po["spread"].to_numpy()))
        if ref_brier is not None and len(po):
            d = po.set_index("origin")["brier"] - ref_brier.reindex(po["origin"]).to_numpy()
            r["dbrier_vs_ref"] = d.mean()
            r["dbrier_lo"], r["dbrier_hi"] = boot_ci(d.to_numpy())
        pc = cal[m].to_numpy()[has_cal]
        r["n_cal"] = int(has_cal.sum())
        r["brier_cal"] = np.mean((pc - y[has_cal]) ** 2) if has_cal.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """coin = 0.5; base_rate = share of up-moves among windows with an earlier origin (walk-forward)."""
    df = df.copy()
    df["coin"] = 0.5
    up = (df["ret_true"] > 0).astype(float)
    by = up.groupby(df["origin"]).agg(["sum", "count"]).sort_index()
    prev = (by["sum"].cumsum() - by["sum"]) / (by["count"].cumsum() - by["count"]).replace(0, np.nan)
    df["base_rate"] = df["origin"].map(prev.fillna(0.5))
    return df


def fmt_table(lb: pd.DataFrame) -> str:
    cols = [("model", "model"), ("brier", "Brier"), ("bss_coin", "BSS vs 0.5"), ("brier_cal", "Brier (calibrated)"),
            ("auc", "AUC"), ("acc", "Acc"), ("ece", "ECE"), ("ic", "IC"), ("ic_t", "IC t"),
            ("spread_pct", "Q5−Q1 %")]
    cols = [(k, h) for k, h in cols if k in lb]
    lines = ["| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in lb.iterrows():
        cells = []
        for k, _ in cols:
            v = r[k]
            if k == "model":
                cells.append(str(v))
            elif k in ("ic_t",):
                cells.append(f"{v:+.2f}" if pd.notna(v) else "–")
            elif k in ("bss_coin", "ic", "spread_pct"):
                cells.append(f"{v:+.3f}" if pd.notna(v) else "–")
            else:
                cells.append(f"{v:.4f}" if pd.notna(v) else "–")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cmd_updown(args) -> None:
    from fintech_agent.config import get_settings
    from fintech_agent.data import LiveDataProvider
    from fintech_agent.data.universe import UNIVERSES
    market = UNIVERSES[args.universe][0]
    src = Path(args.csv or LOGS / f"b50_{args.universe}_h5.csv")
    out = Path(args.out or LOGS / f"dm_{args.universe}_h5")
    grid, wide = load_grid(src, args.min_origin)
    if args.limit_origins:
        keep = np.sort(grid["origin"].unique())[-args.limit_origins:]
        grid = grid[grid["origin"].isin(keep)]
    print(f"{len(grid)} windows · {grid['ticker'].nunique()} stocks · {grid['origin'].nunique()} origins from {src.name}")

    states_path = out.with_name(out.name + "_states.parquet")
    if states_path.exists() and not args.rebuild_states:
        st = pd.read_parquet(states_path)
        st = st.merge(grid[["ticker", "origin"]], on=["ticker", "origin"])
    else:
        prov = LiveDataProvider(get_settings())
        prov.fm.max_wait_s = 0
        prov.chips_stale_ok_days = 3650            # use stored 籌碼 history; never spend quota on this benchmark
        st = build_states(prov, grid, market, chips=not args.no_chips)
        st.to_parquet(states_path)
    print(f"{len(st)} states · median last-price ratio vs backtest {st['last_ratio'].median():.4f}")
    print("example state:\n" + st["state"].iloc[len(st) // 2] + "\n")

    client = SystemOne(args.backend, args.url, args.model, cache=LOGS / "systemone_cache.jsonl",
                       concurrency=args.concurrency)
    t0 = time.time()
    res = client.ask_many([(s, updown_questions()) for s in st["state"]])
    wall = time.time() - t0
    dm = pd.DataFrame({
        "dm_up": [noul_p(r["answers"]["up"]) for r in res],
        "dm_rise": [choice_p(r["answers"]["move"], "rise") for r in res],
        "dm_flat": [choice_p(r["answers"]["move"], "flat") for r in res],
        "dm_fall": [choice_p(r["answers"]["move"], "fall") for r in res],
        "dm_outlook": [score_unit(r["answers"]["outlook"], 5) for r in res],
        "latency_ms": [r.get("latency_ms", np.nan) if not r.get("cached") else np.nan for r in res],
        "input_tokens": [(r.get("usage") or {}).get("input_tokens", np.nan) for r in res],
    })
    st = pd.concat([st.reset_index(drop=True), dm], axis=1)
    st["dm_move"] = st["dm_rise"] + 0.5 * st["dm_flat"]

    order = {}
    if args.order_check:
        sub = st.sample(min(args.order_check, len(st)), random_state=0)
        rev = client.ask_many([(s, updown_questions(reverse=True)) for s in sub["state"]])
        d_up = np.abs(np.array([noul_p(r["answers"]["up"]) for r in rev]) - sub["dm_up"].to_numpy())
        d_rise = np.abs(np.array([choice_p(r["answers"]["move"], "rise") for r in rev]) - sub["dm_rise"].to_numpy())
        top_a = sub[["dm_rise", "dm_flat", "dm_fall"]].to_numpy().argmax(1)
        top_b = np.array([[choice_p(r["answers"]["move"], o) for o in ("rise", "flat", "fall")] for r in rev]).argmax(1)
        order = {"n": len(sub), "mean_abs_diff_up": float(d_up.mean()), "mean_abs_diff_rise": float(d_rise.mean()),
                 "choice_flip_rate": float((top_a != top_b).mean())}
        print(f"order sensitivity: {order}")

    # ------------------------------------------------------------- compare with the forecasting models
    merged = st.merge(wide.reset_index(), on=["ticker", "origin"], how="left")
    ml_cols = [c for c in wide.columns if merged[c].notna().mean() > 0.95]
    for extra in args.extra or []:                 # e.g. logs/lh_tw50_h5_ml.csv:monthly (LightGBM refit monthly)
        path, _, suffix = extra.partition(":")
        _, w2 = load_grid(Path(path), args.min_origin)
        w2.columns = [f"{c}-{suffix or 'extra'}" for c in w2.columns]
        merged = merged.merge(w2.reset_index(), on=["ticker", "origin"], how="left")
        cover = {c: merged[c].notna().mean() for c in w2.columns}
        print(f"{Path(path).name}: coverage " + ", ".join(f"{c} {v:.0%}" for c, v in cover.items()))
        ml_cols += [c for c, v in cover.items() if v > 0.95]
    merged = add_baselines(merged)
    dm_cols = ["dm_up", "dm_move", "dm_outlook"]
    models = ["coin", "base_rate"] + dm_cols + ml_cols
    if "lgbm-cov" in merged:                       # does the decision model add anything to the best ML model?
        rk = lambda c: merged.groupby("origin")[c].rank(pct=True)  # noqa: E731
        merged["combo"] = (rk("dm_up") + rk("lgbm-cov")) / 2
    lb = score_models(merged, models).sort_values("brier").reset_index(drop=True)
    lat = st["latency_ms"].dropna()
    meta = {
        "backend": client.ident, "universe": args.universe, "market": market, "source_csv": src.name,
        "n_windows": int(len(merged)), "n_origins": int(merged["origin"].nunique()),
        "origin_range": [str(merged["origin"].min().date()), str(merged["origin"].max().date())],
        "up_share": float((merged["ret_true"] > 0).mean()),
        "latency_ms_p50": float(lat.median()) if len(lat) else None,
        "latency_ms_p95": float(lat.quantile(0.95)) if len(lat) else None,
        "wall_s": round(wall, 1), "input_tokens_mean": float(st["input_tokens"].mean()),
        "jev_cost_usd_per_1k_calls": float(st["input_tokens"].mean() * 1000 / 1e6 * JEV_COST_PER_M_INPUT),
        "dm_internal_consistency": {
            "corr_up_vs_move": float(st["dm_up"].corr(st["dm_move"])),
            "corr_up_vs_outlook": float(st["dm_up"].corr(st["dm_outlook"])),
            "share_up_gt_half": float((st["dm_up"] > 0.5).mean()),
            "share_choice_rise": float((st[["dm_rise", "dm_flat", "dm_fall"]].to_numpy().argmax(1) == 0).mean()),
        },
        "order_check": order,
        "combo_ic": float(per_origin(merged.dropna(subset=["combo"]), "combo")["ic"].mean())
        if "combo" in merged else None,
        "feature_ic": {k: float(per_origin(merged.dropna(subset=[k]), k)["ic"].mean())
                       for k in ("r5", "r20", "ma20", "rsi", "pos52") if k in merged},
    }
    merged.drop(columns=["state"]).to_csv(out.with_name(out.name + "_rows.csv"), index=False)
    out.with_name(out.name + "_metrics.json").write_text(
        json.dumps({"meta": meta, "leaderboard": lb.round(5).to_dict(orient="records")}, ensure_ascii=False,
                   indent=1, default=str), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1, default=str))
    print(fmt_table(lb))


# =========================================================================================== news
def news_questions(name: str) -> dict:
    return {
        "impact": {"type": "score",
                   "instructions": f"How is this news likely to move {name}'s share price over the next trading day?",
                   "criteria": ["clearly negative", "somewhat negative", "neutral or unclear", "somewhat positive",
                                "clearly positive"]},
        "relevant": {"type": "noul",
                     "instructions": f"Is this headline mainly about {name} itself, rather than a market roundup, "
                                     f"a list of stocks, or another company?"},
    }


def collect_news(prov, market: str, tickers: list[str], path: Path, tw_days: int = 3) -> int:
    from fintech_agent.data import parse_symbol
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                seen.add((r["ticker"], r["title"]))
            except Exception:
                continue
    new = []
    for t in tickers:
        sym = parse_symbol(t, default_market=market)
        items = []
        if market == "US":
            from fintech_agent.data.provider import yahoo_news_raw
            for x in yahoo_news_raw(sym.yf_ticker, count=50):
                c = x.get("content", x) or {}
                if not c.get("title"):
                    continue
                when = c.get("pubDate") or c.get("displayTime") or x.get("providerPublishTime")
                if isinstance(when, (int, float)):
                    when = pd.Timestamp(int(when), unit="s", tz="UTC").isoformat()
                items.append({"published": when, "tz": "UTC", "title": c["title"],
                              "summary": (c.get("summary") or "")[:500],
                              "source": (c.get("provider") or {}).get("displayName") or x.get("publisher")})
        else:
            try:
                df = prov.fm.news(sym.code, days=tw_days)
            except Exception as e:
                print(f"  {t}: news failed ({e})")
                df = pd.DataFrame()
            for _, x in df.iterrows():
                items.append({"published": str(x.get("date")), "tz": "Asia/Taipei", "title": x.get("title"),
                              "summary": "", "source": x.get("source")})
        n0 = len(new)
        for it in items:
            if it["title"] and (sym.code, it["title"]) not in seen and it["published"]:
                seen.add((sym.code, it["title"]))
                new.append({"ticker": sym.code, "market": market, "collected": pd.Timestamp.now().isoformat(), **it})
        print(f"  {t}: {len(items)} items, {len(new) - n0} new", flush=True)
    with path.open("a", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return len(new)


def label_news(news: pd.DataFrame, prices: dict[str, pd.Series], index: pd.Series, market: str,
               now=None) -> pd.DataFrame:
    tz, hh, mm = CLOSE_TIME[market]
    rows = []
    idx_s = index.sort_index()
    # a bar dated today may be an intraday snapshot (prices are cached once a day): only use completed sessions
    today = (pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz=tz).tz_localize(None)).normalize()
    for _, r in news.iterrows():
        px = prices.get(r["ticker"])
        if px is None or len(px) < 5:
            continue
        ts = pd.Timestamp(r["published"])
        ts = ts.tz_localize(r.get("tz") or "UTC") if ts.tzinfo is None else ts
        ts = ts.tz_convert(tz).tz_localize(None)
        closes = px.index + pd.Timedelta(hours=hh, minutes=mm)
        i0 = int(np.searchsorted(closes, ts, side="right")) - 1          # last close at or before publication
        if i0 < 0 or i0 + 1 >= len(px):
            continue
        d0, d1 = px.index[i0], px.index[i0 + 1]
        if d1 >= today:
            continue
        m = idx_s.reindex(idx_s.index.union(px.index)).ffill()
        stock1 = px.iloc[i0 + 1] / px.iloc[i0] - 1
        mkt1 = m.loc[d1] / m.loc[d0] - 1
        row = {**r.to_dict(), "event_day": d1, "reaction": stock1, "reaction_abn": stock1 - mkt1}
        if i0 + 2 < len(px) and px.index[i0 + 2] < today:
            d2 = px.index[i0 + 2]
            stock2 = px.iloc[i0 + 2] / px.iloc[i0 + 1] - 1
            row.update(drift=stock2, drift_abn=stock2 - (m.loc[d2] / m.loc[d1] - 1))
        rows.append(row)
    return pd.DataFrame(rows)


def finbert_scores(texts: list[str]) -> list[float] | None:
    try:
        from transformers import pipeline
        clf = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True)
    except Exception as e:
        print(f"FinBERT unavailable ({e})")
        return None
    out = []
    for i in range(0, len(texts), 32):
        for res in clf(texts[i:i + 32]):
            d = {x["label"].lower(): x["score"] for x in res}
            out.append(d.get("positive", 0.0) - d.get("negative", 0.0))
    return out


PRICE_MOVE = re.compile(r"\d+(?:\.\d+)?\s*[%％]|漲停|跌停|大漲|大跌|暴跌|重挫|狂瀉|勁揚|飆|急漲|急跌|上漲|下跌|走高|走低|收紅|收黑|"
                        r"\b(?:jumps?|soars?|surges?|plunges?|plummets?|sinks?|falls?|drops?|rall(?:y|ies)|tumbles?|"
                        r"slides?|climbs?|rises?|gains?|slumps?|skyrockets?|crash(?:es)?)\b", re.I)


def news_metrics(df: pd.DataFrame, scorer: str, target: str, threshold: float = 0.1) -> dict:
    d = df.dropna(subset=[scorer, target])
    if len(d) < 10:
        return {"scorer": scorer, "target": target, "n": len(d)}
    s, y = d[scorer], d[target]
    covered = s.abs() > threshold
    cluster = d["ticker"].astype(str) + "|" + d["event_day"].astype(str)
    rng = np.random.default_rng(0)
    groups = pd.Series(np.arange(len(d)), index=d.index).groupby(cluster.to_numpy()).apply(list).tolist()
    ics = []
    for _ in range(1000):                                    # cluster bootstrap over (stock, event day)
        take = [i for g in rng.choice(len(groups), len(groups)) for i in groups[g]]
        ss, yy = s.iloc[take], y.iloc[take]
        ics.append(ss.rank().corr(yy.rank()))
    pos, neg = s > threshold, s < -threshold
    return {"scorer": scorer, "target": target, "n": int(len(d)), "n_events": len(groups),
            "ic": float(s.rank().corr(y.rank())), "ic_lo": float(np.nanquantile(ics, 0.025)),
            "ic_hi": float(np.nanquantile(ics, 0.975)), "coverage": float(covered.mean()),
            "hit_rate": float((np.sign(s[covered]) == np.sign(y[covered])).mean()) if covered.any() else None,
            "mean_ret_pos_pct": float(y[pos].mean() * 100) if pos.any() else None,
            "mean_ret_neg_pct": float(y[neg].mean() * 100) if neg.any() else None,
            "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}


def cmd_news(args) -> None:
    from fintech_agent.config import get_settings
    from fintech_agent.data import LiveDataProvider, parse_symbol
    from fintech_agent.data.universe import UNIVERSES
    from fintech_agent.features.sentiment import lexicon_score
    prov = LiveDataProvider(get_settings())
    prov.fm.max_wait_s = 0
    client = SystemOne(args.backend, args.url, args.model, cache=LOGS / "systemone_cache.jsonl",
                       concurrency=args.concurrency)
    out_path = LOGS / "news_bench_metrics.json"
    report = {"backend": client.ident, "markets": {}}
    if out_path.exists():                      # keep the other market's results when only one market is re-run
        try:
            old = json.loads(out_path.read_text(encoding="utf-8"))
            if old.get("backend") == client.ident:
                report["markets"].update(old.get("markets", {}))
        except Exception:
            pass
    for market in args.markets:
        uni = "tw50" if market == "TW" else "us50"
        tickers = UNIVERSES[uni][1][: args.max_tickers]
        path = LOGS / f"news_{market.lower()}.jsonl"
        if not args.no_collect:
            print(f"[{market}] collecting headlines …")
            collect_news(prov, market, tickers, path, tw_days=args.tw_days)
        news = pd.read_json(path, lines=True, dtype={"ticker": str}) if path.exists() else pd.DataFrame()
        if news.empty:
            print(f"[{market}] no headlines")
            continue
        news = news.drop_duplicates(["ticker", "title"])
        prices, names = {}, {}
        for t in news["ticker"].unique():
            sym = parse_symbol(t, default_market=market)
            px = prov.prices(sym, years=1)
            if len(px):
                prices[t] = px["close"]
            try:
                names[t] = prov.profile(sym).get("name") or t
            except Exception:
                names[t] = t
        idx = prov.prices(parse_symbol(MARKET_INDEX[market]), years=1)["close"]
        lab = label_news(news, prices, idx, market)
        print(f"[{market}] {len(news)} headlines · {len(lab)} with a completed reaction window")
        if lab.empty:
            continue
        lab["name"] = lab["ticker"].map(names)
        text = (lab["title"].fillna("") + ". " + lab["summary"].fillna("")).str.strip(" .")
        lab["lexicon"] = [lexicon_score(t) for t in text]
        if market == "US" and not args.no_finbert:
            fb = finbert_scores(lab["title"].fillna("").tolist())
            if fb is not None:
                lab["finbert"] = fb
        items = []
        for _, r in lab.iterrows():
            state = f"Company: {r['name']} (ticker {r['ticker']}).\nHeadline: {r['title']}"
            if isinstance(r.get("summary"), str) and r["summary"].strip():
                state += f"\nSummary: {r['summary'].strip()}"
            items.append((state, news_questions(r["name"])))
        res = client.ask_many(items, progress_every=200)
        lab["dm"] = [score_unit(x["answers"]["impact"], 5) * 2 - 1 for x in res]
        lab["dm_relevant"] = [noul_p(x["answers"]["relevant"]) for x in res]
        scorers = [c for c in ("dm", "lexicon", "finbert") if c in lab]
        # the same story syndicated by several sites counts once; headlines that only report the price move
        # ("... jumps 7%") are trivially aligned with the reaction, so score the rest separately as well
        lab["story"] = lab["title"].fillna("").str.replace(r"\s+-\s+[^-]+$", "", regex=True).str.strip()
        lab["price_move"] = lab["title"].fillna("").str.contains(PRICE_MOVE)
        unique = lab.drop_duplicates(["ticker", "story"])
        subsets = {"all": lab, "relevant": lab[lab["dm_relevant"] > 0.5], "unique": unique,
                   "unique_no_price_move": unique[~unique["price_move"]]}
        res_rows = []
        for target in ("reaction_abn", "drift_abn"):
            for sc in scorers:
                for name, sub in subsets.items():
                    res_rows.append({**news_metrics(sub, sc, target), "subset": name})
        lab.to_csv(LOGS / f"news_{market.lower()}_scored.csv", index=False)
        report["markets"][market] = {
            "n_headlines": int(len(lab)), "n_stories": int(len(unique)), "n_stocks": int(lab["ticker"].nunique()),
            "price_move_share": float(unique["price_move"].mean()),
            "published_range": [str(lab["published"].min())[:19], str(lab["published"].max())[:19]],
            "relevant_share": float((lab["dm_relevant"] > 0.5).mean()),
            "corr_dm_lexicon": float(lab["dm"].corr(lab["lexicon"])),
            "corr_dm_finbert": float(lab["dm"].corr(lab["finbert"])) if "finbert" in lab else None,
            "results": res_rows,
        }
        tbl = pd.DataFrame(res_rows)
        pd.set_option("display.width", 200)
        print(tbl[["subset", "target", "scorer", "n", "ic", "ic_lo", "ic_hi", "coverage", "hit_rate",
                   "mean_ret_pos_pct", "mean_ret_neg_pct"]].round(4).to_string(index=False))
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("updown", "news"):
        p = sub.add_parser(name)
        p.add_argument("--backend", choices=["strands", "typesafe", "mock"], default="strands")
        p.add_argument("--url", default=None)
        p.add_argument("--model", default=None)
        p.add_argument("--concurrency", type=int, default=1)
    u = sub.choices["updown"]
    u.add_argument("--universe", default="tw50", choices=["tw50", "us50"])
    u.add_argument("--csv", default=None, help="backtest windows (default logs/b50_{universe}_h5.csv)")
    u.add_argument("--extra", nargs="*", help="more backtest CSVs to compare, as path:suffix")
    u.add_argument("--min-origin", default="2025-10-01")
    u.add_argument("--limit-origins", type=int, default=None, help="only the most recent N origins (smoke test)")
    u.add_argument("--no-chips", action="store_true")
    u.add_argument("--rebuild-states", action="store_true")
    u.add_argument("--order-check", type=int, default=300, help="re-ask N states with options reversed")
    u.add_argument("--out", default=None)
    n = sub.choices["news"]
    n.add_argument("--markets", nargs="+", default=["US"], choices=["US", "TW"])
    n.add_argument("--max-tickers", type=int, default=50)
    n.add_argument("--no-collect", action="store_true", help="score the stored headlines only")
    n.add_argument("--no-finbert", action="store_true")
    n.add_argument("--tw-days", type=int, default=3, help="FinMind days to collect per stock (1 API call per day)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    LOGS.mkdir(exist_ok=True)
    {"updown": cmd_updown, "news": cmd_news}[args.cmd](args)


if __name__ == "__main__":
    main()
