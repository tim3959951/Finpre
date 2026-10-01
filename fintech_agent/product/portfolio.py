"""風險雷達 (portfolio risk radar).

Holdings -> the distribution of the portfolio's return over the next `horizon` trading days, where it could come
from, and how concentrated it is. Research-mode safe: it states risk facts, it never tells anyone to trade.

Method — filtered historical simulation:
  1. For each holding, take overlapping `horizon`-day log returns over the last ~3 years (keeps fat tails and the
     co-movement between holdings, including USD/TWD for US stocks held in TWD).
  2. Re-centre and re-scale each holding's scenarios so their median and P10–P90 spread match today's model
     forecast (champion model; volatility regimes change, the history alone would be stale).
  3. Portfolio return per scenario = value-weighted sum of simple returns -> quantiles, VaR, expected shortfall,
     probability of a loss worse than 5%, and each holding's share of the tail loss (component ES).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..data.provider import DataProvider
from ..data.symbols import parse_symbol
from ..evaluation.regimes import REGIME_LABELS, market_regime
from .forecast import TickerForecast, forecast_ticker

log = logging.getLogger(__name__)
FX_TICKER = "TWD=X"          # TWD per USD


@dataclass
class Position:
    ticker: str
    shares: float


def _fx_series(provider: DataProvider) -> pd.Series:
    fx = provider.prices(parse_symbol(FX_TICKER))
    return fx["close"].astype(float) if len(fx) else pd.Series(dtype=float)


def portfolio_risk(provider: DataProvider, holdings: dict[str, float], horizon: int = 5, base_ccy: str = "TWD",
                   settings: Settings | None = None, cash: float = 0.0, lookback_days: int = 756,
                   max_positions: int = 60) -> dict:
    s = settings or get_settings()
    if not holdings:
        raise ValueError("請至少輸入一檔持股")
    if base_ccy not in ("TWD", "USD"):
        raise ValueError("base_ccy must be TWD or USD")
    if not (np.isfinite(cash) and cash >= 0):
        raise ValueError("現金必須為非負數")
    bad = [str(t) for t, n in holdings.items() if not (isinstance(n, (int, float)) and np.isfinite(n) and n > 0)]
    if bad:
        raise ValueError(f"持股股數必須為正數：{', '.join(bad[:5])}")
    # one row per security: "2330" and "2330.TW" are the same holding
    merged: dict[str, tuple] = {}
    skipped = []
    for t, shares in holdings.items():
        try:
            sym = provider.symbol(str(t))
        except Exception as e:
            skipped.append({"ticker": str(t), "reason": str(e)[:120]})
            continue
        prev = merged.get(sym.code)
        merged[sym.code] = (sym, (prev[1] if prev else 0.0) + float(shares))
    if len(merged) > max_positions:
        raise ValueError(f"最多 {max_positions} 檔持股")
    fx = _fx_series(provider) if any(sym.currency != base_ccy for sym, _ in merged.values()) else pd.Series(dtype=float)
    fx_last = float(fx.iloc[-1]) if len(fx) else None

    def to_base(ccy: str) -> float:
        if ccy == base_ccy:
            return 1.0
        if fx_last is None:
            raise ValueError("取不到 USD/TWD 匯率，無法換算不同幣別的持股")
        return fx_last if (ccy == "USD" and base_ccy == "TWD") else 1.0 / fx_last

    rows, fcs, closes = [], {}, {}
    for code, (sym, shares) in merged.items():
        try:
            px = provider.prices(sym)
            fc = forecast_ticker(provider, sym, horizon, s, prices=px)
        except Exception as e:
            skipped.append({"ticker": code, "reason": str(e)[:120]})
            continue
        value = shares * fc.last * to_base(sym.currency)
        prof = provider.profile(sym) or {}
        rows.append({"ticker": sym.code, "name": fc.name, "market": sym.market, "currency": sym.currency,
                     "shares": shares, "last": round(fc.last, 4), "value": value,
                     "sector": prof.get("industry") or prof.get("sector") or "其他", "forecast": fc.to_dict()})
        fcs[sym.code] = fc
        closes[sym.code] = px["close"].astype(float)
    if not rows:
        raise ValueError("沒有任何持股能取得價格資料：" + "；".join(x["reason"] for x in skipped))
    total = sum(r["value"] for r in rows) + cash
    for r in rows:
        r["weight"] = r["value"] / total
    tickers = [r["ticker"] for r in rows]
    w = np.array([r["weight"] for r in rows])

    # ---- scenario matrix of horizon-day log returns (aligned calendar, forward-filled across markets)
    panel = pd.DataFrame(closes).sort_index().ffill().dropna(how="all")
    panel = panel.iloc[-(lookback_days + horizon):]
    foreign = [r["currency"] != base_ccy for r in rows]
    if any(foreign) and len(fx):
        fxa = fx.reindex(panel.index.union(fx.index)).ffill().reindex(panel.index)
    lp = np.log(panel)
    scen = (lp.shift(-horizon) - lp).iloc[:-horizon].dropna(how="any")
    if len(scen) < 60:                 # too little overlap (e.g. a new listing): use each asset's own recent history
        arrays = {t: (np.log(closes[t]).shift(-horizon) - np.log(closes[t])).dropna().to_numpy()[-lookback_days:]
                  for t in tickers}
        n = min(len(a) for a in arrays.values())
        if n < 30:
            raise ValueError("持股的共同歷史資料不足，無法估計組合風險")
        scen = pd.DataFrame({t: a[-n:] for t, a in arrays.items()})
    R = scen[tickers].to_numpy()
    # ---- filter: match each asset's scenario median / P10-P90 spread to today's model forecast
    for j, t in enumerate(tickers):
        fc: TickerForecast = fcs[t]
        h10, h50, h90 = np.quantile(R[:, j], [0.1, 0.5, 0.9])
        m10, m50, m90 = (np.log1p(fc.quantile_returns[q]) for q in (0.1, 0.5, 0.9))
        scale = (m90 - m10) / (h90 - h10) if h90 > h10 else 1.0
        R[:, j] = m50 + (R[:, j] - h50) * float(np.clip(scale, 0.25, 4.0))
    simple = np.expm1(R)
    if any(foreign) and len(fx):
        fx_ret = (np.log(fxa).shift(-horizon) - np.log(fxa)).reindex(scen.index).fillna(0.0).to_numpy()
        for j, r in enumerate(rows):
            if foreign[j]:                     # USD asset in TWD: + USD/TWD move; TWD asset in USD: − it
                sign = 1.0 if r["currency"] == "USD" else -1.0
                simple[:, j] = (1 + simple[:, j]) * np.exp(sign * fx_ret) - 1
    port = simple @ w
    q = np.quantile(port, [0.05, 0.1, 0.5, 0.9])
    tail = port <= q[0]
    es = float(-port[tail].mean()) if tail.any() else float(-q[0])
    contrib = (simple[tail] * w).mean(axis=0) if tail.any() else np.zeros(len(w))
    share = contrib / contrib.sum() if contrib.sum() != 0 else np.zeros(len(w))
    for j, r in enumerate(rows):
        r["tail_loss_share"] = round(float(share[j]), 4)
        r["value"] = round(r["value"], 2)
        r["weight"] = round(r["weight"], 4)

    # ---- concentration & diversification
    hhi = float((w ** 2).sum())
    by_sector = pd.Series({r["sector"]: 0.0 for r in rows})
    for r in rows:
        by_sector[r["sector"]] += r["weight"]
    daily = np.log(panel[tickers]).diff().tail(250)
    corr = daily.corr().to_numpy()
    avg_corr = float(corr[np.triu_indices(len(tickers), 1)].mean()) if len(tickers) > 1 else 1.0
    vol_ann = float(np.sqrt(w @ (daily.cov().to_numpy() * 252) @ w)) if len(daily) > 20 else float("nan")

    # ---- market regime per market held
    regimes = {}
    for m in sorted({r["market"] for r in rows}):
        idx = s.get_path(f"data.market_index.{m}")
        try:
            reg = market_regime(provider.prices(parse_symbol(idx))["close"]).dropna()
            regimes[m] = {"index": idx, "regime": reg.iloc[-1], "label": REGIME_LABELS[reg.iloc[-1]],
                          "as_of": str(reg.index[-1].date())}
        except Exception as e:  # pragma: no cover - network
            log.warning("regime for %s unavailable: %s", m, e)

    flags = []
    top = max(rows, key=lambda r: r["weight"])
    if top["weight"] > 0.25:
        flags.append(f"單一持股 {top['name']}（{top['ticker']}）占組合 {top['weight']:.0%}，超過 25%")
    top3 = sorted(w, reverse=True)[:3]
    if len(rows) >= 4 and sum(top3) > 0.6:
        flags.append(f"前三大持股合計占 {sum(top3):.0%}")
    big_sector = by_sector.idxmax()
    if by_sector.max() > 0.4 and len(rows) > 1:
        flags.append(f"產業集中：{big_sector} 占 {by_sector.max():.0%}")
    if len(rows) > 2 and avg_corr > 0.5:
        flags.append(f"持股間平均相關係數 {avg_corr:.2f}，分散效果有限")
    hot = [r for r in rows if r["forecast"]["hv20_ann_pct"] > 50]
    if hot:
        flags.append(f"{len(hot)} 檔持股近 20 日年化波動超過 50%：" + "、".join(r["name"] for r in hot[:5]))
    for m, g in regimes.items():
        if g["regime"] == "bear":
            flags.append(f"{'台股' if m == 'TW' else '美股'}大盤處於空頭狀態（跌破 200 日均線且 60 日下跌）")
    fx_w = sum(r["weight"] for r in rows if r["currency"] != base_ccy)
    if fx_w > 0:
        other = "美元" if base_ccy == "TWD" else "新台幣"
        flags.append(f"{other}資產占 {fx_w:.0%}，已計入 USD/TWD 匯率波動")

    return {
        "as_of": max(fc.as_of for fc in fcs.values()), "horizon": horizon, "base_currency": base_ccy,
        "total_value": round(total, 2), "cash": round(cash, 2), "n_positions": len(rows), "scenarios": int(len(port)),
        "return_pct": {"p05": round(q[0] * 100, 2), "p10": round(q[1] * 100, 2), "p50": round(q[2] * 100, 2),
                       "p90": round(q[3] * 100, 2)},
        "var95_pct": round(-q[0] * 100, 2), "es95_pct": round(es * 100, 2),
        "var95_amount": round(-q[0] * total, 0), "es95_amount": round(es * total, 0),
        "prob_loss_over_5pct": round(float((port < -0.05).mean()), 3),
        "vol_ann_pct": round(vol_ann * 100, 1) if np.isfinite(vol_ann) else None,
        "concentration": {"hhi": round(hhi, 4), "effective_positions": round(1 / hhi, 1),
                          "top_weight": round(float(w.max()), 4), "avg_pairwise_corr": round(avg_corr, 3),
                          "sectors": {k: round(float(v), 4) for k, v in by_sector.sort_values(ascending=False).items()}},
        "regimes": regimes, "flags": flags, "positions": sorted(rows, key=lambda r: -r["weight"]), "skipped": skipped,
        "method": "濾波歷史模擬：近 3 年重疊報酬情境，依今日模型預測的中位數與 P10–P90 區間重新校準；外幣資產含匯率",
    }
