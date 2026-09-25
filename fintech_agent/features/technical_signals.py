"""Rule-based technical evidence + score. Deterministic so the LLM layer can be audited against it."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import add_indicators


def _r(x, nd=2):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else round(v, nd)


def _cross(a: pd.Series, b: pd.Series, lookback: int = 3) -> str | None:
    """'golden' if a crossed above b within lookback bars, 'death' if below."""
    diff = (a - b).dropna()
    if len(diff) < lookback + 1:
        return None
    w = diff.iloc[-(lookback + 1):]
    s = np.sign(w.to_numpy())
    for i in range(len(s) - 1, 0, -1):
        if s[i] > 0 >= s[i - 1]:
            return "golden"
        if s[i] < 0 <= s[i - 1]:
            return "death"
    return None


@dataclass
class TechnicalView:
    score: float                      # -2 (strong bearish) .. +2 (strong bullish)
    confidence: float                 # 0..1
    evidence: dict
    signals: list[tuple[str, float]] = field(default_factory=list)   # (description zh-TW, contribution)


def technical_view(ohlcv: pd.DataFrame) -> TechnicalView:
    if len(ohlcv) < 60:
        return TechnicalView(0.0, 0.1, {"error": "資料不足 60 根K棒"})
    df = add_indicators(ohlcv)
    L, P = df.iloc[-1], df.iloc[-2]
    c = float(L["close"])

    def ret(n):
        return _r((c / df["close"].iloc[-n - 1] - 1) * 100) if len(df) > n else None

    ma = {n: _r(L.get(f"ma{n}")) for n in (5, 10, 20, 60, 120, 240)}
    if all(ma[n] for n in (5, 20, 60)) and ma[5] > ma[20] > ma[60]:
        alignment = "多頭排列"
    elif all(ma[n] for n in (5, 20, 60)) and ma[5] < ma[20] < ma[60]:
        alignment = "空頭排列"
    else:
        alignment = "糾結/混合"

    levels_below, levels_above = [], []
    for name, val in [("20日低點", df["low"].tail(20).min()), ("60日低點", df["low"].tail(60).min()),
                      ("20日高點", df["high"].tail(20).max()), ("60日高點", df["high"].tail(60).max()),
                      ("月線MA20", L.get("ma20")), ("季線MA60", L.get("ma60")), ("年線MA240", L.get("ma240")),
                      ("布林下軌", L.get("bb_lower")), ("布林上軌", L.get("bb_upper"))]:
        if val is None or pd.isna(val):
            continue
        (levels_below if val < c else levels_above).append((name, round(float(val), 2)))
    support = sorted(levels_below, key=lambda x: -x[1])[:3]
    resistance = sorted(levels_above, key=lambda x: x[1])[:3]

    bw = df["bb_width"].tail(120).dropna()
    ev = {
        "as_of": str(df.index[-1].date()),
        "close": _r(c),
        "chg_1d_pct": _r(L["ret_1d"]),
        "ret_5d_pct": ret(5), "ret_20d_pct": ret(20), "ret_60d_pct": ret(60),
        "ma": ma, "ma_alignment": alignment,
        "bias_pct": {n: _r(L.get(f"bias{n}")) for n in (5, 20, 60, 240)},
        "macd": {"dif": _r(L["dif"], 3), "dea": _r(L["dea"], 3), "osc": _r(L["osc"], 3),
                 "osc_rising": bool(L["osc"] > P["osc"]), "cross": _cross(df["dif"], df["dea"])},
        "kd": {"k": _r(L["k"]), "d": _r(L["d"]), "cross": _cross(df["k"], df["d"])},
        "rsi": {"rsi6": _r(L["rsi6"]), "rsi14": _r(L["rsi14"])},
        "bollinger": {"pctb": _r(L["bb_pctb"]), "width": _r(L["bb_width"], 4),
                      "squeeze": bool(len(bw) > 60 and L["bb_width"] <= bw.quantile(0.2))},
        "adx": {"adx": _r(L["adx"]), "plus_di": _r(L["plus_di"]), "minus_di": _r(L["minus_di"])},
        "atr14": _r(L["atr14"]), "atr_pct": _r(L["atr_pct"]), "hv20_pct": _r(L["hv20"]),
        "volume": {"vol_ratio": _r(L["vol_ratio"]),
                   "obv_20d_trend": "up" if df["obv"].iloc[-1] > df["obv"].iloc[-21] else "down"},
        "williams_r": _r(L["willr14"]),
        "pct_from_52w_high": _r((c / L["high_252"] - 1) * 100) if not pd.isna(L["high_252"]) else None,
        "pct_from_52w_low": _r((c / L["low_252"] - 1) * 100) if not pd.isna(L["low_252"]) else None,
        "support": support, "resistance": resistance,
    }

    # ------------------------------------------------------------------ scoring
    sig: list[tuple[str, float]] = []
    trend: list[tuple[str, float]] = []
    if alignment == "多頭排列":
        trend.append(("均線多頭排列 (MA5>MA20>MA60)", 0.6))
    elif alignment == "空頭排列":
        trend.append(("均線空頭排列 (MA5<MA20<MA60)", -0.6))
    if ma[60]:
        trend.append(("站上季線" if c > ma[60] else "跌破季線", 0.2 if c > ma[60] else -0.2))
    if ma[240]:
        trend.append(("位於年線之上 (長多格局)" if c > ma[240] else "位於年線之下 (長空格局)", 0.2 if c > ma[240] else -0.2))
    adx_v = L["adx"] if not pd.isna(L["adx"]) else 20
    mult = 1.3 if adx_v > 25 else (0.7 if adx_v < 20 else 1.0)
    if adx_v > 25:
        sig.append((f"ADX {adx_v:.0f} > 25，趨勢明確，趨勢訊號加權", 0.0))
    elif adx_v < 20:
        sig.append((f"ADX {adx_v:.0f} < 20，盤整格局，趨勢訊號降權", 0.0))
    sig += [(t, v * mult) for t, v in trend]

    m = ev["macd"]
    if m["osc"] is not None:
        sig.append(("MACD 柱狀體翻正" if m["osc"] > 0 else "MACD 柱狀體為負", 0.25 if m["osc"] > 0 else -0.25))
        sig.append(("MACD 柱狀體擴張向上" if m["osc_rising"] else "MACD 柱狀體收斂向下", 0.15 if m["osc_rising"] else -0.15))
    if m["cross"] == "golden":
        sig.append(("MACD 黃金交叉", 0.2))
    elif m["cross"] == "death":
        sig.append(("MACD 死亡交叉", -0.2))

    k = ev["kd"]
    if k["cross"] == "golden" and (k["k"] or 50) < 30:
        sig.append(("KD 低檔黃金交叉", 0.3))
    elif k["cross"] == "death" and (k["k"] or 50) > 70:
        sig.append(("KD 高檔死亡交叉", -0.3))
    elif (k["k"] or 0) > 80 and (k["d"] or 0) > 80:
        sig.append(("KD 高檔鈍化 (強勢但留意過熱)", 0.1))
    elif (k["k"] or 100) < 20 and (k["d"] or 100) < 20:
        sig.append(("KD 低檔鈍化 (弱勢)", -0.1))

    r14 = ev["rsi"]["rsi14"] or 50
    if r14 > 75:
        sig.append((f"RSI14={r14:.0f} 超買", -0.3))
    elif r14 < 25:
        sig.append((f"RSI14={r14:.0f} 超賣，留意反彈", 0.3))
    elif r14 >= 50:
        sig.append((f"RSI14={r14:.0f} 位於多方區", 0.1))
    else:
        sig.append((f"RSI14={r14:.0f} 位於空方區", -0.1))

    pctb, vr = ev["bollinger"]["pctb"], ev["volume"]["vol_ratio"] or 1
    if pctb is not None:
        if pctb > 1:
            sig.append(("帶量突破布林上軌" if vr > 1.5 else "觸及布林上軌但量能不足", 0.2 if vr > 1.5 else -0.1))
        elif pctb < 0:
            sig.append(("帶量跌破布林下軌" if vr > 1.5 else "跌破布林下軌 (乖離過大)", -0.2 if vr > 1.5 else 0.1))
    if ev["bollinger"]["squeeze"]:
        sig.append(("布林通道壓縮，醞釀變盤", 0.0))

    if vr > 1.5 and (ev["chg_1d_pct"] or 0) > 0:
        sig.append((f"爆量上漲 (量比 {vr:.1f})", 0.15))
    elif vr > 1.5 and (ev["chg_1d_pct"] or 0) < 0:
        sig.append((f"爆量下跌 (量比 {vr:.1f})", -0.15))

    b20 = ev["bias_pct"][20] or 0
    if b20 > 10:
        sig.append((f"月線乖離 {b20:.1f}% 過熱", -0.2))
    elif b20 < -10:
        sig.append((f"月線負乖離 {b20:.1f}% 過大，易跌深反彈", 0.2))

    score = float(np.clip(sum(v for _, v in sig), -2, 2))
    pos, neg = sum(v for _, v in sig if v > 0), -sum(v for _, v in sig if v < 0)
    agreement = abs(pos - neg) / (pos + neg) if pos + neg else 0
    confidence = float(np.clip(0.3 + 0.5 * agreement + (0.1 if adx_v > 25 else 0), 0.2, 0.9))
    return TechnicalView(round(score, 3), round(confidence, 3), ev, sig)
