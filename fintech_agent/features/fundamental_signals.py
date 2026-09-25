"""Rule-based fundamental + chip (籌碼) + sentiment + market-regime scoring."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FundamentalView:
    score: float
    confidence: float
    sub_scores: dict
    signals: list[tuple[str, float]] = field(default_factory=list)


def _g(d: dict | None, *path, default=None):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
    return default if d is None else d


def _fundamental_tw(f: dict, sig: list) -> None:
    yoy = _g(f, "monthly_revenue", "yoy_pct")
    if yoy is not None:
        if yoy > 20:
            sig.append((f"月營收年增 {yoy:.1f}% 強勁", 0.4))
        elif yoy > 5:
            sig.append((f"月營收年增 {yoy:.1f}%", 0.2))
        elif yoy < -10:
            sig.append((f"月營收年減 {abs(yoy):.1f}%", -0.3))
        elif yoy < 0:
            sig.append((f"月營收小幅年減 {abs(yoy):.1f}%", -0.1))
    y3 = _g(f, "monthly_revenue", "yoy_3m_avg_pct")
    if y3 is not None:
        sig.append((f"近3月營收平均年增 {y3:.1f}%", float(np.clip(y3 / 100, -0.2, 0.2))))
    if _g(f, "monthly_revenue", "is_24m_high"):
        sig.append(("月營收創近24個月新高", 0.15))
    eps = _g(f, "financials", "eps_ttm_yoy_pct")
    if eps is not None:
        sig.append((f"近四季 EPS 年成長 {eps:.1f}%", 0.3 if eps > 20 else 0.1 if eps > 0 else -0.3 if eps < -10 else -0.1))
    gm = _g(f, "financials", "gross_margin_yoy_pp")
    if gm is not None and abs(gm) > 1:
        sig.append((f"毛利率年{'增' if gm > 0 else '減'} {abs(gm):.1f} 個百分點", 0.1 if gm > 0 else -0.1))
    pct = _g(f, "valuation", "pe_percentile_3y")
    pe = _g(f, "valuation", "pe")
    if pct is not None:
        if pct < 20:
            sig.append((f"本益比 {pe} 位於近3年 {pct:.0f} 百分位 (相對便宜)", 0.25))
        elif pct > 80:
            sig.append((f"本益比 {pe} 位於近3年 {pct:.0f} 百分位 (相對昂貴)", -0.25))
    dy = _g(f, "valuation", "dividend_yield_pct")
    if dy is not None and dy > 5:
        sig.append((f"殖利率 {dy:.1f}% 具防禦性", 0.1))


def _fundamental_us(f: dict, sig: list) -> None:
    y = f.get("yahoo", {})
    rg = y.get("revenue_growth_yoy")
    if rg is not None:
        sig.append((f"營收年增 {rg*100:.1f}%", 0.3 if rg > 0.15 else 0.1 if rg > 0 else -0.3 if rg < -0.05 else -0.1))
    eg = y.get("earnings_growth_yoy")
    if eg is not None:
        sig.append((f"獲利年增 {eg*100:.1f}%", 0.3 if eg > 0.2 else 0.1 if eg > 0 else -0.3 if eg < -0.1 else -0.1))
    pe, fpe = y.get("pe"), y.get("forward_pe")
    if pe and fpe and fpe > 0:
        sig.append((f"預估本益比 {fpe:.1f} {'低於' if fpe < pe else '高於'}歷史本益比 {pe:.1f}", 0.1 if fpe < pe else -0.1))
    rm = y.get("analyst_rating_mean")
    if rm is not None:
        sig.append((f"分析師平均評等 {rm:.1f} (1=強力買進)", 0.2 if rm < 2 else -0.2 if rm > 3 else 0.0))
    peg = y.get("peg")
    if peg is not None and peg > 0:
        sig.append((f"PEG {peg:.2f}", 0.1 if peg < 1 else -0.1 if peg > 2.5 else 0.0))


def _chips_tw(c: dict, ret20: float | None, sig: list) -> None:
    fo = _g(c, "institutional", "foreign", default={})
    tr = _g(c, "institutional", "trust", default={})
    st = fo.get("streak_days", 0) or 0
    if st >= 3:
        sig.append((f"外資連續買超 {st} 日", 0.25))
    elif st <= -3:
        sig.append((f"外資連續賣超 {abs(st)} 日", -0.25))
    n20 = fo.get("net_20d_lots")
    if n20:
        sig.append((f"外資近20日{'買' if n20 > 0 else '賣'}超 {abs(n20):,.0f} 張", 0.15 if n20 > 0 else -0.15))
    tst = tr.get("streak_days", 0) or 0
    if tst >= 3:
        sig.append((f"投信連續買超 {tst} 日 (作帳/認養訊號)", 0.25))
    elif tst <= -3:
        sig.append((f"投信連續賣超 {abs(tst)} 日", -0.2))
    pv = _g(c, "institutional", "net_20d_pct_of_avg_volume")
    if pv is not None and abs(pv) > 5:
        sig.append((f"法人20日淨買賣占均量 {pv:.1f}%", 0.1 if pv > 0 else -0.1))
    mc = _g(c, "margin", "margin_chg_20d_pct")
    if mc is not None and ret20 is not None:
        if mc > 10 and ret20 < 0:
            sig.append((f"股價跌但融資增 {mc:.1f}% (籌碼凌亂)", -0.2))
        elif mc < -5 and ret20 > 0:
            sig.append((f"股價漲但融資減 {abs(mc):.1f}% (籌碼沉澱)", 0.15))
        elif mc > 15 and ret20 > 0:
            sig.append((f"融資大增 {mc:.1f}% (散戶追價，留意多殺多)", -0.1))
    smr = _g(c, "margin", "short_margin_ratio_pct")
    if smr is not None and smr > 30:
        sig.append((f"券資比 {smr:.1f}% 偏高，具軋空潛力", 0.1))
    fr = _g(c, "shareholding", "foreign_ratio_chg_20d")
    if fr is not None and abs(fr) > 0.5:
        sig.append((f"外資持股比例20日{'增' if fr > 0 else '減'} {abs(fr):.2f} 個百分點", 0.1 if fr > 0 else -0.1))


def _chips_us(c: dict, sig: list) -> None:
    sp = c.get("short_pct_float")
    if sp is not None and sp > 0.15:
        sig.append((f"空單占流通股 {sp*100:.1f}% 偏高", -0.15))
    sc = c.get("short_interest_chg_pct")
    if sc is not None and abs(sc) > 20:
        sig.append((f"空單月變化 {sc:+.1f}%", -0.1 if sc > 0 else 0.1))
    ins = c.get("insider_6m") or {}
    if ins.get("buy_count", 0) > ins.get("sell_count", 0):
        sig.append((f"近6月內部人買進次數 ({ins['buy_count']}) 多於賣出", 0.15))
    elif ins.get("sell_value", 0) > 5 * max(ins.get("buy_value", 0), 1) and ins.get("sell_count", 0) >= 5:
        sig.append((f"近6月內部人以賣出為主 ({ins['sell_count']} 筆)", -0.05))


def _market(m: dict, market: str, sig: list) -> None:
    idx = m.get("index") or {}
    if idx.get("above_ma60") is True and (idx.get("ret_20d_pct") or 0) > 0:
        sig.append(("大盤站上季線且月線翻揚，系統性風險低", 0.15))
    elif idx.get("above_ma60") is False:
        sig.append(("大盤跌破季線，系統性風險升高", -0.15))
    vix = m.get("vix")
    if vix is not None and vix > 25:
        sig.append((f"VIX {vix:.1f} 恐慌升溫", -0.15))
    flows = m.get("institutional_flows") or {}
    f5 = flows.get("foreign_net_5d_100m_twd")
    if market == "TW" and f5 is not None:
        sig.append((f"外資近5日對台股{'買' if f5 > 0 else '賣'}超 {abs(f5):.0f} 億", 0.1 if f5 > 0 else -0.1))


def fundamental_view(fund: dict, chips: dict, sentiment: dict, market_ctx: dict, market: str,
                     ret_20d_pct: float | None = None) -> FundamentalView:
    groups: dict[str, list] = {"fundamental": [], "chips": [], "sentiment": [], "market": []}
    if market == "TW":
        _fundamental_tw(fund or {}, groups["fundamental"])
        _chips_tw(chips or {}, ret_20d_pct, groups["chips"])
    else:
        _fundamental_us(fund or {}, groups["fundamental"])
        _chips_us(chips or {}, groups["chips"])
    if sentiment and sentiment.get("n"):
        s = sentiment.get("mean", 0.0)
        groups["sentiment"].append((f"新聞情緒 {s:+.2f} ({sentiment.get('method')}, {sentiment['n']} 則)", 0.4 * s))
    _market(market_ctx or {}, market, groups["market"])

    sub = {g: round(float(np.clip(sum(v for _, v in items), -2, 2)), 3) for g, items in groups.items()}
    all_sig = [s for items in groups.values() for s in items]
    score = float(np.clip(sum(sub.values()), -2, 2))
    coverage = sum(1 for items in groups.values() if items) / len(groups)
    pos, neg = sum(v for _, v in all_sig if v > 0), -sum(v for _, v in all_sig if v < 0)
    agreement = abs(pos - neg) / (pos + neg) if pos + neg else 0
    confidence = float(np.clip(0.2 + 0.4 * coverage + 0.3 * agreement, 0.15, 0.9))
    return FundamentalView(round(score, 3), round(confidence, 3), sub, all_sig)
