"""每日盤後研究報告 (daily after-close research report).

Market regime, the ranking model's top/bottom names, watchlist risk ranges and the model scorecard, in three
renderings: JSON (API), HTML (email / web, self-contained) and Markdown (LINE / chat). Research wording only; the
compliance guard runs over the finished report before it is returned.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime

import numpy as np

from ..config import Settings, get_settings
from ..data.provider import DataProvider
from ..data.summaries import summarize_index
from ..data.symbols import parse_symbol
from ..evaluation.regimes import REGIME_LABELS, market_regime
from .compliance import RESEARCH_DISCLAIMER, ComplianceGuard
from .forecast import forecast_ticker
from .scorecard import backtest_scorecard, live_scorecard

log = logging.getLogger(__name__)
MARKET_NAME = {"TW": "台股", "US": "美股"}


def _ret(close, n):
    return round(float((close.iloc[-1] / close.iloc[-1 - n] - 1) * 100), 2) if len(close) > n else None


def build_report(provider: DataProvider, market: str = "TW", horizon: int = 5, watchlist: list[str] | None = None,
                 settings: Settings | None = None, top_n: int = 10) -> dict:
    s = settings or get_settings()
    idx_symbol = s.get_path(f"data.market_index.{market}")
    idx_px = provider.prices(parse_symbol(idx_symbol))
    close = idx_px["close"].astype(float)
    reg = market_regime(close).dropna()
    overview = {"index": idx_symbol, **summarize_index(close), "ret_1d_pct": _ret(close, 1),
                "regime": reg.iloc[-1] if len(reg) else None,
                "regime_label": REGIME_LABELS.get(reg.iloc[-1]) if len(reg) else None,
                "as_of": str(close.index[-1].date())}

    # ---- ranking snapshot (research list)
    runs = s.resolve_path("evaluation.runs_dir")
    ranking = None
    for h in (horizon, 5, 20):
        p = runs / f"ranking_{market}_h{h}.json"
        if p.exists():
            snap = json.loads(p.read_text())
            enrich = []
            ranks = snap["ranks"]
            for r in ranks[:top_n] + ranks[max(top_n, len(ranks) - 5):]:
                try:
                    c = provider.prices(provider.symbol(r["ticker"]))["close"]
                    enrich.append({**r, "ret_5d_pct": _ret(c, 5), "ret_20d_pct": _ret(c, 20)})
                except Exception:
                    enrich.append(r)
            ranking = {"as_of": snap["as_of"], "model": snap["model"], "universe": snap["universe"],
                       "horizon": snap["horizon"], "n": snap["n"], "backtest": snap.get("backtest", {}),
                       "top": enrich[:top_n], "bottom": enrich[top_n:]}
            break

    # ---- watchlist risk ranges
    items = []
    for t in watchlist or []:
        try:
            fc = forecast_ticker(provider, t, horizon, s)
            c = provider.prices(fc.symbol)["close"]
            row = fc.to_dict() | {"ret_5d_pct": _ret(c, 5), "ret_20d_pct": _ret(c, 20)}
            if fc.symbol.market == "TW":
                inst = (provider.chips(fc.symbol) or {}).get("institutional") or {}
                row["foreign_net_5d_lots"] = (inst.get("foreign") or {}).get("net_5d_lots")
                row["trust_net_5d_lots"] = (inst.get("trust") or {}).get("net_5d_lots")
            items.append(row)
        except Exception as e:
            items.append({"ticker": t, "error": str(e)[:120]})

    card = backtest_scorecard()
    fc_card = next((r for r in card.get("forecasting", []) if r["market"] == market and r["horizon"] == horizon), None)
    rk_card = [r for r in card.get("ranking", []) if r["market"] == market]
    if ranking:
        same = [r for r in rk_card if r["universe"] == ranking["universe"] and r["horizon"] == ranking["horizon"]]
        rk_card = same or rk_card
        age = (np.datetime64(overview["as_of"]) - np.datetime64(ranking["as_of"])).astype(int)
        ranking["age_days"] = int(age)
        ranking["stale"] = bool(age > int(s.get_path("ranking.max_age_days", 7)))
    try:
        live = live_scorecard(s)
        live_rows = [m for m in live.get("models", []) if m["horizon"] == horizon]
        live_card = {"since": live.get("since"), "pending": live.get("pending", 0), "resolved": live.get("resolved", 0),
                     "models": live_rows}
    except Exception as e:  # the report must not fail because the tracking store is unavailable
        log.warning("live scorecard unavailable: %s", e)
        live_card = None
    report = {
        "title": f"{MARKET_NAME.get(market, market)}盤後研究報告", "market": market, "horizon": horizon,
        "date": overview["as_of"], "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overview": overview, "ranking": ranking, "watchlist": items,
        "scorecard": {"forecasting": fc_card, "ranking": rk_card, "live": live_card},
        "highlights": _highlights(overview, ranking, items), "disclaimer": RESEARCH_DISCLAIMER,
    }
    clean, rep = ComplianceGuard("research").apply(report)
    clean["compliance"] = rep.public()
    clean["compliance_full"] = rep.to_dict()      # for the audit log; callers pop it before showing the report
    return clean


def _highlights(ov: dict, ranking: dict | None, items: list[dict]) -> list[str]:
    out = []
    if ov.get("regime_label"):
        out.append(f"大盤狀態：{ov['regime_label']}（近 20 日 {ov.get('ret_20d_pct'):+.1f}%，距一年高點 {ov.get('drawdown_from_1y_high_pct'):+.1f}%）")
    if ranking and ranking["top"]:
        names = "、".join(f"{r['name']}" for r in ranking["top"][:3])
        out.append(f"排序模型分數最高：{names}（{ranking['universe']}，{ranking['horizon']} 日）")
    wide = [i for i in items if "p10_pct" in i and (i["p90_pct"] - i["p10_pct"]) > 12]
    if wide:
        out.append("觀察清單中未來區間最寬（波動最大）：" + "、".join(i["name"] for i in wide[:3]))
    up = [i for i in items if i.get("p_up") is not None and i["p_up"] >= 0.6]
    if up:
        out.append("模型上漲機率 ≥ 60%：" + "、".join(f"{i['name']} {i['p_up']:.0%}" for i in up[:5]))
    return out


def scorecard_lines(rep: dict) -> list[str]:
    """Plain-language track record shown under every report — including the parts that are not flattering."""
    sc = rep.get("scorecard") or {}
    out = []
    f = sc.get("forecasting")
    if f and f.get("lgbm"):
        lg = f["lgbm"]
        out.append(f"預測模型（LightGBM，{f['period']}，{f['tickers']} 檔）的機率預測比 random walk 改善 "
                   f"{lg.get('crps_skill_pct', 0):+.2f}%，{(lg.get('share_beating_naive') or 0):.0%} 的股票勝出，"
                   f"80% 區間實際覆蓋 {(lg.get('cov80') or 0):.0%}；各年度為正：{f.get('lgbm_positive_years', '–')}。")
    for r in (sc.get("ranking") or [])[:1]:
        txt = (f"排序模型（{r['model']}，{r['universe']}，{r['horizon']} 日）回測 IC {r['ic_mean']:+.3f}（t = {r['ic_t']:.1f}，"
               f"{r['ic_positive_share']:.0%} 的期間為正）")
        if r.get("excess_gross") is not None and r.get("excess_net") is not None:
            txt += (f"；前 10 名組合相對等權重的年化超額報酬，未計成本 {r['excess_gross'] * 100:+.1f}%、"
                    f"計入每次來回 {r['cost_bps']:g} bps 交易成本並控制換手後 {r['excess_net'] * 100:+.1f}%")
        out.append(txt + "。排序的統計關聯顯著，但扣除台股交易成本後不足以單獨構成交易策略，宜作為研究篩選的參考。"
                   if r.get("excess_net") is not None and r["excess_net"] <= 0 else txt + "。")
    lv = sc.get("live")
    if lv:
        if lv.get("resolved"):
            best = next((m for m in lv["models"] if m["model"] != "naive"), None)
            txt = f"上線追蹤（{lv.get('since')} 起）：已到期 {lv['resolved']} 筆、等待到期 {lv['pending']} 筆"
            if best:
                hit = "–" if best["dir_acc"] is None else f"{best['dir_acc']:.0%}"
                txt += f"；{best['model']} 的 80% 區間實際覆蓋 {best['cov80']:.0%}、方向命中率 {hit}"
                if best.get("mae_vs_naive_pct") is not None:
                    txt += f"、誤差相對 random walk {best['mae_vs_naive_pct']:+.1f}%"
            out.append(txt + "。")
        elif lv.get("pending"):
            since = f"（{lv['since']} 起）" if lv.get("since") else ""
            out.append(f"上線追蹤{since}：{lv['pending']} 筆預測等待到期，到期後每天自動計分並公開。")
    return out


# ------------------------------------------------------------------------------------------ renderers
def render_markdown(rep: dict) -> str:
    ov = rep["overview"]
    lines = [f"# {rep['title']} · {rep['date']}", "",
             f"**大盤**：{ov['index']} {ov.get('last')}（1 日 {ov.get('ret_1d_pct'):+.2f}%、5 日 {ov.get('ret_5d_pct'):+.2f}%、"
             f"20 日 {ov.get('ret_20d_pct'):+.2f}%）· 狀態 **{ov.get('regime_label') or '–'}**", ""]
    if rep["highlights"]:
        lines += ["## 今日重點"] + [f"- {h}" for h in rep["highlights"]] + [""]
    rk = rep.get("ranking")
    if rk:
        bt = rk.get("backtest") or {}
        lines += [f"## 模型分數排名（{rk['universe']}，{rk['horizon']} 日，資料 {rk['as_of']}）",
                  f"排序模型 {rk['model']} 的歷史回測 IC {bt.get('ic_mean', float('nan')):+.3f}（t = {bt.get('ic_t', float('nan')):.1f}）。"
                  "排名是研究參考，不是買賣建議。" + (f"（注意：排名資料已 {rk['age_days']} 天未更新）" if rk.get("stale") else ""),
                  "", "| 排名 | 股票 | 分數 | 近 5 日漲跌 | 近 20 日漲跌 |", "|---|---|---|---|---|"]
        for r in rk["top"]:
            lines.append(f"| {r['rank']} | {r['name']}（{r['ticker']}） | {r['score']:.3f} | {_p(r.get('ret_5d_pct'))} | {_p(r.get('ret_20d_pct'))} |")
        if rk.get("note"):
            lines.append(f"\n{rk['note']}")
        lines.append("")
    if rep["watchlist"]:
        lines += [f"## 觀察清單：未來 {rep['horizon']} 個交易日的模型區間", "",
                  "| 股票 | 中位數 | 80% 區間 | 上漲機率 | 年化波動 | 模型 |", "|---|---|---|---|---|---|"]
        for i in rep["watchlist"]:
            if "error" in i:
                lines.append(f"| {i['ticker']} | 資料不足 | | | | |")
                continue
            lines.append(f"| {i['name']}（{i['ticker']}） | {_p(i['median_pct'])} | {_p(i['p10_pct'])} ～ {_p(i['p90_pct'])} | "
                         f"{i['p_up']:.0%} | {i['hv20_ann_pct']:.0f}% | {i['model']} |")
        lines.append("")
    sl = scorecard_lines(rep)
    if sl:
        lines += ["## 模型成績單"] + [f"- {x}" for x in sl] + [""]
    lines += [f"> {rep['disclaimer']}"]
    return "\n".join(lines)


def _p(v) -> str:
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1f}%"


_CSS = """
body{margin:0;background:#f4f5f7;color:#15181c;font:15px/1.6 -apple-system,"PingFang TC","Noto Sans TC","Microsoft JhengHei",sans-serif}
.wrap{max-width:760px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.meta{color:#6b727c;font-size:13px}.card{background:#fff;border:1px solid #e2e5e9;border-radius:10px;padding:14px 16px;margin-top:12px}
.pill{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12.5px;border:1px solid #cfd4da}
.bull{background:#e7f4ea;border-color:#b9dfc3}.bear{background:#fbeaea;border-color:#efc2c2}.mixed{background:#f3f1e6;border-color:#e0dbbf}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid #eceef1;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:#6b727c;font-weight:500}.tw{overflow-x:auto}.pos{color:#1a7f37}.neg{color:#b42318}
ul{padding-left:20px;margin:6px 0}.foot{color:#6b727c;font-size:12.5px;margin-top:28px}
"""


def render_html(rep: dict) -> str:
    e = html.escape
    ov = rep["overview"]

    def num(v, d=1):
        if v is None:
            return "–"
        cls = "pos" if v > 0 else "neg" if v < 0 else ""
        return f'<span class="{cls}">{v:+.{d}f}%</span>'

    parts = [f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{e(rep['title'])} {e(rep['date'])}</title><style>{_CSS}</style></head><body><div class='wrap'>",
             f"<h1>{e(rep['title'])}</h1><div class='meta'>資料日期 {e(rep['date'])} · 產生於 {e(rep['generated_at'])}</div>",
             "<div class='card'>",
             f"<b>{e(str(ov['index']))}</b> {ov.get('last')} &nbsp; 1 日 {num(ov.get('ret_1d_pct'), 2)} · 5 日 {num(ov.get('ret_5d_pct'), 2)} · "
             f"20 日 {num(ov.get('ret_20d_pct'), 2)} &nbsp; <span class='pill {e(str(ov.get('regime') or ''))}'>{e(ov.get('regime_label') or '–')}</span>",
             "</div>"]
    if rep["highlights"]:
        parts.append("<h2>今日重點</h2><div class='card'><ul>" + "".join(f"<li>{e(h)}</li>" for h in rep["highlights"]) + "</ul></div>")
    rk = rep.get("ranking")
    if rk:
        bt = rk.get("backtest") or {}
        parts.append(f"<h2>模型分數排名 · {e(str(rk['universe']))} · {int(rk['horizon'])} 日</h2><div class='card'>"
                     f"<div class='meta'>排序模型 {e(str(rk['model']))}，歷史回測 IC {bt.get('ic_mean', float('nan')):+.3f}"
                     f"（t = {bt.get('ic_t', float('nan')):.1f}）。排名是研究參考，不是買賣建議。"
                     + (f"<b>排名資料已 {rk['age_days']} 天未更新。</b>" if rk.get("stale") else "") + "</div><div class='tw'><table>"
                     "<tr><th>排名</th><th>股票</th><th>分數</th><th>近 5 日漲跌</th><th>近 20 日漲跌</th></tr>")
        for r in rk["top"]:
            parts.append(f"<tr><td>{int(r['rank'])}</td><td>{e(str(r['name']))}（{e(str(r['ticker']))}）</td><td>{r['score']:.3f}</td>"
                         f"<td>{num(r.get('ret_5d_pct'))}</td><td>{num(r.get('ret_20d_pct'))}</td></tr>")
        parts.append("</table></div>")
        if rk.get("note"):
            parts.append(f"<div class='meta'>{e(str(rk['note']))}</div>")
        parts.append("</div>")
    if rep["watchlist"]:
        parts.append(f"<h2>觀察清單 · 未來 {int(rep['horizon'])} 個交易日的模型區間</h2><div class='card tw'><table>"
                     "<tr><th>股票</th><th>中位數</th><th>80% 區間</th><th>上漲機率</th><th>年化波動</th><th>外資 5 日（張）</th></tr>")
        for i in rep["watchlist"]:
            if "error" in i:
                parts.append(f"<tr><td>{e(str(i['ticker']))}</td><td colspan='5'>資料不足</td></tr>")
                continue
            fl = i.get("foreign_net_5d_lots")
            parts.append(f"<tr><td>{e(str(i['name']))}（{e(str(i['ticker']))}）</td><td>{num(i['median_pct'])}</td>"
                         f"<td>{i['p10_pct']:+.1f}% ～ {i['p90_pct']:+.1f}%</td><td>{i['p_up']:.0%}</td>"
                         f"<td>{i['hv20_ann_pct']:.0f}%</td><td>{'–' if fl is None else f'{fl:+,.0f}'}</td></tr>")
        parts.append("</table></div>")
    sl = scorecard_lines(rep)
    if sl:
        parts.append("<h2>模型成績單</h2><div class='card'><ul>" + "".join(f"<li>{e(x)}</li>" for x in sl) + "</ul></div>")
    parts.append(f"<div class='foot'>{e(rep['disclaimer'])}</div></div></body></html>")
    return "".join(parts)
