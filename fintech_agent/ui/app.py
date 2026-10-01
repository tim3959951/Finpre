"""Finpre web app.  Run:  streamlit run fintech_agent/ui/app.py"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fintech_agent import __version__  # noqa: E402
from fintech_agent.agents import DISCLAIMER, ClientProfile, InvestmentAdvisor  # noqa: E402
from fintech_agent.config import PROJECT_ROOT, get_settings  # noqa: E402
from fintech_agent.data import LiveDataProvider  # noqa: E402
from fintech_agent.evaluation.abtest import compare  # noqa: E402
from fintech_agent.evaluation.backtest import BacktestConfig, run_backtest  # noqa: E402
from fintech_agent.evaluation.experiments import ExperimentStore  # noqa: E402
from fintech_agent.forecasting.registry import list_models  # noqa: E402
from fintech_agent.llm import PROVIDERS, get_llm  # noqa: E402
from fintech_agent.product.audit import AuditLog  # noqa: E402
from fintech_agent.product.compliance import MODE_LABEL, QUOTE_LABEL, RESEARCH_DISCLAIMER, ComplianceGuard  # noqa: E402
from fintech_agent.product.plans import PLANS, QuotaExceeded, Tenant, TenantStore  # noqa: E402
from fintech_agent.product.portfolio import portfolio_risk  # noqa: E402
from fintech_agent.product.report import build_report, render_html, render_markdown  # noqa: E402
from fintech_agent.product.scorecard import backtest_scorecard, live_scorecard  # noqa: E402
from fintech_agent.ui.charts import SERIES, calibration_scatter, leaderboard_bar, price_chart, return_fan  # noqa: E402

st.set_page_config(page_title="Finpre · 有成績單的 AI 投資研究", layout="wide")
S = get_settings()
# Access control. FA_UI_AUTH=apikey: every visitor signs in with a tenant API key — plan quotas, compliance mode and
# the audit log then apply exactly as on the API. FA_UI_AUTH=none (local development only): no sign-in.
UI_AUTH = os.environ.get("FA_UI_AUTH", str(S.get_path("product.ui.auth", "none")))
ADMIN = os.environ.get("FA_UI_ADMIN", "1" if UI_AUTH == "none" else "0") == "1"   # model lab, LLM choice, champions
ALLOW_ADVISOR = os.environ.get("FA_ALLOW_ADVISOR") == "1"     # local only: deployments run by a licensed firm


@st.cache_resource(show_spinner=False)
def provider():
    return LiveDataProvider(S)


@st.cache_resource(show_spinner=False)
def tenant_store() -> TenantStore:
    return TenantStore(S)


@st.cache_resource(show_spinner=False)
def audit_log() -> AuditLog:
    return AuditLog(S)


@st.cache_resource(show_spinner="載入模型…")
def llm_client(provider_name: str, model: str | None):
    return get_llm("advisor", S, provider=provider_name, model=model or None)


def session_advisor(provider_name: str, model: str | None, mode: str, actor: str) -> InvestmentAdvisor:
    """One advisor per browser session: results and chat state are never shared between visitors."""
    key = (provider_name, model, mode, actor)
    if st.session_state.get("adv_key") != key:
        st.session_state["adv"] = InvestmentAdvisor(provider(), S, llm=llm_client(provider_name, model), mode=mode,
                                                    actor=actor, audit=audit_log())
        st.session_state["adv_key"] = key
    return st.session_state["adv"]


def parse_holdings(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.replace("，", ",").split(",")]
        if parts and parts[0]:
            try:
                out[parts[0].upper()] = {"shares": float(parts[1]) if len(parts) > 1 and parts[1] else None,
                                         "cost": float(parts[2]) if len(parts) > 2 and parts[2] else None}
            except ValueError:
                continue
    return out


def current_tenant() -> Tenant | None:
    tid = st.session_state.get("tenant_id")
    return tenant_store().get(tid) if tid else None


def charge(endpoint: str, analysis: bool = False) -> bool:
    """Meter a web action against the signed-in tenant's plan. False (with a message) when over quota."""
    t = current_tenant()
    if t is None:
        return True
    try:
        tenant_store().charge(t, f"web:{endpoint}", analysis)
        return True
    except QuotaExceeded as e:
        st.error(f"{e}（可在「方案與說明」升級方案）")
        return False


def audit(kind: str, subject: str, request, response, compliance=None) -> None:
    try:
        audit_log().record(actor=ACTOR, mode=MODE, kind=kind, subject=subject, request=request, response=response,
                           compliance=compliance)
    except Exception as e:  # never lose an answer because of the audit log
        st.warning(f"稽核紀錄寫入失敗：{e}")


# ============================================================================ sign-in
if UI_AUTH == "apikey" and current_tenant() is None:
    st.title("Finpre · 有成績單的 AI 投資研究")
    st.caption("請以您的 API 金鑰登入（金鑰由管理者以 scripts/manage_tenants.py 建立）。")
    with st.form("login"):
        key = st.text_input("API 金鑰", type="password", placeholder="fp_...")
        if st.form_submit_button("登入", type="primary"):
            t = tenant_store().authenticate(key.strip())
            if t is None:
                st.error("金鑰無效或已停用")
            else:
                st.session_state["tenant_id"] = t.id
                st.rerun()
    st.caption(RESEARCH_DISCLAIMER)
    st.stop()

TENANT = current_tenant()
PLAN = TENANT.plan if TENANT else None

# ============================================================================ sidebar
with st.sidebar:
    st.markdown(f"### Finpre `v{__version__}`")
    if TENANT:
        MODE = TENANT.mode
        ACTOR = f"tenant:{TENANT.id}"
        use = tenant_store().usage(TENANT.id)
        st.markdown(f"**{TENANT.name}** · {PLAN.name}")
        st.caption(f"本月用量：{use.get('calls', 0):,} / {PLAN.api_calls_per_month:,} 次 · 每日完整分析上限 {PLAN.analyses_per_day}")
        if st.button("登出"):
            for k in ("tenant_id", "adv", "adv_key", "last_result", "chat_msgs", "chat_display", "risk", "report"):
                st.session_state.pop(k, None)
            st.rerun()
    else:
        default_mode = S.get_path("product.mode", "research")
        MODE = (st.radio("服務模式", ["research", "advisor"], index=["research", "advisor"].index(default_mode),
                         format_func=lambda m: MODE_LABEL[m], horizontal=True) if ALLOW_ADVISOR else "research")
        ACTOR = "web:local"
    mode = MODE
    st.info(f"**{MODE_LABEL[mode]}**" + ("：提供訊號、機率與風險區間，不提供個股買賣建議或價位。" if mode == "research"
                                          else "：含操作建議與價位，僅限持牌投顧機構使用。"))
    default_p = S.get_path("llm.provider", "none")
    if ADMIN:
        st.header("設定")
        prov = st.selectbox("LLM 提供者", PROVIDERS, index=PROVIDERS.index(default_p) if default_p in PROVIDERS else 3)
        model = st.text_input("模型（留空用預設）", value="", placeholder=S.get_path(f"llm.defaults.{prov}", ""))
    else:
        prov, model = default_p, ""
    adv = session_advisor(prov, model.strip() or None, mode, ACTOR)
    if ADMIN:
        if adv.llm.enabled:
            st.success(f"LLM：{adv.llm.provider}:{adv.llm.model}")
        else:
            st.warning(f"規則模式：{getattr(adv.llm, 'reason', '')}")
    st.divider()
    st.subheader("投資人資料")
    risk = st.radio("風險屬性", ["保守", "穩健", "積極"], index=1, horizontal=True)
    period = st.selectbox("投資期間", ["短期 (1-4週)", "中期 (1-3個月)", "長期 (6個月以上)"], index=1)
    holdings_txt = st.text_area("目前持股（每行：代號,股數,成本）", value="2330,1000,850\n2317,2000,150\nNVDA,30,120",
                                key="holdings_txt")
    holdings = parse_holdings(holdings_txt)
    client = ClientProfile(risk=risk, horizon=period, holdings=holdings)
    horizon = st.select_slider("預測天數（交易日）", options=[1, 3, 5, 10, 20], value=int(S.get_path("forecasting.default_horizon", 5)))
    st.caption(f"預測模型：{ExperimentStore(S).champion('TW', horizon)}（台股）· {ExperimentStore(S).champion('US', horizon)}（美股）")

_TABS = ["研究助理", "個股分析", "風險雷達", "每日報告", "成績單"] + (["模型實驗室"] if ADMIN else []) + ["方案與說明"]
_tabs = dict(zip(_TABS, st.tabs(_TABS)))
tab_chat, tab_stock, tab_risk, tab_report, tab_card, tab_about = (_tabs[k] for k in ("研究助理", "個股分析", "風險雷達",
                                                                                      "每日報告", "成績單", "方案與說明"))
tab_lab = _tabs.get("模型實驗室")


# ============================================================================ helpers
def render_result(res) -> None:
    ctx, d = res.ctx, res.decision
    c = ctx.prices["close"]
    q = res.reports.get("quant")
    research = res.mode == "research"
    cols = st.columns(5)
    cols[0].metric(f"{ctx.name}（{ctx.symbol.code}）", f"{ctx.close:,.2f}", f"{(c.iloc[-1] / c.iloc[-2] - 1) * 100:+.2f}%")
    if research:
        lo, hi = d.get("return_band_pct") or [None, None]
        cols[1].metric("綜合訊號", d["signal"], f"分數 {d['signal_score']:+.2f}", delta_color="off")
        cols[2].metric("信心", f"{d['conviction']}%")
        cols[3].metric(f"{ctx.horizon} 日報酬 80% 區間", f"{lo:+.1f}% ～ {hi:+.1f}%" if lo is not None else "–")
    else:
        cols[1].metric("建議", d["action"])
        cols[2].metric("信心", f"{d['conviction']}%")
        cols[3].metric("建議部位", f"{d['position_pct']}%")
    if q and q.evidence.get("forecasts"):
        champ = q.evidence["champion"]
        f = q.evidence["forecasts"][champ]
        cols[4].metric(f"{champ} {ctx.horizon}日", f"{f['ret_pct']:+.2f}%", f"上漲機率 {f['p_up']:.0%}", delta_color="off")
    levels = {}
    if not research:
        levels = {"停損": d.get("stop_loss")}
        tp = d.get("take_profit") or []
        if tp:
            levels["目標1"] = tp[0]
    fcs = q.artifacts.get("forecasts") if q else None
    champ_name = q.evidence.get("champion") if q else None
    st.plotly_chart(price_chart(ctx.prices, ctx.symbol.market, None if research else fcs, champ_name, levels=levels),
                    width="stretch")
    if research and fcs:
        st.markdown(f"**未來 {ctx.horizon} 個交易日的報酬率分布**（模型預測，以百分比表示）")
        st.plotly_chart(return_fan(fcs, champ_name, ctx.close), width="stretch")
        st.caption("藍色區域為模型的 80% 統計區間，是機率分布的描述，不是買賣價位或目標。")

    st.subheader("首席研究分析師結論" if research else "首席投資顧問決策")
    a, b = st.columns([3, 2])
    with a:
        st.markdown(d.get("client_message", ""))
        if d.get("thesis"):
            st.markdown(f"**研究論點**：{d['thesis']}" if research else f"**投資論點**：{d['thesis']}")
        if d.get("dissent"):
            st.markdown(f"**專家分歧與權衡**：{d['dissent']}")
        if d.get("advisor_override"):
            st.info(d["advisor_override"])
    with b:
        if research:
            st.markdown(f"- 期間：{d.get('horizon')}\n- 近 20 日年化波動：{d.get('hv20_ann_pct')}%\n"
                        f"- 日均真實波幅（ATR14）：{d.get('atr14_pct')}% 的股價\n- 三位分析師分歧度：{d.get('agent_dispersion')}")
            st.markdown("**觀察指標**\n" + "\n".join(f"- {r}" for r in d.get("watch_items", [])))
        else:
            st.markdown(f"- 進場區間：{d.get('entry_zone')}\n- 停損：{d.get('stop_loss')}\n- 停利：{d.get('take_profit')}\n"
                        f"- 期間：{d.get('horizon')}\n- ATR14：{d.get('atr14')}　年化波動：{d.get('hv20_ann_pct')}%")
            st.markdown("**觀察指標**\n" + "\n".join(f"- {r}" for r in d.get("monitoring", [])))
        st.markdown("**風險**\n" + "\n".join(f"- {r}" for r in d.get("risks", [])))
    st.caption(f"決策引擎：{d.get('advisor_llm')} · 耗時 {res.elapsed_s}s"
               + (f" · 合規過濾移除 {res.compliance.get('removed_sentences', 0)} 句" if research else ""))

    st.subheader("專家報告")
    for col, key in zip(st.columns(3), ["technical", "fundamental", "quant"]):
        r = res.reports.get(key)
        if not r:
            continue
        with col:
            st.markdown(f"#### {r.title}")
            st.markdown(f"**{r.stance}**　分數 {r.score:+.2f}（規則 {r.base_score:+.2f}）　信心 {r.confidence:.0%}")
            st.write(r.summary)
            st.markdown("\n".join(f"- {k}" for k in r.key_points))
            if r.risks:
                st.markdown("**風險**\n" + "\n".join(f"- {k}" for k in r.risks))
            if r.adjustment_reason:
                st.caption(f"LLM 調整理由：{r.adjustment_reason}")
            st.caption(f"{r.llm} · {r.elapsed_s}s")
            with st.expander("規則訊號 / 證據 JSON"):
                st.dataframe(pd.DataFrame(r.rule_signals, columns=["訊號", "貢獻"]), hide_index=True, width="stretch")
                st.json(r.evidence, expanded=False)
    fr = res.reports.get("fundamental")
    heads = (fr.evidence.get("headlines") if fr else None) or []
    if heads:
        with st.expander(f"近期新聞標題與情緒分數（{len(heads)} 則）"):
            st.caption(QUOTE_LABEL)
            st.dataframe(pd.DataFrame(heads).rename(columns={"title": "標題", "date": "日期", "score": "情緒分數"}),
                         hide_index=True, width="stretch")
    if q and q.evidence.get("forecasts"):
        st.subheader("模型預測面板")
        fdf = pd.DataFrame(q.evidence["forecasts"]).T
        names = {"ret_pct": "預測報酬%", "p_up": "上漲機率", "p10_pct": "P10%", "p90_pct": "P90%", "price_h": "預測中位價"}
        fdf = fdf.rename(columns=names)
        st.dataframe(fdf, width="stretch")
        bt = q.evidence.get("backtest_on_this_ticker") or {}
        if bt:
            st.caption("此標的近期 walk-forward 回測（非重疊視窗）")
            st.dataframe(pd.DataFrame(bt).T, width="stretch")
    st.caption(RESEARCH_DISCLAIMER if research else DISCLAIMER)


# ============================================================================ chat
with tab_chat:
    st.caption("與首席研究分析師對話，它會自動調度技術、基本面/籌碼、量化三位專家。例：「2330 最近訊號如何？」「比較 NVDA 用哪個模型預測比較準」")
    st.session_state.setdefault("chat_msgs", [])
    st.session_state.setdefault("chat_display", [])
    for role, text in st.session_state.chat_display:
        with st.chat_message(role):
            st.markdown(text)
    if (prompt := st.chat_input("輸入問題…", max_chars=500)) and charge("chat", analysis=True):
        st.session_state.chat_display.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status("分析中…", expanded=False) as status:
                answer, trace = adv.chat(st.session_state.chat_msgs, prompt, client, on_event=status.write)
                status.update(label=f"完成（{len(trace)} 次工具呼叫）", state="complete")
            st.markdown(answer)
        st.session_state.chat_display.append(("assistant", answer))
        if adv.last_result is not None:
            st.session_state["last_result"] = adv.last_result
    if st.session_state.chat_display and st.button("清除對話"):
        st.session_state.chat_msgs, st.session_state.chat_display = [], []
        st.rerun()

# ============================================================================ single stock
with tab_stock:
    c1, c2 = st.columns([3, 1])
    ticker = c1.text_input("股票代號", value="2330", help="台股：2330、0050、6488；美股：NVDA、AAPL、SPY")
    run = c2.button("執行完整分析", type="primary", width="stretch")
    if run and ticker.strip() and charge("analyze", analysis=True):
        with st.status(f"分析 {ticker}…", expanded=True) as status:
            try:
                st.session_state["last_result"] = adv.analyze(ticker.strip(), horizon, client, on_event=status.write)
                status.update(label="分析完成", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"失敗：{e}", state="error")
    if st.session_state.get("last_result") is not None:
        render_result(st.session_state["last_result"])

# ============================================================================ risk radar
with tab_risk:
    st.caption("輸入持股，估計組合未來的報酬區間、風險值（VaR）、預期短缺（ES）與集中度。持股取自側邊欄，也可在下方修改。")
    rc1, rc2, rc3 = st.columns([3, 1, 1])
    risk_txt = rc1.text_area("持股（代號,股數）", value=st.session_state.get("holdings_txt", ""), key="risk_holdings", height=110)
    risk_h = rc2.selectbox("期間（交易日）", [5, 20], index=0)
    cash = rc3.number_input("現金（新台幣）", min_value=0.0, value=0.0, step=10000.0)
    if st.button("計算組合風險", type="primary"):
        hold = {k: v["shares"] for k, v in parse_holdings(risk_txt).items() if v["shares"] and v["shares"] > 0}
        limit = PLAN.portfolio_positions_max if PLAN else 60
        if len(hold) > limit:
            st.error(f"{PLAN.name if PLAN else ''}最多 {limit} 檔持股")
        elif charge("portfolio"):
            with st.spinner("計算中…"):
                try:
                    rr_ = portfolio_risk(provider(), hold, risk_h, "TWD", S, cash=cash, max_positions=limit)
                    rr_, rep_ = ComplianceGuard(mode).apply(rr_)
                    st.session_state["risk"] = rr_
                    audit("portfolio", ",".join(list(hold)[:20]), {"holdings": hold, "horizon": risk_h, "cash": cash}, rr_,
                          rep_.to_dict())
                except Exception as e:
                    st.error(str(e))
    rr = st.session_state.get("risk")
    if rr:
        m = st.columns(5)
        m[0].metric("組合市值（新台幣）", f"{rr['total_value']:,.0f}")
        m[1].metric(f"{rr['horizon']} 日報酬 80% 區間", f"{rr['return_pct']['p10']:+.1f}% ～ {rr['return_pct']['p90']:+.1f}%")
        m[2].metric("95% VaR", f"{rr['var95_pct']:.1f}%", f"約 {rr['var95_amount']:,.0f} 元", delta_color="off")
        m[3].metric("95% ES（尾端平均損失）", f"{rr['es95_pct']:.1f}%", f"約 {rr['es95_amount']:,.0f} 元", delta_color="off")
        m[4].metric("虧損超過 5% 的機率", f"{rr['prob_loss_over_5pct']:.0%}")
        for f in rr["flags"]:
            st.warning(f)
        pos = pd.DataFrame([{"股票": f"{p['name']}（{p['ticker']}）", "權重": p["weight"], "尾端損失貢獻": p["tail_loss_share"],
                             "中位數%": p["forecast"]["median_pct"], "P10%": p["forecast"]["p10_pct"],
                             "P90%": p["forecast"]["p90_pct"], "年化波動%": p["forecast"]["hv20_ann_pct"],
                             "產業": p["sector"], "模型": p["forecast"]["model"]} for p in rr["positions"]])
        g1, g2 = st.columns([3, 2])
        g1.dataframe(pos.style.format({"權重": "{:.1%}", "尾端損失貢獻": "{:.1%}"}), hide_index=True, width="stretch")
        fig = go.Figure()
        fig.add_bar(y=pos["股票"], x=pos["權重"], orientation="h", name="權重", marker_color=SERIES[0])
        fig.add_bar(y=pos["股票"], x=pos["尾端損失貢獻"], orientation="h", name="尾端損失貢獻", marker_color=SERIES[1])
        fig.update_layout(barmode="group", height=80 + 42 * len(pos), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_tickformat=".0%", legend=dict(orientation="h", y=1.08))
        g2.plotly_chart(fig, width="stretch")
        st.caption(f"方法：{rr['method']}（{rr['scenarios']} 個情境）。市場狀態："
                   + "、".join(f"{k} {v['label']}" for k, v in rr["regimes"].items()) + "。" + RESEARCH_DISCLAIMER)

# ============================================================================ daily report
with tab_report:
    d1, d2, d3 = st.columns([1, 3, 1])
    rep_market = d1.radio("市場", ["TW", "US"], horizontal=True, format_func=lambda m: "台股" if m == "TW" else "美股")
    default_wl = ",".join((S.get_path(f"product.daily.{rep_market}.watchlist", []) or [])[:8])
    wl = d2.text_input("觀察清單（逗號分隔）", value=default_wl, key=f"wl_{rep_market}")
    if d3.button("產生報告", type="primary", width="stretch"):
        wl_list = [x.strip() for x in wl.split(",") if x.strip()]
        if PLAN and len(wl_list) > PLAN.watchlist_max:
            st.error(f"{PLAN.name}觀察清單最多 {PLAN.watchlist_max} 檔")
        elif charge("report"):
            with st.spinner("產生報告中…"):
                try:
                    rep_ = build_report(provider(), rep_market, 5, wl_list, S,
                                        top_n=5 if PLAN and PLAN.key == "free" else 10)
                    if PLAN and PLAN.key == "free" and rep_.get("ranking"):
                        rep_["ranking"]["bottom"] = []
                    full_ = rep_.pop("compliance_full", None)
                    st.session_state["report"] = rep_
                    audit("report", rep_market, {"watchlist": wl_list}, rep_, full_)
                except Exception as e:
                    st.error(str(e))
    rep = st.session_state.get("report")
    if rep:
        html = render_html(rep)
        b1, b2, b3 = st.columns(3)
        b1.download_button("下載 HTML", html, file_name=f"{rep['date']}_{rep['market']}.html", mime="text/html")
        b2.download_button("下載 Markdown", render_markdown(rep), file_name=f"{rep['date']}_{rep['market']}.md")
        b3.download_button("下載 JSON", json.dumps(rep, ensure_ascii=False, indent=1, default=str),
                           file_name=f"{rep['date']}_{rep['market']}.json", mime="application/json")
        if hasattr(st, "iframe"):          # all report fields are html-escaped in render_html
            st.iframe(html, height=1100)
        else:  # older Streamlit
            components.html(html, height=1100, scrolling=True)

# ============================================================================ scorecard
with tab_card:
    st.caption("所有數字都來自 walk-forward 回測（模型只看得到預測當下以前的資料）與上線後的真實追蹤，包含表現不好的年份。")
    bt = backtest_scorecard()
    if bt.get("available"):
        st.subheader("回測成績：2018 → 2026")
        rows = []
        for r in bt["forecasting"]:
            for m in ("lgbm", "timesfm-2.5", "chronos-2"):
                if r.get(m):
                    rows.append({"市場": "台股" if r["market"] == "TW" else "美股", "天期": f"{r['horizon']} 日", "模型": m,
                                 "相對 random walk": f"{r[m]['crps_skill_pct']:+.2f}%",
                                 "勝出股票比例": f"{r[m]['share_beating_naive']:.0%}", "80% 區間覆蓋": f"{r[m]['cov80']:.0%}",
                                 "正值年份": r.get("lgbm_positive_years", "") if m == "lgbm" else ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("**選股排序**")
        st.dataframe(pd.DataFrame([{"股票池": r["universe"], "天期": f"{r['horizon']} 日", "模型": r["model"],
                                    "平均 IC": f"{r['ic_mean']:+.3f}", "IC t 值": f"{r['ic_t']:.1f}",
                                    "IC>0 比例": f"{r['ic_positive_share']:.0%}",
                                    "前 10 名超額（未扣成本）": "–" if r["excess_gross"] is None else f"{r['excess_gross']:+.1%}",
                                    "扣成本＋換手控制後": "–" if r["excess_net"] is None else f"{r['excess_net']:+.1%}"}
                                   for r in bt["ranking"]]), hide_index=True, width="stretch")
        st.caption("超額報酬＝前 10 名等權組合減去整個股票池等權組合（年化）。台股每次來回成本以 58.5 bps 計"
                   "（手續費 0.1425% × 2 + 證交稅 0.3%）。排序與未來報酬的統計關聯顯著，但在逐日重選的流動性前 50 大股票池"
                   "（twse-pit50，較無存活者偏差）中，扣除成本後並未穩定勝過等權組合；tw50 用今天的成分股回推，"
                   "含存活者偏差、數字偏樂觀。這些結果我們照實公開。")
        pngs = sorted((PROJECT_ROOT / "docs" / "benchmarks" / "v0.3").glob("*.png"))
        if pngs:
            with st.expander("所有回測圖表"):
                for p in pngs:
                    st.image(str(p))
    live = live_scorecard(S)
    st.subheader("上線後追蹤")
    if live.get("models"):
        st.dataframe(pd.DataFrame(live["models"]).rename(columns={
            "model": "模型", "horizon": "天期", "n": "已到期預測數", "cov80": "80% 區間覆蓋", "dir_acc": "方向命中率",
            "brier": "上漲機率 Brier 分數", "mae_pct": "平均誤差%", "mae_vs_naive_pct": "誤差相對 random walk 改善%",
            "paired_n": "配對數"}), hide_index=True, width="stretch")
        st.caption(f"自 {live['since']} 起；待到期 {live['pending']} 筆。")
    else:
        st.info(f"目前還沒有到期的預測（待到期 {live.get('pending', 0)} 筆）。每日排程會為追蹤股票池記錄預測，到期後自動計分。")
    card_file = S.resolve_path("evaluation.runs_dir") / "scorecard.json"
    rl = (json.loads(card_file.read_text()).get("ranking_live") or {}) if card_file.exists() else {}
    if rl.get("snapshots_scored"):
        st.markdown(f"**排名的上線後實際 IC**：已計分 {rl['snapshots_scored']} 份排名，平均 IC {rl['mean_ic']:+.3f}，"
                    f"IC 為正的比例 {rl['ic_positive_share']:.0%}")
        st.dataframe(pd.DataFrame(rl["history"]), hide_index=True, width="stretch")

# ============================================================================ model lab (operators only)
if tab_lab is not None:
    with tab_lab:
        st.caption("Walk-forward 回測 + champion/challenger A/B 檢定。Champion 依市場 × 天數分開管理（見 settings.yaml）；naive (random walk) 是必須打敗的基準。")
        with st.expander("選股排序（最新排名與歷史回測）", expanded=False):
            runs_dir = S.resolve_path("evaluation.runs_dir")
            snaps = sorted(runs_dir.glob("ranking_*_h*.json"))
            snaps = [p for p in snaps if not p.name.startswith("ranking_backtest_")]
            if not snaps:
                st.info("尚未產生排名：執行 `python scripts/rank_stocks.py --pool twse --pit-top 50 --horizon 20`。")
            else:
                pick = st.selectbox("排名檔", [p.name for p in snaps])
                snap = json.loads((runs_dir / pick).read_text())
                bt = snap.get("backtest") or {}
                st.markdown(f"**{snap['model']}**・{snap['universe']}・{snap['horizon']} 日・資料日期 {snap['as_of']}　"
                            f"回測 IC {bt.get('ic_mean', float('nan')):+.3f}（t={bt.get('ic_t', float('nan')):.1f}）")
                st.dataframe(pd.DataFrame(snap["ranks"]), width="stretch", hide_index=True)
                st.caption("排名僅供研究；前 10 名不是買進建議。")
        store = ExperimentStore(S)
        specs = list_models(S)
        c1, c2, c3 = st.columns([2, 3, 2])
        market = c1.radio("市場", ["TW", "US"], horizontal=True)
        default_t = "2330,2317,2454,2881,0050" if market == "TW" else "AAPL,MSFT,NVDA,SPY,JPM"
        tickers = [t.strip() for t in c2.text_input("標的（逗號分隔）", default_t).split(",") if t.strip()]
        lab_h = c3.selectbox("預測天數", [1, 5, 10, 20], index=1)
        default_models = [m for m in ["timesfm-2.5", "chronos-2", "chronos-bolt-small", "naive", "drift", "lgbm", "ensemble"]
                          if m in [s.name for s in specs]]
        models = st.multiselect("模型", [s.name for s in specs], default=default_models)
        c4, c5, c6 = st.columns(3)
        windows = c4.slider("每檔視窗數", 10, 150, 40, step=10)
        step = c5.slider("視窗間隔（交易日）", 1, 20, lab_h)
        min_origin = c6.text_input("最早預測起點（避免預訓練洩漏，選填）", "", placeholder="2025-10-01")
        if st.button("開始回測", type="primary"):
            prices = {}
            for t in tickers:
                sym = provider().symbol(t)
                px = provider().prices(sym)
                if len(px):
                    prices[sym.code] = px["close"]
            cfg = BacktestConfig(horizon=lab_h, n_windows=windows, step=step, min_origin=min_origin or None,
                                 context_length=int(S.get_path("forecasting.context_length", 512)),
                                 cost_bps=float(S.get_path(f"evaluation.cost_bps.{market}", 0)))
            bar = st.progress(0.0, "準備中…")
            w, lb = run_backtest(prices, models, cfg, S, progress=lambda k, n, m: bar.progress(k / n, f"執行 {m}…"))
            bar.progress(1.0, "完成")
            run_id = store.log_run("benchmark", market, lab_h, list(prices), models, cfg.__dict__, lb)
            st.session_state["lab"] = (w, lb, market, lab_h, run_id)
        if "lab" in st.session_state:
            w, lb, mk, hh, run_id = st.session_state["lab"]
            st.markdown(f"**Leaderboard**（run `{run_id}`，訓練截止 {lb.attrs.get('train_cutoff', '-')}）")
            show = [c for c in ["model", "n_windows", "crps_rel", "wql", "crps_skill", "mase", "skill_vs_naive", "mape", "dir_acc", "ic",
                                "cov80", "width80_pct", "strat_sharpe", "bh_sharpe", "strat_max_dd", "sec_per_100"] if c in lb]
            st.dataframe(lb[show].style.format(precision=4), width="stretch", hide_index=True)
            g1, g2 = st.columns(2)
            g1.plotly_chart(leaderboard_bar(lb, "crps_rel" if "crps_rel" in lb else "mase"), width="stretch")
            g2.plotly_chart(calibration_scatter(lb), width="stretch")
            champ = store.champion(mk, hh)
            ab_metric = S.get_path("evaluation.ab_metric", "crps_rel")
            st.markdown(f"**A/B 檢定：champion `{champ}` vs challengers**（Diebold-Mariano per ticker + Stouffer 合併，損失 = {ab_metric.upper()}）")
            rows = []
            for m in lb["model"]:
                if m != champ and champ in set(w["model"]):
                    try:
                        rows.append(compare(w, champ, m, ab_metric, hh, step=step).to_dict())
                    except ValueError:
                        pass
            if rows:
                ab = pd.DataFrame(rows)[["challenger", "n", "champion_loss", "challenger_loss", "improvement_pct", "dm_p",
                                         "p_challenger_better", "ticker_win_rate", "decision", "reason"]]
                st.dataframe(ab, width="stretch", hide_index=True)
                baselines = set() if S.get_path("evaluation.promote_baselines", False) else {"naive", "drift"}
                if ((ab["decision"] == "promote") & ab["challenger"].isin(baselines)).any():
                    st.warning("隨機漫步基準勝過 champion：代表目前沒有模型在此指標上具穩定優勢，量化 Agent 會自動降低權重。")
                winners = ab[(ab["decision"] == "promote") & ~ab["challenger"].isin(baselines)].sort_values("improvement_pct", ascending=False)
                if len(winners):
                    best = winners.iloc[0]
                    if st.button(f"將 {best['challenger']} 升級為 {mk} {hh}日 champion"):
                        store.set_champion(mk, hh, best["challenger"], best["reason"], run_id)
                        st.success("已更新 champion")
            elif champ not in set(w["model"]):
                st.info(f"本次回測未包含 champion `{champ}`，無法做 A/B。")
        with st.expander("線上 shadow A/B（實盤預測記錄）"):
            if st.button("結算已到期預測"):
                n = store.resolve(lambda t: provider().prices(provider().symbol(t))["close"])
                st.write(f"結算 {n} 筆")
            st.dataframe(store.online_scores(), width="stretch")
        with st.expander("歷史實驗"):
            st.dataframe(store.runs(), width="stretch", hide_index=True)

# ============================================================================ about
with tab_about:
    st.subheader("方案")
    st.dataframe(pd.DataFrame([{"方案": p.name, "月費（新台幣）": f"{p.price_twd_month:,}", "每日完整分析": p.analyses_per_day,
                                "每月 API 呼叫": f"{p.api_calls_per_month:,}", "觀察清單": p.watchlist_max,
                                "風險雷達持股數": p.portfolio_positions_max, "顧問模式": "持牌機構" if p.advisor_mode else "–"}
                               for p in PLANS.values()]), hide_index=True, width="stretch")
    st.markdown("""
### 合規設計
- **研究模式**（預設）：只提供訊號、分數、機率、以百分比表示的報酬率區間與模型歷史成績；系統在輸出前逐句過濾買進、賣出、目標價、停損、支撐壓力等字眼。
- **顧問模式**：含操作建議與價位，僅開放給持牌證券投資顧問事業（企業版，依合約由其負責建議）。
- **稽核紀錄**：每一次輸出都寫入防竄改的稽核紀錄（雜湊鏈），可匯出給法遵。
""")
    ok, bad = audit_log().verify()
    st.caption(f"稽核紀錄雜湊鏈：{'完整' if ok else f'第 {bad} 筆起不一致，請通知管理者'}")
    st.markdown("""
### 架構
**首席投資顧問 Agent** 調度三位專家（平行執行），再以信心加權 + LLM 判斷做最終決策：
- **技術分析師**：均線、MACD、KD、RSI、布林、ADX、ATR、量價、支撐壓力
- **基本面／籌碼／情緒分析師**：月營收、EPS、本益比百分位、三大法人、融資券、外資持股、新聞情緒、大盤行情
- **量化 ML 工程師**：各市場 champion（LightGBM / TimesFM 2.5）+ Chronos-2 / Bolt + 統計基準，含此標的即時回測與模型技能折減，以及選股排序模型的橫斷面排名

每位專家都是「規則引擎算分（可稽核）→ LLM 解讀並在 ±0.5 內調整」，沒有 LLM 時仍可運作（規則模式）。
""")
    st.dataframe(pd.DataFrame([{"模型": s.name, "類型": s.family, "授權": s.license, "可商用": s.commercial_ok,
                                "需訓練": s.trainable, "說明": s.description} for s in list_models(S, True)]),
                 width="stretch", hide_index=True)
    st.caption(RESEARCH_DISCLAIMER)
