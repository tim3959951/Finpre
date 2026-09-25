"""Streamlit client UI.  Run:  streamlit run fintech_agent/ui/app.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fintech_agent.agents import DISCLAIMER, ClientProfile, InvestmentAdvisor  # noqa: E402
from fintech_agent.config import get_settings  # noqa: E402
from fintech_agent.data import LiveDataProvider  # noqa: E402
from fintech_agent.evaluation.abtest import compare  # noqa: E402
from fintech_agent.evaluation.backtest import BacktestConfig, run_backtest  # noqa: E402
from fintech_agent.evaluation.experiments import ExperimentStore  # noqa: E402
from fintech_agent.forecasting.registry import list_models  # noqa: E402
from fintech_agent.llm import PROVIDERS, get_llm  # noqa: E402
from fintech_agent.ui.charts import calibration_scatter, leaderboard_bar, price_chart  # noqa: E402

st.set_page_config(page_title="Fintech Agent · TimesFM", layout="wide")
S = get_settings()


@st.cache_resource(show_spinner=False)
def provider():
    return LiveDataProvider(S)


@st.cache_resource(show_spinner="載入 agents…")
def advisor(provider_name: str, model: str | None) -> InvestmentAdvisor:
    llm = get_llm("advisor", S, provider=provider_name, model=model or None)
    return InvestmentAdvisor(provider(), S, llm=llm)


# ============================================================================ sidebar
with st.sidebar:
    st.header("設定")
    default_p = S.get_path("llm.provider", "none")
    prov = st.selectbox("LLM 提供者", PROVIDERS, index=PROVIDERS.index(default_p) if default_p in PROVIDERS else 3)
    model = st.text_input("模型（留空用預設）", value="", placeholder=S.get_path(f"llm.defaults.{prov}", ""))
    adv = advisor(prov, model.strip() or None)
    if adv.llm.enabled:
        st.success(f"LLM：{adv.llm.provider}:{adv.llm.model}")
    else:
        st.warning(f"規則模式：{getattr(adv.llm, 'reason', '')}")
    st.divider()
    st.subheader("客戶資料")
    risk = st.radio("風險屬性", ["保守", "穩健", "積極"], index=1, horizontal=True)
    period = st.selectbox("投資期間", ["短期 (1-4週)", "中期 (1-3個月)", "長期 (6個月以上)"], index=1)
    holdings_txt = st.text_area("目前持股（每行：代號,股數,成本）", placeholder="2330,1000,850\nNVDA,20,120")
    holdings = {}
    for line in holdings_txt.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if parts and parts[0]:
            holdings[parts[0].upper()] = {"shares": float(parts[1]) if len(parts) > 1 and parts[1] else None,
                                          "cost": float(parts[2]) if len(parts) > 2 and parts[2] else None}
    client = ClientProfile(risk=risk, horizon=period, holdings=holdings)
    horizon = st.select_slider("預測天數（交易日）", options=[1, 3, 5, 10, 20], value=int(S.get_path("forecasting.default_horizon", 5)))
    st.caption(f"Champion 模型：{ExperimentStore(S).champion('TW', horizon)} (TW) · {ExperimentStore(S).champion('US', horizon)} (US)")

tab_chat, tab_stock, tab_lab, tab_about = st.tabs(["投資顧問對話", "個股分析", "模型實驗室", "系統說明"])


# ============================================================================ helpers
def render_result(res) -> None:
    ctx, d = res.ctx, res.decision
    c = ctx.prices["close"]
    q = res.reports.get("quant")
    cols = st.columns(5)
    cols[0].metric(f"{ctx.name}（{ctx.symbol.code}）", f"{ctx.close:,.2f}", f"{(c.iloc[-1] / c.iloc[-2] - 1) * 100:+.2f}%")
    cols[1].metric("建議", d["action"])
    cols[2].metric("信心", f"{d['conviction']}%")
    cols[3].metric("建議部位", f"{d['position_pct']}%")
    if q and q.evidence.get("forecasts"):
        champ = q.evidence["champion"]
        f = q.evidence["forecasts"][champ]
        cols[4].metric(f"{champ} {ctx.horizon}日", f"{f['ret_pct']:+.2f}%", f"上漲機率 {f['p_up']:.0%}", delta_color="off")
    levels = {"停損": d.get("stop_loss")}
    tp = d.get("take_profit") or []
    if tp:
        levels["目標1"] = tp[0]
    st.plotly_chart(price_chart(ctx.prices, ctx.symbol.market, (q.artifacts.get("forecasts") if q else None),
                                q.evidence.get("champion") if q else None, levels=levels), width="stretch")

    st.subheader("首席投資顧問決策")
    a, b = st.columns([3, 2])
    with a:
        st.markdown(d.get("client_message", ""))
        if d.get("thesis"):
            st.markdown(f"**投資論點**：{d['thesis']}")
        if d.get("dissent"):
            st.markdown(f"**專家分歧與權衡**：{d['dissent']}")
        if d.get("advisor_override"):
            st.info(d["advisor_override"])
    with b:
        st.markdown(f"- 進場區間：{d.get('entry_zone')}\n- 停損：{d.get('stop_loss')}\n- 停利：{d.get('take_profit')}\n"
                    f"- 期間：{d.get('horizon')}\n- ATR14：{d.get('atr14')}　年化波動：{d.get('hv20_ann_pct')}%")
        st.markdown("**風險**\n" + "\n".join(f"- {r}" for r in d.get("risks", [])))
        st.markdown("**觀察指標**\n" + "\n".join(f"- {r}" for r in d.get("monitoring", [])))
    st.caption(f"決策引擎：{d.get('advisor_llm')} · 權重 {d.get('draft', {}).get('effective_weights')} · 耗時 {res.elapsed_s}s")

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
    if q and q.evidence.get("forecasts"):
        st.subheader("模型預測面板")
        fdf = pd.DataFrame(q.evidence["forecasts"]).T
        fdf.columns = ["預測報酬%", "上漲機率", "P10%", "P90%", "目標價"]
        st.dataframe(fdf, width="stretch")
        bt = q.evidence.get("backtest_on_this_ticker") or {}
        if bt:
            st.caption("此標的近期 walk-forward 回測（非重疊視窗）")
            st.dataframe(pd.DataFrame(bt).T, width="stretch")
    st.caption(DISCLAIMER)


# ============================================================================ chat
with tab_chat:
    st.caption("與首席投資顧問對話，它會自動調度技術、基本面/籌碼、量化三位專家。例：「2330 現在適合進場嗎？」「比較 NVDA 用哪個模型預測比較準」")
    st.session_state.setdefault("chat_msgs", [])
    st.session_state.setdefault("chat_display", [])
    for role, text in st.session_state.chat_display:
        with st.chat_message(role):
            st.markdown(text)
    if prompt := st.chat_input("輸入問題…"):
        st.session_state.chat_display.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status("顧問思考中…", expanded=False) as status:
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
    if run and ticker.strip():
        with st.status(f"分析 {ticker}…", expanded=True) as status:
            try:
                st.session_state["last_result"] = adv.analyze(ticker.strip(), horizon, client, on_event=status.write)
                status.update(label="分析完成", state="complete", expanded=False)
            except Exception as e:
                status.update(label=f"失敗：{e}", state="error")
    if st.session_state.get("last_result") is not None:
        render_result(st.session_state["last_result"])

# ============================================================================ model lab
with tab_lab:
    st.caption("Walk-forward 回測 + champion/challenger A/B 檢定。TimesFM 2.5 為 v1 champion；naive (random walk) 是必須打敗的基準。")
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
    st.markdown("""
### 架構
**首席投資顧問 Agent** 調度三位專家（平行執行），再以信心加權 + LLM 判斷做最終決策：
- **技術分析師**：均線、MACD、KD、RSI、布林、ADX、ATR、量價、支撐壓力
- **基本面／籌碼／情緒分析師**：月營收、EPS、本益比百分位、三大法人、融資券、外資持股、新聞情緒、大盤行情
- **量化 ML 工程師**：TimesFM 2.5（champion）+ Chronos-2 / Bolt + 統計基準，含此標的即時回測與模型技能折減

每位專家都是「規則引擎算分（可稽核）→ LLM 解讀並在 ±0.5 內調整」，沒有 LLM 時仍可運作（規則模式）。
""")
    st.dataframe(pd.DataFrame([{"模型": s.name, "類型": s.family, "授權": s.license, "可商用": s.commercial_ok,
                                "需訓練": s.trainable, "說明": s.description} for s in list_models(S, True)]),
                 width="stretch", hide_index=True)
    st.caption(DISCLAIMER)
