import numpy as np
import pandas as pd
import pytest

from fintech_agent.data import SyntheticProvider, parse_symbol
from fintech_agent.data.covariates import align_series, chip_columns, covariate_features
from fintech_agent.evaluation.backtest import BacktestConfig, run_backtest
from fintech_agent.forecasting.base import QUANTILES
from fintech_agent.forecasting.baselines import NaiveForecaster


def test_align_strictly_before_avoids_same_day_us_close():
    tw = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    us = pd.Series([100.0, 200.0, 300.0], index=pd.DatetimeIndex(["2026-01-02", "2026-01-05", "2026-01-06"]))
    same = align_series(us, tw, strictly_before=False)
    before = align_series(us, tw, strictly_before=True)
    assert list(same) == [200.0, 300.0]
    assert list(before) == [100.0, 200.0]          # TW 1/6 only knows the US close of 1/5


def test_chip_columns_cumulate_and_scale():
    idx = pd.bdate_range("2026-01-01", periods=5)
    vol = pd.Series(1000.0, index=idx)
    inst = pd.DataFrame([{"date": d, "name": "Foreign_Investor", "buy": 600, "sell": 100} for d in idx])
    marg = pd.DataFrame([{"date": d, "MarginPurchaseTodayBalance": 100 + i} for i, d in enumerate(idx)])
    out = chip_columns(idx, vol, inst, marg)
    assert out["foreign_flow_lvl"].iloc[-1] == pytest.approx(2.5)   # 5 days x 500 / 1000
    assert "margin_z" in out


def test_build_covariates_shape_and_features():
    p = SyntheticProvider()
    s = parse_symbol("2330")
    px = p.prices(s)
    cov = p.covariates(s, px)
    assert len(cov) == len(px) and not cov.isna().any().any()
    f = covariate_features(cov, len(cov) - 1)
    assert len(f) == 1 + 3 * (len(cov.columns) - 1)                  # vol_z raw + 3 lags per *_lvl


class _PeekCov(NaiveForecaster):
    name, uses_covariates = "peek-cov", True

    def __init__(self):
        self.last_rows = []

    def predict(self, contexts, horizon, covariates=None):
        self.last_rows += [c.index[-1] for c in covariates]
        return super().predict(contexts, horizon)


def test_backtest_passes_only_past_covariates():
    p = SyntheticProvider()
    prices, covs = {}, {}
    for t in ["2330", "2317"]:
        px = p.prices(parse_symbol(t))
        prices[t], covs[t] = px["close"], p.covariates(parse_symbol(t), px)
    peek = _PeekCov()
    w, _ = run_backtest(prices, {"naive": NaiveForecaster(), "peek": peek}, BacktestConfig(horizon=5, n_windows=10),
                        covariates=covs)
    rows = w[w["model"] == "peek"].reset_index(drop=True)
    for (_, r), last in zip(rows.iterrows(), peek.last_rows):
        assert last < r["origin"]                                     # strictly before the first forecast day


def test_lgbm_cov_trains_and_predicts():
    pytest.importorskip("lightgbm")
    from fintech_agent.forecasting.ml import LGBMCovForecaster
    p = SyntheticProvider()
    ser, covs = [], []
    for t in ["A1", "B2", "C3"]:
        px = p.prices(parse_symbol(t))
        ser.append(px["close"].to_numpy())
        covs.append(p.covariates(parse_symbol(t), px))
    m = LGBMCovForecaster(horizon=5, n_estimators=20).fit([s[:-50] for s in ser], [c.iloc[:-50] for c in covs])
    r = m.predict([s[-300:] for s in ser], 5, covariates=[c.iloc[-305:] for c in covs])
    assert len(r) == 3 and set(r[0].quantiles) == set(QUANTILES)


def test_lgbm_checkpoint_roundtrip_and_load_trained(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    from fintech_agent.config import get_settings
    from fintech_agent.forecasting import registry
    from fintech_agent.forecasting.ml import LGBMCovForecaster
    s = get_settings()
    monkeypatch.setitem(s["forecasting"], "checkpoints_dir", str(tmp_path))
    p = SyntheticProvider()
    ser, covs = [], []
    for t in ["A1", "B2"]:
        px = p.prices(parse_symbol(t))
        ser.append(px["close"].to_numpy())
        covs.append(p.covariates(parse_symbol(t), px))
    m = LGBMCovForecaster(horizon=5, n_estimators=10).fit(ser, covs)
    m.save(registry.checkpoint_path(s, "lgbm-cov", "TW", 5))
    assert registry.load_trained("lgbm-cov", "US", 5, s) is None
    loaded = registry.load_trained("lgbm-cov", "TW", 5, s)
    a = m.predict([ser[0][-300:]], 5, covariates=[covs[0].iloc[-305:]])[0]
    b = loaded.predict([ser[0][-300:]], 5, covariates=[covs[0].iloc[-305:]])[0]
    assert np.allclose(a.point, b.point)


def test_champion_lookup_order(tmp_path, monkeypatch):
    from fintech_agent.config import get_settings
    from fintech_agent.evaluation.experiments import ExperimentStore
    s = get_settings()
    monkeypatch.setitem(s["evaluation"], "runs_dir", str(tmp_path))
    monkeypatch.setitem(s["forecasting"], "champions", {"TW": {5: "lgbm-cov"}})
    st = ExperimentStore(s)
    assert st.champion("TW", 5) == "lgbm-cov"             # settings table
    assert st.champion("TW", 10) == st.default_champion   # horizon not listed
    assert st.champion("US", 5) == st.default_champion    # market not listed
    st.set_champion("TW", 5, "chronos-2", "test")
    assert st.champion("TW", 5) == "chronos-2"            # local promotion wins


def test_untrained_champion_falls_back(tmp_path, monkeypatch):
    from fintech_agent.agents.base import AnalysisContext
    from fintech_agent.agents.specialists import QuantAgent
    from fintech_agent.config import get_settings
    s = get_settings()
    monkeypatch.setitem(s["evaluation"], "runs_dir", str(tmp_path / "runs"))
    monkeypatch.setitem(s["forecasting"], "checkpoints_dir", str(tmp_path / "ck"))
    monkeypatch.setitem(s["forecasting"], "champions", {"TW": {5: "lgbm"}})
    monkeypatch.setitem(s["forecasting"], "panel", ["naive"])
    prov = SyntheticProvider()
    q = QuantAgent(prov, None, s, skill_windows=3)
    q.store.default_champion = "drift"                    # keep the test offline (no model download)
    sym = parse_symbol("2330")
    ctx = AnalysisContext(sym, prov.prices(sym), prov.profile(sym), 5)
    ev = q.gather(ctx)[0]
    assert ev["champion"] == "drift"
    assert "lgbm" not in ev["forecasts"]


def test_rolling_backtest_refits_monthly_without_lookahead(monkeypatch):
    pytest.importorskip("lightgbm")
    from fintech_agent.forecasting.ml import LGBMForecaster
    p = SyntheticProvider()
    prices = {t: p.prices(parse_symbol(t))["close"] for t in ["A1", "B2", "C3"]}
    fits = []
    m = LGBMForecaster(horizon=5, n_estimators=10)
    orig = m.fit_rows

    def spy(X, Y):
        fits.append(len(X))
        return orig(X, Y)

    monkeypatch.setattr(m, "fit_rows", spy)
    idx = prices["A1"].index
    cfg = BacktestConfig(horizon=5, n_windows=10_000, step=5, min_origin=str(idx[-130].date()), retrain="M")
    w, lb = run_backtest(prices, {"lgbm": m, "naive": NaiveForecaster()}, cfg)
    n_months = pd.DatetimeIndex(w[w["model"] == "lgbm"]["origin"]).to_period("M").nunique()
    assert len(fits) == n_months >= 5
    assert fits == sorted(fits) and fits[-1] > fits[0]          # expanding window grows every month
    assert set(w["model"]) == {"lgbm", "naive"}


def test_rolling_rows_never_use_future_targets():
    from fintech_agent.forecasting.ml import LGBMForecaster
    x = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 400)))
    X, Y, K, T = LGBMForecaster(horizon=5).build_rows([x])
    assert (T < len(x)).all() and (np.diff(T) > 0).all() and len(X) == len(Y) == len(K)


def test_ranking_panel_and_backtest():
    pytest.importorskip("lightgbm")
    from fintech_agent.evaluation.regimes import market_regime
    from fintech_agent.ranking import LGBMRanker, RankConfig, build_panel, factor_scorers, rank_backtest, summarize
    p = SyntheticProvider()
    tick = [f"{1100 + i}" for i in range(24)]
    prices = {t: p.prices(parse_symbol(t)) for t in tick}
    mkt = p.prices(parse_symbol("^TWII"))["close"]
    panel = build_panel(prices, mkt, 5)
    # label = return from close[d+1] to close[d+6]
    dates = panel.index.get_level_values(0).unique()
    t0, d0 = tick[0], dates[300]
    c = np.log(prices[t0]["close"])
    k = c.index.get_loc(d0)
    assert np.isclose(panel.loc[(d0, t0), "fwd_ret"], c.iloc[k + 6] - c.iloc[k + 1])
    assert panel["r20"].between(0, 1).all() or panel["r20"].isna().any()      # percentile features
    cfg = RankConfig(horizon=5, start=str(dates[550].date()), top_n=5)
    per = rank_backtest(panel, [LGBMRanker(n_estimators=20)] + factor_scorers(panel), cfg, regime=market_regime(mkt))
    s = summarize(per, cfg)
    assert {"xs-lgbm", "momentum", "reversal"} <= set(s["scorer"])
    assert per.groupby("scorer")["date"].apply(lambda d: d.diff().dropna().min()).min() >= pd.Timedelta(days=5)
    assert s["ic_mean"].abs().max() < 0.2                     # random walks: no real edge


def test_quant_agent_uses_fresh_ranking_snapshot(tmp_path, monkeypatch):
    import json
    from fintech_agent.agents.base import AnalysisContext
    from fintech_agent.agents.specialists import QuantAgent
    from fintech_agent.config import get_settings
    s = get_settings()
    runs = tmp_path / "runs"
    monkeypatch.setitem(s["evaluation"], "runs_dir", str(runs))
    prov = SyntheticProvider()
    sym = parse_symbol("2330")
    px = prov.prices(sym)
    runs.mkdir()
    snap = {"as_of": str(px.index[-1].date()), "model": "xs-lgbm-chips", "universe": "tw50", "horizon": 5, "n": 50,
            "backtest": {"ic_mean": 0.05, "ic_t": 4.5, "excess_ann": 0.08, "top_n": 10},
            "ranks": [{"ticker": "2330", "name": "台積電", "rank": 1, "score": 0.6}]}
    (runs / "ranking_TW_h5.json").write_text(json.dumps(snap))
    monkeypatch.setitem(s["ranking"], "weight", 0.5)
    q = QuantAgent(prov, None, s, panel=["naive", "drift"], skill_windows=3)
    ev, score, conf, signals, _ = q.gather(AnalysisContext(sym, px, prov.profile(sym), 5))
    r = ev["cross_sectional_rank"]
    assert r["rank"] == 1 and r["contribution"] == pytest.approx(0.5)
    assert any("選股排序" in t for t, _ in signals)
    snap["as_of"] = str((px.index[-1] - pd.Timedelta(days=30)).date())     # stale -> ignored
    (runs / "ranking_TW_h5.json").write_text(json.dumps(snap))
    ev2 = q.gather(AnalysisContext(sym, px, prov.profile(sym), 5))[0]
    assert "cross_sectional_rank" not in ev2


def test_point_in_time_members_and_panel():
    from fintech_agent.data.universe import point_in_time_members
    from fintech_agent.ranking import RankConfig, build_panel, factor_scorers, rank_backtest, summarize
    p = SyntheticProvider()
    tick = [f"{1200 + i}" for i in range(16)]
    prices = {t: p.prices(parse_symbol(t)) for t in tick}
    prices[tick[0]] = prices[tick[0]].assign(volume=prices[tick[0]]["volume"] * 1000)   # always most liquid
    mem = point_in_time_members(prices, top=8, min_history=100)
    assert mem.iloc[:99].sum().sum() == 0                     # nobody before min_history
    assert (mem.iloc[150:].sum(axis=1) == 8).all() and mem[tick[0]].iloc[150:].all()
    mkt = p.prices(parse_symbol("^TWII"))["close"]
    panel = build_panel(prices, mkt, 5, members=mem)
    assert panel.groupby(level="date").size().max() == 8
    dates = panel.index.get_level_values(0).unique()
    cfg = RankConfig(horizon=5, start=str(dates[300].date()), top_n=3, buffer_rank=5)
    per = rank_backtest(panel, factor_scorers(panel), cfg)
    s = summarize(per, cfg)
    assert (s["buf_turnover"] <= s["turnover"] + 1e-9).all()     # the buffer never trades more


def test_regime_at_handles_repeated_dates():
    from fintech_agent.evaluation.regimes import market_regime, regime_at
    c = pd.Series(np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.01, 500))),
                  index=pd.bdate_range("2020-01-01", periods=500))
    r = market_regime(c)
    d = list(c.index[300:305]) * 3                             # many tickers share an origin date
    out = regime_at(r, d)
    assert len(out) == 15 and list(out[:5]) == list(out[5:10])


def test_incremental_history_fetches_only_new_days(tmp_path):
    from fintech_agent.data.history import IncrementalHistory
    calls = []

    def fetch(a, b):
        calls.append((a, b))
        d = pd.bdate_range(a, b)
        return pd.DataFrame({"date": d.strftime("%Y-%m-%d"), "name": "Foreign_Investor", "buy": 1.0, "sell": 0.0})

    legacy = tmp_path / "cache"
    legacy.mkdir()
    fetch("2025-01-01", "2025-03-31").to_parquet(legacy / "inst_hist_2330_2025-01-01_2025-03-31_abcd1234.parquet")
    calls.clear()
    h = IncrementalHistory(tmp_path / "hist", legacy_dir=legacy)
    df = h.get("inst_hist", "2330", "2025-01-01", "2025-04-30", fetch, keys=("date", "name"))
    assert len(calls) == 1 and calls[0][0] >= "2025-03-20"          # adopted the old file, fetched only the tail
    assert df["date"].is_monotonic_increasing and not df.duplicated(["date", "name"]).any()
    calls.clear()
    h.get("inst_hist", "2330", "2025-01-01", "2025-04-30", fetch, keys=("date", "name"))
    assert calls == []                                              # nothing new: no network
