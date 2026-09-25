import json

import numpy as np
import pandas as pd
import pytest

from fintech_agent.data import SyntheticProvider, parse_symbol
from fintech_agent.data import summaries as S
from fintech_agent.evaluation.abtest import compare, diebold_mariano, direction_significance
from fintech_agent.evaluation.backtest import BacktestConfig, make_origins, run_backtest
from fintech_agent.evaluation.metrics import strategy_stats, window_metrics
from fintech_agent.features.indicators import add_indicators, kd, rsi
from fintech_agent.features.sentiment import lexicon_score
from fintech_agent.features.technical_signals import technical_view
from fintech_agent.forecasting.base import ForecastResult, QUANTILES
from fintech_agent.forecasting.baselines import DriftForecaster, EnsembleForecaster, NaiveForecaster


# ------------------------------------------------------------------ symbols
@pytest.mark.parametrize("raw,market,first", [("2330", "TW", "2330.TW"), ("6488.TWO", "TW", "6488.TWO"),
                                              ("00878", "TW", "00878.TW"), ("nvda", "US", "NVDA"),
                                              ("BRK.B", "US", "BRK-B"), ("^TWII", "TW", "^TWII")])
def test_parse_symbol(raw, market, first):
    s = parse_symbol(raw)
    assert s.market == market and s.yf_ticker == first


# ------------------------------------------------------------------ indicators
def test_rsi_bounds_and_trend():
    up = pd.Series(np.linspace(10, 20, 100))
    assert rsi(up, 14).iloc[-1] > 99
    noisy = pd.Series(100 + np.random.default_rng(0).normal(0, 1, 300).cumsum())
    r = rsi(noisy, 14).dropna()
    assert r.between(0, 100).all()


def test_kd_range():
    px = SyntheticProvider().prices(parse_symbol("2330"))
    k = kd(px["high"], px["low"], px["close"]).dropna()
    assert k["k"].between(0, 100).all() and k["d"].between(0, 100).all()


def test_add_indicators_columns():
    df = add_indicators(SyntheticProvider().prices(parse_symbol("AAPL")))
    for c in ["ma20", "ma240", "dif", "dea", "osc", "k", "d", "rsi14", "bb_pctb", "adx", "atr14", "obv", "hv20"]:
        assert c in df


def test_technical_view_score_bounds():
    tv = technical_view(SyntheticProvider().prices(parse_symbol("2317")))
    assert -2 <= tv.score <= 2 and 0 < tv.confidence <= 1
    assert tv.evidence["ma_alignment"] in {"多頭排列", "空頭排列", "糾結/混合"}


# ------------------------------------------------------------------ summaries
def test_institutional_summary_and_streak():
    dates = pd.bdate_range("2026-01-01", periods=6)
    rows = []
    for i, d in enumerate(dates):
        rows.append({"date": d, "name": "Foreign_Investor", "buy": 5000 + i, "sell": 1000})
        rows.append({"date": d, "name": "Investment_Trust", "buy": 0, "sell": 2000})
    out = S.summarize_institutional(pd.DataFrame(rows))
    assert out["foreign"]["streak_days"] == 6
    assert out["trust"]["streak_days"] == -6
    assert out["foreign"]["net_5d_lots"] > 0


def test_month_revenue_yoy():
    rows = [{"revenue_year": y, "revenue_month": m, "revenue": 100 * (1.2 if y == 2026 else 1.0), "date": pd.Timestamp(y, m, 1)}
            for y in (2025, 2026) for m in range(1, 13) if not (y == 2026 and m > 6)]
    out = S.summarize_month_revenue(pd.DataFrame(rows))
    assert out["latest_month"] == "2026-06"
    assert abs(out["yoy_pct"] - 20) < 1e-6


def test_lexicon_sentiment():
    assert lexicon_score("台積電營收創新高 法人看好") > 0
    assert lexicon_score("Company misses estimates, shares plunge") < 0


# ------------------------------------------------------------------ forecasts & metrics
def test_forecast_result_probability():
    fc = ForecastResult("x", np.array([101.0]), {q: np.array([100 + (q - 0.5) * 10]) for q in QUANTILES})
    assert abs(fc.prob_above(100.0) - 0.5) < 1e-6
    assert fc.prob_above(90.0) > 0.85 and fc.prob_above(110.0) < 0.15


def test_naive_and_drift_shapes():
    x = SyntheticProvider().prices(parse_symbol("2330"))["close"].to_numpy()
    for f in (NaiveForecaster(), DriftForecaster()):
        r = f.predict([x, x[:300]], 7)
        assert len(r) == 2 and r[0].point.shape == (7,) and set(r[0].quantiles) == set(QUANTILES)
        assert np.all(np.diff(np.stack([r[0].quantiles[q] for q in QUANTILES]), axis=0) >= -1e-9)
    ens = EnsembleForecaster([NaiveForecaster(), DriftForecaster()]).predict([x], 5)[0]
    assert ens.point.shape == (5,)


def test_window_metrics_naive_direction_undefined():
    ctx = np.linspace(90, 100, 50)
    fc = NaiveForecaster().predict([ctx], 3)[0]
    m = window_metrics(np.array([101, 102, 103.0]), fc, ctx)
    assert np.isnan(m["dir_hit"]) and m["mae"] == pytest.approx(2.0)


def test_strategy_stats_costs():
    s = strategy_stats(np.array([0.01, 0.01]), np.array([0.02, -0.01]), 5, cost_bps=50)
    assert s["n_trades"] == 2 and s["strat_total_ret"] == pytest.approx((1.015) * (0.985) - 1)


# ------------------------------------------------------------------ backtest / AB
def test_origins_respect_min_context():
    cfg = BacktestConfig(horizon=5, n_windows=500, step=1, min_context=100)
    o = make_origins(300, cfg)
    assert min(o) >= 100 and max(o) == 295


class _Peek(NaiveForecaster):
    """Records the contexts it receives to prove no future data leaks in."""
    name = "peek"

    def __init__(self):
        self.seen = []

    def predict(self, contexts, horizon):
        self.seen += [c[-1] for c in contexts]
        return super().predict(contexts, horizon)


def test_backtest_no_leakage_and_random_walk_sanity():
    p = SyntheticProvider()
    prices = {t: p.prices(parse_symbol(t))["close"] for t in ["A1", "B2"]}
    peek = _Peek()
    cfg = BacktestConfig(horizon=5, n_windows=30, step=5)
    w, lb = run_backtest(prices, {"naive": NaiveForecaster(), "drift": DriftForecaster(), "peek": peek}, cfg)
    for _, r in w[w["model"] == "peek"].iterrows():  # context ends exactly one bar before the origin
        s = prices[r["ticker"]]
        assert r["last"] == pytest.approx(s.iloc[s.index.get_loc(r["origin"]) - 1])
    assert set(lb["model"]) == {"naive", "drift", "peek"}
    assert lb.loc[lb["model"] == "naive", "skill_vs_naive"].iloc[0] == 0


def test_diebold_mariano_detects_difference():
    rng = np.random.default_rng(1)
    a = np.abs(rng.normal(0, 1, 200)) + 0.5
    b = np.abs(rng.normal(0, 1, 200))
    stat, p = diebold_mariano(a, b, h=1)
    assert stat > 0 and p < 0.01
    assert diebold_mariano(a, a)[1] == 1.0


def test_compare_decision():
    rows = []
    rng = np.random.default_rng(2)
    for t in ["X", "Y", "Z"]:
        for i in range(60):
            e = abs(rng.normal())
            rows.append({"model": "champ", "ticker": t, "origin": i, "mase": e + 0.4, "dir_hit": 1.0})
            rows.append({"model": "chal", "ticker": t, "origin": i, "mase": e, "dir_hit": 1.0})
    r = compare(pd.DataFrame(rows), "champ", "chal", "mase", 1)
    assert r.decision == "promote" and r.improvement_pct > 0


def test_direction_significance():
    assert direction_significance(np.ones(30))["p_value"] < 0.001


def test_dlinear_trains_and_is_calibrated_monotone():
    pytest.importorskip("torch")
    from fintech_agent.forecasting.ml import DLinearForecaster
    p = SyntheticProvider()
    series = [p.prices(parse_symbol(t))["close"].to_numpy() for t in ["A", "B", "C"]]
    m = DLinearForecaster(max_horizon=5, epochs=3, device="cpu").fit(series)
    assert 0.5 <= m.train_info["spread_scale"] <= 4.0
    r = m.predict([s[-200:] for s in series], 5)[0]
    q = np.stack([r.quantiles[k] for k in QUANTILES])
    assert r.point.shape == (5,) and np.all(np.diff(q, axis=0) >= 0)
