"""Decision-model benchmark: states must be past-only and anonymized; metrics must run end to end."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from fintech_agent.data import SyntheticProvider, parse_symbol

_spec = importlib.util.spec_from_file_location(
    "dmb", Path(__file__).resolve().parents[1] / "scripts" / "decision_model_bench.py")
dmb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dmb)


def _grid(prov, tickers, n_origins=12, h=5):
    rows = []
    for t in tickers:
        px = prov.prices(parse_symbol(t))
        for o in range(len(px) - h - 5 * n_origins, len(px) - h, 5):
            rows.append({"ticker": t, "origin": px.index[o], "ret_true": px["close"].iloc[o + h - 1] /
                         px["close"].iloc[o - 1] - 1, "last": px["close"].iloc[o - 1]})
    return pd.DataFrame(rows)


def test_state_uses_only_past_data_and_no_identifiers():
    prov = SyntheticProvider()
    px = prov.prices(parse_symbol("2330"))
    mkt = prov.prices(parse_symbol("^TWII"))["close"]
    t = len(px) - 20
    a = dmb.render_state(dmb.feature_frame(px, mkt, None).iloc[t], "TW")
    future = px.copy()
    future.iloc[t + 1:, :] *= 1.5                    # change everything after the snapshot
    b = dmb.render_state(dmb.feature_frame(future, mkt, None).iloc[t], "TW")
    assert a == b
    assert "2330" not in a and "台積" not in a and str(px.index[t].year) not in a
    assert "RSI(14)" in a and "Market index" in a


def test_updown_metrics_end_to_end():
    prov = SyntheticProvider()
    grid = _grid(prov, ["2330", "2317", "2454", "2881", "2412", "2308", "2303", "2882", "2891", "1301"])
    st = dmb.build_states(prov, grid, "TW", chips=False)
    assert len(st) == len(grid)
    assert np.allclose(st["last_ratio"], 1.0)
    client = dmb.SystemOne("mock")
    res = client.ask_many([(s, dmb.updown_questions()) for s in st["state"]])
    st["dm_up"] = [dmb.noul_p(r["answers"]["up"]) for r in res]
    st["oracle"] = (st["ret_true"] > 0).astype(float) * 0.6 + 0.2
    df = dmb.add_baselines(st)
    lb = dmb.score_models(df, ["coin", "base_rate", "dm_up", "oracle"]).set_index("model")
    assert lb.loc["coin", "brier"] == 0.25 and lb.loc["coin", "bss_coin"] == 0
    assert lb.loc["oracle", "auc"] == 1.0 and lb.loc["oracle", "ic"] > 0.5
    assert lb.loc["oracle", "brier_cal"] < lb.loc["oracle", "brier"]      # calibration learns the 0.2/0.8 map
    assert lb["n_cal"].nunique() == 1


def test_answer_parsing_tolerates_api_variants():
    assert dmb.noul_p({"noul": 0.7}) == 0.7
    assert dmb.noul_p({"noul": True, "probabilities": {"true": 0.8, "false": 0.2}}) == 0.8
    assert dmb.choice_p({"choice": "rise"}, "rise") == 1.0
    assert dmb.score_unit({"score": 2.0}, 5) == 0.5
    assert dmb.score_unit({"probabilities": {"0": 0, "1": 0, "2": 0, "3": 0, "4": 1}}, 5) == 1.0


def test_news_labels_use_the_close_after_publication():
    idx = pd.bdate_range("2026-09-01", periods=6)
    px = pd.Series([100, 101, 102, 103, 104, 105.0], index=idx)
    mkt = pd.Series(100.0, index=idx)
    news = pd.DataFrame([
        {"ticker": "AAPL", "published": "2026-09-02T14:00:00Z", "tz": "UTC", "title": "a"},   # 10:00 ET, intraday
        {"ticker": "AAPL", "published": "2026-09-02T21:30:00Z", "tz": "UTC", "title": "b"},   # after the close
    ])
    lab = dmb.label_news(news, {"AAPL": px}, mkt, "US", now="2026-09-20").set_index("title")
    assert lab.loc["a", "event_day"] == idx[1] and np.isclose(lab.loc["a", "reaction"], 101 / 100 - 1)
    assert lab.loc["b", "event_day"] == idx[2] and np.isclose(lab.loc["b", "reaction"], 102 / 101 - 1)


def test_finmind_news_walks_back_one_day_per_request():
    from fintech_agent.data.finmind import FinMindClient
    calls = []

    class Fake(FinMindClient):
        def fetch(self, dataset, data_id=None, start=None, end=None):
            calls.append((start, end))
            return pd.DataFrame({"date": [start] * 3, "title": ["a", "b", "c"]})
    df = Fake().news("2330", days=5, limit=7)
    assert len(calls) == 3 and all(e is None for _, e in calls)          # stops once the limit is reached
    assert len(df) == 9 and len(Fake().news("2330", days=2)) == 9


def test_news_labels_skip_sessions_that_have_not_closed():
    idx = pd.bdate_range("2026-09-01", periods=6)
    px = pd.Series([100, 101, 102, 103, 104, 105.0], index=idx)
    news = pd.DataFrame([{"ticker": "AAPL", "published": "2026-09-01T21:30:00Z", "tz": "UTC", "title": "a"}])
    assert dmb.label_news(news, {"AAPL": px}, px, "US", now="2026-09-02").empty          # 09-02 bar is intraday
    lab = dmb.label_news(news, {"AAPL": px}, px, "US", now="2026-09-03")              # reaction known, drift not
    assert len(lab) == 1 and ("drift" not in lab or lab["drift"].isna().all())
    assert "drift" in dmb.label_news(news, {"AAPL": px}, px, "US", now="2026-09-04")
