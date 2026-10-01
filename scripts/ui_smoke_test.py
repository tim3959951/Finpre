#!/usr/bin/env python
"""Headless smoke test of the Streamlit app (no browser): runs one full analysis and one chat turn.

  python scripts/ui_smoke_test.py 2330
  python scripts/ui_smoke_test.py 2330 --key fp_...      # signed-in (FA_UI_AUTH=apikey) customer view
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "fintech_agent" / "ui" / "app.py")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ticker = args[0] if args else "2330"
    key = sys.argv[sys.argv.index("--key") + 1] if "--key" in sys.argv else None
    if key:
        os.environ["FA_UI_AUTH"] = "apikey"
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if key:
        assert not at.tabs, "content must not render before sign-in"
        at.text_input[0].set_value("fp_wrong")
        at.button[0].click().run()
        assert any("無效" in e.value for e in at.error), "a wrong key must be rejected"
        at.text_input[0].set_value(key)
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert at.tabs, "sign-in with the given key failed"
        print("signed in:", [m.value for m in at.sidebar.markdown][:2])
        assert "模型實驗室" not in [t.label for t in at.tabs], "operators-only tab visible to a customer"
    next(t for t in at.text_input if t.label == "股票代號").set_value(ticker)
    t0 = time.time()
    next(b for b in at.button if b.label == "執行完整分析").click().run()
    assert not at.exception, [e.value for e in at.exception]
    print(f"analysis OK in {time.time() - t0:.1f}s")
    for m in at.metric:
        print(f"  {m.label}: {m.value}  {m.delta or ''}")
    print("  sections:", [s.value for s in at.subheader])
    t0 = time.time()
    at.chat_input[0].set_value(f"{ticker} 可以買嗎？").run()
    assert not at.exception, [e.value for e in at.exception]
    print(f"chat OK in {time.time() - t0:.1f}s ({len(at.chat_message)} messages)")
    t0 = time.time()
    next(b for b in at.button if b.label == "計算組合風險").click().run()
    assert not at.exception, [e.value for e in at.exception]
    print(f"risk radar OK in {time.time() - t0:.1f}s:", [f"{m.label}={m.value}" for m in at.metric if "VaR" in m.label])
    t0 = time.time()
    next(b for b in at.button if b.label == "產生報告").click().run()
    assert not at.exception, [e.value for e in at.exception]
    print(f"daily report OK in {time.time() - t0:.1f}s")
    print("  tabs:", [t.label for t in at.tabs])


if __name__ == "__main__":
    main()
