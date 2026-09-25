#!/usr/bin/env python
"""Headless smoke test of the Streamlit app (no browser): runs one full analysis and one chat turn.

  python scripts/ui_smoke_test.py 2330
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "fintech_agent" / "ui" / "app.py")


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "2330"
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
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


if __name__ == "__main__":
    main()
