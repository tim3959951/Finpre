#!/usr/bin/env python
"""Run the full multi-agent analysis from the command line.

  python scripts/analyze.py 2330
  python scripts/analyze.py NVDA --horizon 20 --risk 積極 --provider ollama --model qwen3:8b
  python scripts/analyze.py 2330 --provider none        # rule-only mode, no LLM needed
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fintech_agent.agents import ClientProfile, InvestmentAdvisor  # noqa: E402
from fintech_agent.config import get_settings  # noqa: E402
from fintech_agent.data import LiveDataProvider  # noqa: E402
from fintech_agent.llm import get_llm  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--risk", default="穩健", choices=["保守", "穩健", "積極"])
    ap.add_argument("--provider", default=None, help="anthropic | openai | ollama | none")
    ap.add_argument("--model", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    s = get_settings()
    llm = get_llm("advisor", s, provider=args.provider, model=args.model) if args.provider else None
    adv = InvestmentAdvisor(LiveDataProvider(s), s, llm=llm)
    print(f"LLM: {adv.llm.provider}:{adv.llm.model}" + ("" if adv.llm.enabled else f" ({getattr(adv.llm, 'reason', '')})"))
    res = adv.analyze(args.ticker, args.horizon, ClientProfile(risk=args.risk), on_event=lambda m: print("·", m))
    if args.json:
        print(json.dumps(res.brief(), ensure_ascii=False, indent=1, default=str))
        return
    d = res.decision
    print(f"\n=== {res.ctx.name} ({res.ctx.symbol.code}) 收盤 {res.ctx.close:.2f} ===")
    for r in res.reports.values():
        print(f"\n[{r.title}] {r.stance} {r.score:+.2f} (信心 {r.confidence:.0%}, {r.llm})\n  {r.summary}")
        for k in r.key_points:
            print(f"   - {k}")
    print(f"\n[首席投資顧問] {d['action']} · 信心 {d['conviction']}% · 部位 {d['position_pct']}%")
    print(f"  進場 {d['entry_zone']}  停損 {d['stop_loss']}  停利 {d['take_profit']}")
    print(f"  {d.get('client_message', '')}")
    print(f"\n{d['disclaimer']}")


if __name__ == "__main__":
    main()
