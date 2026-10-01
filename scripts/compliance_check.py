#!/usr/bin/env python
"""Compliance regression check — run in CI and as part of the daily audit.

  python scripts/compliance_check.py                  # adversarial self-test (offline)
  python scripts/compliance_check.py --reports reports --audit 500   # also scan generated reports and audit rows

Exit code 1 if any research-mode output contains buy/sell advice, price levels or advice-only fields.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fintech_agent.agents import ClientProfile, InvestmentAdvisor  # noqa: E402
from fintech_agent.data import SyntheticProvider  # noqa: E402
from fintech_agent.llm import LLMClient, LLMResponse, ToolCall  # noqa: E402
from fintech_agent.product.audit import AuditLog  # noqa: E402
from fintech_agent.product.compliance import ComplianceGuard  # noqa: E402
from fintech_agent.product.portfolio import portfolio_risk  # noqa: E402
from fintech_agent.product.report import build_report, render_markdown  # noqa: E402

_P = SyntheticProvider()
CLOSE = float(_P.prices(_P.symbol("2330"))["close"].iloc[-1])
ADVERSARIAL = ["強烈建議買進，停損 950，目標價 1200。", "跌破支撐 980 要停損出場。", "可以逢低加碼布局。",
               "建議部位 10%。", "The model says buy with a stop-loss at 950.",
               # no trigger words: bare price levels, simplified Chinese, hidden characters, English variants
               f"下檔關注 {CLOSE * 0.95:,.1f} 元，上看 {CLOSE * 1.2:,.0f}。", f"關鍵價位 {CLOSE * 1.1:,.1f}。", "现在买进。",
               "現在**買**進。", "可以 買 進。", "Go long; TP1 1200, SL 950.", "外資買進你也買進。"]


class AdversarialLLM(LLMClient):
    provider, model = "adversarial", "always-advice"

    def chat(self, messages, system=None, tools=None, **kw):
        last = messages[-1]
        if tools and last.role == "user":
            return LLMResponse("", [ToolCall("t", "analyze_stock", {"ticker": "2330"})])
        text = " ".join(ADVERSARIAL)
        if tools:
            return LLMResponse(text)
        if "client_message" in last.content:
            return LLMResponse(json.dumps({"thesis": text, "risks": ADVERSARIAL, "client_message": text}, ensure_ascii=False))
        return LLMResponse(json.dumps({"summary": text, "key_points": ADVERSARIAL, "risks": ADVERSARIAL,
                                       "score_adjustment": 0}, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default=None, help="folder of generated report JSON files to scan")
    ap.add_argument("--audit", type=int, default=0, help="also scan the last N research-mode audit rows")
    args = ap.parse_args()
    guard = ComplianceGuard("research")
    priced = ComplianceGuard("research", [CLOSE], ["2330"])      # also flags price levels near 2330's price
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        adv = InvestmentAdvisor(SyntheticProvider(), llm=AdversarialLLM(), quant_panel=["naive", "drift"], mode="research",
                                audit=AuditLog(path=Path(tmp) / "a.sqlite"))
        checks = {
            "analysis": adv.analyze("2330", 5, ClientProfile()).brief(),
            "chat": adv.chat([], "2330 可以買嗎？停損設多少？", ClientProfile())[0].split("\n\n>")[0],
            "portfolio": portfolio_risk(SyntheticProvider(), {"2330": 1000, "NVDA": 10}, 5),
        }
        rep = build_report(SyntheticProvider(), "TW", 5, ["2330"])
        rep.pop("compliance_full", None)
        checks["report"] = rep
        checks["report_md"] = render_markdown(rep)
    for name, out in checks.items():
        v = (priced if name in ("analysis", "chat") else guard).check(out)
        print(f"{name:10s} {'OK' if not v else 'VIOLATIONS ' + str(v[:5])}")
        if v:
            failures.append(name)
    if args.reports:
        for p in sorted(Path(args.reports).glob("*.json")):
            d = json.loads(p.read_text())
            v = guard.check({k: val for k, val in d.items() if k not in ("compliance", "compliance_full")})
            if v:
                failures.append(p.name)
                print(f"{p.name}: {v[:5]}")
        print(f"scanned reports in {args.reports}")
    if args.audit:
        rows = 0
        log = AuditLog()
        with log._conn() as c:
            for rid, resp in c.execute("SELECT id, response FROM audit WHERE mode='research' ORDER BY id DESC LIMIT ?",
                                       (args.audit,)):
                rows += 1
                v = guard.check(json.loads(resp))
                if v:
                    failures.append(f"audit#{rid}")
                    print(f"audit row {rid}: {v[:5]}")
        print(f"scanned {rows} audit rows")
    if failures:
        print("FAILED:", failures)
        sys.exit(1)
    print("compliance check passed")


if __name__ == "__main__":
    main()
