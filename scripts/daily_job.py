#!/usr/bin/env python
"""After-close daily job for one market.

  python scripts/daily_job.py --market TW          # ranking refresh, tracking forecasts, scoring, report, audit check
  python scripts/daily_job.py --market US --skip-ranking

Steps
  1. refresh the ranking snapshot (scripts/rank_stocks.py, point-in-time universe for TW) and archive it
  2. log today's champion + random-walk forecasts for the tracking universe (live scorecard)
  3. fill in actual prices for matured predictions
  4. build the daily research report -> reports/{date}_{market}.{json,html,md}
  5. write runs/scorecard.json (backtest + live + ranking live IC), scan the reports for compliance, anchor and
     verify the audit log

On a market holiday (no new index bar today) the job does nothing unless --force. Exit code 0 = all good,
2 = audit chain broken or a report failed the compliance scan, 3 = some step failed (see the log).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fintech_agent.config import get_settings  # noqa: E402
from fintech_agent.data import LiveDataProvider  # noqa: E402
from fintech_agent.data.universe import UNIVERSES  # noqa: E402
from fintech_agent.evaluation.experiments import ExperimentStore  # noqa: E402
from fintech_agent.data.symbols import parse_symbol  # noqa: E402
from fintech_agent.product.audit import AuditLog  # noqa: E402
from fintech_agent.product.compliance import ComplianceGuard, find_violations  # noqa: E402
from fintech_agent.product.report import build_report, render_html, render_markdown  # noqa: E402
from fintech_agent.product.scorecard import backtest_scorecard, live_scorecard, ranking_live, track_universe  # noqa: E402


def step(name):
    print(f"\n== {name}", flush=True)
    return time.time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", choices=["TW", "US"], required=True)
    ap.add_argument("--skip-ranking", action="store_true")
    ap.add_argument("--skip-tracking", action="store_true")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--force", action="store_true", help="run even if the market has no new bar today (holiday)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    s = get_settings()
    cfg = s.get_path(f"product.daily.{args.market}", {}) or {}
    prov = LiveDataProvider(s)
    prov.fm.max_wait_s = 3600
    summary: dict = {"market": args.market}
    failed: list[str] = []
    tz = {"TW": "Asia/Taipei", "US": "America/New_York"}[args.market]
    today = pd.Timestamp.now(tz=tz).date()
    try:
        last_bar = prov.prices(parse_symbol(s.get_path(f"data.market_index.{args.market}")))["close"].index[-1].date()
    except Exception as e:
        last_bar = None
        print(f"index check failed: {e}")
    if last_bar is not None and last_bar < today and not args.force:
        print(f"no {args.market} bar for {today} (last {last_bar}): market holiday or data not yet published — skipping"
              " (use --force to run anyway)")
        return
    close_at = {"TW": (14, 0), "US": (16, 30)}[args.market]          # close + time for the data to settle
    now = pd.Timestamp.now(tz=tz)
    if last_bar == today and (now.hour, now.minute) < close_at and not args.force:
        print(f"{args.market} session still open ({now:%H:%M} {tz}): today's bar is intraday — skipping (use --force)")
        return

    if not args.skip_ranking:
        t0 = step("ranking snapshot")
        env = {**os.environ, "FA_LGBM_JOBS": os.environ.get("FA_LGBM_JOBS", "4"),
               "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
        for h in (5, 20):
            cmd = [sys.executable, str(ROOT / "scripts" / "rank_stocks.py"), *cfg.get("ranking", []), "--horizon", str(h)]
            try:
                r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                                   timeout=int(cfg.get("ranking_timeout_s", 5400)))
                ok, err = r.returncode == 0, r.stderr[-2000:]
            except subprocess.TimeoutExpired:
                ok, err = False, "timed out (data source rate limit?) — the previous ranking snapshot stays in use"
            print(f"  h={h}: {'ok' if ok else 'FAILED'} ({time.time() - t0:.0f}s)")
            if not ok:
                print(err)
                failed.append(f"ranking_h{h}")
            summary[f"ranking_h{h}"] = ok

    if not args.skip_tracking:
        t0 = step("tracking forecasts")
        try:
            uni = cfg.get("tracking_universe")
            tickers = UNIVERSES[uni][1] if uni in UNIVERSES else []
            n = track_universe(prov, tickers, (5, 20), s)
            print(f"  logged {n} forecasts for {len(tickers)} tickers ({time.time() - t0:.0f}s)")
            summary["tracked"] = n
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append("tracking")

    t0 = step("resolve matured predictions")
    try:
        n = ExperimentStore(s).resolve(lambda t: prov.prices(prov.symbol(t))["close"])
        print(f"  resolved {n} ({time.time() - t0:.0f}s)")
        summary["resolved"] = n
    except Exception as e:
        print(f"  FAILED: {e}")
        failed.append("resolve")

    t0 = step("daily report")
    leaks: list[str] = []
    try:
        rep = build_report(prov, args.market, 5, cfg.get("watchlist", []), s)
        full = rep.pop("compliance_full", None)
        out = ROOT / args.out
        out.mkdir(parents=True, exist_ok=True)
        stem = out / f"{rep['date']}_{args.market}"
        md, page = render_markdown(rep), render_html(rep)
        stem.with_suffix(".json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
        stem.with_suffix(".html").write_text(page)
        stem.with_suffix(".md").write_text(md)
        AuditLog(s).record(actor="system:daily_job", mode="research", kind="report", subject=args.market,
                           request={"watchlist": cfg.get("watchlist", [])}, response=rep, compliance=full)
        # last line of defence: the published files themselves must be clean
        leaks = find_violations(md) + ComplianceGuard("research").check({k: v for k, v in rep.items() if k != "compliance"})
        print(f"  → {stem}.{{json,html,md}} ({time.time() - t0:.0f}s); compliance filtered "
              f"{rep['compliance']['removed_sentences']} sentences; scan of published files: "
              f"{'clean' if not leaks else 'LEAKS ' + str(leaks[:5])}")
        summary["report"] = str(stem)
    except Exception as e:
        print(f"  FAILED: {e}")
        failed.append("report")

    t0 = step("scorecard + audit")
    try:
        card = {"backtest": backtest_scorecard(), "live": live_scorecard(s),
                "ranking_live": ranking_live(lambda t: prov.prices(prov.symbol(t))["close"], s)}
        (s.resolve_path("evaluation.runs_dir") / "scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=1, default=str))
        print(f"  scorecard: {card['live'].get('resolved', 0)} resolved live forecasts")
    except Exception as e:
        print(f"  scorecard FAILED: {e}")
        failed.append("scorecard")
    log = AuditLog(s)
    ok, bad = log.verify()
    if ok:
        a = log.anchor()
        print(f"  audit chain intact; anchored {a['count']} rows (head {a['head_hash'][:12]}…)")
    else:
        print(f"  AUDIT CHAIN BROKEN at {bad}")
    summary.update(audit_ok=ok, report_leaks=len(leaks), failed=failed)
    print("\n" + json.dumps(summary, ensure_ascii=False))
    if not ok or leaks:
        sys.exit(2)
    if failed:
        sys.exit(3)


if __name__ == "__main__":
    main()
