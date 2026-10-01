#!/usr/bin/env python
"""Tiny scheduler for the daily jobs (used by the `scheduler` service in docker-compose).

Runs `scripts/daily_job.py --market M` at `product.daily.M.run_at` (Asia/Taipei) on weekdays. Exchange holidays
are handled by the job itself (it exits without doing anything when the market index has no bar for the day).
For a server with cron you can instead use:  40 15 * * 1-5  cd /app && python scripts/daily_job.py --market TW
"""
from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fintech_agent.config import get_settings  # noqa: E402

TZ = ZoneInfo("Asia/Taipei")


def next_runs(now: datetime) -> list[tuple[datetime, str]]:
    out = []
    for market, cfg in (get_settings().get_path("product.daily", {}) or {}).items():
        hh, mm = map(int, str(cfg.get("run_at", "15:40")).split(":"))
        t = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        # TW runs Mon-Fri; the US job runs Tue-Sat morning Taipei time (after Mon-Fri US sessions)
        days = {0, 1, 2, 3, 4} if market == "TW" else {1, 2, 3, 4, 5}
        while t <= now or t.weekday() not in days:
            t += timedelta(days=1)
        out.append((t, market))
    return sorted(out)


def main() -> None:
    print("scheduler started", flush=True)
    while True:
        when, market = next_runs(datetime.now(TZ))[0]
        wait = (when - datetime.now(TZ)).total_seconds()
        print(f"next: {market} at {when:%Y-%m-%d %H:%M} ({wait / 3600:.1f} h)", flush=True)
        time.sleep(max(1, wait))
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "daily_job.py"), "--market", market], cwd=ROOT).returncode
        print(f"{market} daily job finished with exit code {rc}", flush=True)


if __name__ == "__main__":
    main()
