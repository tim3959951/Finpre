#!/usr/bin/env python
"""Manage API customers (tenants), plans and keys.

  python scripts/manage_tenants.py create --name "示範投顧" --plan enterprise --licensed --mode advisor
  python scripts/manage_tenants.py create --name "王小明" --plan research
  python scripts/manage_tenants.py list
  python scripts/manage_tenants.py set-plan t_1234abcd --plan pro
  python scripts/manage_tenants.py usage t_1234abcd
  python scripts/manage_tenants.py set-plan t_1234abcd --plan enterprise --no-licensed     # revoke a licence
  python scripts/manage_tenants.py rotate-key t_1234abcd       # new key, old one stops working
  python scripts/manage_tenants.py deactivate t_1234abcd
  python scripts/manage_tenants.py audit-verify
  python scripts/manage_tenants.py audit-anchor                # record the log head (daily job does this)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fintech_agent.product.audit import AuditLog  # noqa: E402
from fintech_agent.product.plans import PLANS, TenantStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--plan", choices=sorted(PLANS), default="free")
    c.add_argument("--licensed", action="store_true", help="licensed investment-consulting firm (required for advisor mode)")
    c.add_argument("--mode", choices=["research", "advisor"], default="research")
    sub.add_parser("list")
    sp = sub.add_parser("set-plan")
    sp.add_argument("tenant_id")
    sp.add_argument("--plan", choices=sorted(PLANS), required=True)
    sp.add_argument("--licensed", action=argparse.BooleanOptionalAction, default=None)
    sp.add_argument("--mode", choices=["research", "advisor"])
    u = sub.add_parser("usage")
    u.add_argument("tenant_id")
    u.add_argument("--month")
    d = sub.add_parser("deactivate")
    d.add_argument("tenant_id")
    rk = sub.add_parser("rotate-key")
    rk.add_argument("tenant_id")
    sub.add_parser("audit-verify")
    sub.add_parser("audit-anchor")
    args = ap.parse_args()
    store = TenantStore()
    if args.cmd == "create":
        t, key = store.create(args.name, args.plan, args.licensed, args.mode)
        print(json.dumps(t.to_dict(), ensure_ascii=False, indent=1))
        print(f"\nAPI key（只顯示這一次，請妥善保存）：\n{key}")
    elif args.cmd == "list":
        print(json.dumps(store.list(), ensure_ascii=False, indent=1))
    elif args.cmd == "set-plan":
        print(json.dumps(store.set_plan(args.tenant_id, args.plan, args.licensed, args.mode).to_dict(), ensure_ascii=False, indent=1))
    elif args.cmd == "usage":
        print(json.dumps(store.usage(args.tenant_id, args.month), ensure_ascii=False, indent=1))
    elif args.cmd == "deactivate":
        store.deactivate(args.tenant_id)
        print("deactivated")
    elif args.cmd == "rotate-key":
        print(f"新的 API key（只顯示這一次）：\n{store.rotate_key(args.tenant_id)}")
    elif args.cmd == "audit-anchor":
        print(json.dumps(AuditLog().anchor()))
    elif args.cmd == "audit-verify":
        ok, bad = AuditLog().verify()
        print("audit chain intact" if ok else f"AUDIT CHAIN BROKEN at row {bad}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
