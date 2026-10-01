"""Append-only, tamper-evident audit log of everything the product tells a user.

Record keeping is what a compliance officer asks for first. Each row stores a keyed hash (HMAC-SHA256) of its
content plus the previous row's hash, so editing or deleting a row in the middle of the log breaks the chain, and
without the key nobody can re-compute a consistent chain after an edit. `anchor()` (run daily by the scheduler)
writes the current head (row count + last hash) to an anchors file; `verify()` checks the chain AND every anchor, so
rows deleted from the end are detected too once an anchor covers them. For production, set `FA_AUDIT_KEY` and copy
the anchors file somewhere the database host cannot rewrite (e-mail, object storage with retention lock).

Writers take an IMMEDIATE transaction around "read previous hash + insert", so several API workers can share one
log without forking the chain. The API exposes no update/delete; exports are JSON Lines.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..config import Settings, get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  actor TEXT NOT NULL,
  mode TEXT NOT NULL,
  kind TEXT NOT NULL,
  subject TEXT,
  request TEXT,
  response TEXT,
  compliance TEXT,
  prev_hash TEXT NOT NULL,
  hash TEXT NOT NULL
);
"""
GENESIS = "0" * 64
ALG = "hmac1"            # rows written before keyed hashing carry alg = 'sha256' and are verified as such


def _dump(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, default=str)


class AuditLog:
    def __init__(self, settings: Settings | None = None, path: Path | str | None = None, key: bytes | str | None = None):
        s = settings or get_settings()
        self.path = Path(path) if path else s.resolve_path("evaluation.runs_dir") / "audit.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.anchors_path = self.path.with_name(self.path.stem + "_anchors.jsonl")
        self.key = self._load_key(key)
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(audit)")}
            if "alg" not in cols:
                c.execute("ALTER TABLE audit ADD COLUMN alg TEXT NOT NULL DEFAULT 'sha256'")

    def _load_key(self, key) -> bytes:
        if key:
            return key.encode() if isinstance(key, str) else key
        env = os.environ.get("FA_AUDIT_KEY")
        if env:
            return env.encode()
        kf = self.path.with_name(".audit_key")             # per-install fallback; production should set FA_AUDIT_KEY
        if not kf.exists():
            kf.write_text(secrets.token_hex(32))
            try:
                kf.chmod(0o600)
            except OSError:
                pass
        return kf.read_text().strip().encode()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    @contextmanager
    def _write(self):
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        try:
            c.execute("BEGIN IMMEDIATE")          # serialises writers across processes
            yield c
            c.execute("COMMIT")
        except Exception:
            c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    def _hash(self, alg: str, prev: str, ts: str, actor: str, mode: str, kind: str, subject: str, request: str,
              response: str, compliance: str) -> str:
        h = hmac.new(self.key, digestmod=hashlib.sha256) if alg == ALG else hashlib.sha256()
        for part in (prev, ts, actor, mode, kind, subject, request, response, compliance):
            h.update(part.encode("utf-8"))
            h.update(b"\x1f")
        return h.hexdigest()

    def record(self, *, actor: str, mode: str, kind: str, subject: str = "", request: Any = None, response: Any = None,
               compliance: Any = None) -> str:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        req, resp, comp = _dump(request), _dump(response), _dump(compliance)
        with self._write() as c:
            row = c.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
            prev = row[0] if row else GENESIS
            h = self._hash(ALG, prev, ts, actor, mode, kind, subject or "", req, resp, comp)
            c.execute("INSERT INTO audit (ts, actor, mode, kind, subject, request, response, compliance, prev_hash, hash, alg) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?)", (ts, actor, mode, kind, subject or "", req, resp, comp, prev, h, ALG))
        return h

    def rows(self, limit: int = 100, actor: str | None = None) -> list[dict]:
        q = "SELECT id, ts, actor, mode, kind, subject, compliance, hash FROM audit"
        args: tuple = ()
        if actor:
            q += " WHERE actor = ?"
            args = (actor,)
        q += " ORDER BY id DESC LIMIT ?"
        with self._conn() as c:
            cur = c.execute(q, args + (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def export(self, actor: str | None = None) -> Iterator[str]:
        with self._conn() as c:
            cur = (c.execute("SELECT * FROM audit WHERE actor = ? ORDER BY id", (actor,)) if actor
                   else c.execute("SELECT * FROM audit ORDER BY id"))
            cols = [d[0] for d in cur.description]
            for r in cur:
                yield _dump(dict(zip(cols, r)))

    # ------------------------------------------------------------------ integrity
    def head(self) -> dict:
        with self._conn() as c:
            n, = c.execute("SELECT COUNT(*) FROM audit").fetchone()
            row = c.execute("SELECT id, hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        return {"count": int(n), "head_id": row[0] if row else 0, "head_hash": row[1] if row else GENESIS}

    def anchor(self) -> dict:
        """Append the current head to the anchors file (daily). Later deletions of anchored rows become detectable."""
        a = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **self.head()}
        with open(self.anchors_path, "a") as f:
            f.write(json.dumps(a) + "\n")
        return a

    def anchors(self) -> list[dict]:
        if not self.anchors_path.exists():
            return []
        return [json.loads(x) for x in self.anchors_path.read_text().splitlines() if x.strip()]

    def verify(self) -> tuple[bool, int | None]:
        """(True, None) if the chain and every anchor are intact, else (False, id of the first bad row / anchor head)."""
        prev, keyed, seen = GENESIS, False, {}
        with self._conn() as c:
            for n, r in enumerate(c.execute("SELECT id, ts, actor, mode, kind, subject, request, response, compliance, "
                                            "prev_hash, hash, alg FROM audit ORDER BY id"), start=1):
                rid, ts, actor, mode, kind, subject, req, resp, comp, prev_hash, h, alg = r
                if alg == ALG:
                    keyed = True
                elif keyed:                       # an unkeyed row after keyed ones = a rewritten tail
                    return False, rid
                if prev_hash != prev or self._hash(alg, prev, ts, actor, mode, kind, subject, req, resp, comp) != h:
                    return False, rid
                prev = h
                seen[rid] = (n, h)
        for a in self.anchors():
            if a["head_id"] == 0:
                continue
            got = seen.get(a["head_id"])
            if got is None or got[1] != a["head_hash"] or got[0] != a["count"]:
                return False, a["head_id"]
        return True, None
