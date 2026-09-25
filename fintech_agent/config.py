"""Settings loader: YAML + .env with ${VAR} / ${VAR:-default} expansion."""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            return os.environ.get(m.group(1)) or (m.group(2) or "")
        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class Settings(dict):
    """dict with dotted-path access: settings.get_path('forecasting.champion')."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve_path(self, path: str) -> Path:
        """Resolve a directory setting relative to the project root and create it."""
        p = Path(self.get_path(path))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=4)
def get_settings(config_path: str | None = None) -> Settings:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
    path = Path(config_path or os.environ.get("FA_CONFIG", PROJECT_ROOT / "config" / "settings.yaml"))
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Settings(_expand(raw))
