"""Ticker normalisation for Taiwan (TWSE/TPEx) and US markets."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_TW_CODE = re.compile(r"^\d{4,6}[A-Z]?$")          # 2330, 00878, 2881A
_TW_FULL = re.compile(r"^(\d{4,6}[A-Z]?)\.(TW|TWO)$", re.I)

TW_INDEX = {"^TWII", "^TWOII"}


@dataclass(frozen=True)
class Symbol:
    raw: str
    market: str                 # "TW" | "US"
    code: str                   # FinMind / exchange code: "2330", "AAPL"
    yf_candidates: tuple[str, ...] = field(default_factory=tuple)
    is_index: bool = False

    @property
    def yf_ticker(self) -> str:
        return self.yf_candidates[0]

    @property
    def currency(self) -> str:
        return "TWD" if self.market == "TW" else "USD"

    def __str__(self) -> str:  # pragma: no cover
        return self.code if self.market == "US" else f"{self.code}.{self.market}"


def parse_symbol(text: str, default_market: str | None = None) -> Symbol:
    """Parse user input into a Symbol.

    >>> parse_symbol("2330").yf_candidates
    ('2330.TW', '2330.TWO')
    >>> parse_symbol("aapl").market
    'US'
    """
    raw = text.strip()
    t = raw.upper()
    if t.startswith("^"):
        market = "TW" if t in TW_INDEX else "US"
        return Symbol(raw, market, t, (t,), is_index=True)
    m = _TW_FULL.match(t)
    if m:
        code, suffix = m.group(1), m.group(2).upper()
        other = "TWO" if suffix == "TW" else "TW"
        return Symbol(raw, "TW", code, (f"{code}.{suffix}", f"{code}.{other}"))
    if _TW_CODE.match(t) and default_market != "US":
        return Symbol(raw, "TW", t, (f"{t}.TW", f"{t}.TWO"))
    # US tickers: BRK.B -> BRK-B for Yahoo
    yf = t.replace(".", "-")
    return Symbol(raw, "US", t, (yf,))
