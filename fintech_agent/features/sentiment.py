"""Headline sentiment. Lexicon fallback (zh-TW + EN); the fundamental agent upgrades to LLM scoring when available."""
from __future__ import annotations

import re

POS = ["創新高", "成長", "看好", "調升", "上修", "買超", "強勁", "優於預期", "利多", "擴產", "大單", "漲停", "突破", "回溫",
       "獲利", "配息", "增加", "年增", "旺季", "受惠", "樂觀",
       "beat", "beats", "surge", "record", "upgrade", "raises", "strong", "growth", "bullish", "outperform",
       "rally", "jumps", "soars", "tops", "expands", "buyback", "profit"]
NEG = ["衰退", "下修", "調降", "賣超", "利空", "虧損", "跌停", "疲弱", "低於預期", "減產", "砍單", "警訊", "裁員", "下滑",
       "年減", "淡季", "悲觀", "違約", "調查", "罰", "關稅",
       "miss", "misses", "plunge", "downgrade", "cuts", "weak", "loss", "bearish", "underperform", "falls",
       "slumps", "lawsuit", "probe", "recall", "tariff", "layoffs", "warning", "decline"]
_POS = [re.compile(re.escape(w), re.I) if not w.isascii() else re.compile(rf"\b{w}\b", re.I) for w in POS]
_NEG = [re.compile(re.escape(w), re.I) if not w.isascii() else re.compile(rf"\b{w}\b", re.I) for w in NEG]


def lexicon_score(text: str) -> float:
    """Return sentiment in [-1, 1]."""
    if not text:
        return 0.0
    p = sum(1 for r in _POS if r.search(text))
    n = sum(1 for r in _NEG if r.search(text))
    return 0.0 if p + n == 0 else (p - n) / (p + n)


def score_headlines(items: list[dict]) -> dict:
    scored = [(it, lexicon_score(f"{it.get('title', '')} {it.get('summary', '')}")) for it in items]
    vals = [s for _, s in scored]
    return {
        "method": "lexicon",
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 3) if vals else 0.0,
        "positive": sum(v > 0 for v in vals),
        "negative": sum(v < 0 for v in vals),
        "items": [{"title": it.get("title"), "date": it.get("date"), "score": round(s, 2)} for it, s in scored],
    }
