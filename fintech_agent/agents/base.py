"""Agent base: deterministic evidence + rule score, then LLM narration (bounded score adjustment)."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings, get_settings
from ..data.provider import DataProvider
from ..data.symbols import Symbol
from ..llm.base import LLMClient, Message, NullLLM, extract_json
from . import prompts as P

log = logging.getLogger(__name__)


def stance_of(score: float) -> str:
    if score >= 1.0:
        return "強烈看多"
    if score >= 0.35:
        return "看多"
    if score > -0.35:
        return "中性"
    if score > -1.0:
        return "看空"
    return "強烈看空"


@dataclass
class ClientProfile:
    risk: str = "穩健"                 # 保守 | 穩健 | 積極
    horizon: str = "中期 (1-3個月)"
    capital_twd: float | None = None
    holdings: dict = field(default_factory=dict)   # {"2330": {"shares": 1000, "cost": 900}}
    notes: str = ""

    @property
    def max_position_pct(self) -> float:
        return {"保守": 5.0, "穩健": 10.0, "積極": 20.0}.get(self.risk, 10.0)

    @property
    def atr_stop_mult(self) -> float:
        return {"保守": 1.5, "穩健": 2.0, "積極": 2.5}.get(self.risk, 2.0)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnalysisContext:
    """Data shared by all agents in one analysis run (fetched once)."""
    symbol: Symbol
    prices: pd.DataFrame
    profile: dict
    horizon: int = 5
    client: ClientProfile = field(default_factory=ClientProfile)
    market: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.profile.get("name") or self.symbol.code)

    @property
    def close(self) -> float:
        return float(self.prices["close"].iloc[-1])


@dataclass
class AgentReport:
    agent: str
    title: str
    ticker: str
    score: float
    base_score: float
    confidence: float
    stance: str
    summary: str
    key_points: list[str]
    risks: list[str]
    evidence: dict
    rule_signals: list[tuple[str, float]]
    llm: str
    adjustment_reason: str = ""
    elapsed_s: float = 0.0
    artifacts: dict = field(default_factory=dict, repr=False)   # non-serialisable extras (forecast objects) for UI

    def brief(self) -> dict:
        """Compact form passed to the advisor / chat tools."""
        return {"agent": self.agent, "title": self.title, "score": round(self.score, 2), "stance": self.stance,
                "confidence": round(self.confidence, 2), "summary": self.summary, "key_points": self.key_points[:6],
                "risks": self.risks[:4]}

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("artifacts", None)
        return d


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


class BaseAgent:
    key = "base"
    title = "Agent"
    role = "分析"
    persona = ""

    def __init__(self, provider: DataProvider, llm: LLMClient | None = None, settings: Settings | None = None):
        self.provider = provider
        self.llm = llm or NullLLM()
        self.settings = settings or get_settings()

    # subclasses implement: returns (evidence, base_score, confidence, rule_signals, artifacts)
    def gather(self, ctx: AnalysisContext) -> tuple[dict, float, float, list[tuple[str, float]], dict]:  # pragma: no cover
        raise NotImplementedError

    def fallback_summary(self, ctx: AnalysisContext, evidence: dict, score: float,
                         signals: list[tuple[str, float]]) -> tuple[str, list[str], list[str]]:
        pos = [s for s, v in sorted(signals, key=lambda x: -x[1]) if v > 0][:4]
        neg = [s for s, v in sorted(signals, key=lambda x: x[1]) if v < 0][:3]
        summary = f"{self.title}規則引擎評分 {score:+.2f}（{stance_of(score)}）。"
        if pos:
            summary += "偏多因素：" + "、".join(pos) + "。"
        if neg:
            summary += "偏空因素：" + "、".join(neg) + "。"
        return summary, [s for s, _ in signals if s][:6], neg or ["無明顯反向訊號，但仍須留意系統性風險"]

    def analyze(self, ctx: AnalysisContext) -> AgentReport:
        t0 = time.time()
        evidence, base, conf, signals, artifacts = self.gather(ctx)
        evidence = _jsonable(evidence)
        score, reason = base, ""
        summary, key_points, risks = self.fallback_summary(ctx, evidence, base, signals)
        llm_name = "rule-only"
        if self.llm.enabled:
            prompt = P.NARRATE_TEMPLATE.format(
                name=ctx.name, code=ctx.symbol.code, market=ctx.symbol.market, role=self.role, score=base,
                confidence=conf, signals="\n".join(f"- {s} ({v:+.2f})" for s, v in signals) or "- (無)",
                evidence=json.dumps(evidence, ensure_ascii=False, indent=1)[:12000])
            try:
                resp = self.llm.chat([Message("user", prompt)], system=self.persona)
                data = extract_json(resp.text) or {}
                if data.get("summary"):
                    summary = str(data["summary"])
                    key_points = [str(x) for x in data.get("key_points", key_points)][:6]
                    risks = [str(x) for x in data.get("risks", risks)][:4]
                    adj = float(np.clip(float(data.get("score_adjustment") or 0.0), -0.5, 0.5))
                    score = float(np.clip(base + adj, -2, 2))
                    reason = str(data.get("adjustment_reason") or "") if adj else ""
                    llm_name = f"{self.llm.provider}:{self.llm.model}"
                else:  # never pretend the LLM wrote a narrative it didn't
                    llm_name = f"rule-only（{self.llm.provider}:{self.llm.model} 回覆無法解析）"
            except Exception as e:
                log.warning("%s LLM narration failed, using rule summary: %s", self.key, e)
                llm_name = f"rule-only (LLM error: {type(e).__name__})"
        return AgentReport(self.key, self.title, ctx.symbol.code, round(score, 3), round(base, 3), round(conf, 3),
                           stance_of(score), summary, key_points, risks, evidence, signals, llm_name, reason,
                           round(time.time() - t0, 2), artifacts)
