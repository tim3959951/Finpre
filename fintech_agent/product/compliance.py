"""Compliance modes: what the system may say to whom.

Taiwan's Securities Investment Trust and Consulting Act treats paid, individual-stock buy/sell advice to the public
as securities investment consulting (a licensed business); SITCA guidance also flags apps that give buy/sell price
levels, support/resistance or stop-loss/take-profit levels for individual stocks. So the product runs in one of two
modes, enforced in code at the output boundary (not only in prompts):

  research  (default, for the general public): signals, scores, probabilities, return ranges in %, model track
            records. No buy/sell/add/reduce wording, no position sizes, no entry/stop/target/support/resistance
            price levels — and no stock prices other than the current one.
  advisor   (licensed investment-consulting firms only, by contract): the full decision incl. action and levels.

`ComplianceGuard.apply(obj)` deep-cleans any JSON-like output: drops advice-only keys and removes every sentence that
contains advice wording or a price level. What was removed is returned for the audit log only — it is never sent
back to the user (`ComplianceReport.public()`).

Matching runs on a normalised copy of each sentence (NFKC, HTML entities, zero-width and markdown characters,
spaces between CJK characters, common simplified characters) so that "買 進", "ＢＵＹ", "买进" or "**買**進" are
caught. Institutional-flow facts ("外資連 3 日買進", "投信加碼") are allowed only when the clause names who traded
and carries no advice cue ("你", "跟著", "建議", "可以", …).
"""
from __future__ import annotations

import html as _html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable

MODES = ("research", "advisor")
MODE_LABEL = {"research": "研究模式", "advisor": "顧問模式（持牌機構）"}

RESEARCH_DISCLAIMER = ("本服務為投資研究工具，內容由統計模型與 AI 根據公開資料自動產生，僅呈現訊號、機率與歷史回測結果，"
                       "不提供個股買賣建議、買賣價位或持股配置建議，亦不構成任何要約；投資有風險，過去績效不代表未來表現，"
                       "請自行判斷並承擔風險。")

# Keys that only exist to carry advice. Removed wholesale in research mode (at any depth).
ADVICE_KEYS = {"action", "position_pct", "entry_zone", "stop_loss", "take_profit", "advisor_override", "draft",
               "support", "resistance", "monitoring", "model_band_p10_p90", "effective_weights", "dissent",
               "price_h",            # a model's forecast price reads like a price target; research shows % only
               "analyst_target_mean", "analyst_target_high", "analyst_target_low", "target_mean_price",
               "analyst_rating", "recommendation_key", "recommendation"}
# Third-party news headlines are shown as a labelled quotation, not scrubbed (silently deleting "外資調升評等" would
# bias the news view). Only at `...evidence.headlines`, only list items of {title, date, score}; LLM text written
# about them is still scrubbed.
QUOTED_KEYS = {"headlines", "news"}
QUOTE_PARENTS = {"evidence"}
QUOTE_FIELDS = ("title", "date", "score", "source")
QUOTE_LABEL = "以下為第三方新聞標題原文引述，不代表本服務觀點，亦非買賣建議。"

# ------------------------------------------------------------------------------------------ normalisation
_S2T_PAIRS = ("买買 卖賣 进進 码碼 减減 损損 标標 价價 撑撐 压壓 议議 仓倉 场場 资資 应應 该該 设設 点點 时時 机機 线線 区區 "
              "际際 观觀 势勢 涨漲 险險 实實 战戰 稳穩 获獲 结結 续續 满滿 单單 购購 盘盤 车車 层層 档檔 盈盈 亏虧 轻輕")
_S2T = str.maketrans({p[0]: p[1] for p in _S2T_PAIRS.split()})
_ZERO_WIDTH = re.compile(r"[​-‏ - ⁠-⁯﻿­]")
_MARKDOWN = re.compile(r"[*`~#>|]")
_CJK_GAP = re.compile(r"(?<=[㐀-鿿])[\s　]+(?=[㐀-鿿])")
_BRANDS = re.compile(r"\bBest\s+Buy\b", re.IGNORECASE)        # company names that contain a trigger word


def normalize(text: str) -> str:
    t = _html.unescape(text or "")
    t = unicodedata.normalize("NFKC", t)
    t = _ZERO_WIDTH.sub("", t)
    t = _MARKDOWN.sub("", t).replace("_", " ")
    t = _CJK_GAP.sub("", t)
    t = _BRANDS.sub("BestBuyCo", t)
    return t.translate(_S2T)


# ------------------------------------------------------------------------------------------ advice wording
# Action verbs: advice when addressed to the reader, a fact when someone else did it ("外資加碼", "投信賣出").
_ACTIONS = r"買進|買入|賣出|加碼|減碼|布局|佈局|進場|出場|做多|做空|放空|逢低|逢高|追高|抄底|承接|停損|停利|止損|止盈|" \
           r"加倉|減倉|建倉|清倉|增持|減持"
# Always advice, whoever the subject is (price levels, sizing, explicit recommendations).
_ALWAYS = [
    r"目標價", r"(?:第[一二三]|短線|中線|長線|上檔|下檔)目標", r"目標\s*[:：]\s*\d", r"目標區",
    r"(?:停損|停利|止損|止盈|防守|進場|出場)(?:價|點|位|區)", r"防守價", r"買點", r"賣點", r"低接", r"上車", r"出清",
    r"獲利了結", r"續抱", r"抱牢", r"關鍵價", r"價位\s*[:：]?\s*\$?\d", r"(?:進場|出場|買賣|操作)時機",
    r"(?:上檔|下檔|短線|技術)?(?:壓力|支撐)(?:位|區|價|帶|線|關卡)", r"(?:上檔|下檔)(?:壓力|支撐)",
    r"(?:跌破|突破|守住|站穩|回測|挑戰|測試)\S{0,4}(?:壓力|支撐)", r"(?:壓力|支撐)\s*(?:在|於|約|為|是|落在)?\s*\$?\d",
    r"(?:上看|下看)\s*\$?\d[\d,.]*(?![\d,.]|\s*(?:億|萬|兆|%))",
    r"建議部位", r"部位\s*[:：]?\s*\d+(?:\.\d+)?\s*%", r"(?:建議|配置|調整|提高|降低)\S{0,4}持股比例",
    r"(?:建議|可|宜)\S{0,2}配置", r"(?:配置|投入)\s*\d+(?:\.\d+)?\s*%", r"資金\s*\d+(?:\.\d+)?\s*%",
    r"建議(?:您|你)?[^\s，。,；;！？!?、]{0,2}(?:買|賣|持有|觀望|避開|減|加|布局|佈局|分批|逢|續抱|出|進|停|上車|介入|承接|低接|空)",
    r"(?:可以|可(?!能|望|觀|見|知|說|以看)|應該|不妨|(?<!便)宜|請|最好|務必)[^\s，。,；;！？!?、]{0,2}"
    r"(?:買|賣|加碼|減碼|布局|佈局|進場|出場|介入|承接|低接|逢|分批|停損|停利|避開|出清|續抱|空)",
    # English
    r"\bbuy(?:s|ing)?\b(?![- ]?(?:back|side))", r"\bsell(?:s|ing)?\b(?![- ]?(?:off|side))", r"\bgo(?:ing)? (?:long|short)\b",
    r"\bshort (?:it|the stock|this|shares)\b", r"\baccumulat\w*", r"\b(?:over|under)weight\b", r"\btrim(?:ming)?\b",
    r"\bexit (?:at|now|the position|positions)\b", r"\bentry (?:point|price|zone|level|at)\b", r"\btarget (?:price|of|at|to)\b",
    r"\bprice target\b", r"\bstop[- ]?loss\b", r"\bstop (?:at|out)\b", r"\btake[- ]?profit\b", r"\bTP\s*\d?\s*[:=@]?\s*\$?\d",
    r"\bSL\s*[:=@]?\s*\$?\d", r"\bstrong ?buy\b", r"\b(?:support|resistance) (?:at|level|of|around|near|zone)\b",
    r"\bposition siz\w+", r"\ballocate\b", r"\bload up\b",
]
ALWAYS = re.compile("|".join(_ALWAYS), re.IGNORECASE)
ACTION = re.compile(_ACTIONS)
FORBIDDEN = re.compile("|".join(_ALWAYS + [_ACTIONS]), re.IGNORECASE)     # kept for callers that only need a quick test

FACT_SUBJECT = re.compile(r"外資|投信|自營商|三大法人|法人|內部人|董監|大股東|股東|主力|散戶|大戶|機構|基金|壽險|國安基金|"
                          r"央行|聯準會|Fed|官員|公司|企業|集團|融資|融券|資金|買盤|賣壓|多方|空方|市場|海外|全球|產能|產業|"
                          r"供應鏈|期貨|政府|投資人們|外國|陸資|港資", re.IGNORECASE)
ADVICE_CUE = re.compile(r"建議|可以|可(?!能|望)|應該|應(?!用|收|付|變|對|徵)|不妨|(?<!便)宜|請|你|您|就該|跟著|跟進|一起|快(?!速)|吧|"
                        r"值得|考慮|適合|趁|不如|最好|務必|別錯過")
CLAUSE_SPLIT = re.compile(r"[。，,；;！？!?\n、：:]")


def _clause(text: str, start: int, end: int) -> tuple[str, str]:
    """(clause text before the match, rest of the clause after it)."""
    head = text[:start]
    cut = max((m.end() for m in CLAUSE_SPLIT.finditer(head)), default=0)
    tail = text[end:]
    m = CLAUSE_SPLIT.search(tail)
    return head[cut:], tail[:m.start()] if m else tail


def _advice_words(t: str) -> list[str]:
    out = [m.group(0) for m in ALWAYS.finditer(t)]
    for m in ACTION.finditer(t):
        before, after = _clause(t, m.start(), m.end())
        if FACT_SUBJECT.search(before) and not ADVICE_CUE.search(before + after):
            continue                      # "外資連 3 日買進" — a fact about someone else's trades
        out.append(m.group(0))
    return out


# ------------------------------------------------------------------------------------------ price levels
_NUM = re.compile(r"(?P<pre>(?:NT|US|HK)?\$|＄)?\s?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)")
_NOT_PRICE_UNIT = re.compile(r"\s*(?:%|％|bps?\b|基點|億|萬|千|百萬|兆|張|股|口|倍|x\b|個|日|天|週|周|月|年|季|檔|名|次|筆|期|"
                             r"家|人|項|點|分|秒|歲|位|成|折|°|[KMB]\b|bn\b|million|billion|trillion)", re.IGNORECASE)
_PRICE_UNIT = re.compile(r"\s*(?:元|塊|美元|美金|台幣|新台幣|dollars?\b|USD\b|NTD\b|TWD\b)", re.IGNORECASE)
_DATE_OR_TIME = re.compile(r"(?:[-/.]\d{1,2}(?:[-/.]\d{1,2})?\b|:\d{2}\b)")
_INDICATOR_BEFORE = re.compile(r"(?:RSI|KD|K值|D值|MACD|DIF|DEA|ADX|DMI|ATR|CCI|OBV|威廉|乖離|IC|t\s*=|n\s*=|分數|信心|樣本|"
                               r"排名|S&P|標普|Nasdaq|那斯達克|道瓊|Dow|羅素|Russell|Top|TOP)\s*[:：=]?\s*$")
_AMOUNT_BEFORE = re.compile(r"(?:EPS|每股盈餘|每股淨值|淨值|股利|股息|配息|現金股利|營收|淨利|獲利|毛利|市值|成交(?:金額|值|量)|"
                            r"資本額|股本|現金流|負債|資產|薪|成本|費用|手續費|稅|保證金|餘額|合約|訂單|募資|賠償|罰)"
                            r"[^。，,；;！？!?]{0,10}$", re.IGNORECASE)


def price_levels(text: str, ref_prices: Iterable[float] = (), ignore: Iterable[str] = ()) -> list[str]:
    """Numbers in `text` that read as a stock price other than the current one.

    A number is a price when it carries a currency (`$`, 元, 美元 …) and is not a fundamentals amount (EPS 12 元,
    營收 1,000 億元), or — when the current prices are known — when it lies within ±50% of one of them. Percentages,
    counts, dates, indicator values, ticker codes and the current price itself (±0.5%) are allowed."""
    refs = [float(r) for r in ref_prices if r and r > 0]
    ign = {str(x) for x in ignore}
    out = []
    for m in _NUM.finditer(text):
        raw, pre = m.group("num"), m.group("pre")
        after, before = text[m.end():], text[:m.start()]
        if raw in ign or _NOT_PRICE_UNIT.match(after):
            continue
        prev = before[-1:]
        if not pre and (re.search(r"[A-Za-z_]$", before) or prev in ".:/" or
                        (prev in "+-−±" and not before[-2:-1].isdigit())):
            continue                                         # MA20, v2.5, -0.35 (a signed score), 09/30
        if _DATE_OR_TIME.match(after) and (len(raw) == 4 or float(raw.replace(",", "")) <= 31):
            continue                                         # 2026-09-30, 9/30, 13:30
        if re.search(r"[（(\[]\s*$", before[-2:]) and re.match(r"\s*[)）\]]", after) and "." not in raw:
            continue                                         # 台積電（2330）
        val = float(raw.replace(",", ""))
        if "," not in raw and "." not in raw and 1990 <= val <= 2100 and not pre:
            continue                                         # a year
        if any(abs(val / r - 1) <= 0.005 for r in refs):
            continue                                         # the current price may be quoted
        if pre or _PRICE_UNIT.match(after):
            if not _AMOUNT_BEFORE.search(before[-16:]):
                out.append(m.group(0).strip())
        elif any(0.5 * r <= val <= 1.5 * r for r in refs) and not _INDICATOR_BEFORE.search(before[-10:]):
            out.append(m.group(0).strip())
    return out


# ------------------------------------------------------------------------------------------ public helpers
def find_violations(text: str, ref_prices: Iterable[float] = (), ignore: Iterable[str] = ()) -> list[str]:
    t = normalize(text or "")
    return _advice_words(t) + [f"price:{p}" for p in price_levels(t, ref_prices, ignore)]


_SPLIT = re.compile(r"(?<=[。！？!?；;\n])|(?<=\.)(?=\s)")


def sentences(text: str) -> list[str]:
    return [s for s in _SPLIT.split(text or "") if s]


def scrub_text(text: str, ref_prices: Iterable[float] = (), ignore: Iterable[str] = ()) -> tuple[str, list[str]]:
    """Drop every sentence that contains advice wording or a price level. Returns (clean text, removed sentences)."""
    if not text:
        return text, []
    refs = list(ref_prices)
    kept, removed = [], []
    for s in sentences(text):
        (removed if find_violations(s, refs, ignore) else kept).append(s)
    return "".join(kept).strip(), [r.strip() for r in removed if r.strip()]


def signal_label(score: float) -> str:
    return next(lbl for th, lbl in SIGNAL_LABELS if score >= th)


SIGNAL_LABELS = [(1.0, "偏多（強）"), (0.35, "偏多"), (-0.35, "中性"), (-1.0, "偏空"), (-99, "偏空（強）")]
NEUTRAL_WATCH = ["量化模型的上漲機率跌破 45% 或升破 60%", "三大法人買賣超方向反轉", "月營收年增率轉向",
                 "大盤由多頭轉為盤整或空頭（200 日均線與 60 日動能）"]


@dataclass
class ComplianceReport:
    mode: str
    removed_keys: list[str] = field(default_factory=list)
    removed_text: list[str] = field(default_factory=list)
    quoted: list[str] = field(default_factory=list)       # paths of third-party quotations passed through as-is

    @property
    def clean(self) -> bool:
        return not self.removed_text

    def merge(self, other: "ComplianceReport") -> "ComplianceReport":
        self.removed_keys += other.removed_keys
        self.removed_text += other.removed_text
        self.quoted += other.quoted
        return self

    def to_dict(self) -> dict:
        """Full record, removed sentences included — for the audit log ONLY, never for the user."""
        d = {"mode": self.mode, "removed_keys": sorted(set(self.removed_keys)), "removed_text": self.removed_text[:100]}
        if self.quoted:
            d["quoted"] = sorted(set(self.quoted))
        return d

    def public(self) -> dict:
        """What the user may see: how much was filtered, never what."""
        d = {"mode": self.mode, "removed_sentences": len(self.removed_text), "removed_fields": len(set(self.removed_keys))}
        if self.quoted:
            d["quote_note"] = QUOTE_LABEL
        return d


def public_view(d: dict | None) -> dict:
    """Turn a stored full report dict (to_dict) into the public form."""
    d = d or {}
    out = {"mode": d.get("mode"), "removed_sentences": len(d.get("removed_text") or []),
           "removed_fields": len(d.get("removed_keys") or [])}
    if d.get("quoted"):
        out["quote_note"] = QUOTE_LABEL
    return out


class ComplianceGuard:
    def __init__(self, mode: str = "research", ref_prices: Iterable[float] = (), ignore: Iterable[str] = ()):
        if mode not in MODES:
            raise ValueError(f"unknown compliance mode {mode!r}; use one of {MODES}")
        self.mode = mode
        self.ref_prices = [float(x) for x in ref_prices if x]
        self.ignore = {str(x) for x in ignore}

    @property
    def research(self) -> bool:
        return self.mode == "research"

    def with_refs(self, ref_prices: Iterable[float] = (), ignore: Iterable[str] = ()) -> "ComplianceGuard":
        return ComplianceGuard(self.mode, [*self.ref_prices, *ref_prices], {*self.ignore, *map(str, ignore)})

    def apply(self, obj: Any) -> tuple[Any, ComplianceReport]:
        rep = ComplianceReport(self.mode)
        if not self.research:
            return obj, rep
        return self._clean(obj, rep, ""), rep

    def _violations(self, text: str) -> list[str]:
        return find_violations(text, self.ref_prices, self.ignore)

    @staticmethod
    def _quotable(path: str, k: str, v: Any) -> bool:
        parent = path.rstrip(".").rsplit(".", 1)[-1] if path else ""
        return k in QUOTED_KEYS and parent in QUOTE_PARENTS and isinstance(v, (list, tuple))

    def _quote(self, items, rep: ComplianceReport, path: str) -> list:
        out = []
        for it in items:
            if isinstance(it, dict) and isinstance(it.get("title"), str):
                out.append({f: it[f] for f in QUOTE_FIELDS if f in it and isinstance(it[f], (str, int, float, type(None)))})
            else:
                rep.removed_keys.append(f"{path}[non-quote]")
        rep.quoted.append(path.rstrip("."))
        return out

    def _clean(self, obj: Any, rep: ComplianceReport, path: str) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                ks = str(k)
                if ks in ADVICE_KEYS:
                    rep.removed_keys.append(f"{path}{ks}")
                    continue
                if self._violations(ks):                     # advice smuggled in a dict key
                    rep.removed_keys.append(f"{path}<key>")
                    rep.removed_text.append(ks)
                    continue
                if self._quotable(path, ks, v):
                    out[k] = self._quote(v, rep, f"{path}{ks}.")
                    continue
                out[k] = self._clean(v, rep, f"{path}{ks}.")
            return out
        if isinstance(obj, (list, tuple)):
            items = [self._clean(v, rep, path) for v in obj]
            return [v for v in items if not (isinstance(v, str) and v == "")]
        if isinstance(obj, str):
            if not self._violations(obj):
                return obj
            kept, removed = [], []
            for s in sentences(obj):
                (removed if self._violations(s) else kept).append(s)
            rep.removed_text += [r.strip() for r in removed if r.strip()]
            return "".join(kept).strip()
        return obj

    def check(self, obj: Any) -> list[str]:
        """Violations left in an (already cleaned) output — used by tests, CI and the daily audit sample."""
        if not self.research:
            return []
        found: list[str] = []

        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    ks = str(k)
                    if ks in ADVICE_KEYS:
                        found.append(f"key:{path}{ks}")
                    found.extend(self._violations(ks))
                    if self._quotable(path, ks, v):
                        bad = [x for x in v if not (isinstance(x, dict) and set(x) <= set(QUOTE_FIELDS))]
                        found.extend(f"quote:{path}{ks}" for _ in bad)
                        continue
                    walk(v, f"{path}{ks}.")
            elif isinstance(o, (list, tuple)):
                for v in o:
                    walk(v, path)
            elif isinstance(o, str):
                found.extend(self._violations(o))
        walk(obj)
        return found


def as_text(v: Any, limit: int = 2000) -> str:
    """LLM fields that must be plain text (a dict or list here could smuggle content past the guard)."""
    if isinstance(v, str):
        return v[:limit]
    if isinstance(v, (list, tuple)):
        return "；".join(as_text(x, limit) for x in v)[:limit]
    if isinstance(v, dict):
        return "；".join(f"{as_text(k)}：{as_text(x)}" for k, x in v.items())[:limit]
    return "" if v is None else str(v)[:limit]


def as_text_list(v: Any, n: int = 6) -> list[str]:
    if isinstance(v, (list, tuple)):
        return [as_text(x, 500) for x in v][:n]
    return [as_text(v, 500)] if v else []


def research_decision(draft: dict, *, name: str, horizon: int, close: float, llm_text: dict | None = None) -> dict:
    """Research-mode replacement for the advisor decision: signal + probabilities + ranges in %, no advice."""
    score = float(draft.get("score", 0.0))
    conv = int(round(float(draft.get("confidence", 0.0)) * 100))
    band = draft.get("model_band_p10_p90") or [None, None]
    lo = (band[0] / close - 1) * 100 if band[0] else None
    hi = (band[1] / close - 1) * 100 if band[1] else None
    sig = signal_label(score)
    rng = f"模型估計未來 {horizon} 個交易日報酬率的 80% 區間約為 {lo:+.1f}% ～ {hi:+.1f}%，" if lo is not None else ""
    message = (f"綜合技術、基本面／籌碼與量化三位分析師，{name} 目前的綜合訊號為「{sig}」（分數 {score:+.2f}，信心 {conv}%）。"
               f"{rng}近 20 日年化波動約 {draft.get('hv20_ann_pct', '–')}%。以上為統計模型的研究結果，並非買賣建議。")
    out = {"mode": "research", "signal": sig, "signal_score": round(score, 3), "conviction": conv,
           "return_band_pct": [round(lo, 2) if lo is not None else None, round(hi, 2) if hi is not None else None],
           "hv20_ann_pct": draft.get("hv20_ann_pct"),
           "atr14_pct": round(float(draft["atr14"]) / close * 100, 2) if draft.get("atr14") else None,
           "agent_dispersion": draft.get("agent_dispersion"), "watch_items": list(NEUTRAL_WATCH),
           "client_message": message, "disclaimer": RESEARCH_DISCLAIMER}
    for k in ("thesis", "risks", "client_message", "horizon"):
        if llm_text and llm_text.get(k):
            out[k] = llm_text[k]
    return out
