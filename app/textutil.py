from __future__ import annotations

import hashlib
import re
from typing import Iterable

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "and", "or", "if", "as", "at",
    "by", "from", "into", "over", "after", "about", "that", "this", "these",
    "those", "it", "its", "my", "your", "our", "we", "you", "me", "i", "do",
    "does", "did", "what", "which", "who", "whom", "whose", "where", "when",
    "why", "how", "please", "there", "their", "them", "his", "her", "she",
    "he", "they", "have", "has", "had", "not", "but", "so", "than", "then",
    "的", "了", "吗", "呢", "是", "我", "你", "他", "她", "它", "在", "有",
    "和", "与", "就", "都", "而", "及", "或", "什么", "哪里", "怎么", "如何",
    "请问", "一个", "这个", "那个",
}

SYNONYM_GROUPS = (
    {"live", "lives", "living", "lived", "residence", "city", "home", "moved", "move", "stay", "staying", "住", "住在", "搬到", "搬去", "搬家", "居住", "家"},
    {"like", "likes", "love", "loved", "prefer", "prefers", "favorite", "favourite", "enjoy", "喜欢", "偏好", "最爱"},
    {"name", "called", "call", "名字", "姓名", "叫做"},
    {"work", "works", "job", "occupation", "employed", "职业", "工作"},
    {"meet", "met", "meeting", "见面", "遇到", "认识"},
    {"rule", "must", "never", "always", "constraint", "规则", "必须", "不要", "不能"},
    {"before", "previously", "earlier", "formerly", "used", "以前", "曾经", "原来", "过去"},
    {"now", "current", "currently", "latest", "today", "现在", "目前", "最新"},
)

HISTORY_RE = re.compile(
    r"\b(used to|previously|previous|before|formerly|earlier|originally|no longer|anymore)\b|以前|曾经|原来|过去|不再|之前|曾经的",
    re.I,
)
QUOTE_RE = re.compile(
    r"\b(say|said|mention|mentioned|told|asked|quote|exactly)\b|说过|提到|原话|怎么说|那句",
    re.I,
)
BROAD_RE = re.compile(
    r"\b(know about|profile|preferences|summarize|summary|who am i|about me)\b|了解我|个人偏好|总结|个人信息|我是谁",
    re.I,
)
STATE_RE = re.compile(
    r"\b(where|who|what is|what's|current|now|latest|today|which)\b|现在|目前|哪里|是谁|最新|什么",
    re.I,
)
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
NUMBER_RE = re.compile(r"\d+(?:[:./-]\d+)*")


def is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def tokenize(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text or "")
    tokens: list[str] = []
    cjk_buf: list[str] = []

    def flush_cjk() -> None:
        if not cjk_buf:
            return
        tokens.extend(cjk_buf)
        tokens.extend(a + b for a, b in zip(cjk_buf, cjk_buf[1:]))
        cjk_buf.clear()

    for piece in raw:
        if len(piece) == 1 and is_cjk(piece):
            cjk_buf.append(piece)
            continue
        flush_cjk()
        lower = piece.lower()
        tokens.append(lower)
        if "_" in lower or any(ch.isupper() for ch in piece):
            for part in CAMEL_RE.findall(piece):
                part = part.lower()
                if part and part != lower:
                    tokens.append(part)
            for part in lower.split("_"):
                if part and part != lower:
                    tokens.append(part)
    flush_cjk()
    return tokens


def content_tokens(text: str) -> list[str]:
    return [tok for tok in tokenize(text) if tok not in STOPWORDS and len(tok) > 1 or is_cjk_token(tok)]


def is_cjk_token(token: str) -> bool:
    return bool(token) and all(is_cjk(ch) for ch in token)


def query_terms(text: str) -> list[str]:
    seen: list[str] = []
    bag = set()
    for tok in tokenize(text):
        if tok in STOPWORDS:
            continue
        if len(tok) == 1 and not is_cjk_token(tok):
            continue
        if tok not in bag:
            bag.add(tok)
            seen.append(tok)
    expanded = list(seen)
    present = set(tokenize(text))
    for group in SYNONYM_GROUPS:
        if present & group or bag & group:
            for word in group:
                if word not in bag and (len(word) > 1 or is_cjk_token(word)):
                    bag.add(word)
                    expanded.append(word)
                if len(expanded) >= len(seen) + 16:
                    break
    return expanded


def classify_intent(text: str) -> str:
    if HISTORY_RE.search(text or ""):
        return "history"
    if QUOTE_RE.search(text or ""):
        return "quote"
    if BROAD_RE.search(text or ""):
        return "broad"
    if STATE_RE.search(text or ""):
        return "state"
    return "general"


def numbers_in(text: str) -> list[str]:
    return NUMBER_RE.findall(text or "")


def user_token(user_id: str) -> str:
    digest = hashlib.sha1(user_id.encode("utf-8")).hexdigest()[:16]
    return f"u{digest}"


def looks_like_code(text: str) -> bool:
    if not text:
        return False
    symbols = sum(text.count(ch) for ch in "{}();=<>")
    lines = text.count("\n")
    return lines >= 6 and symbols / max(len(text), 1) > 0.02


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a = set(left)
    b = set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_ratio(query: Iterable[str], doc: Iterable[str]) -> float:
    q = [tok for tok in query if tok not in STOPWORDS]
    if not q:
        return 0.0
    docs = set(doc)
    return sum(1 for tok in q if tok in docs) / len(q)


def flatten_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    parts.append(str(part["text"]))
                elif part.get("type") == "image_url":
                    parts.append("[image]")
        return "\n".join(parts)
    return str(content or "")
