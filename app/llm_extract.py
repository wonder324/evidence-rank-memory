from __future__ import annotations

import json
import logging
import time

import httpx

from app.config import Settings
from app.extract import Fact
from app.textutil import tokenize

logger = logging.getLogger("memory.llm")


class LLMExtractionError(RuntimeError):
    """The configured extractor did not produce a usable fact payload."""


def _unwrap_gateway_json(content: str) -> str:
    """Undo only the gateway's observed envelope and non-JSON bracket escapes.

    Backslashes inside JSON strings remain untouched, including Windows paths
    and literal bracket escapes in a remembered value. No facts are inferred.
    """
    text = content.strip()
    if text.startswith("```json\n") or text.startswith("```\n"):
        first_newline = text.index("\n")
        if text.endswith("```"):
            text = text[first_newline + 1:-3].strip()
    # Some responses put a prose/code block before the transport envelope.
    # Never treat '<escape' inside a JSON string as a transport envelope.
    if not text.startswith(("{", "[", '"')):
        marker = text.find("<escape")
        if marker >= 0:
            text = text[marker + len("<escape"):].strip()
    out = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "\\" and index + 1 < len(text) and text[index + 1] in "[]":
                index += 1
                char = text[index]
            out.append(char)
        index += 1
    return "".join(out)


def parse_fact_payload(content: str) -> dict:
    if not isinstance(content, str):
        raise LLMExtractionError("extractor content must be text")
    try:
        normalized = _unwrap_gateway_json(content)
        if normalized == content.strip() or "<escape" not in content:
            payload = json.loads(normalized)
        else:
            payload, end = json.JSONDecoder().raw_decode(normalized)
            # Only the actually observed envelope endings are accepted.
            if normalized[end:].strip() not in {"", ">", "\\</", "</escape>"}:
                raise ValueError("unexpected trailing extractor output")
    except (ValueError, TypeError) as exc:
        raise LLMExtractionError("extractor returned invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), list):
        raise LLMExtractionError("extractor payload must contain a facts array")
    return payload

_LLM_MODES = {"replace", "append", "retract", "archive", "correct"}
_LLM_ATTRS = {
    "residence",
    "name",
    "occupation",
    "preference",
    "rule",
    "event",
    "other",
    "mentor",
    "colleague",
    "friend",
    "classmate",
    "manager",
    "student",
    "met",
    "phone",
    "email",
    "city",
    "employee_id",
    "department",
    "id_card",
    "allergy",
    "nickname",
    "address",
    "medical",
}

SYSTEM = (
    "You extract atomic personal memories from a conversation snippet. "
    "Do not answer questions and do not invent facts that are not stated. "
    "Return a JSON object {\"facts\": [{\"subject\": str, \"attribute\": str, "
    "\"value\": str, \"mode\": \"replace\"|\"append\"|\"retract\"|\"archive\"|\"correct\", "
    "\"replaces\": str}]}. "
    "attribute is one of residence, name, occupation, preference, rule, event, "
    "mentor, colleague, friend, classmate, manager, student, met, phone, email, "
    "city, employee_id, department, id_card, allergy, nickname, address, medical, other. "
    "Use mode=replace for one current value. Use mode=archive for a past value that "
    "must not become current. Use mode=append for preferences, rules, relations, and events. "
    "Use mode=retract when the speaker cancels a fact, and value='*' to cancel a whole attribute. "
    "Use mode=correct with replaces set to the wrong value. "
    "Number ordered stops as value '1:place'. Do not store hearsay as the user's own residence."
)


def llm_facts(settings: Settings, text: str) -> list[Fact]:
    api_key = settings.openai_api_key
    if not settings.llm_extract or not text.strip():
        return []
    if not api_key or settings.llm_model != "gpt-4o-mini":
        if getattr(settings, "llm_extract_strict", False):
            raise LLMExtractionError("extractor configuration is incomplete")
        return []
    snippet = text[:6000]
    started = time.monotonic()
    try:
        response = httpx.post(
            f"{settings.openai_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": settings.llm_model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": snippet},
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=25,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMExtractionError("extractor output was truncated")
        if choice.get("message", {}).get("refusal"):
            raise LLMExtractionError("extractor refused the request")
        content = choice["message"]["content"]
        payload = parse_fact_payload(content)
    except Exception as exc:
        # Log neither the memory text nor the gateway's response body/URL.
        status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
        reason = str(exc) if isinstance(exc, LLMExtractionError) else type(exc).__name__
        logger.warning("llm fact extraction failed type=%s status=%s reason=%s seconds=%.3f",
                       type(exc).__name__, status, reason, time.monotonic() - started)
        if getattr(settings, "llm_extract_strict", False):
            raise LLMExtractionError("fact extraction unavailable") from exc
        return []
    facts: list[Fact] = []
    for item in payload.get("facts", []):
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or "").strip()
        attribute = str(item.get("attribute") or "other").strip().lower()
        mode = str(item.get("mode") or "append").strip().lower()
        subject = str(item.get("subject") or "user").strip() or "user"
        if not value or mode not in _LLM_MODES:
            continue
        if attribute not in _LLM_ATTRS:
            attribute = "other"
        replaces = str(item.get("replaces") or "").strip()[:80]
        facts.append(
            Fact(
                subject=subject[:80],
                attribute=attribute,
                value=value[:240],
                evidence=snippet[:240],
                mode=mode,
                entities=tokenize(f"{subject} {value}")[:12],
                replaces=replaces,
            )
        )
    return facts
