from __future__ import annotations

import json
import logging

import httpx

from app.config import Settings
from app.extract import Fact
from app.textutil import tokenize

logger = logging.getLogger("memory.llm")

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
    if not settings.llm_extract or not api_key or settings.llm_model != "gpt-4o-mini" or not text.strip():
        return []
    snippet = text[:6000]
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
        content = response.json()["choices"][0]["message"]["content"]
        payload = json.loads(content)
    except Exception:
        logger.warning("llm fact extraction failed", exc_info=True)
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
