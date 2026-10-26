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


def _has_transport_brackets(text: str) -> bool:
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "\\" and text[index + 1:index + 2] in {"[", "]"}:
            return True
    return False


def _strip_gateway_envelope(content: str) -> tuple[str, bool]:
    text = content.strip().lstrip("\ufeff").strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        label = text[3:first_newline].strip().lower() if first_newline >= 0 else None
        if label in {"", "json"} and text.endswith("```"):
            text = text[first_newline + 1:-3].strip()
    # Observed gateway replies may prefix JSON with a fenced prose explanation.
    # Discard that complete block only when a recognizable payload follows it.
    if text.startswith("```"):
        first_newline = text.find("\n")
        fence_end = text.find("```", first_newline + 1) if first_newline >= 0 else -1
        if fence_end >= 0:
            rest = text[fence_end + 3:].strip()
            if rest.startswith(("{", "<escape", "<{")):
                text = rest
    # Some responses put a prose/code block before the transport envelope.
    # Never treat '<escape' inside a JSON string as a transport envelope.
    envelope = False
    if text.startswith("<{"):
        text = text[1:]
        envelope = True
    elif not text.startswith(("{", "[", '"')):
        marker = text.find("<escape")
        if marker >= 0:
            text = text[marker + len("<escape"):].strip()
            envelope = True
    elif text.startswith("{") and text.endswith("\\</escape") and _has_transport_brackets(text):
        # Observed missing opening marker: require independent transport evidence.
        envelope = True
    return text, envelope


def _unwrap_gateway_json(content: str) -> str:
    """Undo transport bracket escapes while preserving all legal JSON escapes."""
    text, _ = _strip_gateway_envelope(content)
    out = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            if not escaped and char == "\\" and text[index + 1:index + 2] in {"[", "]"}:
                # These two escape sequences are illegal in JSON. A paired
                # backslash (e.g. a literal Windows path) is never removed.
                index += 1
                char = text[index]
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


def _repair_json_syntax(text: str) -> str:
    """Repair separators outside strings; never invent a key, value or bracket.

    Only a colon run following a quoted token and a comma immediately before
    a container end are eligible. The strict decoder still validates the full
    document, so missing values and duplicated fragments remain errors.
    """
    out = []
    in_string = False
    escaped = False
    previous = ""
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
                previous = '"'
            index += 1
            continue
        if char == '"':
            in_string = True
        elif previous == '"' and (char == ":" or text.startswith("\\:", index)):
            end = index
            while end < len(text):
                if text[end] == ":":
                    end += 1
                elif text.startswith("\\:", end):
                    end += 2
                else:
                    lookahead = end
                    while lookahead < len(text) and text[lookahead].isspace():
                        lookahead += 1
                    if lookahead < len(text) and (text[lookahead] == ":" or text.startswith("\\:", lookahead)):
                        end = lookahead
                    else:
                        break
            out.append(":")
            previous = ":"
            index = end
            continue
        elif char == "," and previous not in {"", "{", "[", ",", ":"}:
            end = index + 1
            while end < len(text) and text[end].isspace():
                end += 1
            if end < len(text) and text[end] in "]}":
                index += 1
                continue
        out.append(char)
        if not char.isspace() and not in_string:
            previous = char
        index += 1
    return "".join(out)


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            if _canonical_json(result[key]) != _canonical_json(value):
                raise ValueError("conflicting duplicate JSON key")
            continue
        result[key] = value
    return result


def _canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _reject_constant(value: str):
    raise ValueError("non-JSON numeric constant")


def _allowed_tail(tail: str, envelope: bool) -> bool:
    return not tail or envelope and tail in {">", "\\</", "\\</escape", "</escape>"}


def _decode_repeated_documents(text: str, decoder, envelope: bool) -> dict:
    """Collapse complete consecutive documents only when their values agree."""
    payload, end = decoder.raw_decode(text)
    identity = _canonical_json(payload)
    for _ in range(8):
        tail = text[end:].strip()
        if _allowed_tail(tail, envelope):
            return payload
        if not tail.startswith("{"):
            raise ValueError("unexpected trailing extractor output")
        duplicate, consumed = decoder.raw_decode(tail)
        if _canonical_json(duplicate) != identity:
            raise ValueError("conflicting repeated JSON documents")
        end = len(text) - len(tail) + consumed
    raise ValueError("too many repeated JSON documents")


def _recovery_fact(fact) -> bool:
    required = {"subject", "attribute", "value", "mode"}
    return (isinstance(fact, dict) and required <= fact.keys()
            and fact.keys() <= required | {"replaces"}
            and all(isinstance(value, str) for value in fact.values())
            and all(fact[key] for key in required)
            and fact["attribute"] in _LLM_ATTRS and fact["mode"] in _LLM_MODES)


def _prefix_before_restart(prefix: str, decoder):
    """Read complete facts and scalar fields before an interrupted fact value.

    No closing quote/bracket/value is supplied. The restart must follow a real
    colon. Evidence is a complete preceding fact, or an already complete
    subject/attribute/value triple in the interrupted first fact.
    """
    size = len(prefix)
    pos = 0

    def whitespace(index):
        while index < size and prefix[index].isspace():
            index += 1
        return index

    def symbol(index, expected):
        index = whitespace(index)
        if index >= size or prefix[index] != expected:
            raise ValueError("not a fact-prefix restart")
        return whitespace(index + 1)

    pos = symbol(pos, "{")
    key, pos = decoder.raw_decode(prefix, pos)
    if key != "facts":
        raise ValueError("restart root must contain only facts")
    pos = symbol(pos, ":")
    if pos == size:
        # A repeated root header contains no discarded fact or partial value.
        return [], {}, None
    pos = symbol(pos, "[")
    complete = []
    while pos < size:
        pos = symbol(pos, "{")
        fields = {}
        while pos < size:
            key, pos = decoder.raw_decode(prefix, pos)
            if key not in {"subject", "attribute", "value", "mode", "replaces"} or key in fields:
                raise ValueError("ambiguous interrupted fact")
            pos = symbol(pos, ":")
            if pos == size:
                first_fact_evidence = (
                    {"subject", "attribute", "value"} <= fields.keys()
                    and all(fields[k] for k in ("subject", "attribute", "value"))
                    and fields["attribute"] in _LLM_ATTRS
                )
                if not fields.get("subject") or (not complete and not first_fact_evidence):
                    raise ValueError("not enough duplicated prefix evidence")
                return complete, fields, key
            value, pos = decoder.raw_decode(prefix, pos)
            if not isinstance(value, str):
                raise ValueError("interrupted fact values must be strings")
            fields[key] = value
            pos = whitespace(pos)
            if pos < size and prefix[pos] == "}":
                if not _recovery_fact(fields):
                    raise ValueError("incomplete prefix fact")
                complete.append(fields)
                pos = symbol(pos + 1, ",")
                break
            pos = symbol(pos, ",")
    raise ValueError("no interrupted scalar value")


def _recover_restarted_payload(text: str, decoder, envelope: bool) -> dict:
    """Recover a complete repeated facts object that agrees with its prefix.

    This covers the observed generation restart inside an unfinished fact.
    An unrelated nested object, a conflicting fact or a lost prefix is refused.
    """
    if len(text) > 65536:
        raise ValueError("restart recovery size limit")
    starts = []
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{" and index:
            starts.append(index)
            if len(starts) > 64:
                raise ValueError("restart recovery candidate limit")
    recovered = {}
    for start in starts:
        try:
            payload, end = decoder.raw_decode(text, start)
            if not _allowed_tail(text[end:].strip(), envelope):
                continue
            if not isinstance(payload, dict) or set(payload) != {"facts"}:
                continue
            facts = payload["facts"]
            if not isinstance(facts, list) or not facts or not all(_recovery_fact(fact) for fact in facts):
                continue
            complete, partial, pending_key = _prefix_before_restart(text[:start], decoder)
            if len(facts) <= len(complete):
                continue
            if _canonical_json(facts[:len(complete)]) != _canonical_json(complete):
                continue
            continuation = facts[len(complete)]
            if pending_key is not None and (pending_key not in continuation or any(continuation.get(k) != v for k, v in partial.items())):
                continue
            recovered[_canonical_json(payload)] = payload
        except (ValueError, TypeError):
            continue
    if len(recovered) != 1:
        raise ValueError("no unique, consistent restart payload")
    return next(iter(recovered.values()))


def parse_fact_payload(content: str) -> dict:
    if not isinstance(content, str):
        raise LLMExtractionError("extractor content must be text")
    try:
        normalized = _repair_json_syntax(_unwrap_gateway_json(content))
        decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        _, envelope = _strip_gateway_envelope(content)
        try:
            payload = _decode_repeated_documents(normalized, decoder, envelope)
        except ValueError:
            payload = _recover_restarted_payload(normalized, decoder, envelope)
    except (ValueError, TypeError, RecursionError, OverflowError) as exc:
        raise LLMExtractionError("extractor returned invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), list):
        raise LLMExtractionError("extractor payload must contain a facts array")
    for fact in payload["facts"]:
        if not isinstance(fact, dict):
            raise LLMExtractionError("facts entries must be objects")
        for key in {"subject", "attribute", "value", "mode", "replaces"} & fact.keys():
            if not isinstance(fact[key], str):
                raise LLMExtractionError("fact fields must be strings")
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
        # At most one fresh extraction after a malformed payload. Reuse the
        # original prompt and spend only the remaining read-timeout budget.
        for attempt in range(2):
            remaining = 25.0 - (time.monotonic() - started)
            if remaining <= 0:
                raise LLMExtractionError("extractor retry budget exhausted")
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
                timeout=remaining,
            )
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") in {"length", "content_filter"}:
                raise LLMExtractionError("extractor output was truncated or filtered")
            if choice.get("message", {}).get("refusal"):
                raise LLMExtractionError("extractor refused the request")
            content = choice["message"]["content"]
            try:
                payload = parse_fact_payload(content)
            except LLMExtractionError:
                if attempt == 0 and 25.0 - (time.monotonic() - started) >= 10.0:
                    logger.warning("llm payload invalid; retrying once within remaining budget")
                    continue
                raise
            if attempt:
                logger.info("llm payload recovered on bounded retry")
            break
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
