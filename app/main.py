from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import hashlib
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import SYSTEM_NAME, SYSTEM_VERSION, get_settings
from app.embedder import Embedder
from app.extract import Fact
from app.llm_extract import LLMExtractionError, llm_facts
from app.store import ContractError, MemoryStore
from app.textutil import flatten_content

logger = logging.getLogger("uvicorn.error.memory")
settings = get_settings()
store = MemoryStore(settings, Embedder(settings), settings.weights())
app = FastAPI(title="Agent Memory Challenge", version="1.0.0")
_slots = threading.BoundedSemaphore(int(os.environ.get("MAX_INFLIGHT", "8")))


def _reason(status: int, reason: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": {"reason": reason}})


def _authorized(request: Request) -> bool:
    expected = settings.api_key
    if not expected:
        return True
    header = request.headers.get("authorization", "")
    supplied = request.headers.get("x-api-key", "")
    if header.lower().startswith("bearer "):
        supplied = supplied or header[7:].strip()
    elif header.lower().startswith("token "):
        supplied = supplied or header[6:].strip()
    return supplied == expected


def _busy() -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": {"reason": "memory service is busy"}}, headers={"Retry-After": "2"})


@app.get("/health")
def health():
    return {"status": "ok", "system": SYSTEM_NAME, "version": SYSTEM_VERSION}


def _handle_add(payload: dict):
    started = time.monotonic()
    ref = hashlib.sha256(str(payload.get("request_id", "")).encode()).hexdigest()[:16]
    logger.info("add start ref=%s", ref)
    try:
        result = _handle_add_inner(payload)
        status = result.status_code if isinstance(result, JSONResponse) else 200
        logger.info("add complete ref=%s status=%s seconds=%.3f", ref, status, time.monotonic() - started)
        return result
    except Exception:
        logger.exception("add unhandled ref=%s seconds=%.3f", ref, time.monotonic() - started)
        raise


def _handle_add_inner(payload: dict):
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return _reason(422, "messages must be a list")
    request_id = str(payload.get("request_id") or "")
    user_id = str(payload.get("user_id") or "")
    session_id = str(payload.get("session_id") or "")
    if not request_id or not user_id or not session_id or not messages:
        return _reason(422, "request_id, user_id, session_id and messages are required")
    try:
        # Validation still runs on retries; successful replays require no model API.
        for message in messages:
            store._prepare_message(message)
        if store.add_completed(request_id, user_id, session_id):
            return {"success": True, "request_id": payload["request_id"],
                    "user_id": payload["user_id"], "session_id": payload["session_id"]}
    except ContractError as exc:
        return _reason(exc.status, exc.reason)
    extra: list[Fact] = []
    if settings.llm_extract:
        blob = "\n".join(flatten_content(message.get("content")) for message in messages if isinstance(message, dict))
        try:
            extra = llm_facts(settings, blob)
        except LLMExtractionError:
            # Raw source messages still go through the normal atomic persistence
            # and indexing path below. Never acknowledge before store.add commits.
            logger.warning("llm extraction unavailable; persisting indexed source messages")
            extra = []
    try:
        store.add(
            request_id=str(payload.get("request_id") or ""),
            user_id=str(payload.get("user_id") or ""),
            session_id=str(payload.get("session_id") or ""),
            messages=messages,
            extra_facts=extra,
        )
    except ContractError as exc:
        return _reason(exc.status, exc.reason)
    except Exception:
        logger.exception("add failed")
        return _reason(500, "add failed")
    return {
        "success": True,
        "request_id": payload["request_id"],
        "user_id": payload["user_id"],
        "session_id": payload["session_id"],
    }


def _handle_search(payload: dict):
    started = time.monotonic()
    ref = uuid.uuid4().hex[:12]
    logger.info("search start ref=%s", ref)
    try:
        result = _handle_search_inner(payload)
        status = result.status_code if isinstance(result, JSONResponse) else 200
        logger.info("search complete ref=%s status=%s seconds=%.3f", ref, status, time.monotonic() - started)
        return result
    except Exception:
        logger.exception("search unhandled ref=%s seconds=%.3f", ref, time.monotonic() - started)
        raise


def _handle_search_inner(payload: dict):
    options = payload.get("options")
    if options is not None and not isinstance(options, list):
        return _reason(422, "options must be a string array")
    try:
        data = store.search(
            user_id=str(payload.get("user_id") or ""),
            query=payload.get("query"),
            top_k=payload.get("top_k"),
            options=options,
        )
    except ContractError as exc:
        return _reason(exc.status, exc.reason)
    except Exception:
        logger.exception("search failed")
        return _reason(500, "search failed")
    return {"data": data}


@app.post("/add")
async def add(request: Request):
    if not _authorized(request):
        return _reason(401, "invalid memory system key")
    if not _slots.acquire(blocking=False):
        return _busy()
    try:
        try:
            payload = await request.json()
        except Exception:
            return _reason(400, "request body must be JSON")
        return await asyncio.to_thread(_handle_add, payload)
    finally:
        _slots.release()


@app.post("/search")
async def search(request: Request):
    if not _authorized(request):
        return _reason(401, "invalid memory system key")
    if not _slots.acquire(blocking=False):
        return _busy()
    try:
        try:
            payload = await request.json()
        except Exception:
            return _reason(400, "request body must be JSON")
        return await asyncio.to_thread(_handle_search, payload)
    finally:
        _slots.release()
