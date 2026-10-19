from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = tempfile.mkdtemp(prefix="aml-contract-")
os.environ["DATA_DIR"] = TMP
os.environ["EMBEDDER"] = "hash"
os.environ["LLM_EXTRACT"] = "0"
os.environ["RERANKER"] = "0"
os.environ["MEMORY_API_KEY"] = "secret"
os.environ["FUSION_WEIGHTS"] = str(Path(__file__).resolve().parents[1] / "models" / "fusion_weights.json")

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> None:
    health = client.get("/health")
    check(health.status_code == 200, "health failed")

    denied = client.post("/add", json={"request_id": "x", "user_id": "u", "session_id": "s", "messages": []})
    check(denied.status_code == 401, "missing key should be 401")

    headers = {"Authorization": "Bearer secret"}
    created = client.post(
        "/add",
        headers=headers,
        json={
            "request_id": "r1",
            "user_id": "eval:run:locomo:conv-0",
            "session_id": "eval:run:sample:0",
            "messages": [
                {"role": "user", "timestamp": 1704067200000, "content": "I used to live in New York but now I live in Boston."},
                {"role": "user", "timestamp": 1704153600000, "content": "I met Bob on Tuesday. Bob lives in Paris."},
            ],
        },
    )
    check(created.status_code == 200, created.text)
    body = created.json()
    check(body["success"] is True, "success must be true")
    check(body["request_id"] == "r1", "request_id echo")
    check(body["user_id"] == "eval:run:locomo:conv-0", "user_id echo")

    again = client.post(
        "/add",
        headers=headers,
        json={
            "request_id": "r1",
            "user_id": "eval:run:locomo:conv-0",
            "session_id": "eval:run:sample:0",
            "messages": [{"role": "user", "content": "I used to live in New York but now I live in Boston."}],
        },
    )
    check(again.status_code == 200, "idempotent add should succeed")

    current = client.post(
        "/search",
        headers=headers,
        json={"query": "Which city is home now?", "user_id": "eval:run:locomo:conv-0", "top_k": 5},
    )
    check(current.status_code == 200, current.text)
    current_blob = "\n".join(str(item["content"]) for item in current.json()["data"])
    check("Boston" in current_blob, f"current city missing: {current_blob}")
    check(all(item["id"] and item["content"] for item in current.json()["data"]), "id/content required")
    check(len(current.json()["data"]) <= 5, "top_k exceeded")

    history = client.post(
        "/search",
        headers={"X-Api-Key": "secret"},
        json={"query": "Where did I live before?", "user_id": "eval:run:locomo:conv-0", "top_k": 8},
    )
    history_blob = "\n".join(str(item["content"]) for item in history.json()["data"])
    check("New York" in history_blob, f"history missing: {history_blob}")

    hop = client.post(
        "/search",
        headers=headers,
        json={
            "query": "Where does the person I met on Tuesday live?",
            "options": ["A. Paris", "B. Boston"],
            "user_id": "eval:run:locomo:conv-0",
            "top_k": 8,
        },
    )
    hop_blob = "\n".join(str(item["content"]) for item in hop.json()["data"])
    check("Paris" in hop_blob and "Bob" in hop_blob, f"hop missing: {hop_blob}")
    check(all("A. Paris" not in str(item["content"]) for item in hop.json()["data"]), "option text was stored as memory")

    other = client.post(
        "/search",
        headers=headers,
        json={"query": "Where does Bob live?", "user_id": "someone-else", "top_k": 5},
    )
    check(other.json()["data"] == [], "user isolation failed")

    zh = client.post(
        "/add",
        headers=headers,
        json={
            "request_id": "r-zh",
            "user_id": "user-zh",
            "session_id": "s-zh",
            "messages": [{"role": "user", "content": "我以前住在杭州，现在住在南京。"}],
        },
    )
    check(zh.status_code == 200, zh.text)
    zh_now = client.post(
        "/search",
        headers=headers,
        json={"query": "我现在住在哪里？", "user_id": "user-zh", "top_k": 5},
    )
    zh_blob = "\n".join(item["content"] for item in zh_now.json()["data"])
    check("南京" in zh_blob, f"chinese current missing: {zh_blob}")

    image = client.post(
        "/add",
        headers=headers,
        json={
            "request_id": "r-img",
            "user_id": "user-img",
            "session_id": "s-img",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "The poster shows a red bicycle."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                    ],
                }
            ],
        },
    )
    check(image.status_code == 200, image.text)
    found = client.post(
        "/search",
        headers=headers,
        json={"query": "What does the poster show?", "user_id": "user-img", "top_k": 3},
    )
    check(any("bicycle" in str(item["content"]) for item in found.json()["data"]), found.text)

    empty = client.post("/add", headers=headers, json={"request_id": "", "user_id": "u", "session_id": "s", "messages": []})
    check(empty.status_code == 422, "empty add should be 422")
    print("contract check passed")


if __name__ == "__main__":
    main()
