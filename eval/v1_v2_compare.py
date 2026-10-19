"""Search-only comparison of v1 and v2 on the same stored PerLTQA memories.

v1 is the untouched tree. v2 is this copy, including the cross-encoder.
Neither side rewrites the canonical database; each opens its own copy.
"""

from __future__ import annotations

import argparse
import array
import ast
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

COPY = Path("/Users/wonder/Desktop/agent-memory-challenge- v1_副本")
V1 = Path("/Users/wonder/Desktop/agent-memory-challenge- v1")

PROFILE_ZH = {
    "Gender": "性别",
    "Nickname": "昵称",
    "Title": "头衔",
    "Age": "年龄",
    "Occupation": "职业",
    "Nationality": "国籍",
    "Physical Characteristics": "外貌",
    "Hobbies": "爱好",
    "Achievements": "成就",
    "Ethnic Background": "民族",
    "Education Background": "教育背景",
    "Employer": "工作单位",
    "Awards and Role Models": "榜样与获奖",
    "Personality": "性格",
    "Birthday": "生日",
    "Birthplace": "出生地",
    "Residence": "居住地",
    "Marital Status": "婚姻状况",
    "Family": "家庭",
}


def _load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_refs(raw) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw]
    text = str(raw).strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [text]


def dialogue_text(dialogue: dict) -> str:
    lines: list[str] = []
    contents = dialogue.get("contents") or {}
    for stamp, turns in contents.items():
        lines.append(str(stamp))
        if isinstance(turns, list):
            lines.extend(str(turn) for turn in turns)
        else:
            lines.append(str(turns))
    return "\n".join(lines)


def memory_docs(memory: dict) -> dict[str, str]:
    profile = memory["profile"]
    name = profile["Protagonist"]
    docs: dict[str, str] = {}
    for key, value in profile.items():
        if key == "Protagonist" or value in (None, ""):
            continue
        label = PROFILE_ZH.get(key, key)
        docs[f"profile:{key}"] = f"{name}的{label}是{value}。"
    description = (memory.get("profile_description") or "").strip()
    if description:
        docs["profile:description"] = description
    for sid, rel in (memory.get("social_relationship") or {}).items():
        person = rel.get("Supporting Characters") or rel.get("配角") or ""
        relation = rel.get("Relationship") or rel.get("关系") or ""
        desc = rel.get("Description") or rel.get("描述") or ""
        docs[str(sid)] = f"{name}的{relation}是{person}。{desc}"
    for eid, event in (memory.get("events") or {}).items():
        docs[str(eid)] = str(event.get("content") or event.get("summary") or "")
    for did, dialogue in (memory.get("dialogues") or {}).items():
        docs[str(did)] = dialogue_text(dialogue)
    return {key: value for key, value in docs.items() if value.strip()}


def needle(text: str) -> str:
    compact = " ".join(text.split())
    if len(compact) <= 80:
        return compact
    return compact[24:104]


def iter_questions(body: dict):
    for item in body.get("profile") or []:
        yield "profile", item
    for section in ("social_relationship", "events", "dialogues"):
        block = body.get(section) or {}
        groups = block.values() if isinstance(block, dict) else [block]
        for group in groups:
            for item in group:
                yield section, item


def gold_ids(section: str, item: dict) -> list[str]:
    refs = parse_refs(item.get("Reference Memory"))
    if section == "profile":
        return [f"profile:{ref}" for ref in refs]
    return refs


def rank_of(rows: list[dict], snippet: str) -> int | None:
    if not snippet:
        return None
    key = "".join(str(snippet).split())
    for index, row in enumerate(rows):
        if key in "".join(str(row["content"]).split()):
            return index
    return None


class CacheEmbedder:
    def __init__(self, inner, path: Path):
        self.inner = inner
        self.conn = sqlite3.connect(path)
        self.conn.execute("CREATE TABLE IF NOT EXISTS vec (h TEXT PRIMARY KEY, blob BLOB)")
        self.mem: dict[str, array.array] = {}

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def _load(self, text: str):
        key = self._hash(text)
        cached = self.mem.get(key)
        if cached is not None:
            return cached
        row = self.conn.execute("SELECT blob FROM vec WHERE h = ?", (key,)).fetchone()
        if row is None:
            return None
        vector = array.array("f")
        vector.frombytes(row[0])
        self.mem[key] = vector
        return vector

    def embed_many(self, texts):
        output = [None] * len(texts)
        missing = []
        for index, text in enumerate(texts):
            cached = self._load(text)
            if cached is None:
                missing.append((index, text))
            else:
                output[index] = cached
        for start in range(0, len(missing), 10):
            chunk = missing[start : start + 10]
            vectors = self.inner.embed_many([text for _, text in chunk])
            for (index, text), vector in zip(chunk, vectors):
                key = self._hash(text)
                self.mem[key] = vector
                self.conn.execute("INSERT OR REPLACE INTO vec(h, blob) VALUES (?, ?)", (key, vector.tobytes()))
                output[index] = vector
            self.conn.commit()
        return output

    def embed(self, text: str):
        return self.embed_many([text])[0]


def _rates(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "recall_at_1": round(sum(row["recall_at_1"] for row in rows) / n, 4),
        "recall_at_5": round(sum(row["recall_at_5"] for row in rows) / n, 4),
        "mrr": round(sum(row["mrr"] for row in rows) / n, 4),
    }


def _prepare_app(side: str):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["RERANKER"] = "0" if side == "v1" else "1"
    os.environ["RERANKER_CONTEXT"] = "1" if side == "v3" else "0"
    os.environ["RERANKER_TOP_N"] = "20" if side == "v4" else "5"
    os.environ["RERANKER_BLEND"] = "0.45" if side == "v4" else "0.8"
    os.environ["CHRONOS"] = "1" if side == "v5" else "0"
    if side == "v3":
        os.environ["RERANKER_MODEL"] = str(COPY / "models" / "v3-reranker")
    elif side == "v4":
        os.environ["RERANKER_MODEL"] = "BAAI/bge-reranker-base"
    root = V1 if side == "v1" else COPY
    sys.path.insert(0, str(root))
    from app.config import get_settings
    from app.embedder import Embedder
    from app.store import MemoryStore

    return get_settings, Embedder, MemoryStore


def run_side(side: str, limit: int, data_dir: Path, out_path: Path) -> None:
    _load_env(COPY / ".env")
    get_settings, Embedder, MemoryStore = _prepare_app(side)
    qa = json.loads((COPY / "eval/external/perltqa_zh.json").read_text(encoding="utf-8"))
    mem_rows = json.loads((COPY / "eval/external/perltmem_zh.json").read_text(encoding="utf-8"))
    memories = {row["profile"]["Protagonist"]: row for row in mem_rows}
    characters = [(name, body) for item in qa for name, body in item.items()]
    settings = get_settings()
    settings.data_dir = data_dir
    if side in {"v2", "v3", "v4"}:
        settings.reranker = True
        settings.reranker_context = side == "v3"
        settings.reranker_top_n = 20 if side == "v4" else 5
        settings.reranker_blend = 0.45 if side == "v4" else 0.8
        if side == "v3":
            settings.reranker_model = str(COPY / "models" / "v3-reranker")
        elif side == "v4":
            settings.reranker_model = "BAAI/bge-reranker-base"
    if side == "v5":
        settings.reranker = False
        settings.chronos = True
    embedder = CacheEmbedder(Embedder(settings), data_dir / "embed-cache.sqlite")
    store = MemoryStore(settings, embedder, settings.weights())
    if side in {"v2", "v3", "v4"}:
        store.reranker.use_context = side == "v3"
        store.reranker.model_name = settings.reranker_model
        store.reranker.top_n = settings.reranker_top_n
        store.reranker.blend = settings.reranker_blend
        store.reranker._load()
    seen = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with out_path.open("w", encoding="utf-8") as handle:
        for name, body in characters:
            if seen >= limit:
                break
            docs = memory_docs(memories[name])
            user_id = f"perltqa-{name}"
            questions = list(iter_questions(body))
            embedder.embed_many([item["Question"] for _, item in questions])
            for section, item in questions:
                if seen >= limit:
                    break
                refs = gold_ids(section, item)
                snippets = [needle(docs[ref]) for ref in refs if docs.get(ref)]
                if not snippets:
                    continue
                rows = store.search(user_id, item["Question"], 100)
                ranks = [rank_of(rows, snippet) for snippet in snippets]
                worst = None if any(rank is None for rank in ranks) else max(ranks)
                record = {
                    "i": seen,
                    "character": name,
                    "section": section,
                    "recall_at_1": 1.0 if worst == 0 else 0.0,
                    "recall_at_5": 1.0 if worst is not None and worst < 5 else 0.0,
                    "mrr": 0.0 if worst is None else 1.0 / (worst + 1),
                    "rank": worst,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                seen += 1
            handle.flush()
            print(f"{side} {name} n={seen} elapsed={time.time() - started:.0f}s", flush=True)
    store.close()
    print(f"{side} DONE n={seen}", flush=True)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def merge(v1_path: Path, v2_path: Path, out_path: Path) -> None:
    left = _read_jsonl(v1_path)
    right = _read_jsonl(v2_path)
    if len(left) != len(right):
        raise SystemExit(f"row mismatch v1={len(left)} v2={len(right)}")
    paired = []
    for a, b in zip(left, right):
        if a["i"] != b["i"] or a["section"] != b["section"] or a["character"] != b["character"]:
            raise SystemExit(f"alignment mismatch at {a['i']}")
        paired.append((a, b))
    sections = ("profile", "social_relationship", "events", "dialogues")
    by_section = {}
    for section in sections:
        group = [(a, b) for a, b in paired if a["section"] == section]
        by_section[section] = {
            "v1": _rates([a for a, _ in group]),
            "v2": _rates([b for _, b in group]),
        }
    characters = []
    names = []
    for a, _b in paired:
        if a["character"] not in names:
            names.append(a["character"])
    for name in names:
        group = [(a, b) for a, b in paired if a["character"] == name]
        characters.append(
            {
                "character": name,
                "v1": _rates([a for a, _ in group]),
                "v2": _rates([b for _, b in group]),
            }
        )
    both = v2_only = v1_only = neither = 0
    for a, b in paired:
        if a["recall_at_1"] and b["recall_at_1"]:
            both += 1
        elif b["recall_at_1"]:
            v2_only += 1
        elif a["recall_at_1"]:
            v1_only += 1
        else:
            neither += 1
    report = {
        "dataset": "PerLTQA zh",
        "mode": "search-only on memories stored 2026-10-07",
        "v1": "agent-memory-challenge- v1 @ 0a18d65",
        "v2": "副本 with cross-encoder blend 0.8",
        "n": len(paired),
        "overall": {"v1": _rates([a for a, _ in paired]), "v2": _rates([b for _, b in paired])},
        "rank1_overlap": {"both": both, "v2_only": v2_only, "v1_only": v1_only, "neither": neither},
        "by_section": by_section,
        "by_character": characters,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"n": report["n"], "overall": report["overall"], "rank1_overlap": report["rank1_overlap"]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--side", choices=("v1", "v2", "v3", "v4", "v5"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--v1-jsonl", type=Path)
    parser.add_argument("--v2-jsonl", type=Path)
    args = parser.parse_args()
    if args.merge:
        merge(args.v1_jsonl, args.v2_jsonl, args.out)
        return
    if not args.side or not args.data_dir or not args.out:
        raise SystemExit("need --side, --data-dir, and --out")
    run_side(args.side, args.limit, args.data_dir, args.out)


if __name__ == "__main__":
    main()
