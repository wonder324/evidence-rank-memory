"""Fit retrieval fusion weights on synthetic conversations.

This does not touch leaderboard data. The official rules allow a memory system
to be tuned offline, and they forbid training on evaluation traffic.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from app.config import DEFAULT_WEIGHTS, ROOT, Settings
from app.embedder import Embedder
from app.store import MemoryStore

OUT = ROOT / "models" / "fusion_weights.json"


def episodes() -> list[dict]:
    return [
        {
            "user": "u-city",
            "messages": [
                (1, "I used to live in New York but now I live in Boston."),
                (2, "My favorite drink is tea."),
            ],
            "probes": [
                ("Which city is home now?", ["Boston"], 5),
                ("Where did I live before?", ["New York"], 5),
                ("What do you know about my preferences?", ["tea"], 5),
            ],
        },
        {
            "user": "u-hop",
            "messages": [
                (1, "I met Bob on Tuesday."),
                (2, "Bob lives in Paris."),
                (3, "I like hiking and I don't like olives."),
            ],
            "probes": [
                ("Where does the person I met on Tuesday live?", ["Bob", "Paris"], 8),
                ("What food do I not like?", ["olives"], 5),
            ],
        },
        {
            "user": "u-zh",
            "messages": [
                (1, "我以前住在杭州，现在住在南京。"),
                (2, "我叫周宁，我喜欢川菜。"),
            ],
            "probes": [
                ("我现在住在哪里？", ["南京"], 5),
                ("我以前住在哪里？", ["杭州"], 5),
                ("我叫什么？", ["周宁"], 5),
            ],
        },
        {
            "user": "u-rule",
            "messages": [
                (1, "Never book a meeting on Friday."),
                (2, "The appointment with Mina is at 15:30 on Wednesday."),
                (3, "Please remember the passphrase purple umbrella."),
            ],
            "probes": [
                ("What constraint do I have about Friday?", ["Friday"], 5),
                ("When is the appointment with Mina?", ["15:30"], 5),
                ("What passphrase did I mention?", ["purple umbrella"], 5),
            ],
        },
        {
            "user": "u-code",
            "messages": [
                (
                    1,
                    "def rebuild_index(path):\n    if not path:\n        return None\n    items = load_rows(path)\n    return sorted(items)\n\n"
                    "class Store:\n    def search(self, query):\n        return rebuild_index(query)\n",
                ),
            ],
            "probes": [
                ("Where is rebuild_index defined?", ["rebuild_index"], 3),
            ],
        },
        {
            "user": "u-distract",
            "messages": [
                (1, "I visited Boston for a day."),
                (2, "The Boston conference ran late, and Boston came up again in the notes."),
                (3, "I talked about Boston with the Boston office."),
                (4, "I now live in Lisbon."),
            ],
            "probes": [
                ("Which city is home now?", ["Lisbon"], 5),
            ],
        },
    ]


def load(store: MemoryStore) -> list[dict]:
    probes = []
    for episode in episodes():
        messages = [
            {"role": "user", "timestamp": 1_700_000_000_000 + index * 86_400_000, "content": text}
            for index, text in episode["messages"]
        ]
        store.add(
            request_id=f"add-{episode['user']}",
            user_id=episode["user"],
            session_id=f"s-{episode['user']}",
            messages=messages,
        )
        for query, golds, k in episode["probes"]:
            probes.append({"user": episode["user"], "query": query, "golds": golds, "k": k})
    return probes


def recall(store: MemoryStore, probes: list[dict]) -> float:
    if not probes:
        return 0.0
    total = 0.0
    for probe in probes:
        rows = store.search(probe["user"], probe["query"], max(probe["k"], 10))
        contents = [str(row["content"]) for row in rows]
        gold_scores = []
        for gold in probe["golds"]:
            rank = next((index for index, content in enumerate(contents) if gold in content), None)
            gold_scores.append(0.0 if rank is None else 1.0 / (rank + 1))
        total += sum(gold_scores) / len(gold_scores)
    return total / len(probes)


def sample_weights(rng: random.Random) -> dict[str, float]:
    weights = {}
    for key, base in DEFAULT_WEIGHTS.items():
        weights[key] = max(0.0, base * rng.uniform(0.15, 2.4))
    return weights


def main() -> None:
    settings = Settings(data_dir=ROOT / "data" / "train", embedder="hash", llm_extract=False)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    db = settings.data_dir / "memory.db"
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(db) + suffix)
        if path.exists():
            path.unlink()
    store = MemoryStore(settings, Embedder(settings), DEFAULT_WEIGHTS)
    probes = load(store)
    holdout = probes[::4]
    train = [probe for probe in probes if probe not in holdout]
    rng = random.Random(7)
    candidates = [dict(DEFAULT_WEIGHTS)]
    candidates.extend(sample_weights(rng) for _ in range(80))
    best = dict(DEFAULT_WEIGHTS)
    best_score = -1.0
    for weights in candidates:
        store.weights = weights
        score = recall(store, train)
        if score > best_score:
            best_score = score
            best = dict(weights)
    keys = list(DEFAULT_WEIGHTS)
    for _ in range(2):
        for key in keys:
            current = best[key]
            for factor in (0.5, 0.8, 1.0, 1.25, 1.6):
                trial = dict(best)
                trial[key] = current * factor
                store.weights = trial
                score = recall(store, train)
                if score > best_score:
                    best_score = score
                    best = trial
    store.weights = best
    train_score = recall(store, train)
    holdout_score = recall(store, holdout)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "embedder": "hash",
                "weights": {key: round(value, 4) for key, value in best.items()},
                "train_mrr": round(train_score, 4),
                "holdout_mrr": round(holdout_score, 4),
                "probe_count": len(probes),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"train_mrr={train_score:.3f} holdout_mrr={holdout_score:.3f}")
    print(f"wrote {OUT}")
    store.close()


if __name__ == "__main__":
    main()
