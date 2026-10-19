"""Fine-tune the cross-encoder so evidence outranks BM25 hard negatives.

Each example's query already contains the question and five BM25 passages.
One positive and four negatives share that query. The loss is the log loss of
the positive's share of exp(score).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


class Pairs(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def collate(rows: list[dict], tokenizer, max_length: int):
    queries, docs = [], []
    for row in rows:
        docs_for_row = [row["positive"], *row["negatives"]]
        queries.extend([row["query"]] * len(docs_for_row))
        docs.extend(docs_for_row)
    encoded = tokenizer(
        queries,
        docs,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return encoded, len(rows[0]["negatives"]) + 1


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total = 0.0
    seen = 0
    for batch, width in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        scores = model(**batch).logits.squeeze(-1).view(-1, width)
        loss = torch.nn.functional.cross_entropy(scores, torch.zeros(scores.size(0), dtype=torch.long, device=device))
        total += float(loss) * scores.size(0)
        seen += scores.size(0)
    model.train()
    return total / max(seen, 1)


def main() -> None:
    root = Path(__file__).resolve().parent / "v3"
    model_path = Path(__file__).resolve().parent / "v3" / "base"
    out = root / "model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_data = Pairs(root / "train.jsonl")
    dev_data = Pairs(root / "dev.jsonl")
    rng = random.Random(7)
    order = list(range(len(train_data)))
    rng.shuffle(order)

    def train_loader():
        rows = [train_data[index] for index in order]
        return DataLoader(rows, batch_size=8, shuffle=False, collate_fn=lambda batch: collate(batch, tokenizer, 256))

    dev_loader = DataLoader(list(dev_data), batch_size=8, shuffle=False, collate_fn=lambda batch: collate(batch, tokenizer, 256))
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps = 2 * max(len(train_data) // 8, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    best = None
    model.train()
    for epoch in range(2):
        rng.shuffle(order)
        running = 0.0
        seen = 0
        for batch, width in train_loader():
            batch = {key: value.to(device) for key, value in batch.items()}
            scores = model(**batch).logits.squeeze(-1).view(-1, width)
            loss = torch.nn.functional.cross_entropy(scores, torch.zeros(scores.size(0), dtype=torch.long, device=device))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss) * scores.size(0)
            seen += scores.size(0)
        dev = evaluate(model, dev_loader, device)
        print(f"epoch={epoch} train_loss={running / max(seen, 1):.4f} dev_loss={dev:.4f}", flush=True)
        if best is None or dev < best:
            best = dev
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)
    print(f"best_dev_loss={best:.4f} wrote {out}", flush=True)


if __name__ == "__main__":
    main()
