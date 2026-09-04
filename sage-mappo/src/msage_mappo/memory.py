from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import time

import numpy as np

from .embedding import HashTextEmbedder


@dataclass
class SemanticMemoryItem:
    scenario: str
    cause: str
    bad_action: str
    good_action: str
    constraint_tags: list[str]
    priority: str
    outcome: str
    stats: dict[str, float] = field(default_factory=dict)
    writer_backend: str = "unknown"
    raw_llm_output: str = ""
    created_at: float = field(default_factory=time.time)

    def to_text(self) -> str:
        return (
            f"scenario: {self.scenario}. cause: {self.cause}. "
            f"bad_action: {self.bad_action}. good_action: {self.good_action}. "
            f"constraint_tags: {' '.join(self.constraint_tags)}. "
            f"priority: {self.priority}. outcome: {self.outcome}."
        )


class SemanticMemoryBank:
    def __init__(self, dim: int = 64, top_k: int = 3, constraint_bonus: float = 0.25):
        self.embedder = HashTextEmbedder(dim=dim)
        self.dim = dim
        self.top_k = top_k
        self.constraint_bonus = constraint_bonus
        self.items: list[SemanticMemoryItem] = []
        self.embeddings = np.zeros((0, dim), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.items)

    def add(self, item: SemanticMemoryItem) -> None:
        emb = self.embedder.encode(item.to_text())
        self.items.append(item)
        self.embeddings = np.vstack([self.embeddings, emb[None, :]])

    def add_many(self, items: list[SemanticMemoryItem]) -> None:
        for item in items:
            self.add(item)

    def retrieve(self, query_text: str, constraint_context: dict, top_k: int | None = None):
        if len(self.items) == 0:
            k = top_k or self.top_k
            return np.zeros((k, self.dim), dtype=np.float32), []

        query = self.embedder.encode(query_text)
        sims = self.embeddings @ query
        current_tags = set(constraint_context.get("tags", []))
        bonus = np.zeros_like(sims)
        for idx, item in enumerate(self.items):
            overlap = current_tags.intersection(item.constraint_tags)
            if overlap:
                bonus[idx] = self.constraint_bonus * len(overlap)
        scores = sims + bonus
        k = min(top_k or self.top_k, len(self.items))
        order = np.argsort(scores)[-k:][::-1]
        return self.embeddings[order].astype(np.float32), [self.items[int(i)] for i in order]

    def save_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path, dim: int = 64, top_k: int = 3):
        bank = cls(dim=dim, top_k=top_k)
        path = Path(path)
        if not path.exists():
            return bank
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    bank.add(SemanticMemoryItem(**json.loads(line)))
        return bank
