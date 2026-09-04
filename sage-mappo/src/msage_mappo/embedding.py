from __future__ import annotations

import hashlib
import re

import numpy as np


class HashTextEmbedder:
    """Dependency-light text embedder for memory retrieval experiments."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        if not tokens:
            return vec
        for tok in tokens:
            digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def encode_constraint_context(self, context: dict) -> np.ndarray:
        tags = context.get("tags", [])
        fields = [
            f"deadline_margin_{bucket(context.get('deadline_margin', 0.0))}",
            f"queue_pressure_{bucket(context.get('queue_pressure', 0.0))}",
            f"power_usage_{bucket(context.get('avg_power', 0.0))}",
        ]
        return self.encode(" ".join(fields + list(tags)))


def bucket(value: float) -> str:
    if value < 0.2:
        return "low"
    if value < 0.6:
        return "medium"
    return "high"
