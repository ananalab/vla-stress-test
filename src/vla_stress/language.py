"""Instruction canonicalisation: map a free-form instruction to the closest training instruction.

SmolVLA follows the instructions it was trained on but not paraphrases of them. Before the
instruction reaches the policy, we embed it with a small sentence encoder and replace it by the
most similar of the 40 LIBERO training instructions. Below a similarity threshold, fixed before
looking at any result, the instruction is left unchanged: an unrelated request ("sing a song")
must not be turned into a robot task.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from vla_stress.env_utils import task_instructions

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
THRESHOLD = 0.5


class Canonicalizer:
    def __init__(self, threshold: float = THRESHOLD, encoder: str = ENCODER):
        self.threshold = threshold
        self.tok = AutoTokenizer.from_pretrained(encoder)
        self.model = AutoModel.from_pretrained(encoder).eval()
        self.candidates = sorted({i for s in SUITES for i in task_instructions(s)})
        self.cand_emb = self.embed(self.candidates)

    @torch.no_grad()
    def embed(self, texts: list[str]) -> np.ndarray:
        """Mean-pooled, L2-normalised sentence embeddings (the encoder's intended usage)."""
        b = self.tok(texts, padding=True, truncation=True, return_tensors="pt")
        h = self.model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        e = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(e, dim=-1).numpy()

    def similarity(self, a: str, b: str) -> float:
        ea, eb = self.embed([a, b])
        return float(ea @ eb)

    def __call__(self, text: str) -> tuple[str, float, bool]:
        """Return (instruction to use, similarity to the closest candidate, accepted)."""
        if not text.strip():
            return text, 0.0, False
        sims = self.cand_emb @ self.embed([text])[0]
        k = int(np.argmax(sims))
        accepted = bool(sims[k] >= self.threshold)
        return (self.candidates[k] if accepted else text), float(sims[k]), accepted
