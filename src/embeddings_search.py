"""
embeddings_search.py
---------------------
Bonus feature: semantic search over extracted clauses.

Builds a small in-memory embedding index over every clause excerpt
extracted across all processed contracts, so you can ask things like
"which contracts let the customer terminate without cause?" and get back
the most semantically similar clause excerpts, ranked by cosine similarity,
instead of relying on exact keyword matches.

Uses a local sentence-transformers model (all-MiniLM-L6-v2) so semantic
search works offline / without extra API calls or cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ClauseRecord:
    contract_id: str
    clause_type: str
    excerpt: str


class ClauseSearchIndex:
    """A minimal embedding index over clause excerpts."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy-loaded, sentence-transformers import is slow
        self.records: list[ClauseRecord] = []
        self.embeddings: Optional[np.ndarray] = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build(self, results: list[dict]) -> None:
        """
        Build the index from pipeline results.

        Parameters
        ----------
        results : list[dict]
            Each dict is expected to have "contract_id" and clause fields
            (termination_clause / confidentiality_clause / liability_clause),
            each with an "excerpts" list -- i.e. the same shape produced by
            pipeline.run_pipeline().
        """
        model = self._load_model()

        self.records = []
        for row in results:
            contract_id = row["contract_id"]
            for clause_type in ("termination_clause", "confidentiality_clause", "liability_clause"):
                clause = row.get(clause_type) or {}
                for excerpt in clause.get("excerpts", []):
                    if excerpt:
                        self.records.append(ClauseRecord(contract_id, clause_type, excerpt))

        if not self.records:
            self.embeddings = np.zeros((0, 384))
            return

        texts = [r.excerpt for r in self.records]
        self.embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k clause excerpts most semantically similar to `query`."""
        if self.embeddings is None or len(self.records) == 0:
            return []

        model = self._load_model()
        query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]

        scores = self.embeddings @ query_vec  # cosine similarity (vectors are normalized)
        top_idx = np.argsort(-scores)[:top_k]

        return [
            {
                "contract_id": self.records[i].contract_id,
                "clause_type": self.records[i].clause_type,
                "excerpt": self.records[i].excerpt,
                "score": float(scores[i]),
            }
            for i in top_idx
        ]
