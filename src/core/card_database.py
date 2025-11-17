from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.config_utils import load_config
from src.core.sqlite_store import load_embeddings_with_meta


class SimpleCardDB:
    """SQLite-basierte Embedding-Datenbank fuer die Laufzeit-Erkennung."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        config_path: str = "config.yaml",
        mode: str = "runtime",
        emb_dim: Optional[int] = None,
        load_existing: bool = True,
    ):
        if db_path:
            self.db_path = Path(db_path)
        else:
            cfg = load_config(config_path)
            self.db_path = Path(cfg.get("database", {}).get("sqlite_path", "tcg_database/database/karten.db"))
            emb_dim = emb_dim or cfg.get("encoder", {}).get("emb_dim") or cfg.get("model", {}).get("embed_dim")
        self.mode = mode
        self.emb_dim = int(emb_dim or 1024)
        self.cards: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.meta: Dict[str, Any] = {"mode": self.mode, "sqlite_path": str(self.db_path)}
        if load_existing:
            self.load_from_sqlite()

    def load_from_sqlite(self) -> None:
        if not self.db_path.exists():
            self.cards = []
            self.embeddings = None
            return
        embeddings_by_card, meta_by_card = load_embeddings_with_meta(str(self.db_path), self.mode, self.emb_dim)
        cards: List[Dict[str, Any]] = []
        vectors: List[np.ndarray] = []
        for cid, vecs in embeddings_by_card.items():
            meta = meta_by_card.get(cid, {})
            base = {
                "card_uuid": cid,
                "name": meta.get("name") or cid,
                "set_code": meta.get("set") or "",
                "collector_number": meta.get("collector_number") or "",
                "image_path": (meta.get("image_paths") or [None])[0] or "",
                "lang": meta.get("lang"),
            }
            for vec in vecs:
                cards.append(base.copy())
                vectors.append(np.asarray(vec, dtype=np.float32).flatten())
        self.cards = cards
        if vectors:
            arr = np.stack(vectors, axis=0)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            self.embeddings = arr / np.clip(norms, 1e-12, None)
        else:
            self.embeddings = None

    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.7) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.cards) == 0:
            return []
        query = query_embedding.flatten().reshape(1, -1)
        db_norm = np.linalg.norm(self.embeddings, axis=1, keepdims=True).clip(min=1e-12)
        q_norm = np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-12)
        normalized_db = self.embeddings / db_norm
        normalized_q = query / q_norm
        sims = (normalized_q @ normalized_db.T)[0]
        indices = np.argsort(sims)[::-1]
        results: List[Dict[str, Any]] = []
        for idx in indices[:top_k]:
            sim = float(sims[idx])
            if sim < threshold:
                continue
            card = self.cards[idx].copy()
            card["similarity"] = sim
            card["distance"] = 1.0 - sim
            results.append(card)
        return results
