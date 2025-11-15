from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.config_utils import load_config


class SimpleCardDB:
    """Einfache JSON-basierte Datenbank für Karten + Embeddings."""

    def __init__(self, db_path: Optional[str] = None, config_path: str = "config.yaml", load_existing: bool = True):
        if db_path:
            self.db_path = Path(db_path)
        else:
            cfg = load_config(config_path)
            self.db_path = Path(cfg.get("database", {}).get("path", "./data/cards.json"))
        self.cards: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.meta: Dict[str, Any] = {}
        if load_existing:
            self.load_from_file()

    def load_from_file(self) -> None:
        if not self.db_path.exists():
            return
        with self.db_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "cards" in payload and "embeddings" in payload:
            self.cards = payload.get("cards", [])
            embeddings = payload.get("embeddings", [])
            self.embeddings = np.array(embeddings, dtype=np.float32) if embeddings else None
            self.meta = payload.get("meta", {})
            return
        if isinstance(payload, dict):
            cards: List[Dict[str, Any]] = []
            embeddings: List[List[float]] = []
            for key, value in payload.items():
                if not isinstance(value, list):
                    continue
                cards.append(
                    {
                        "card_uuid": key,
                        "name": key,
                        "set_code": "",
                        "collector_number": "",
                        "image_path": key,
                    }
                )
                embeddings.append(value)
            self.cards = cards
            self.embeddings = np.array(embeddings, dtype=np.float32) if embeddings else None
            self.meta = {}
            return

        self.cards = []
        self.embeddings = None
        self.meta = {}

    def save_to_file(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cards": self.cards,
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else [],
            "meta": self.meta,
        }
        with self.db_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def add_card(
        self,
        card_uuid: str,
        name: str,
        set_code: str,
        collector_number: str,
        image_path: str,
        embedding: np.ndarray,
    ) -> None:
        for idx, card in enumerate(self.cards):
            if card["card_uuid"] == card_uuid:
                self.cards[idx] = {
                    "card_uuid": card_uuid,
                    "name": name,
                    "set_code": set_code,
                    "collector_number": collector_number,
                    "image_path": image_path,
                }
                if self.embeddings is not None:
                    self.embeddings[idx] = embedding.flatten()
                return

        self.cards.append(
            {
                "card_uuid": card_uuid,
                "name": name,
                "set_code": set_code,
                "collector_number": collector_number,
                "image_path": image_path,
            }
        )
        embedding_flat = embedding.flatten()
        if self.embeddings is None:
            self.embeddings = embedding_flat.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding_flat])

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
