from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple
import os
import json
import yaml
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Lädt YAML-Konfiguration direkt"""
    if not os.path.exists(config_path):
        return {"database": {"path": "./data/cards.json"}}  # Fallback
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


class SimpleCardDB:
    """Einfache In-Memory Kartendatenbank mit JSON-Persistierung"""
    
    def __init__(
        self,
        db_path: str = None,
        config_path: str = "config.yaml",
        load_existing: bool = True,
    ):
        # Entweder direkter Pfad oder aus Config
        if db_path:
            self.db_path = Path(db_path)
        else:
            config = load_config(config_path)
            self.db_path = Path(config.get('database', {}).get('path', './data/cards.json'))

        self.cards: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.meta: Dict[str, Any] = {}
        if load_existing:
            self.load_from_file()
    
    def load_from_file(self):
        """Lädt Kartendaten aus JSON-Datei"""
        if not self.db_path.exists():
            return
        with open(self.db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.cards = data.get('cards', [])
        embeddings_list = data.get('embeddings', [])
        self.embeddings = np.array(embeddings_list, dtype=np.float32) if embeddings_list else None
        self.meta = data.get('meta', {})
    
    def save_to_file(self):
        """Speichert Kartendaten in JSON-Datei"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'cards': self.cards,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else [],
            'meta': self.meta,
        }
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def add_card(self, card_uuid: str, name: str, set_code: str, 
                 collector_number: str, image_path: str, embedding: np.ndarray):
        """Fügt eine neue Karte hinzu"""
        # Prüfe ob Karte bereits existiert
        for i, card in enumerate(self.cards):
            if card['card_uuid'] == card_uuid:
                # Update existierende Karte
                self.cards[i] = {
                    'card_uuid': card_uuid,
                    'name': name,
                    'set_code': set_code,
                    'collector_number': collector_number,
                    'image_path': image_path
                }
                self.embeddings[i] = embedding.flatten()
                return
        
        # Neue Karte hinzufügen
        self.cards.append({
            'card_uuid': card_uuid,
            'name': name,
            'set_code': set_code,
            'collector_number': collector_number,
            'image_path': image_path
        })
        
        # Embedding hinzufügen
        embedding_flat = embedding.flatten()
        if self.embeddings is None:
            self.embeddings = embedding_flat.reshape(1, -1)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding_flat])
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Sucht ähnliche Karten basierend auf Embedding"""
        if self.embeddings is None or len(self.cards) == 0:
            return []
        
        query_flat = query_embedding.flatten().reshape(1, -1)
        similarities = cosine_similarity(query_flat, self.embeddings)[0]
        
        # Sortiere nach Ähnlichkeit (höher = ähnlicher)
        indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in indices[:top_k]:
            similarity = similarities[idx]
            if similarity >= threshold:
                result = self.cards[idx].copy()
                result['similarity'] = float(similarity)
                result['distance'] = 1.0 - similarity  # Für Kompatibilität
                results.append(result)
        
        return results
    
    def get_card_count(self) -> int:
        """Gibt Anzahl der gespeicherten Karten zurück"""
        return len(self.cards)


# Globale Datenbankinstanz
_db_instance = None

def get_db() -> SimpleCardDB:
    """Gibt Datenbankinstanz zurück (Singleton)"""
    global _db_instance
    if _db_instance is None:
        _db_instance = SimpleCardDB()  # Verwendet Standard-Config-Pfad
    return _db_instance


def ensure_schema():
    """Erstellt Datenbankschema (für Kompatibilität)"""
    db = get_db()
    # Für JSON-DB nichts zu tun
    pass


def upsert_card(card_uuid: str, name: str, set_code: str, 
               collector_number: str, image_path: str, embedding: np.ndarray):
    """Fügt Karte in Datenbank ein"""
    db = get_db()
    db.add_card(card_uuid, name, set_code, collector_number, image_path, embedding)
    db.save_to_file()


def query_topk(query_embedding: np.ndarray, top_k: int = 3, 
              threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Führt Top-K Ähnlichkeitssuche durch"""
    db = get_db()
    threshold = threshold or 0.5  # Default threshold
    return db.search_similar(query_embedding, top_k, threshold)
