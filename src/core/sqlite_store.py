from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS karten (
    id TEXT PRIMARY KEY,
    oracle_id TEXT,
    name TEXT,
    "set" TEXT,
    set_name TEXT,
    collector_number TEXT,
    lang TEXT,
    type_line TEXT,
    oracle_text TEXT,
    mana_cost TEXT,
    cmc INTEGER,
    colors TEXT,
    color_identity TEXT,
    rarity TEXT,
    image_uris TEXT,
    legalities TEXT
);

CREATE TABLE IF NOT EXISTS card_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scryfall_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    source TEXT NOT NULL,
    language TEXT,
    is_training INTEGER NOT NULL DEFAULT 1,
    UNIQUE (scryfall_id, file_path),
    FOREIGN KEY (scryfall_id) REFERENCES karten(id)
);

CREATE TABLE IF NOT EXISTS card_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scryfall_id TEXT NOT NULL,
    image_id INTEGER,
    mode TEXT NOT NULL,
    aug_index INTEGER NOT NULL,
    emb BLOB NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (scryfall_id) REFERENCES karten(id),
    FOREIGN KEY (image_id) REFERENCES card_images(id)
);
"""


def _json_dumps(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def _json_loads(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


class SqliteEmbeddingStore:
    def __init__(self, db_path: str, emb_dim: int):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.emb_dim = int(emb_dim)
        self.ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def upsert_card(self, card: Dict[str, Any]) -> None:
        payload = {
            "id": card.get("id"),
            "oracle_id": card.get("oracle_id"),
            "name": card.get("name"),
            "set": card.get("set"),
            "set_name": card.get("set_name"),
            "collector_number": card.get("collector_number"),
            "lang": card.get("lang"),
            "type_line": card.get("type_line"),
            "oracle_text": card.get("oracle_text"),
            "mana_cost": card.get("mana_cost"),
            "cmc": card.get("cmc"),
            "colors": _json_dumps(card.get("colors")),
            "color_identity": _json_dumps(card.get("color_identity")),
            "rarity": card.get("rarity"),
            "image_uris": _json_dumps(card.get("image_uris")),
            "legalities": _json_dumps(card.get("legalities")),
        }
        if not payload["id"]:
            raise ValueError("Card muss ein 'id' Feld besitzen (Scryfall ID).")
        columns = ", ".join(payload.keys())
        quoted_columns = ", ".join([f"\"{col}\"" for col in payload.keys()])
        placeholders = ", ".join([":" + key for key in payload.keys()])
        update_clause = ", ".join([f"\"{col}\"=excluded.\"{col}\"" for col in payload.keys() if col != "id"])
        sql = (
            f"INSERT INTO karten ({quoted_columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {update_clause}"
        )
        with self._connect() as conn:
            conn.execute(sql, payload)
            conn.commit()

    def get_or_create_image(
        self,
        scryfall_id: str,
        file_path: str,
        source: str,
        language: Optional[str],
        is_training: bool,
    ) -> int:
        if not scryfall_id:
            raise ValueError("scryfall_id ist erforderlich.")
        payload = {
            "scryfall_id": scryfall_id,
            "file_path": file_path,
            "source": source,
            "language": language,
            "is_training": int(bool(is_training)),
        }
        sql = """
        INSERT INTO card_images (scryfall_id, file_path, source, language, is_training)
        VALUES (:scryfall_id, :file_path, :source, :language, :is_training)
        ON CONFLICT(scryfall_id, file_path)
        DO UPDATE SET language=excluded.language, source=excluded.source, is_training=excluded.is_training
        """
        with self._connect() as conn:
            cur = conn.execute(sql, payload)
            if cur.lastrowid:
                conn.commit()
                return int(cur.lastrowid)
            # Row existed, fetch id
            cur = conn.execute(
                "SELECT id FROM card_images WHERE scryfall_id = ? AND file_path = ?",
                (scryfall_id, file_path),
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError("Konnte card_images Datensatz nicht abrufen.")
            return int(row["id"])

    def clear_embeddings(self, mode: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM card_embeddings WHERE mode = ?", (mode,))
            conn.commit()

    def add_embedding(
        self,
        scryfall_id: str,
        vec: np.ndarray,
        mode: str,
        aug_index: int,
        image_id: Optional[int] = None,
    ) -> int:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size != self.emb_dim:
            raise ValueError(f"Erwartete Embedding-Dimension {self.emb_dim}, erhalten {arr.size}")
        payload = {
            "scryfall_id": scryfall_id,
            "image_id": image_id,
            "mode": mode,
            "aug_index": int(aug_index),
            "emb": arr.astype(np.float32, copy=False).tobytes(),
        }
        sql = """
        INSERT INTO card_embeddings (scryfall_id, image_id, mode, aug_index, emb)
        VALUES (:scryfall_id, :image_id, :mode, :aug_index, :emb)
        """
        with self._connect() as conn:
            cur = conn.execute(sql, payload)
            conn.commit()
            return int(cur.lastrowid)


def _decode_meta_row(row: sqlite3.Row) -> Dict[str, Any]:
    colors = _json_loads(row["colors"]) or []
    color_identity = _json_loads(row["color_identity"]) or []
    image_uris = _json_loads(row["image_uris"]) or {}
    legalities = _json_loads(row["legalities"]) or {}
    return {
        "scryfall_id": row["id"],
        "name": row["name"],
        "set": row["set"],
        "set_name": row["set_name"],
        "collector_number": row["collector_number"],
        "lang": row["lang"],
        "mana_cost": row["mana_cost"],
        "cmc": row["cmc"],
        "colors": colors,
        "color_identity": color_identity,
        "rarity": row["rarity"],
        "type_line": row["type_line"],
        "oracle_text": row["oracle_text"],
        "image_uris": image_uris,
        "legalities": legalities,
    }


def load_embeddings_with_meta(
    sqlite_path: str,
    mode: str,
    emb_dim: int,
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Dict[str, Any]]]:
    """
    Laedt alle Embeddings aus card_embeddings fuer den gegebenen mode.
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    query = """
    SELECT
        ce.scryfall_id,
        ce.emb,
        ce.aug_index,
        ci.file_path AS image_path,
        ci.language AS image_language,
        k.*
    FROM card_embeddings AS ce
    LEFT JOIN card_images AS ci ON ce.image_id = ci.id
    LEFT JOIN karten AS k ON ce.scryfall_id = k.id
    WHERE ce.mode = ?
    ORDER BY ce.scryfall_id, ce.aug_index, ce.id
    """
    embeddings_by_card: Dict[str, List[np.ndarray]] = {}
    meta_by_card: Dict[str, Dict[str, Any]] = {}
    with conn:
        for row in conn.execute(query, (mode,)):
            cid = row["scryfall_id"]
            if cid is None:
                continue
            vec = np.frombuffer(row["emb"], dtype=np.float32)
            if vec.size != emb_dim:
                raise ValueError(f"Falsche Embedding-Dimension fuer {cid}: {vec.size} != {emb_dim}")
            embeddings_by_card.setdefault(cid, []).append(vec)
            if cid not in meta_by_card:
                meta = _decode_meta_row(row) if row["id"] else {"scryfall_id": cid}
                meta.setdefault("image_paths", [])
                meta_by_card[cid] = meta
            meta = meta_by_card[cid]
            paths: List[str] = meta.setdefault("image_paths", [])
            img_path = row["image_path"]
            if img_path and img_path not in paths:
                paths.append(img_path)
            if not meta.get("lang") and row["image_language"]:
                meta["lang"] = row["image_language"]
        # Fallback: falls kein Bildpfad vorhanden (z. B. nur Zentroiden), hole erstmals verfügbares card_image
        missing_paths = [cid for cid, meta in meta_by_card.items() if not meta.get("image_paths")]
        if missing_paths:
            placeholder = ",".join("?" for _ in missing_paths)
            img_query = f"SELECT scryfall_id, file_path, language FROM card_images WHERE scryfall_id IN ({placeholder})"
            for img_row in conn.execute(img_query, missing_paths):
                cid = img_row[0]
                meta = meta_by_card.get(cid)
                if meta is None:
                    continue
                paths: List[str] = meta.setdefault("image_paths", [])
                if img_row[1] and img_row[1] not in paths:
                    paths.append(img_row[1])
                if not meta.get("lang") and img_row[2]:
                    meta["lang"] = img_row[2]
    conn.close()
    return embeddings_by_card, meta_by_card


def load_flat_samples(
    sqlite_path: str,
    mode: str,
    emb_dim: int,
) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    embeddings_by_card, meta_by_card = load_embeddings_with_meta(sqlite_path, mode, emb_dim)
    all_vectors: List[np.ndarray] = []
    labels: List[str] = []
    metas: List[Dict[str, Any]] = []
    for cid, vectors in embeddings_by_card.items():
        card_meta = meta_by_card.get(cid, {"scryfall_id": cid})
        label = card_meta.get("name") or cid
        for vec in vectors:
            all_vectors.append(vec)
            labels.append(label)
            metas.append(card_meta)
    if not all_vectors:
        return np.zeros((0, emb_dim), dtype=np.float32), labels, metas
    X = np.stack(all_vectors, axis=0)
    return X, labels, metas
