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
    legalities TEXT,
    price_usd REAL,
    price_usd_foil REAL,
    price_eur REAL,
    price_eur_foil REAL,
    price_tix REAL
);

CREATE TABLE IF NOT EXISTS card_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scryfall_id TEXT NOT NULL,
    oracle_id TEXT,
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
    oracle_id TEXT,
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


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cur.fetchall())


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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
            # Nachziehen neuer Spalten ohne bestehende Daten zu verlieren.
            if not _column_exists(conn, "card_images", "oracle_id"):
                conn.execute("ALTER TABLE card_images ADD COLUMN oracle_id TEXT")
            if not _column_exists(conn, "card_embeddings", "oracle_id"):
                conn.execute("ALTER TABLE card_embeddings ADD COLUMN oracle_id TEXT")
            # Unique-Index auf oracle_id + file_path, damit reprints zusammenlaufen.
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_card_images_oracle_file "
                "ON card_images(oracle_id, file_path)"
            )
            # Vorhandene Daten mit oracle_id aus karten befüllen.
            conn.execute(
                """
                UPDATE card_images
                SET oracle_id = (
                    SELECT oracle_id FROM karten WHERE karten.id = card_images.scryfall_id
                )
                WHERE oracle_id IS NULL OR oracle_id = ''
                """
            )
            conn.execute(
                """
                UPDATE card_embeddings
                SET oracle_id = (
                    SELECT oracle_id FROM karten WHERE karten.id = card_embeddings.scryfall_id
                )
                WHERE oracle_id IS NULL OR oracle_id = ''
                """
            )
            # Preise nachziehen, falls Spalten fehlen.
            for col in (
                ("price_usd", "REAL"),
                ("price_usd_foil", "REAL"),
                ("price_eur", "REAL"),
                ("price_eur_foil", "REAL"),
                ("price_tix", "REAL"),
            ):
                if not _column_exists(conn, "karten", col[0]):
                    conn.execute(f"ALTER TABLE karten ADD COLUMN {col[0]} {col[1]}")
            conn.commit()

    def upsert_card(self, card: Dict[str, Any]) -> None:
        prices = card.get("prices") or {}
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
            "price_usd": _to_float(prices.get("usd")),
            "price_usd_foil": _to_float(prices.get("usd_foil")),
            "price_eur": _to_float(prices.get("eur")),
            "price_eur_foil": _to_float(prices.get("eur_foil")),
            "price_tix": _to_float(prices.get("tix")),
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
        scryfall_id: Optional[str],
        file_path: str,
        source: str,
        language: Optional[str],
        is_training: bool,
        oracle_id: Optional[str] = None,
    ) -> int:
        if not scryfall_id and not oracle_id:
            raise ValueError("scryfall_id oder oracle_id ist erforderlich.")
        # oracle_id dient als gemeinsamer Schluessel; falls keine Print-ID vorhanden ist,
        # verwenden wir sie auch als scryfall_id, um die NOT NULL-Constraint zu erfuellen.
        payload = {
            "scryfall_id": scryfall_id or oracle_id,
            "oracle_id": oracle_id or scryfall_id,
            "file_path": file_path,
            "source": source,
            "language": language,
            "is_training": int(bool(is_training)),
        }
        sql = """
        INSERT INTO card_images (scryfall_id, oracle_id, file_path, source, language, is_training)
        VALUES (:scryfall_id, :oracle_id, :file_path, :source, :language, :is_training)
        ON CONFLICT(oracle_id, file_path)
        DO UPDATE SET language=excluded.language, source=excluded.source, is_training=excluded.is_training
        """
        with self._connect() as conn:
            cur = conn.execute(sql, payload)
            if cur.lastrowid:
                conn.commit()
                return int(cur.lastrowid)
            # Row existed, fetch id
            cur = conn.execute(
                "SELECT id FROM card_images WHERE oracle_id = ? AND file_path = ?",
                (payload["oracle_id"], file_path),
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
        scryfall_id: Optional[str],
        vec: np.ndarray,
        mode: str,
        aug_index: int,
        image_id: Optional[int] = None,
        oracle_id: Optional[str] = None,
    ) -> int:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if arr.size != self.emb_dim:
            raise ValueError(f"Erwartete Embedding-Dimension {self.emb_dim}, erhalten {arr.size}")
        payload = {
            "scryfall_id": scryfall_id or oracle_id,
            "oracle_id": oracle_id or scryfall_id,
            "image_id": image_id,
            "mode": mode,
            "aug_index": int(aug_index),
            "emb": arr.astype(np.float32, copy=False).tobytes(),
        }
        sql = """
        INSERT INTO card_embeddings (scryfall_id, oracle_id, image_id, mode, aug_index, emb)
        VALUES (:scryfall_id, :oracle_id, :image_id, :mode, :aug_index, :emb)
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
        "oracle_id": row["oracle_id"] or row["id"],
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
        "price_usd": row["price_usd"],
        "price_usd_foil": row["price_usd_foil"],
        "price_eur": row["price_eur"],
        "price_eur_foil": row["price_eur_foil"],
        "price_tix": row["price_tix"],
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
        ce.oracle_id AS emb_oracle_id,
        ce.emb,
        ce.aug_index,
        ci.file_path AS image_path,
        ci.language AS image_language,
        ci.oracle_id AS img_oracle_id,
        k.*,
        k.oracle_id AS card_oracle_id
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
            oracle_id = row["emb_oracle_id"] or row["img_oracle_id"] or row["card_oracle_id"]
            cid = oracle_id or row["scryfall_id"]
            if cid is None:
                continue
            vec = np.frombuffer(row["emb"], dtype=np.float32)
            if vec.size != emb_dim:
                raise ValueError(f"Falsche Embedding-Dimension fuer {cid}: {vec.size} != {emb_dim}")
            embeddings_by_card.setdefault(cid, []).append(vec)
            if cid not in meta_by_card:
                if row["id"]:
                    meta = _decode_meta_row(row)
                    if oracle_id:
                        meta["oracle_id"] = oracle_id
                else:
                    meta = {"scryfall_id": cid, "oracle_id": cid}
                meta.setdefault("image_paths", [])
                meta_by_card[cid] = meta
            meta = meta_by_card[cid]
            paths: List[str] = meta.setdefault("image_paths", [])
            img_path = row["image_path"]
            if img_path and img_path not in paths:
                paths.append(img_path)
            if not meta.get("lang") and row["image_language"]:
                meta["lang"] = row["image_language"]
        # Fallback: falls kein Bildpfad vorhanden (z. B. nur Zentroiden), hole erstmals verfuegbares card_image
        missing_paths = [cid for cid, meta in meta_by_card.items() if not meta.get("image_paths")]
        if missing_paths:
            placeholder = ",".join("?" for _ in missing_paths)
            img_query = (
                f"SELECT COALESCE(oracle_id, scryfall_id) AS card_id, file_path, language "
                f"FROM card_images WHERE (oracle_id IN ({placeholder}) OR scryfall_id IN ({placeholder}))"
            )
            img_query = "".join(img_query)
            params = tuple(missing_paths) + tuple(missing_paths)
            for img_row in conn.execute(img_query, params):
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
