from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Tuple

import numpy as np


def ensure_oracle_quality_table(sqlite_path: str) -> None:
    conn = sqlite3.connect(sqlite_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS oracle_quality (
                oracle_id TEXT PRIMARY KEY,
                suspect_overlap INTEGER NOT NULL DEFAULT 0,
                suspect_cluster_spread INTEGER NOT NULL DEFAULT 0,
                suspect_metadata_conflict INTEGER NOT NULL DEFAULT 0,
                nearest_neighbor_oracle_id TEXT,
                nearest_neighbor_dist REAL,
                intra_mean_dist REAL,
                intra_max_dist REAL,
                meta_info TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _json_loads(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except Exception:
        return None


def load_embeddings_grouped_by_oracle(
    sqlite_path: str,
    mode: str,
    emb_dim: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
    """
    Laedt alle Embeddings aus card_embeddings fuer den gegebenen mode und
    gruppiert sie per oracle_id. Liefert:
      - Dict[oracle_id, np.ndarray] mit Shape [N, emb_dim]
      - Dict[oracle_id, Dict] mit Metadaten-Aggregaten (z.B. name, mana_cost, colors ...)
    
    WICHTIG: Diese Funktion ist speziell für Oracle-basierte Analysen (z.B. Cluster-Spread).
    Für die normale Vektorsuche verwenden Sie load_embeddings_with_meta(), welches nach
    scryfall_id gruppiert (siehe sqlite_store.py).
    """
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    query = (
        """
        SELECT
            ce.scryfall_id,
            COALESCE(ce.oracle_id, k1.oracle_id) AS oracle_id,
            ce.emb,
            COALESCE(k1.name, k2.name) AS name,
            COALESCE(k1.mana_cost, k2.mana_cost) AS mana_cost,
            COALESCE(k1.cmc, k2.cmc) AS cmc,
            COALESCE(k1.colors, k2.colors) AS colors,
            COALESCE(k1.color_identity, k2.color_identity) AS color_identity,
            COALESCE(k1."set", k2."set") AS set_code,
            COALESCE(k1.set_name, k2.set_name) AS set_name,
            COALESCE(k1.lang, k2.lang) AS lang,
            ci.file_path AS image_path,
            ci.language AS image_language
        FROM card_embeddings AS ce
        LEFT JOIN karten AS k1 ON ce.scryfall_id = k1.id
        LEFT JOIN karten AS k2 ON COALESCE(ce.oracle_id, k1.oracle_id) = k2.oracle_id
        LEFT JOIN card_images AS ci ON ce.image_id = ci.id
        WHERE ce.mode = ?
        ORDER BY oracle_id
        """
    )
    groups: Dict[str, List[np.ndarray]] = {}
    meta: Dict[str, Dict] = {}
    try:
        for row in conn.execute(query, (mode,)):
            oracle_id = row["oracle_id"] or row["scryfall_id"]
            if not oracle_id:
                # falls nichts verknuepft ist, ueberspringen
                continue
            vec = np.frombuffer(row["emb"], dtype=np.float32)
            if vec.size != emb_dim:
                raise ValueError(f"Falsche Embedding-Dimension fuer {oracle_id}: {vec.size} != {emb_dim}")
            groups.setdefault(oracle_id, []).append(vec)

            # Metadaten-Aggregation: sammle Werte als Sets, um Konflikte zu erkennen
            m = meta.setdefault(
                oracle_id,
                {
                    "names": set(),
                    "mana_costs": set(),
                    "cmcs": set(),
                    "colors": set(),  # als JSON-Strings, spaeter wieder zu Liste wandelbar
                    "color_identities": set(),
                    "set_codes": set(),
                    "set_names": set(),
                    "langs": set(),
                    "image_paths": set(),
                },
            )
            if row["name"]:
                m["names"].add(str(row["name"]))
            if row["mana_cost"] is not None:
                m["mana_costs"].add(str(row["mana_cost"]))
            if row["cmc"] is not None:
                try:
                    m["cmcs"].add(float(row["cmc"]))
                except Exception:
                    pass
            cols = _json_loads(row["colors"]) if isinstance(row["colors"], str) else row["colors"]
            colid = _json_loads(row["color_identity"]) if isinstance(row["color_identity"], str) else row["color_identity"]
            if cols is not None:
                try:
                    m["colors"].add(json.dumps(cols, ensure_ascii=False, sort_keys=True))
                except Exception:
                    pass
            if colid is not None:
                try:
                    m["color_identities"].add(json.dumps(colid, ensure_ascii=False, sort_keys=True))
                except Exception:
                    pass
            if row["set_code"]:
                m["set_codes"].add(str(row["set_code"]))
            if row["set_name"]:
                m["set_names"].add(str(row["set_name"]))
            if row["lang"]:
                m["langs"].add(str(row["lang"]))
            if row["image_path"]:
                try:
                    m["image_paths"].add(str(row["image_path"]))
                except Exception:
                    pass
    finally:
        conn.close()

    # in Arrays wandeln und Metadaten vereinheitlichen
    arrays: Dict[str, np.ndarray] = {}
    meta_out: Dict[str, Dict] = {}
    for oid, vecs in groups.items():
        if not vecs:
            continue
        arrays[oid] = np.stack(vecs, axis=0)
        m = meta.get(oid, {})
        # Ableiten eines repr. Eintrags
        name = next(iter(m.get("names", [])), None)
        mana_cost = next(iter(m.get("mana_costs", [])), None)
        cmc = next(iter(m.get("cmcs", [])), None)
        colors_json = next(iter(m.get("colors", [])), None)
        color_identity_json = next(iter(m.get("color_identities", [])), None)
        set_code = next(iter(m.get("set_codes", [])), None)
        set_name = next(iter(m.get("set_names", [])), None)
        image_path = next(iter(m.get("image_paths", [])), None)
        colors = json.loads(colors_json) if colors_json else []
        color_identity = json.loads(color_identity_json) if color_identity_json else []
        meta_out[oid] = {
            "oracle_id": oid,
            "name": name,
            "mana_cost": mana_cost,
            "cmc": cmc,
            "set_code": set_code,
            "set_name": set_name,
            "image_path": image_path,
            "colors": colors,
            "color_identity": color_identity,
            "names_all": sorted(list(m.get("names", []))),
            "mana_costs_all": sorted(list(m.get("mana_costs", []))),
            "cmcs_all": sorted(list(m.get("cmcs", []))),
            "colors_all": sorted(list(m.get("colors", []))),
            "color_identities_all": sorted(list(m.get("color_identities", []))),
            "set_codes_all": sorted(list(m.get("set_codes", []))),
            "set_names_all": sorted(list(m.get("set_names", []))),
            "langs_all": sorted(list(m.get("langs", []))),
            "image_paths_all": sorted(list(m.get("image_paths", []))),
        }

    return arrays, meta_out