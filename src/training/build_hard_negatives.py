"""
Erzeugt Hard-Negatives nur für verdächtige Oracle-IDs.
Quelle für Verdacht: oracle_quality aus SQLite oder, falls vorhanden, oracle_summary_file (JSON).
Embeddings werden aus der SQLite (mode=analysis) geladen.
"""
import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, Set

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config
from src.core.embedding_utils import compute_centroid
from src.core.sqlite_store import load_embeddings_with_meta


def _load_suspects_from_json(summary_path: Path) -> Set[str]:
    suspects: Set[str] = set()
    if not summary_path.exists():
        return suspects
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return suspects

    # Erwartete Struktur: { oracle_id: { suspect_overlap: 0/1, suspect_cluster_spread: 0/1, cards: [...] }, ... }
    if isinstance(data, dict):
        for oid, info in data.items():
            if not isinstance(info, dict):
                continue
            if info.get("suspect_overlap") == 1 or info.get("suspect_cluster_spread") == 1:
                suspects.add(str(oid))
    return suspects


def _load_suspects_from_db(sqlite_path: Path) -> Set[str]:
    suspects: Set[str] = set()
    if not sqlite_path.exists():
        return suspects
    try:
        with sqlite3.connect(sqlite_path) as conn:
            cur = conn.execute(
                "SELECT oracle_id FROM oracle_quality WHERE suspect_overlap=1 OR suspect_cluster_spread=1"
            )
            for row in cur.fetchall():
                if row[0]:
                    suspects.add(str(row[0]))
    except Exception:
        pass
    return suspects


def _build_card_to_oracle(meta_by_card: Dict[str, Dict]) -> Dict[str, str]:
    return {cid: meta.get("oracle_id", cid) for cid, meta in meta_by_card.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--groups",
        default="suspect_overlap,suspect_cluster_spread",
        help="Flags, die berücksichtigt werden (informativ).",
    )
    parser.add_argument(
        "--top-k",
        dest="top_k",
        type=int,
        default=None,
        help="Override für Top-K Hard-Negatives (sonst Wert aus Config).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary_path = Path(cfg.get("analysis", {}).get("oracle_summary_file", "debug/oracle_cluster_summary.json"))
    out_path = Path(cfg["training"]["fine"]["hard_negatives"]["file"])
    top_k = args.top_k or cfg["training"]["fine"]["hard_negatives"]["top_k"]

    sqlite_path = Path(cfg.get("database", {}).get("sqlite_path", "tcg_database/database/karten.db"))
    emb_dim = int(cfg.get("encoder", {}).get("emb_dim", cfg.get("model", {}).get("embed_dim", 1024)))

    # Verdächtige Oracle-IDs aus JSON oder DB holen
    suspects = _load_suspects_from_json(summary_path)
    if not suspects:
        suspects = _load_suspects_from_db(sqlite_path)
    if not suspects:
        raise SystemExit("[HardNegatives] Keine verdächtigen Oracle-IDs gefunden (weder JSON noch DB).")
    print(f"[HardNegatives] {len(suspects)} verdächtige Oracle-IDs gefunden.")

    # Embeddings aus SQLite laden (analysis-Mode)
    embeddings_by_card, meta_by_card = load_embeddings_with_meta(str(sqlite_path), mode="analysis", emb_dim=emb_dim)
    if not embeddings_by_card:
        raise SystemExit("[HardNegatives] Keine Embeddings im mode=analysis gefunden.")

    card_to_oracle = _build_card_to_oracle(meta_by_card)

    # Optional: Mapping aus Summary ergänzen, falls vorhanden
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            if isinstance(summary, dict):
                for oid, info in summary.items():
                    if not isinstance(info, dict):
                        continue
                    for cid in info.get("cards", []):
                        card_to_oracle.setdefault(cid, oid)
        except Exception:
            pass

    # Karte -> Embedding (Zentroid je Karte)
    embeddings = {}
    for cid, vecs in embeddings_by_card.items():
        if not vecs:
            continue
        embeddings[cid] = compute_centroid(vecs)

    difficult_cards = [cid for cid, o in card_to_oracle.items() if o in suspects and cid in embeddings]
    print(f"[HardNegatives] {len(difficult_cards)} schwierige Karten für Hard-Negatives.")
    if not difficult_cards:
        raise SystemExit("[HardNegatives] Keine schwierigen Karten gefunden – Abbruch.")

    matrix = np.stack([embeddings[cid] for cid in difficult_cards])
    ids = difficult_cards
    N = len(ids)

    hard_map: Dict[str, list] = {}
    for i in range(N):
        anchor = matrix[i]
        sims = matrix @ anchor  # Cosine (normalisiert)
        order = np.argsort(-sims)
        order = order[order != i]
        neighbors = [ids[j] for j in order[:top_k]]
        hard_map[ids[i]] = neighbors

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(hard_map, f, indent=2)

    print(f"[HardNegatives] Datei geschrieben: {out_path}")


if __name__ == "__main__":
    main()
