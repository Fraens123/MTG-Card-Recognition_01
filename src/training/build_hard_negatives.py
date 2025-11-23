"""
Erzeugt Hard-Negatives nur fuer verdächtige Oracle-IDs.
Quelle fuer Verdacht: oracle_quality aus SQLite oder, falls vorhanden, oracle_summary_file (JSON).
Embeddings werden aus der SQLite (mode=analysis) geladen.
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config
from src.core.embedding_utils import compute_centroid
from src.core.sqlite_store import load_embeddings_with_meta


def _load_suspects_from_summary(summary_path: Path, top_spread: int) -> Tuple[Set[str], List[str]]:
    """Liest Oracle-Summary und liefert Overlaps + Top-N Spread-Kandidaten."""
    overlap_ids: Set[str] = set()
    spread_candidates: List[Tuple[float, str]] = []
    if not summary_path.exists():
        return overlap_ids, []
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return overlap_ids, []

    if isinstance(data, dict):
        top_block = data.get("top", {})
        if isinstance(top_block, dict):
            overlap_entries = top_block.get("overlap", [])
            if isinstance(overlap_entries, list):
                for entry in overlap_entries:
                    if not isinstance(entry, dict):
                        continue
                    flag = entry.get("flag", entry.get("suspect_overlap"))
                    if flag == 1 or flag is True:
                        oid = entry.get("oracle_id")
                        if oid:
                            overlap_ids.add(str(oid))
            spread_entries = top_block.get("spread", [])
            if isinstance(spread_entries, list):
                for entry in spread_entries:
                    if not isinstance(entry, dict):
                        continue
                    flag = entry.get("flag", entry.get("suspect_cluster_spread"))
                    if not (flag == 1 or flag is True):
                        continue
                    oid = entry.get("oracle_id")
                    intra = entry.get("intra_mean_dist") or entry.get("intra_mean")
                    if oid is None or intra is None:
                        continue
                    try:
                        intra_val = float(intra)
                    except Exception:
                        continue
                    spread_candidates.append((intra_val, str(oid)))

        # Legacy-Format: { oracle_id: { suspect_overlap: 0/1, suspect_cluster_spread: 0/1, ... } }
        if not overlap_ids and not spread_candidates:
            for oid, info in data.items():
                if not isinstance(info, dict):
                    continue
                if info.get("suspect_overlap") == 1:
                    overlap_ids.add(str(oid))
                if info.get("suspect_cluster_spread") == 1:
                    intra = info.get("intra_mean_dist") or info.get("intra_mean")
                    if intra is None:
                        continue
                    try:
                        intra_val = float(intra)
                    except Exception:
                        continue
                    spread_candidates.append((intra_val, str(oid)))

    spread_candidates.sort(key=lambda x: x[0], reverse=True)
    spread_top = [oid for _, oid in spread_candidates[: max(0, int(top_spread))]]
    return overlap_ids, spread_top


def _load_overlap_from_db(sqlite_path: Path, scenario: str) -> Set[str]:
    overlap_ids: Set[str] = set()
    if not sqlite_path.exists():
        return overlap_ids
    try:
        with sqlite3.connect(sqlite_path) as conn:
            cur = conn.execute(
                "SELECT oracle_id FROM oracle_quality WHERE suspect_overlap=1 AND scenario = ?",
                (scenario,),
            )
            for row in cur.fetchall():
                if row[0]:
                    overlap_ids.add(str(row[0]))
    except Exception:
        pass
    return overlap_ids


def _load_top_spread_from_db(sqlite_path: Path, top_spread: int, scenario: str) -> List[str]:
    spread_ids: List[str] = []
    if not sqlite_path.exists() or top_spread <= 0:
        return spread_ids
    try:
        with sqlite3.connect(sqlite_path) as conn:
            cur = conn.execute(
                """
                SELECT oracle_id
                FROM oracle_quality
                WHERE suspect_cluster_spread=1 AND scenario = ?
                ORDER BY intra_mean_dist DESC
                LIMIT ?
                """,
                (scenario, int(top_spread)),
            )
            for row in cur.fetchall():
                if row[0]:
                    spread_ids.append(str(row[0]))
    except Exception:
        pass
    return spread_ids


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
    hard_cfg = cfg.get("training", {}).get("fine", {}).get("hard_negatives", {})
    top_k_cfg = hard_cfg.get("top_k", 20)
    top_k = args.top_k or top_k_cfg
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 20
    try:
        top_spread = int(hard_cfg.get("spread_top_n", 20))
    except Exception:
        top_spread = 20

    sqlite_path = Path(cfg.get("database", {}).get("sqlite_path", "tcg_database/database/karten.db"))
    scenario = cfg.get("database", {}).get("scenario") or "default"
    emb_dim = int(cfg.get("encoder", {}).get("emb_dim", cfg.get("model", {}).get("embed_dim", 1024)))

    overlap_ids, spread_ids = _load_suspects_from_summary(summary_path, top_spread)
    if not overlap_ids:
        overlap_ids = _load_overlap_from_db(sqlite_path, scenario=scenario)
    if not spread_ids:
        spread_ids = _load_top_spread_from_db(sqlite_path, top_spread, scenario=scenario)

    suspects = set(overlap_ids) | set(spread_ids)
    if not suspects:
        raise SystemExit("[HardNegatives] Keine verdächtigen Oracle-IDs gefunden (weder JSON noch DB).")
    print(
        f"[HardNegatives] Overlap={len(overlap_ids)} | Spread-Top={len(spread_ids)} "
        f"(N={top_spread}) -> Gesamt={len(suspects)} verdächtige Oracle-IDs."
    )

    embeddings_by_card, meta_by_card = load_embeddings_with_meta(
        str(sqlite_path), mode="analysis", emb_dim=emb_dim, scenario=scenario
    )
    if not embeddings_by_card:
        raise SystemExit("[HardNegatives] Keine Embeddings im mode=analysis gefunden.")

    card_to_oracle = _build_card_to_oracle(meta_by_card)

    # Optional: Mapping aus Summary ergänzen, falls vorhanden (legacy-Struktur mit cards-Liste)
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            if isinstance(summary, dict):
                for oid, info in summary.items():
                    if not isinstance(info, dict):
                        continue
                    for cid in info.get("cards", []):
                        card_to_oracle.setdefault(cid, str(oid))
        except Exception:
            pass

    embeddings = {}
    for cid, vecs in embeddings_by_card.items():
        if not vecs:
            continue
        embeddings[cid] = compute_centroid(vecs)

    difficult_cards = [cid for cid, o in card_to_oracle.items() if o in suspects and cid in embeddings]
    print(f"[HardNegatives] {len(difficult_cards)} schwierige Karten für Hard-Negatives.")
    if not difficult_cards:
        raise SystemExit("[HardNegatives] Keine schwierigen Karten gefunden - Abbruch.")

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
