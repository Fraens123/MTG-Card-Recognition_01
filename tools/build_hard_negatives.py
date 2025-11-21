import argparse
import json
import os
import numpy as np

from src.core.config_utils import load_config
from src.core.embedding_utils import l2_normalize


def load_embeddings(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = {}
    for card in data["cards"]:
        card_id = card["card_id"]
        vec = np.array(card["embedding"], dtype=np.float32)
        embeddings[card_id] = l2_normalize(vec)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--embeddings", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary_path = cfg["analysis"]["oracle_summary_file"]
    out_path = cfg["training"]["fine"]["hard_negatives"]["file"]
    top_k = cfg["training"]["fine"]["hard_negatives"]["top_k"]

    # 1) Oracle-Summary laden
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Oracle summary file fehlt: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        oracle_summary = json.load(f)

    # Herausfiltern: nur suspect_overlap und suspect_cluster_spread
    difficult_oracles = []
    for oracle_id, info in oracle_summary.items():
        if info.get("suspect_overlap") == 1 or info.get("suspect_cluster_spread") == 1:
            difficult_oracles.append(oracle_id)

    print(f"[HardNegatives] {len(difficult_oracles)} schwierige Oracle-IDs gefunden.")

    # 2) Embeddings laden
    embeddings = load_embeddings(args.embeddings)

    # 3) Karte → Oracle-ID Mapping umdrehen
    card_to_oracle = {}
    for oracle_id, info in oracle_summary.items():
        for card_id in info.get("cards", []):
            card_to_oracle[card_id] = oracle_id

    # 4) Nur Karten auswählen, die zu schwierigen Oracles gehören
    difficult_cards = [
        cid for cid, oracle in card_to_oracle.items() if oracle in difficult_oracles and cid in embeddings
    ]

    print(f"[HardNegatives] {len(difficult_cards)} schwierige Karten für Hard-Negatives.")

    if not difficult_cards:
        raise SystemExit("[HardNegatives] Keine schwierigen Karten gefunden – Abbruch.")

    # 5) Embedding-Matrix nur für diese Karten
    matrix = np.stack([embeddings[cid] for cid in difficult_cards])
    ids = difficult_cards
    N = len(ids)

    hard_map: dict = {}

    # 6) Distanzen berechnen
    for i in range(N):
        anchor = matrix[i]
        sims = matrix @ anchor  # Cosine (weil normalisiert)
        order = np.argsort(-sims)  # höchste Ähnlichkeit zuerst
        order = order[order != i]  # sich selbst entfernen

        neighbors = [ids[j] for j in order[:top_k]]
        hard_map[ids[i]] = neighbors

    # 7) JSON speichern
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hard_map, f, indent=2)

    print(f"[HardNegatives] Datei geschrieben: {out_path}")


if __name__ == "__main__":
    main()
