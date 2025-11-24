from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.embedding_utils import compute_centroid, l2_normalize
from tools.analyze_embeddings import (
    compute_glide_quality,
    compute_intra_card_metrics,
    group_embeddings_by_card,
    load_embeddings_and_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Glide-Style Heatmap der Karten-Zentroiden.")
    parser.add_argument(
        "--mode",
        choices=["runtime", "analysis"],
        default="runtime",
        help="Welchen Export-Modus aus der config laden (default: runtime).",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Pfad zur Konfigurationsdatei.")
    parser.add_argument(
        "--database",
        "--db-path",
        dest="database",
        type=str,
        default=None,
        help="Direkter Pfad zur SQLite-Embedding-Datenbank (ueberschreibt config).",
    )
    parser.add_argument("--max-cards", type=int, default=500, help="Maximale Anzahl Karten fuer die Heatmap.")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Pfad zum Speichern der Heatmap (default: ./debug/heatmap_boundary.png).",
    )
    parser.add_argument("--no-show", action="store_true", help="Heatmap nicht anzeigen, nur speichern.")
    return parser.parse_args()


def _filter_cards(
    by_card: Dict[str, List[np.ndarray]],
    meta: Dict[str, Dict],
    max_cards: int,
) -> Tuple[List[str], Dict[str, List[np.ndarray]], Dict[str, Dict]]:
    card_ids = sorted(by_card.keys())
    if max_cards > 0 and len(card_ids) > max_cards:
        card_ids = card_ids[:max_cards]
    filtered_groups = {cid: by_card[cid] for cid in card_ids}
    filtered_meta = {cid: meta.get(cid, {}) for cid in card_ids}
    return card_ids, filtered_groups, filtered_meta


def _build_centroid_matrix(card_ids: List[str], groups: Dict[str, List[np.ndarray]]) -> np.ndarray:
    centroids: List[np.ndarray] = []
    for cid in card_ids:
        vecs = [np.asarray(v, dtype=np.float32) for v in groups[cid]]
        c = compute_centroid(vecs)
        centroids.append(c)
    if not centroids:
        return np.empty((0, 0), dtype=np.float32)
    stacked = np.stack(centroids)
    return np.asarray([l2_normalize(v) for v in stacked], dtype=np.float32)


def _compute_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    if centroids.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    sims = centroids @ centroids.T
    dist = 1.0 - sims
    np.fill_diagonal(dist, 0.0)
    return dist


def _sort_by_min_boundary(boundary: np.ndarray) -> List[int]:
    if boundary.size == 0:
        return []
    n = boundary.shape[0]
    scores: List[Tuple[float, int]] = []
    for i in range(n):
        row = np.delete(boundary[i], i) if n > 1 else np.array([0.0], dtype=np.float32)
        min_val = float(np.min(row)) if row.size else 0.0
        scores.append((min_val, i))
    scores.sort(key=lambda x: x[0])
    return [idx for _, idx in scores]


def main() -> None:
    args = parse_args()
    db_path, cfg, cards, embeddings, _ = load_embeddings_and_config(args.config, args.database, mode=args.mode)
    print(f"[LOAD] {args.mode.capitalize()}-DB: {db_path}")

    # Norm absichern
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    by_card, meta = group_embeddings_by_card(cards, embeddings)
    card_ids, groups, meta = _filter_cards(by_card, meta, args.max_cards)
    print(f"[INFO] Karten fuer Heatmap: {len(card_ids)} (max={args.max_cards})")
    if not card_ids:
        raise SystemExit("Keine Karten vorhanden.")

    centroids = _build_centroid_matrix(card_ids, groups)
    dist_matrix = _compute_distance_matrix(centroids)

    # Kennzahlen
    intra_cos, intra_dist, intra_stats, _ = compute_intra_card_metrics({cid: groups[cid] for cid in card_ids})
    D_aug = intra_stats["mean_dist"]
    d_aug_safe = D_aug if D_aug == D_aug else 0.0
    if dist_matrix.size:
        tri = np.triu_indices_from(dist_matrix, k=1)
        Dist_med = float(np.median(dist_matrix[tri])) if tri[0].size else float("nan")
    else:
        Dist_med = float("nan")
    Q = compute_glide_quality(D_aug, Dist_med)

    boundary = dist_matrix - d_aug_safe
    np.fill_diagonal(boundary, 0.0)

    # Sortierung nach minimaler Boundary-Distanz
    order = _sort_by_min_boundary(boundary)
    card_ids = [card_ids[i] for i in order]
    dist_matrix = dist_matrix[np.ix_(order, order)]
    boundary = boundary[np.ix_(order, order)]
    centroids = centroids[order]

    # Visualisierung
    vmin = -d_aug_safe if d_aug_safe > 0 else float(boundary.min())
    vmax = d_aug_safe * 3.0 if d_aug_safe > 0 else float(boundary.max() if boundary.size else 1.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(boundary, interpolation="nearest", cmap="seismic", vmin=vmin, vmax=vmax)
    ax.set_title(f"Boundary distances (D_aug={D_aug:.3f}, Dist_med={Dist_med:.3f}, Q={Q:.2f})")
    ax.set_xlabel("Karten-Index (sortiert nach min boundary)")
    ax.set_ylabel("Karten-Index (sortiert nach min boundary)")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Boundary distance = dist - D_aug")
    plt.tight_layout()

    debug_dir = Path(cfg.get("paths", {}).get("debug_dir", "./debug"))
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_path = Path(args.save) if args.save else debug_dir / "heatmap_boundary.png"
    plt.savefig(save_path, dpi=200)

    if not args.no_show:
        plt.show()

    # Kritische Kartenpaare
    tri = np.triu_indices_from(boundary, k=1)
    boundary_vals = boundary[tri]
    sorted_idx = np.argsort(boundary_vals)
    print("[HEATMAP] Kritische Kartenpaare (Boundary < 0 zuerst):")
    for rank, idx in enumerate(sorted_idx[:20], start=1):
        i = int(tri[0][idx])
        j = int(tri[1][idx])
        b_val = float(boundary_vals[idx])
        dist_val = float(dist_matrix[i, j])
        cos_val = 1.0 - dist_val
        cid_a = card_ids[i]
        cid_b = card_ids[j]
        name_a = meta.get(cid_a, {}).get("name", cid_a)
        name_b = meta.get(cid_b, {}).get("name", cid_b)
        flag = " !!!" if b_val < 0 else ""
        print(
            f"  {rank:02d}) {name_a} ~ {name_b}  (cos={cos_val:.4f}, dist={dist_val:.4f}, boundary={b_val:.4f}){flag}"
        )

    print("[DONE] Heatmap gespeichert:", save_path)
    print(f"[STATS] Karten: {len(card_ids)} | D_aug={D_aug:.4f} | Dist_med={Dist_med:.4f} | Q={Q:.2f}")


if __name__ == "__main__":
    main()
