from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # required for 3D projection

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config
from src.core.sqlite_store import load_flat_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="t-SNE 3D visualization of card embeddings")
    parser.add_argument("--config", type=str, default="config.yaml", help="Pfad zu config.yaml")
    parser.add_argument("--database", type=str, default=None, help="Pfad zu einer SQLite-Embedding-DB")
    parser.add_argument("--mode", type=str, default="runtime", choices=["runtime", "analysis"], help="Welcher Modus soll geladen werden")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity")
    parser.add_argument("--max-cards", type=int, default=1000, help="maximale Anzahl Karten fuer t-SNE")
    parser.add_argument("--save", type=str, default=None, help="Optionaler Pfad zum Speichern des Plots (PNG)")
    return parser.parse_args()


def _resolve_database_path(cfg: dict, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    db_path = cfg.get("database", {}).get("sqlite_path") or "tcg_database/database/karten.db"
    return Path(db_path)


def load_embeddings_from_sqlite(path: Path, mode: str, emb_dim: int) -> Tuple[List[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding-DB nicht gefunden: {path}")
    X, labels, metas = load_flat_samples(str(path), mode, emb_dim)
    names: List[str] = []
    for label, meta in zip(labels, metas):
        names.append((meta or {}).get("name") or label)
    if X.size > 0:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.clip(norms, 1e-12, None)
    return names, X


def compute_tsne_3d(X: np.ndarray, perplexity: float = 30.0, random_state: int = 42) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise SystemExit("scikit-learn nicht installiert (benoetigt fuer t-SNE).") from exc

    tsne = TSNE(
        n_components=3,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        metric="cosine",
        random_state=random_state,
    )
    return tsne.fit_transform(X)


def plot_tsne_3d(X_tsne: np.ndarray, names: List[str] | None = None, save_path: Path | None = None) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], s=10, alpha=0.8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
    ax.set_title("t-SNE 3D map of card embeddings")
    ax.view_init(elev=20, azim=45)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        print(f"[SAVE] t-SNE 3D plot saved to {save_path}")

    # Hinweis: Bei Bedarf koennen einzelne Punkte (names) hervorgehoben werden, z. B. Top-N nach Distanz.
    plt.show()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else {}
    db_path = _resolve_database_path(cfg, args.database)
    emb_dim = int(cfg.get("encoder", {}).get("emb_dim") or cfg.get("model", {}).get("embed_dim", 1024))

    names, X = load_embeddings_from_sqlite(db_path, args.mode, emb_dim)
    print(f"[INFO] Loaded {len(names)} embeddings from {db_path} (dim={X.shape[1] if X.ndim == 2 else 'unknown'})")
    if X.size == 0:
        raise SystemExit("Keine Embeddings geladen.")

    if X.shape[0] > args.max_cards > 0:
        rng = np.random.default_rng(42)
        idx = rng.choice(X.shape[0], size=args.max_cards, replace=False)
        X = X[idx]
        names = [names[i] for i in idx]
        print(f"[INFO] Downsampled to {len(names)} Karten fuer t-SNE (max-cards={args.max_cards})")

    print(f"[INFO] Running t-SNE (perplexity={args.perplexity}) ...")
    X_tsne = compute_tsne_3d(X, perplexity=args.perplexity)
    plot_tsne_3d(X_tsne, names=names, save_path=Path(args.save) if args.save else None)


if __name__ == "__main__":
    main()
