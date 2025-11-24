import argparse
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torchvision.transforms as T
import yaml
from PIL import Image

sys.path.insert(0, ".")
from src.core.model_builder import load_encoder  # noqa: E402
from src.core.embedding_utils import build_card_embedding  # noqa: E402
from src.core.sqlite_store import load_flat_samples  # noqa: E402
from src.datasets.card_datasets import parse_scryfall_filename  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
EMBEDDINGS_TSV = "projector_vectors.tsv"
METADATA_TSV = "projector_metadata.tsv"
PROJECTOR_CONFIG = "projector_config.pbtxt"
DEFAULT_METADATA_FIELDS = ("name", "set_code", "collector_number", "card_uuid")
SOURCE_CHOICES = ("db", "images")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exportiert Karten-Embeddings + Metadaten f\u00fcr den TensorBoard Projector."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(REPO_ROOT / "config.yaml"),
        help="Pfad zur config.yaml (Standard: %(default)s)",
    )
    parser.add_argument(
        "--database",
        "--cards-json",
        dest="database",
        type=str,
        help="Pfad zur SQLite-Datenbank. Falls nicht angegeben, wird der Pfad aus der Config genutzt.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Pfad zu den Modell-Gewichten. Standard ist model.weights_path aus der Config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SCRIPT_DIR),
        help="Zielverzeichnis f\u00fcr TSV-Dateien und projector_config.pbtxt.",
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="db",
        help="Welche Embeddings exportiert werden sollen: Datenbank (Standard) oder erneut aus Bildern berechnen.",
    )
    parser.add_argument(
        "--mode",
        choices=["runtime", "analysis"],
        default="runtime",
        help="Embedding-Mode fuer die DB-Quelle (nur source=db).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Szenario-Name; wenn nicht gesetzt, wird database.scenario aus der Config genutzt (Fallback: 'default').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional: Anzahl der Karten begrenzen (z.B. f\u00fcr schnelle Tests).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Erzwingt CPU-Inferenz, selbst wenn CUDA verf\u00fcgbar ist.",
    )
    parser.add_argument(
        "--launch-tensorboard",
        action="store_true",
        help="Startet nach dem Export TensorBoard und \u00f6ffnet den Browser.",
    )
    parser.add_argument(
        "--tensorboard-port",
        type=int,
        default=6006,
        help="Port f\u00fcr TensorBoard (nur mit --launch-tensorboard relevant).",
    )
    parser.add_argument(
        "--tensorboard-host",
        type=str,
        default="localhost",
        help="Host/IP f\u00fcr TensorBoard (nur mit --launch-tensorboard relevant).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config nicht gefunden: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        raise ValueError(f"Config leer: {config_path}")
    return data


def resolve_path(path_str: str | Path, base_dir: Path) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else (base_dir / path).resolve()


def build_transform(target_size: Sequence[int]) -> T.Compose:
    return T.Compose(
        [
            T.Resize(tuple(target_size), antialias=True),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def encode_search_embedding(model, image: Image.Image, transform: T.Compose, device: torch.device):
    """Repliziert die Embedding-Pipeline aus export_embeddings.py fuer Full-Card-Bilder."""
    tensor_full = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = build_card_embedding(model, tensor_full)
    return emb.cpu().tolist()


def sanitize_text(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("\t", " ").replace("\n", " ").strip()


def resolve_image_path(card: dict, repo_root: Path, images_root: Path) -> Path | None:
    raw_path = card.get("image_path")
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    candidate = (repo_root / candidate).resolve()
    if candidate.exists():
        return candidate
    fallback = (images_root / Path(raw_path).name).resolve()
    return fallback if fallback.exists() else None


def build_cards_from_samples(labels: Sequence[str], metas: Sequence[dict | None]) -> List[dict]:
    cards: List[dict] = []
    for label, meta in zip(labels, metas):
        meta = meta or {}
        cards.append(
            {
                "card_uuid": meta.get("scryfall_id") or label,
                "name": meta.get("name") or label,
                "set_code": meta.get("set") or meta.get("set_code") or "",
                "collector_number": meta.get("collector_number") or "",
                "image_path": (meta.get("image_paths") or [None])[0],
            }
        )
    return cards


def list_image_cards(images_dir: Path) -> List[dict]:
    cards: List[dict] = []
    for path in sorted(images_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        meta = parse_scryfall_filename(path.name)
        if meta:
            card_uuid, set_code, collector_number, card_name = meta
        else:
            card_uuid = path.stem
            set_code = ""
            collector_number = ""
            card_name = path.stem
        cards.append(
            {
                "card_uuid": card_uuid,
                "name": card_name.replace("_", " "),
                "set_code": set_code,
                "collector_number": collector_number,
                "image_path": str(path),
            }
        )
    return cards


def ensure_projector_config(output_dir: Path) -> None:
    config_path = output_dir / PROJECTOR_CONFIG
    content = (
        'embeddings {\n'
        f'  tensor_path: "{EMBEDDINGS_TSV}"\n'
        f'  metadata_path: "{METADATA_TSV}"\n'
        "}\n"
    )
    config_path.write_text(content, encoding="utf-8")


def launch_tensorboard(logdir: Path, host: str, port: int) -> None:
    import socket
    def is_port_free(port: int, host: str = "localhost") -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            try:
                s.bind((host, port))
                return True
            except OSError:
                return False

    orig_port = port
    max_tries = 20
    for i in range(max_tries):
        if is_port_free(port, host):
            break
        port += 1
    else:
        print(f"[ERROR] Kein freier Port für TensorBoard gefunden (ab {orig_port}).")
        return

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        host,
        "--port",
        str(port),
    ]
    print(f"[INFO] Starte TensorBoard: {' '.join(cmd)} (Port: {port})")
    try:
        proc = subprocess.Popen(cmd)
    except FileNotFoundError as exc:
        print(f"[WARN] TensorBoard konnte nicht gestartet werden: {exc}")
        return
    url = f"http://{host}:{port}/#projector"
    print(f"[INFO] TensorBoard PID={proc.pid}. Öffne Browser unter {url}")
    try:
        webbrowser.open(url)
    except Exception as exc:
        print(f"[WARN] Browser konnte nicht geöffnet werden: {exc}")


def export_embeddings(
    model,
    cards: Iterable[dict],
    transform: T.Compose,
    device: torch.device,
    output_dir: Path,
    repo_root: Path,
    images_dir: Path,
    limit: int | None = None,
) -> tuple[int, int]:
    embeddings_path = output_dir / EMBEDDINGS_TSV
    metadata_path = output_dir / METADATA_TSV
    processed = 0
    skipped = 0

    with embeddings_path.open("w", encoding="utf-8") as emb_fh, metadata_path.open("w", encoding="utf-8") as meta_fh:
        meta_fh.write("\t".join(DEFAULT_METADATA_FIELDS) + "\n")

        for card in cards:
            if limit is not None and processed >= limit:
                break

            img_path = resolve_image_path(card, repo_root, images_dir)
            if not img_path or not img_path.exists():
                skipped += 1
                continue

            with Image.open(img_path).convert("RGB") as img:
                embedding = encode_search_embedding(model, img, transform, device)

            emb_fh.write("\t".join(f"{value:.8f}" for value in embedding) + "\n")
            meta_row = [sanitize_text(card.get(key, "")) for key in DEFAULT_METADATA_FIELDS]
            meta_fh.write("\t".join(meta_row) + "\n")
            processed += 1

    return processed, skipped


def export_db_embeddings(
    cards: Sequence[dict],
    embeddings: Sequence[Sequence[float]],
    output_dir: Path,
    limit: int | None = None,
) -> tuple[int, int]:
    embeddings_path = output_dir / EMBEDDINGS_TSV
    metadata_path = output_dir / METADATA_TSV
    processed = 0
    skipped = 0

    with embeddings_path.open("w", encoding="utf-8") as emb_fh, metadata_path.open("w", encoding="utf-8") as meta_fh:
        meta_fh.write("\t".join(DEFAULT_METADATA_FIELDS) + "\n")

        for card, emb in zip(cards, embeddings):
            if limit is not None and processed >= limit:
                break
            if emb is None:
                skipped += 1
                continue
            emb_fh.write("\t".join(f"{float(value):.8f}" for value in emb) + "\n")
            meta_row = [sanitize_text(card.get(key, "")) for key in DEFAULT_METADATA_FIELDS]
            meta_fh.write("\t".join(meta_row) + "\n")
            processed += 1

    return processed, skipped


def main():
    args = parse_args()
    if len(sys.argv) == 1:
        print("[INFO] Play-Button Modus: verwende Standardoptionen (--source db --cpu --launch-tensorboard).")
        args.source = "db"
        args.cpu = True
        args.launch_tensorboard = True
        args.tensorboard_host = getattr(args, "tensorboard_host", "localhost")
        args.tensorboard_port = getattr(args, "tensorboard_port", 6006)

    config = load_config(resolve_path(args.config, REPO_ROOT))
    sqlite_cfg = config.get("database", {}).get("sqlite_path") or "tcg_database/database/karten.db"
    sqlite_path = resolve_path(args.database or sqlite_cfg, REPO_ROOT)
    emb_dim = int(config.get("encoder", {}).get("emb_dim") or config.get("model", {}).get("embed_dim", 1024))
    scenario = args.scenario or config.get("database", {}).get("scenario") or "default"

    output_dir = resolve_path(args.output_dir, REPO_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "db":
        if not sqlite_path.exists():
            raise FileNotFoundError(f"SQLite-DB nicht gefunden: {sqlite_path}")
        X, labels, metas = load_flat_samples(str(sqlite_path), args.mode, emb_dim, scenario=scenario)
        cards = build_cards_from_samples(labels, metas)
        print(f"[INFO] Lade {len(cards)} DB-Eintraege aus {sqlite_path} (mode={args.mode}, scenario={scenario})")
        processed, skipped = export_db_embeddings(
            cards=cards,
            embeddings=X,
            output_dir=output_dir,
            limit=args.limit,
        )
    else:
        paths_cfg = config.get("paths", {})
        default_model_path = Path(paths_cfg.get("models_dir", "./models")) / "encoder_fine.pt"
        model_path = resolve_path(args.model or default_model_path, REPO_ROOT)
        images_dir = resolve_path(paths_cfg.get("scryfall_dir", "./data/scryfall_images"), REPO_ROOT)
        if not images_dir.exists():
            raise FileNotFoundError(f"Scryfall-Verzeichnis nicht gefunden: {images_dir}")

        images_cfg = config.get("images", {})
        full_size = images_cfg.get("full_card_size", [224, 320])
        resize_hw = tuple(reversed(full_size))

        if args.cpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Lade Modell {model_path} auf {device}")
        model = load_encoder(str(model_path), cfg=config, device=device)

        cards = list_image_cards(images_dir)
        print(f"[INFO] Karten im Bilder-Verzeichnis: {len(cards)}")
        transform = build_transform(resize_hw)

        processed, skipped = export_embeddings(
            model=model,
            cards=cards,
            transform=transform,
            device=device,
            output_dir=output_dir,
            repo_root=REPO_ROOT,
            images_dir=images_dir,
            limit=args.limit,
        )

    ensure_projector_config(output_dir)

    print(
        f"[DONE] Quelle={args.source} | Embeddings={processed} | Skipped={skipped} | Dateien: "
        f"{output_dir / EMBEDDINGS_TSV}, {output_dir / METADATA_TSV}, {output_dir / PROJECTOR_CONFIG}"
    )
    if args.limit:
        print("[WARN] Limit aktiv - starte ohne --limit f\u00fcr alle Karten.")
    if args.launch_tensorboard:
        launch_tensorboard(output_dir, args.tensorboard_host, args.tensorboard_port)


if __name__ == "__main__":
    main()
