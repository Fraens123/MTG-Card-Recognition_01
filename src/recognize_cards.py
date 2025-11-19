#!/usr/bin/env python3
"""
MTG Card Recognition CLI

Vergleicht Pi-Cam-Bilder mit der Scryfall-Datenbank anhand von Cosine-Similarities,
erstellt Vergleichsplots und fasst die Ergebnisse statistisch zusammen.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.core.model_builder import load_encoder
from src.core.card_database import SimpleCardDB
from src.core.image_ops import (
    build_resize_normalize_transform,
    crop_card_art,
    get_full_art_crop_cfg,
    resolve_resize_hw,
)
from src.core.embedding_utils import build_card_embedding
from src.ocr import run_ocr_for_card_image, select_print_with_ocr, score_candidate_with_ocr


def crop_card_roi(img: Image.Image, roi_cfg: Optional[Dict]) -> Image.Image:
    """
    Schneidet die MTG-Karte aus dem Pi-Cam-Bild anhand fester relativer Koordinaten.
    roi_cfg: Dict mit x_min, y_min, x_max, y_max in [0.0, 1.0].
    Wenn roi_cfg None ist oder unvollständig, wird das Bild unverändert zurückgegeben.
    """
    if not roi_cfg:
        return img

    w, h = img.size
    try:
        x_min = float(roi_cfg.get("x_min", 0.0))
        y_min = float(roi_cfg.get("y_min", 0.0))
        x_max = float(roi_cfg.get("x_max", 1.0))
        y_max = float(roi_cfg.get("y_max", 1.0))
    except Exception:
        return img

    x0 = int(x_min * w)
    y0 = int(y_min * h)
    x1 = int(x_max * w)
    y1 = int(y_max * h)

    # Bounds sichern
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))

    return img.crop((x0, y0, x1, y1))


def load_config(config_path: str = "config.yaml") -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not data:
        raise ValueError(f"Config-Datei ist leer: {config_path}")
    return data


def get_inference_transforms(resize_hw: Tuple[int, int]) -> T.Compose:
    """Identische Resize/Normalize-Pipeline wie beim Embedding-Export."""
    return build_resize_normalize_transform(resize_hw)


class CropDebugger:
    def __init__(self, directory: Optional[Path], limit: int):
        self.directory = directory
        self.limit = max(0, int(limit or 0))
        self.count = 0
        if self.limit > 0 and self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, card_img: Optional[Image.Image], art_img: Image.Image, source: str):
        if self.limit <= 0 or self.directory is None or self.count >= self.limit:
            return
        stem = Path(source).stem
        art_path = self.directory / f"{stem}_art_{self.count:02d}.png"
        card_path = self.directory / f"{stem}_card_{self.count:02d}.png" if card_img else None
        try:
            art_img.save(art_path)
            if card_img and card_path:
                card_img.save(card_path)
            self.count += 1
        except Exception as exc:
            print(f"[WARN] Konnte Kamera-Crop nicht speichern ({source}): {exc}")


def create_comparison_plot(
    camera_img_path: str,
    camera_img: Image.Image,
    best_match: Dict,
    similarity_score: float,
    total_time: float,
    cnn_time_ms: float,
    ocr_time_ms: float,
    output_path: str,
) -> None:
    """Erstellt eine Side-by-Side-Abbildung zwischen Kameraaufnahme und Scryfall-Referenz."""
    import matplotlib

    try:
        matplotlib.rcParams["font.family"] = "Noto Sans Symbols"
    except Exception:
        pass

    plt.style.use("default")
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.3,
        wspace=0.2,
        height_ratios=[0.1, 0.8, 0.1],
        width_ratios=[1, 1],
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.5,
        "MTG Card Recognition - Similarity Search Results",
        fontsize=20,
        fontweight="bold",
        ha="center",
        va="center",
    )

    camera_ax = fig.add_subplot(gs[1, 0])
    camera_ax.imshow(camera_img)
    camera_ax.set_title("Pi Camera Image", fontsize=16, fontweight="bold", pad=20)
    camera_ax.axis("off")

    scryfall_ax = fig.add_subplot(gs[1, 1])
    if os.path.exists(best_match["image_path"]):
        ref_img = Image.open(best_match["image_path"]).convert("RGB")
        scryfall_ax.imshow(ref_img)
    else:
        scryfall_ax.text(
            0.5,
            0.5,
            "Referenzbild fehlt",
            ha="center",
            va="center",
            fontsize=14,
            transform=scryfall_ax.transAxes,
        )
    scryfall_ax.set_title("Best Match (Scryfall)", fontsize=16, fontweight="bold", pad=20)
    scryfall_ax.axis("off")

    info_ax = fig.add_subplot(gs[2, :])
    info_ax.axis("off")

    camera_filename = Path(camera_img_path).name
    info_text = (
        f"SEARCH RESULTS:\n"
        f"* Camera File: {camera_filename}\n"
        f"* Best Match: {best_match['name']} (Set: {best_match['set_code']}, #{best_match['collector_number']})\n"
        f"* Cosine Similarity: {similarity_score:.4f} ({similarity_score*100:.1f}%)\n"
        f"* Total Time: {total_time*1000:.2f} ms\n"
        f"* CNN Time: {cnn_time_ms:.2f} ms\n"
        f"* OCR Time: {ocr_time_ms:.2f} ms\n"
        f"* Card UUID: {best_match['card_uuid'][:8]}...\n"
        f"* Image Path: {Path(best_match['image_path']).name}"
    )

    sel = best_match.get("selection_info")
    if sel:
        info_text += "\n\nOCR DETAILS:\n"
        info_text += f"* Strategy: {sel.get('strategy','-')}\n"
        oracle = sel.get('oracle', {})
        if oracle:
            info_text += f"* Oracle-ID: {str(oracle.get('id',''))[:8]}...  (prints: {oracle.get('candidates',0)})\n"
        ocr = sel.get('ocr', {})
        if ocr:
            info_text += (
                f"* OCR Name: '{(ocr.get('best_name') or '')[:30]}'\n"
                f"* OCR Collector: {ocr.get('collector_clean') or ''}\n"
                f"* OCR SetID: {ocr.get('setid_clean') or ''}\n"
                f"* OCR Quality: {ocr.get('collector_set_score',0)}\n"
            )
        top_scores = sel.get('top_ocr_scores') or []
        if top_scores:
            best = top_scores[0]
            info_text += (
                f"* Top OCR Candidate: {best.get('name','')}"
                f" [{best.get('set_code','')}] #{best.get('collector_number','')}"
                f" (score={best.get('score',0):.3f})\n"
            )

    if similarity_score > 0.8:
        color, status = ("green", "EXCELLENT MATCH")
    elif similarity_score > 0.6:
        color, status = ("orange", "GOOD MATCH")
    elif similarity_score > 0.4:
        color, status = ("red", "POOR MATCH")
    else:
        color, status = ("darkred", "NO MATCH")

    info_ax.text(
        0.02,
        0.95,
        info_text,
        fontsize=12,
        ha="left",
        va="top",
        transform=info_ax.transAxes,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
    )
    info_ax.text(
        0.98,
        0.95,
        status,
        fontsize=14,
        fontweight="bold",
        ha="right",
        va="top",
        transform=info_ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.8, edgecolor="black"),
    )

    bar_width = 0.6
    bar_x = 0.2
    bar_y = 0.2
    info_ax.add_patch(
        patches.Rectangle(
            (bar_x, bar_y),
            bar_width,
            0.15,
            facecolor="lightgray",
            edgecolor="black",
            linewidth=2,
        )
    )
    info_ax.add_patch(
        patches.Rectangle(
            (bar_x, bar_y),
            bar_width * similarity_score,
            0.15,
            facecolor=color,
            alpha=0.8,
            edgecolor="black",
            linewidth=2,
        )
    )
    info_ax.text(
        bar_x + bar_width / 2,
        bar_y + 0.075,
        f"{similarity_score:.3f}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="white",
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[PLOT] Vergleichsgrafik gespeichert: {output_path}")


def search_camera_image(
    model: torch.nn.Module,
    transform: T.Compose,
    db: SimpleCardDB,
    camera_img_path: str,
    device: torch.device,
    config: dict,
    art_crop_cfg: Optional[dict],
    debug_recorder=None,
    use_ocr: bool = True,
) -> Tuple[Optional[Dict], float, float, Dict[str, float]]:
    """
    Berechnet das Query-Embedding und sucht den besten Match in der Datenbank.
    
    Zwei-stufige Erkennung:
    1. CNN-Embedding-Suche findet Top-1-Match (scryfall_id)
    2. OCR verfeinert Auswahl unter allen Prints derselben Oracle-ID
    
    Args:
        use_ocr: True = OCR-Verfeinerung aktivieren, False = nur CNN
    """
    start_time = time.time()
    cnn_time_ms = 0.0
    ocr_time_ms = 0.0

    with Image.open(camera_img_path) as img:
        rgb = img.convert("RGB")
        # NEU: Zuerst die Karte per fester ROI aus dem Pi-Cam-Bild schneiden
        camera_cfg = config.get("camera", {})
        card_roi_cfg = camera_cfg.get("card_roi", None)
        card_img = crop_card_roi(rgb, card_roi_cfg)

        # DANN wie bisher: Artwork-Crop auf der normierten Karte
        art_img = crop_card_art(card_img, art_crop_cfg)
        if debug_recorder:
            debug_recorder(card_img, art_img, camera_img_path)
        full_tensor = transform(art_img).unsqueeze(0).to(device)

    with torch.no_grad():
        t0_cnn = time.time()
        query_embedding = build_card_embedding(model, full_tensor).cpu().numpy().flatten()
        search_cfg = config.get("recognition", {})
        top_k = search_cfg.get("top_k", 5)
        threshold = search_cfg.get("threshold", 0.88)
        results = db.search_similar(query_embedding, top_k=top_k, threshold=threshold)
        cnn_time_ms = (time.time() - t0_cnn) * 1000.0

    for rank, hit in enumerate(results, start=1):
        print(f"[Rank {rank}] {hit['name']} - cos={hit['similarity']:.3f}")

    if not results:
        elapsed = time.time() - start_time
        return None, 0.0, elapsed, {"cnn_ms": cnn_time_ms, "ocr_ms": ocr_time_ms}

    best_match_cnn = results[0]
    
    # === STUFE 2: OCR-Verfeinerung ===
    if use_ocr:
        oracle_id = best_match_cnn.get("oracle_id")
        if oracle_id:
            print(f"[OCR] Lade alle Prints für Oracle-ID: {oracle_id[:8]}...")
            oracle_candidates = db.get_cards_by_oracle_id(oracle_id)
            print(f"[OCR] Gefunden: {len(oracle_candidates)} Print-Varianten")
            
            if len(oracle_candidates) > 1:
                # OCR auf Kamera-Bild ausführen
                t0_ocr = time.time()
                ocr_result = run_ocr_for_card_image(camera_img_path, config)
                print(f"[OCR] Erkannt: Name='{ocr_result.best_name[:30]}...', "
                      f"Collector={ocr_result.collector_clean}, Set={ocr_result.setid_clean}")
                
                # Besten Print anhand OCR-Daten auswählen
                best_match_ocr = select_print_with_ocr(best_match_cnn, oracle_candidates, ocr_result)
                # Scoring-Übersicht berechnen (Top-3) für Dokumentation
                scored = []
                try:
                    for cand in oracle_candidates:
                        s = float(score_candidate_with_ocr(cand, ocr_result))
                        entry = {
                            "name": cand.get("name"),
                            "set_code": cand.get("set_code"),
                            "collector_number": cand.get("collector_number"),
                            "scryfall_id": cand.get("card_uuid") or cand.get("scryfall_id"),
                            "score": s,
                        }
                        scored.append(entry)
                    scored.sort(key=lambda e: -e["score"])
                except Exception:
                    scored = []
                if best_match_ocr:
                    print(f"[OCR] OCR wählt: {best_match_ocr['name']} "
                          f"(Set: {best_match_ocr['set_code']}, #{best_match_ocr['collector_number']})")
                    # Similarity vom CNN-Best-Match übernehmen
                    best_match_ocr["similarity"] = best_match_cnn["similarity"]
                    # Auswahl-Dokumentation anreichern
                    best_match_ocr["selection_info"] = {
                        "strategy": (
                            "ocr-confirmed" if (best_match_ocr.get("card_uuid") == best_match_cnn.get("card_uuid")) else "ocr-selected"
                        ),
                        "oracle": {"id": oracle_id, "candidates": len(oracle_candidates)},
                        "ocr": {
                            "best_name": ocr_result.best_name,
                            "collector_clean": ocr_result.collector_clean,
                            "setid_clean": ocr_result.setid_clean,
                            "collector_set_score": ocr_result.collector_set_score,
                        },
                        "top_ocr_scores": scored[:3],
                    }
                    ocr_time_ms = (time.time() - t0_ocr) * 1000.0
                    elapsed = time.time() - start_time
                    return best_match_ocr, best_match_ocr["similarity"], elapsed, {"cnn_ms": cnn_time_ms, "ocr_ms": ocr_time_ms}
                else:
                    print("[OCR] OCR-Auswahl fehlgeschlagen, verwende CNN-Best-Match")
                    best_match_cnn["selection_info"] = {
                        "strategy": "cnn",
                        "oracle": {"id": oracle_id, "candidates": len(oracle_candidates)},
                        "reason": "ocr_selection_failed",
                    }
                    ocr_time_ms = (time.time() - t0_ocr) * 1000.0
            else:
                print("[OCR] Nur 1 Print vorhanden, OCR übersprungen")
                best_match_cnn["selection_info"] = {
                    "strategy": "cnn",
                    "oracle": {"id": oracle_id, "candidates": len(oracle_candidates)},
                    "reason": "single_print_in_oracle",
                }
        else:
            print("[OCR] Keine Oracle-ID verfügbar, verwende CNN-Best-Match")
            best_match_cnn["selection_info"] = {
                "strategy": "cnn",
                "reason": "no_oracle_id",
            }

    elapsed = time.time() - start_time
    return best_match_cnn, best_match_cnn["similarity"], elapsed, {"cnn_ms": cnn_time_ms, "ocr_ms": ocr_time_ms}


def test_all_camera_images(
    config: dict,
    model_path: str,
    camera_dir: str,
    output_dir: str,
    device: torch.device,
    config_path: str,
) -> None:
    """Iteriert ueber alle Kamera-Bilder, erstellt Vergleichsplots und Statistiken."""
    print(f"[LOAD] Lade trainiertes Modell: {model_path}")
    model = load_encoder(model_path, cfg=config, device=device)
    model.eval()
    print(f"[INFO] Erkennung verwendet Modell: {model_path}")

    print("[INFO] Lade Embedding-Datenbank ...")
    db_path = config.get("database", {}).get("sqlite_path", "tcg_database/database/karten.db")
    db = SimpleCardDB(db_path=db_path, config_path=config_path)
    print(f"[INFO] Erkennung verwendet Embedding-DB (SQLite): {db.db_path}")
    if len(db.cards) == 0:
        print("[WARN] Database ist leer. Bitte zuerst export_embeddings.py ausfuehren.")
        return
    print(f"[INFO] Database geladen: {len(db.cards)} Karten")
    db_meta = getattr(db, "meta", {})
    expected_model = os.path.abspath(model_path)
    db_model = db_meta.get("model_path")
    if db_model and os.path.abspath(db_model) != expected_model:
        print(
            "[WARN] Embedding-DB wurde mit einem anderen Modell erzeugt: "
            f"{db_model}. Bitte export_embeddings.py mit dem aktuellen Modell neu ausfuehren."
        )

    paths_cfg = config.get("paths", {})
    resize_hw = resolve_resize_hw(config, paths_cfg.get("scryfall_dir"))
    target_height, target_width = resize_hw
    transform = get_inference_transforms(resize_hw)
    print(f"[INFO] Inference-Bildgroesse (Breite x Hoehe): {target_width}x{target_height}")

    art_cfg = get_full_art_crop_cfg(config)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    camera_path = Path(camera_dir)
    camera_files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        camera_files.extend(camera_path.glob(ext))
    camera_files.sort()

    if not camera_files:
        print(f"[WARN] Keine Camera-Bilder gefunden in: {camera_dir}")
        return

    print(f"[INFO] Gefunden: {len(camera_files)} Camera-Bilder")
    print(f"[INFO] Output-Verzeichnis: {output_dir}")

    total_time = 0.0
    match_scores: List[float] = []
    total_searches = len(camera_files)

    print("\n[RUN] Starte Similarity-Search fuer alle Camera-Bilder ...")
    # Kamera-Crop-Dumps sind deaktiviert (nicht mehr benötigt).
    crop_dump = None

    for idx, camera_file in enumerate(tqdm(camera_files, desc="Camera-Bilder", unit="img"), start=1):
        try:
            print(f"\n[{idx}/{total_searches}] [RUN] Verarbeite: {camera_file.name}")
            best_match, similarity_score, search_time, timings = search_camera_image(
                model,
                transform,
                db,
                str(camera_file),
                device,
                config,
                art_cfg,
                None,
            )
            total_time += search_time
            match_scores.append(float(similarity_score))

            if best_match:
                print(f"   [MATCH] {best_match['name']} (cos={similarity_score:.4f})")
                print(f"   [TIME]  total={search_time*1000:.2f} ms | cnn={timings.get('cnn_ms',0.0):.2f} ms | ocr={timings.get('ocr_ms',0.0):.2f} ms")
                with Image.open(camera_file) as cam_img_raw:
                    camera_img = cam_img_raw.convert("RGB")
                    comparison_path = output_path / f"{camera_file.stem}_comparison.png"
                    create_comparison_plot(
                        str(camera_file),
                        camera_img,
                        best_match,
                        similarity_score,
                        search_time,
                        timings.get('cnn_ms', 0.0),
                        timings.get('ocr_ms', 0.0),
                        str(comparison_path),
                    )
            else:
                print("   [WARN] Kein Match gefunden.")
        except Exception as exc:
            print(f"   [ERROR] Fehler bei {camera_file.name}: {exc}")

    if not match_scores:
        print("[WARN] Keine Similarity-Werte gesammelt.")
        return

    print("\n" + "=" * 60)
    print("[SUMMARY] SIMILARITY SEARCH")
    print("=" * 60)
    print(f"[INFO] Durchsuchte Bilder: {total_searches}")
    print(f"[INFO] Gesamtzeit: {total_time:.2f}s")
    avg_time_ms = (total_time / max(total_searches, 1)) * 1000
    print(f"[INFO] Durchschnitt pro Bild: {avg_time_ms:.2f} ms")
    print(f"[INFO] Beste Similarity: {max(match_scores):.4f}")
    print(f"[INFO] Schlechteste Similarity: {min(match_scores):.4f}")
    print(f"[INFO] Durchschnittliche Similarity: {np.mean(match_scores):.4f}")
    print(f"[INFO] Vergleichsgrafiken gespeichert in: {output_dir}")

    summary_path = output_path / "summary_statistics.png"
    create_summary_plot(match_scores, total_time, total_searches, str(summary_path))


def create_summary_plot(
    match_scores: Sequence[float],
    total_time: float,
    total_searches: int,
    output_path: str,
) -> None:
    """Erstellt Histogramm, Boxplot und weitere Kennzahlen als Uebersicht."""
    if not match_scores:
        print("[WARN] Keine Daten fuer Statistik-Plot vorhanden.")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    ax1.hist(match_scores, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.set_title("Similarity Score Verteilung", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Cosine Similarity")
    ax1.set_ylabel("Anzahl Bilder")
    ax1.axvline(
        np.mean(match_scores),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Durchschnitt: {np.mean(match_scores):.3f}",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.boxplot(match_scores, vert=True, patch_artist=True, boxprops=dict(facecolor="lightgreen", alpha=0.7))
    ax2.set_title("Similarity Score Box Plot", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Cosine Similarity")
    ax2.grid(True, alpha=0.3)

    avg_time_ms = (total_time / max(total_searches, 1)) * 1000
    performance_labels = ["Durchschnitt\nZeit/Bild", "Gesamt-\nzeit (ms)"]
    performance_values = [avg_time_ms, total_time * 1000]
    bars = ax3.bar(performance_labels, performance_values, color=["orange", "red"], alpha=0.7, edgecolor="black")
    ax3.set_title("Performance-Metriken", fontsize=14, fontweight="bold")
    ax3.set_ylabel("Zeit (ms)")
    ax3.grid(True, alpha=0.3)
    for bar, value in zip(bars, performance_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(performance_values) * 0.01,
            f"{value:.1f} ms",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    excellent = sum(1 for score in match_scores if score > 0.8)
    good = sum(1 for score in match_scores if 0.6 < score <= 0.8)
    poor = sum(1 for score in match_scores if 0.4 < score <= 0.6)
    no_match = sum(1 for score in match_scores if score <= 0.4)
    categories = ["Excellent\n(>0.8)", "Good\n(0.6-0.8)", "Poor\n(0.4-0.6)", "No Match\n(<=0.4)"]
    counts = [excellent, good, poor, no_match]
    colors = ["green", "orange", "red", "darkred"]
    ax4.pie(counts, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax4.set_title("Match Quality Verteilung", fontsize=14, fontweight="bold")

    fig.suptitle(
        f"MTG Card Recognition - Similarity Search Statistiken ({total_searches} Bilder)",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] Statistik-Grafik gespeichert: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MTG Card Recognition - Erkennung von Karten aus Pi Camera-Bildern"
    )
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Pfad zur Config-Datei.")
    parser.add_argument("--camera-dir", type=str, default=None, help="Override fuer das Kamera-Verzeichnis.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override fuer das Output-Verzeichnis.")
    parser.add_argument("--model-path", type=str, default=None, help="Override fuer das Encoder-Gewicht.")
    return parser.parse_args()


def main() -> None:
    play_button_mode = len(sys.argv) == 1
    args = parse_args()
    if play_button_mode:
        print("[INFO] Play-Button Modus: nutze Parameter aus config.yaml")

    try:
        config = load_config(args.config)
        paths_cfg = config.get("paths", {})
        default_model_path = os.path.join(paths_cfg.get("models_dir", "./models"), "encoder_fine.pt")
        model_path = args.model_path or default_model_path
        camera_dir = args.camera_dir or paths_cfg.get("camera_dir", "./data/camera_images")
        output_dir = args.output_dir or paths_cfg.get("output_dir", "./output_matches")

        device = torch.device("cuda" if torch.cuda.is_available() and config["hardware"]["use_cuda"] else "cpu")
        print("[INFO] MTG Card Similarity Search")
        print(f"[INFO] Device: {device.type.upper()}")
        if device.type == "cuda":
            print(f"       GPU: {torch.cuda.get_device_name()}")
        print(f"[INFO] Config: {args.config}")
        print(f"[INFO] Modell: {model_path}")
        print(f"[INFO] Camera-Bilder: {camera_dir}")
        print(f"[INFO] Output: {output_dir}")

        if not os.path.exists(model_path):
            print(f"[ERROR] Modell nicht gefunden: {model_path}")
            print("        Bitte zuerst train_coarse.py und train_fine.py ausfuehren.")
            return

        camera_path = Path(camera_dir)
        if not camera_path.exists():
            print(f"[ERROR] Camera-Verzeichnis nicht gefunden: {camera_dir}")
            print("        Erstelle Verzeichnis und kopiere Testbilder hinein.")
            camera_path.mkdir(parents=True, exist_ok=True)
            return

        db_path = config.get("database", {}).get("sqlite_path", "tcg_database/database/karten.db")
        if not os.path.exists(db_path):
            print(f"[ERROR] Database nicht gefunden: {db_path}")
            print("        Bitte zuerst Embeddings generieren (export_embeddings.py).")
            return

        preview_files: List[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            preview_files.extend(camera_path.glob(ext))
        if not preview_files:
            print(f"[ERROR] Keine Bilder im Camera-Verzeichnis gefunden: {camera_dir}")
            print("        Kopiere einige Testbilder ins Camera-Verzeichnis.")
            return

        test_all_camera_images(config, model_path, camera_dir, output_dir, device, args.config)
    except Exception as exc:
        print(f"[ERROR] Fehler: {exc}")
        print("\n[INFO] Verwendung:")
        print("   Direkt: python src/cardscanner/recognize_cards.py")
        print("   Mit Parametern: python src/cardscanner/recognize_cards.py --camera-dir data/camera_images")
        return


if __name__ == "__main__":
    main()
