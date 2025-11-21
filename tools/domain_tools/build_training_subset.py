import csv
import os
import random
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "tcg_database" / "database" / "karten.db"
# Quelle der Bilder: gesamter Bildbestand (z.B. All Cards oder Unique Artwork)
# Passe diesen Pfad bei Bedarf an.
ALL_IMAGES_ROOT = PROJECT_ROOT / "data" / "scryfall_images"
# Zielordner für das Subset
SUBSET_DIR = PROJECT_ROOT / "data" / "scryfall_images" / "subsets" / "train_5k_en_de"
# CSV mit Pflichtkarten
CSV_PATH = PROJECT_ROOT / "data" / "TCG Sorter.csv"

LANGS = {"en", "de"}
TOTAL_CARDS = 5000
RANDOM_SEED = 1337


def read_required_scryfall_ids(csv_path: Path) -> List[str]:
    ids: List[str] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("Scryfall ID") or "").strip()
            if sid:
                ids.append(sid)
    return ids


def load_available_images(db_path: Path, langs: Set[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Liest aus card_images alle Bilder der gewünschten Sprachen.
    Returns: dict[scryfall_id] -> List[(abs_path, language)]
    """
    mapping: Dict[str, List[Tuple[str, str]]] = {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        q = (
            "SELECT scryfall_id, file_path, language FROM card_images "
            "WHERE language IS NOT NULL"
        )
        for row in cur.execute(q):
            lang = (row["language"] or "").lower()
            if langs and lang not in langs:
                continue
            rel = row["file_path"]
            # file_path ist relativ zum Projekt (ensure_relative_path), ggf. absolut verwenden
            abs_path = (PROJECT_ROOT / rel).resolve() if not os.path.isabs(rel) else Path(rel)
            if abs_path.exists():
                mapping.setdefault(row["scryfall_id"], []).append((str(abs_path), lang))
    finally:
        conn.close()
    return mapping


def load_oracle_to_images(db_path: Path, langs: Set[str]) -> Dict[str, List[Tuple[str, str]]]:
    """Alternative Zuordnung ueber oracle_id, falls ein konkreter Print keine EN/DE-Bilder hat.
    Returns: dict[oracle_id] -> List[(abs_path, language)]
    """
    mapping: Dict[str, List[Tuple[str, str]]] = {}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        q = (
            "SELECT ci.oracle_id, ci.file_path, ci.language FROM card_images ci "
            "WHERE ci.language IS NOT NULL AND ci.oracle_id IS NOT NULL"
        )
        for row in cur.execute(q):
            lang = (row["language"] or "").lower()
            if langs and lang not in langs:
                continue
            rel = row["file_path"]
            abs_path = (PROJECT_ROOT / rel).resolve() if not os.path.isabs(rel) else Path(rel)
            if abs_path.exists():
                mapping.setdefault(row["oracle_id"], []).append((str(abs_path), lang))
    finally:
        conn.close()
    return mapping


def get_oracle_id_for_print(db_path: Path, scryfall_id: str) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT oracle_id FROM karten WHERE id = ?", (scryfall_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] else ""
    finally:
        conn.close()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Versuche Hardlink (spart Platz, gleicher Datenträger nötig)
        os.link(src, dst)
    except Exception:
        # Fallback: Kopieren
        shutil.copy2(src, dst)


def pick_one_path(paths: List[Tuple[str, str]]) -> str:
    # Bevorzuge EN, dann DE, sonst beliebig
    if not paths:
        return ""
    for pref in ("en", "de"):
        for p, lang in paths:
            if lang == pref:
                return p
    return paths[0][0]


def main():
    random.seed(RANDOM_SEED)
    ensure_dir(SUBSET_DIR)

    print(f"Projekt: {PROJECT_ROOT}")
    print(f"DB:      {DB_PATH}")
    print(f"Quellbilder: {ALL_IMAGES_ROOT}")
    print(f"Subset-Ziel: {SUBSET_DIR}")

    required_ids = read_required_scryfall_ids(CSV_PATH)
    print(f"Pflichtkarten aus CSV: {len(required_ids)} IDs")

    available = load_available_images(DB_PATH, LANGS)
    oracle_images = load_oracle_to_images(DB_PATH, LANGS)
    all_ids = list(available.keys())
    print(f"Verfügbare Karten (EN/DE): {len(all_ids)}")

    # 1) Immer alle Pflichtkarten einplanen (falls verfügbar)
    selected: List[str] = []
    chosen_paths: Dict[str, str] = {}

    for sid in required_ids:
        paths = available.get(sid)
        if not paths:
            # Fallback über oracle_id
            oid = get_oracle_id_for_print(DB_PATH, sid)
            o_paths = oracle_images.get(oid, []) if oid else []
            if o_paths:
                paths = o_paths
        if not paths:
            print(f"WARN: Keine EN/DE-Bilder gefunden für Pflichtkarte {sid}")
            continue
        selected.append(sid)
        chosen_paths[sid] = pick_one_path(paths)

    # 2) Rest zufällig auffüllen
    remaining = TOTAL_CARDS - len(selected)
    if remaining <= 0:
        print(f"Pflichtkarten >= {TOTAL_CARDS}, kürze auf {TOTAL_CARDS}")
        selected = selected[:TOTAL_CARDS]
    else:
        pool = [sid for sid in all_ids if sid not in selected]
        random.shuffle(pool)
        selected.extend(pool[:remaining])
        for sid in pool[:remaining]:
            chosen_paths[sid] = pick_one_path(available.get(sid, []))

    print(f"Ausgewählte Karten gesamt: {len(selected)}")

    # 3) Dateien verlinken/kopieren
    for sid in selected:
        src_path = Path(chosen_paths.get(sid, ""))
        if not src_path or not src_path.exists():
            print(f"SKIP: Quelle fehlt für {sid}")
            continue
        # Zieldateiname: originaler Name beibehalten
        dst_path = SUBSET_DIR / src_path.name
        try:
            if dst_path.exists():
                continue
            link_or_copy(src_path, dst_path)
        except Exception as e:
            print(f"FEHLER bei {sid}: {e}")

    print("Fertig. Setze paths.scryfall_dir in config auf:")
    print(SUBSET_DIR)


if __name__ == "__main__":
    main()
