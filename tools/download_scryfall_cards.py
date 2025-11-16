#!/usr/bin/env python3
r"""
Lädt für jede Karte aus der CSV:
- alle Scryfall-Prints
- aber nur in den Sprachen EN und DE
und speichert die Bilder nach:
C:\Users\Fraens\Documents\Fraens\Youtube\TCG Sorter\CNN-Test\CardScannerCNN_02\data\All_image_prints
"""

import csv
import time
import argparse
import re
from pathlib import Path

import requests


SCRYFALL_API_BASE = "https://api.scryfall.com"

# Fester Output-Pfad
OUTPUT_DIR = Path(
    r"C:\Users\Fraens\Documents\Fraens\Youtube\TCG Sorter\CNN-Test\CardScannerCNN_02\data\All_image_prints"
)

# Nur diese Sprachen laden
ALLOWED_LANGS = {"en", "de"}


def slugify_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name)
    return name[:80] or "card"


def read_csv_unique_keys(csv_path: Path):
    """
    CSV einlesen und eindeutige Karten bestimmen.
    Priorität: oracle_id > id/scryfall_id > name
    """
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"CSV {csv_path} ist leer.")

    fieldnames = {c.lower(): c for c in (reader.fieldnames or [])}

    def col_exists(name: str) -> bool:
        return name.lower() in fieldnames

    keys = {}

    if col_exists("oracle_id"):
        col = fieldnames["oracle_id"]
        for row in rows:
            val = row.get(col, "").strip()
            if val:
                keys[("oracle_id", val)] = {
                    "display_name": row.get(fieldnames.get("name", col), val)
                }
        key_type = "oracle_id"

    elif col_exists("id") or col_exists("scryfall_id"):
        col = fieldnames.get("id") or fieldnames["scryfall_id"]
        for row in rows:
            val = row.get(col, "").strip()
            if val:
                keys[("id", val)] = {
                    "display_name": row.get(fieldnames.get("name", col), val)
                }
        key_type = "id"

    elif col_exists("name"):
        col = fieldnames["name"]
        for row in rows:
            val = row.get(col, "").strip()
            if val:
                keys[("name", val)] = {"display_name": val}
        key_type = "name"

    else:
        raise RuntimeError("CSV enthält keine Spalten: oracle_id / id / name")

    print(f"[INFO] Eindeutige Karten: {len(keys)} (Schlüsseltyp: {key_type})")
    return keys


def fetch_card_by_id(session: requests.Session, card_id: str) -> dict:
    url = f"{SCRYFALL_API_BASE}/cards/{card_id}"
    r = session.get(url)
    r.raise_for_status()
    return r.json()


def fetch_card_by_oracle_id(session: requests.Session, oracle_id: str) -> dict:
    """
    Eine repräsentative Karte über oracle_id holen, um an prints_search_uri zu kommen.
    """
    params = {
        "q": f"oracleid:{oracle_id}",
        "order": "released",
        "unique": "prints",
        "include_multilingual": "true",
        "page": 1,
    }
    r = session.get(f"{SCRYFALL_API_BASE}/cards/search", params=params)
    r.raise_for_status()
    data = r.json()
    if not data.get("data"):
        raise RuntimeError(f"Keine Karte für oracle_id={oracle_id}")
    return data["data"][0]


def fetch_card_by_name(session: requests.Session, name: str) -> dict:
    """
    Karte über exakten Namen holen.
    """
    r = session.get(f"{SCRYFALL_API_BASE}/cards/named", params={"exact": name})
    r.raise_for_status()
    return r.json()


def iter_prints(session: requests.Session, prints_search_uri: str):
    """
    prints_search_uri paginieren.
    include_multilingual=true, unique=prints
    """
    if "?" in prints_search_uri:
        url = prints_search_uri + "&include_multilingual=true&unique=prints"
    else:
        url = prints_search_uri + "?include_multilingual=true&unique=prints"

    while url:
        r = session.get(url)
        r.raise_for_status()
        payload = r.json()

        for card in payload.get("data", []):
            yield card

        url = payload.get("next_page") if payload.get("has_more") else None
        time.sleep(0.1)


def get_image_uri(card: dict):
    """
    Geeignete Bild-URL aus Card-Objekt extrahieren.
    """
    if "image_uris" in card:
        img_uris = card["image_uris"]
    elif "card_faces" in card and card["card_faces"]:
        face = card["card_faces"][0]
        img_uris = face.get("image_uris", {})
    else:
        return None

    for key in ["border_crop", "png", "large", "normal", "small"]:
        if key in img_uris:
            return img_uris[key]
    return None


def download_image(session: requests.Session, url: str, out_path: Path):
    """
    Bild herunterladen (wenn noch nicht vorhanden).
    """
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = session.get(url, stream=True)
    r.raise_for_status()
    with out_path.open("wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)


def main():
    parser = argparse.ArgumentParser(
        description="Lädt alle Scryfall-Prints in EN und DE anhand einer CSV."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=r"C:\Users\Fraens\Documents\Fraens\Youtube\TCG Sorter\CNN-Test\CardScannerCNN_02\data\TCG-Sorter.csv",
        help="Pfad zur TCG-Sorter CSV",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.12,
        help="Pause (Sekunden) zwischen Bild-Downloads",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    keys = read_csv_unique_keys(csv_path)

    session = requests.Session()
    session.headers.update({"User-Agent": "TCG-Sorter/1.0"})

    downloaded_ids = set()

    for (key_type, value), meta in keys.items():
        name_display = meta.get("display_name", value)
        print(f"\n[INFO] Processing: {name_display} [{key_type}={value}]")

        try:
            if key_type == "oracle_id":
                base_card = fetch_card_by_oracle_id(session, value)
            elif key_type == "id":
                base_card = fetch_card_by_id(session, value)
            else:  # name
                base_card = fetch_card_by_name(session, value)

            prints_uri = base_card.get("prints_search_uri")
            if not prints_uri:
                print(f"[WARN] Kein prints_search_uri für {name_display}")
                continue

            for pc in iter_prints(session, prints_uri):
                cid = pc.get("id")
                if not cid or cid in downloaded_ids:
                    continue

                lang = pc.get("lang", "xx")

                # 🔴 Sprachfilter: nur EN & DE
                if lang not in ALLOWED_LANGS:
                    continue

                set_code = pc.get("set", "xx")
                nr = pc.get("collector_number", "0")
                cname = pc.get("name", "unknown")

                img_url = get_image_uri(pc)
                if not img_url:
                    print(f"[WARN] Kein Bild für {cname} / {set_code} / {nr} / {lang}")
                    continue

                slug = slugify_name(cname)
                filename = f"{set_code}_{nr}_{slug}_{lang}_{cid}.jpg"
                out_path = OUTPUT_DIR / filename

                print(f"   [DL] {filename}")
                download_image(session, img_url, out_path)
                downloaded_ids.add(cid)

                time.sleep(args.delay)

        except Exception as e:
            print(f"[ERROR] {name_display}: {e}")
            continue

    print(f"\n[OK] Fertig! {len(downloaded_ids)} Bilder (EN/DE) heruntergeladen.")


if __name__ == "__main__":
    main()
