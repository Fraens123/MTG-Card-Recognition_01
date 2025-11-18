#!/usr/bin/env python3
"""
Lädt Scryfall-Prints (EN/DE) basierend auf einer CSV und pflegt die SQLite-Datenbank.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
import sys
from typing import Dict, Iterable, Optional, Tuple

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.sqlite_store import SqliteEmbeddingStore


SCRYFALL_API_BASE = "https://api.scryfall.com"
ALLOWED_LANGS = {"en"}


def _read_csv(csv_path: Path) -> Tuple[list[dict], Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = {name.lower(): name for name in (reader.fieldnames or [])}
    if not rows:
        raise RuntimeError(f"CSV {csv_path} ist leer.")
    return rows, fieldnames


def _normalize_set(row: dict, fieldnames: Dict[str, str]) -> Optional[str]:
    for key in ("set", "set_code", "setcode"):
        if key in fieldnames:
            val = row.get(fieldnames[key], "")
            return val.strip().lower() or None
    return None


def _resolve_name(row: dict, fieldnames: Dict[str, str]) -> Optional[str]:
    for key in ("name", "card_name"):
        if key in fieldnames:
            val = row.get(fieldnames[key], "")
            if val:
                return val.strip()
    return None


def _iter_rows(rows: Iterable[dict], fieldnames: Dict[str, str]):
    for row in rows:
        scryfall_id = row.get(fieldnames.get("scryfall_id", ""), "").strip() if "scryfall_id" in fieldnames else ""
        if not scryfall_id and "id" in fieldnames:
            scryfall_id = row.get(fieldnames["id"], "").strip()
        oracle_id = row.get(fieldnames.get("oracle_id", ""), "").strip() if "oracle_id" in fieldnames else ""
        name = _resolve_name(row, fieldnames)
        set_code = _normalize_set(row, fieldnames)
        yield {
            "scryfall_id": scryfall_id,
            "oracle_id": oracle_id,
            "name": name,
            "set": set_code,
        }


def _fetch(session: requests.Session, url: str, params: Optional[dict] = None) -> dict:
    resp = session.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def fetch_base_card(session: requests.Session, row: dict) -> dict:
    if row.get("scryfall_id"):
        return _fetch(session, f"{SCRYFALL_API_BASE}/cards/{row['scryfall_id']}")
    if row.get("oracle_id"):
        query = f"oracleid:{row['oracle_id']}"
    else:
        if not row.get("name"):
            raise ValueError("CSV-Zeile benötigt entweder scryfall_id, oracle_id, oder mindestens einen Namen.")
        if row.get("set"):
            query = f'!"{row["name"]}" set:{row["set"]}'
        else:
            # Fallback auf exakten Namen, falls kein Set angegeben ist.
            return _fetch(session, f"{SCRYFALL_API_BASE}/cards/named", params={"exact": row["name"]})
    params = {
        "q": query,
        "order": "released",
        "unique": "prints",
        "include_multilingual": "true",
        "page": 1,
    }
    data = _fetch(session, f"{SCRYFALL_API_BASE}/cards/search", params=params)
    cards = data.get("data") or []
    if not cards:
        raise RuntimeError(f"Keine Karte gefunden für Query: {query}")
    return cards[0]


def iter_prints(session: requests.Session, prints_search_uri: str, delay: float):
    url = prints_search_uri
    if "?" in url:
        url += "&include_multilingual=true&unique=prints"
    else:
        url += "?include_multilingual=true&unique=prints"
    while url:
        payload = _fetch(session, url)
        for card in payload.get("data", []):
            yield card
        url = payload.get("next_page") if payload.get("has_more") else None
        time.sleep(delay)


def get_image_uri(card: dict) -> Optional[str]:
    if "image_uris" in card:
        img_uris = card["image_uris"]
    elif "card_faces" in card and card["card_faces"]:
        face = card["card_faces"][0]
        img_uris = face.get("image_uris", {})
    else:
        return None
    for key in ("border_crop", "large", "normal"):
        if key in img_uris:
            return img_uris[key]
    return None


def build_filename(card: dict) -> str:
    set_code = (card.get("set") or "UNK").upper()
    collector = (card.get("collector_number") or "0").replace("/", "-")
    lang = card.get("lang") or "xx"
    cid = card.get("oracle_id") or card.get("id") or "unknown"
    return f"{set_code}_{collector}_{lang}_{cid}.jpg"


def ensure_relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def download_image(session: requests.Session, url: str, out_path: Path) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = session.get(url, stream=True)
    resp.raise_for_status()
    with out_path.open("wb") as handle:
        for chunk in resp.iter_content(8192):
            if chunk:
                handle.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lädt alle Scryfall-Prints in EN/DE anhand einer CSV.")
    parser.add_argument("--csv", type=str, default="data/TCG Sorter.csv", help="Pfad zur TCG-Sorter CSV")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/scryfall_images",
        help="Zielordner für die Bilder (wird angelegt, falls er fehlt).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="tcg_database/database/karten.db",
        help="Pfad zur SQLite-Datenbank.",
    )
    parser.add_argument("--delay", type=float, default=0.1, help="Pause (Sekunden) zwischen Requests.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nicht gefunden: {csv_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_dim = 1024
    cfg_path = PROJECT_ROOT / "config.yaml"
    if cfg_path.exists():
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            emb_dim = int(cfg.get("encoder", {}).get("emb_dim", emb_dim))
        except Exception:
            pass

    store = SqliteEmbeddingStore(args.db, emb_dim=emb_dim)
    rows, fieldnames = _read_csv(csv_path)
    session = requests.Session()
    session.headers.update({"User-Agent": "TCG-Sorter/1.0"})

    downloaded = set()
    for row in _iter_rows(rows, fieldnames):
        display = row.get("name") or row.get("scryfall_id") or row.get("oracle_id") or "unknown"
        try:
            base_card = fetch_base_card(session, row)
        except Exception as exc:
            print(f"[WARN] Überspringe {display}: {exc}")
            continue

        prints_uri = base_card.get("prints_search_uri")
        if not prints_uri:
            print(f"[WARN] Kein prints_search_uri für {display}")
            continue

        print(f"[INFO] Lade Prints für {display}")
        for card in iter_prints(session, prints_uri, args.delay):
            cid = card.get("id")
            oracle_id = card.get("oracle_id") or base_card.get("oracle_id")
            lang = card.get("lang")
            card_key = cid
            if not card_key or lang not in ALLOWED_LANGS:
                continue
            if (card_key, lang) in downloaded:
                continue

            img_url = get_image_uri(card)
            if not img_url:
                continue

            try:
                store.upsert_card(card)
            except Exception as exc:
                print(f"[WARN] Konnte Karte nicht speichern ({cid}): {exc}")
                continue

            filename = build_filename(card)
            out_path = out_dir / filename
            try:
                download_image(session, img_url, out_path)
            except Exception as exc:
                print(f"[WARN] Download fehlgeschlagen für {filename}: {exc}")
                continue

            rel_path = ensure_relative_path(out_path)
            try:
                store.get_or_create_image(
                    scryfall_id=cid,
                    oracle_id=oracle_id or cid,
                    file_path=rel_path,
                    source="scryfall",
                    language=lang,
                    is_training=True,
                )
            except Exception as exc:
                print(f"[WARN] Konnte card_image nicht speichern ({cid}): {exc}")
            downloaded.add((card_key, lang))
            time.sleep(args.delay)

    print(f"[OK] Fertig. {len(downloaded)} Bilder gespeichert in {out_dir}")


if __name__ == "__main__":
    main()
