import requests
import sqlite3
import json
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    import ijson
except Exception:
    ijson = None

from PIL import Image
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.sqlite_store import SqliteEmbeddingStore

# Konfiguration
KARTEN_DB = "tcg_database/database/karten.db"
SCRYFALL_BULK_DATA_API = "https://api.scryfall.com/bulk-data"
DEFAULT_OUT_DIR = r"C:\Users\Fraens\Documents\Fraens\Youtube\TCG Sorter\CNN-Test\CardScannerCNN_02\data\scryfall_images\All Cards"

def setup_logging(log_level=logging.INFO):
    """Konfiguriert das Logging-System mit detaillierter Ausgabe."""
    # Erstelle Log-Verzeichnis falls es nicht existiert
    log_dir = "tcg_database/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Zeitstempel für Log-Datei
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"update_karten_{timestamp}.log")
    
    # Logging-Konfiguration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()  # Auch in Konsole ausgeben
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialisiert. Log-Datei: {log_file}")
    return logger

def get_bulk_catalog() -> List[Dict]:
    """Holt den Bulk-Katalog von Scryfall (Liste aller Bulk-Datensätze)."""
    logger = logging.getLogger(__name__)
    logger.info("Rufe aktuelle Bulk-Data-Informationen von Scryfall ab...")
    
    try:
        response = requests.get(SCRYFALL_BULK_DATA_API, timeout=30)
        response.raise_for_status()
        logger.info(f"Scryfall API erfolgreich kontaktiert. Status: {response.status_code}")
        payload = response.json()
        items = payload.get("data", [])
        logger.debug(f"Bulk-Data Antwort enthält {len(items)} Einträge")
        return items
        
    except requests.RequestException as e:
        logger.error(f"Fehler beim Abrufen der Scryfall API: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Fehler beim Parsen der Scryfall API Antwort: {e}")
        raise


def resolve_bulk_entry(catalog: List[Dict], bulk_type: str) -> Dict:
    for item in catalog:
        if item.get("type") == bulk_type:
            return item
    raise ValueError(f"Bulk-Typ '{bulk_type}' nicht im Scryfall-Katalog gefunden.")

def download_bulk_json(url: str, out_dir: str) -> str:
    """
    Lädt die Bulk-JSON-Datei herunter, aber nur wenn sie neuer ist als die vorhandene.
    Verwendet den ursprünglichen Dateinamen von Scryfall (der bereits einen Zeitstempel enthält).
    """
    logger = logging.getLogger(__name__)
    logger.info("Prüfe ob neue Version verfügbar ist...")
    
    try:
        # Extrahiere Dateinamen aus der URL
        from urllib.parse import urlparse
        parsed_url = urlparse(url)
        original_filename = os.path.basename(parsed_url.path)
        
        if not original_filename or not original_filename.endswith('.json'):
            # Fallback falls der Dateiname nicht aus der URL extrahiert werden kann
            original_filename = "default-cards.json"
            logger.warning("Konnte Dateinamen nicht aus URL extrahieren, verwende Fallback")
        
        logger.info(f"Server-Dateiname: {original_filename}")
        new_filepath = os.path.join(out_dir, original_filename)
        
        # Prüfe ob bereits eine Datei mit diesem Namen existiert
        if os.path.exists(new_filepath):
            file_size = os.path.getsize(new_filepath)
            logger.info(f"Aktuelle Version bereits vorhanden: {original_filename} ({file_size / (1024*1024):.1f} MB)")
            return new_filepath
        
        # Suche nach vorhandenen default-cards*.json Dateien (mit verschiedenen Zeitstempeln)
        import glob
        existing_files = glob.glob(os.path.join(out_dir, "default-cards*.json"))
        
        if existing_files:
            logger.info(f"Gefundene ältere Versionen: {[os.path.basename(f) for f in existing_files]}")
            # Lösche alte Versionen
            for old_file in existing_files:
                old_size = os.path.getsize(old_file)
                logger.info(f"Lösche alte Version: {os.path.basename(old_file)} ({old_size / (1024*1024):.1f} MB)")
                os.remove(old_file)
        
        # Lade neue Version herunter
        logger.info(f"Starte Download von {url}")
        logger.info(f"Speichere als: {original_filename}")
        
        start_time = datetime.now()
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        
        total_size = int(r.headers.get('content-length', 0))
        logger.info(f"Download-Größe: {total_size / (1024*1024):.1f} MB")
        
        # Progress-Bar für Download
        with open(new_filepath, "wb") as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=f"Download {original_filename}") as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        final_size = os.path.getsize(new_filepath)
        avg_speed = final_size / duration / (1024*1024)
        
        logger.info(f"Download abgeschlossen: {original_filename}")
        logger.info(f"Finale Größe: {final_size / (1024*1024):.1f} MB, Dauer: {duration:.1f}s, Durchschnittsgeschwindigkeit: {avg_speed:.1f} MB/s")
        
        return new_filepath
        
    except Exception as e:
        logger.error(f"Fehler beim Download: {e}")
        raise

def load_cards_from_file(file_path: str) -> List[dict]:
    logger = logging.getLogger(__name__)
    logger.info(f"Lese Karten aus {file_path} (streamweise verarbeitung für große Dateien)...")
    
    file_size = os.path.getsize(file_path)
    logger.info(f"Dateigröße: {file_size / (1024*1024):.1f} MB")
    
    if file_size > 50 * 1024 * 1024 and ijson is not None:  # Über 50MB - verwende ijson, falls vorhanden
        logger.info("Datei ist größer als 50MB - verwende streamweise Verarbeitung")
        return load_cards_streaming(file_path)
    else:
        # Normale Verarbeitung für kleinere Dateien
        logger.info("Verwende normale JSON-Verarbeitung für kleinere Datei")
        try:
            start_time = datetime.now()
            with open(file_path, "r", encoding="utf-8") as f:
                cards = json.load(f)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"{len(cards)} Karten geladen in {duration:.1f} Sekunden.")
            return cards
        except MemoryError as e:
            logger.warning(f"Speicher-Fehler bei normaler Verarbeitung: {e}")
            logger.info("Wechsle zu streamweiser Verarbeitung...")
            return load_cards_streaming(file_path)

def load_cards_streaming(file_path: str) -> List[dict]:
    """Lädt Karten streamweise aus einer großen JSON-Datei mit ijson."""
    logger = logging.getLogger(__name__)
    logger.info("Verwende ijson für streamweise JSON-Verarbeitung...")
    
    cards = []
    start_time = datetime.now()
    last_log_time = start_time
    
    if ijson is None:
        raise RuntimeError("ijson ist nicht installiert. Bitte 'pip install ijson' ausführen oder kleinere Datei verwenden.")
    try:
        with open(file_path, "rb") as f:
            # Parse das JSON-Array streamweise - jedes Element ist ein vollständiges Kartenobjekt
            card_count = 0
            
            logger.info("Starte ijson.items parsing...")
            
            # Verwende ijson.items um vollständige Objekte zu extrahieren
            for card in ijson.items(f, 'item'):
                cards.append(card)
                card_count += 1
                
                # Debug: Zeige erste Karte
                if card_count == 1:
                    logger.debug(f"Erste Karte (Typ: {type(card)}): {str(card)[:200]}...")
                    if isinstance(card, dict):
                        logger.info(f"Erste Karte - ID: {card.get('id')}, Name: {card.get('name')}")
                
                # Fortschritts-Logging
                current_time = datetime.now()
                if card_count % 5000 == 0:
                    duration_since_start = (current_time - start_time).total_seconds()
                    duration_since_last = (current_time - last_log_time).total_seconds()
                    cards_per_sec = 5000 / duration_since_last if duration_since_last > 0 else 0
                    logger.info(f"{card_count} Karten gelesen ({cards_per_sec:.1f} Karten/s, Gesamt: {duration_since_start:.1f}s)")
                    last_log_time = current_time
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        avg_speed = len(cards) / total_duration if total_duration > 0 else 0
        
        logger.info(f"Streamweise Verarbeitung abgeschlossen: {len(cards)} Karten geladen in {total_duration:.1f} Sekunden")
        logger.info(f"Durchschnittliche Geschwindigkeit: {avg_speed:.1f} Karten/s")
        
        # Zusätzliche Debug-Informationen
        if cards:
            logger.debug(f"Letzte Karte - ID: {cards[-1].get('id', 'N/A')}, Name: {cards[-1].get('name', 'N/A')}")
        
        return cards
        
    except Exception as e:
        logger.error(f"Fehler beim streamweisen Laden der Karten: {e}")
        raise

# BUG: Bei Karte Herbology Instructor werden keine image_uris geladen. Grund dafür ist im JSON gibt es zwei image_uris
def upsert_cards(cards: List[dict], db_path: Optional[str] = None):
    logger = logging.getLogger(__name__)
    
    if db_path is None:
        db_path = KARTEN_DB
    
    logger.info(f"Starte Datenbankoperationen mit {len(cards)} Karten")
    logger.info(f"Datenbank-Pfad: {db_path}")
        
    # Stelle sicher, dass der database Ordner existiert
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        # Verwende den bestehenden Store (stellt Schema sicher, inkl. karten/card_images)
        store = SqliteEmbeddingStore(db_path, emb_dim=1024)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM karten")
        existing_count = cur.fetchone()[0]
        logger.info(f"Aktuelle Anzahl Karten in Datenbank: {existing_count}")
        
        logger.info(f"Verarbeite {len(cards)} Karten in Batches...")
        
        count = 0
        inserted = 0
        updated = 0
        errors = 0
        batch_size = 1000  # Verarbeite in kleineren Batches
        start_time = datetime.now()
        
        # Progress-Bar für Karten-Import
        with tqdm(total=len(cards), desc="Importiere Karten in DB", unit="Karten") as pbar:
            for i in range(0, len(cards), batch_size):
                batch = cards[i:i + batch_size]
                batch_errors = 0
                
                for card in batch:
                    # Prüfe ob card ein Dictionary ist
                    if not isinstance(card, dict):
                        logger.warning(f"Karte ist kein Dictionary: {type(card)} - {str(card)[:100]}...")
                        batch_errors += 1
                        continue
                    
                    try:
                        # Nutze Store, der Preise/JSON-Spalten sauber setzt (UPSERT)
                        store.upsert_card(card)
                        inserted += 1
                    except Exception as e:
                        logger.warning(f"Fehler bei upsert_card für {card.get('id','unbekannt')}: {e}")
                        batch_errors += 1
                
                count += len(batch)
                errors += batch_errors
                
                # Commit nach jedem Batch für bessere Performance
                conn.commit()
                
                # Update Progress-Bar
                pbar.update(len(batch))
        
        # Finale Statistiken
        cur.execute("SELECT COUNT(*) FROM karten")
        final_count = cur.fetchone()[0]
        new_cards = final_count - existing_count
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logger.info("=== DATENBANKOPERATION ABGESCHLOSSEN ===")
        logger.info(f"Verarbeitete Karten: {count}")
        logger.info(f"Fehler: {errors}")
        logger.info(f"Karten in DB vorher: {existing_count}")
        logger.info(f"Karten in DB nachher: {final_count}")
        logger.info(f"Neue/Aktualisierte Karten: {new_cards}")
        logger.info(f"Gesamtdauer: {total_duration:.1f} Sekunden")
        logger.info(f"Durchschnittsgeschwindigkeit: {count / total_duration:.1f} Karten/s")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Kritischer Fehler bei Datenbankoperationen: {e}")
        raise

def analyze_json_structure(file_path: str):
    """Analysiert die Struktur der JSON-Datei um das richtige ijson Pattern zu finden."""
    print("Analysiere JSON-Struktur...")
    
    with open(file_path, "rb") as f:
        # Lese nur die ersten paar Events um die Struktur zu verstehen
        event_count = 0
        for prefix, event, value in ijson.parse(f):
            print(f"Event {event_count}: prefix='{prefix}', event='{event}', value='{str(value)[:50]}...'")
            event_count += 1
            if event_count > 20:  # Zeige nur die ersten 20 Events
                break
    
    print("Struktur-Analyse beendet.")

def get_image_uri(card: dict, version_order: Optional[List[str]] = None) -> Optional[str]:
    order = ["large", "normal", "border_crop"] if not version_order else version_order
    if "image_uris" in card and card.get("image_uris"):
        img_uris = card["image_uris"]
    elif "card_faces" in card and card.get("card_faces"):
        face = card["card_faces"][0]
        img_uris = face.get("image_uris", {})
    else:
        return None
    for key in order:
        if key in img_uris and img_uris.get(key):
            return img_uris.get(key)
    return None


def build_filename(card: dict) -> str:
    """Erstellt Dateinamen mit Scryfall-ID (id) als Identifier, nicht oracle_id."""
    set_code = (card.get("set") or "UNK").upper()
    collector = (card.get("collector_number") or "0").replace("/", "-")
    lang = card.get("lang") or "xx"
    # Wichtig: Verwende Scryfall-ID (id), nicht oracle_id
    scryfall_id = card.get("id") or "unknown"
    return f"{set_code}_{collector}_{lang}_{scryfall_id}.jpg"


def ensure_relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def download_image(session: requests.Session, url: str, out_path: Path) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    resp = session.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with out_path.open("wb") as handle:
        for chunk in resp.iter_content(8192):
            if chunk:
                handle.write(chunk)


def main():
    # Initialisiere Logging als erstes
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Lädt Scryfall Bulk-Daten (unique_artwork) und optional Bilder; befüllt die SQLite-DB.")
    parser.add_argument("--force-download", action="store_true", 
                       help="Erzwingt den Download einer neuen JSON-Datei, auch wenn bereits eine existiert")
    parser.add_argument("--db-path", type=str, default=KARTEN_DB,
                       help=f"Pfad zur Datenbank (Standard: {KARTEN_DB})")
    parser.add_argument("--bulk-data-dir", type=str, default=str(Path(DEFAULT_OUT_DIR)/"bulk"),
                       help="Verzeichnis für Bulk-Daten (Standard: Unterordner 'bulk' im Bildpfad)")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                       help="Zielordner für Scryfall-Bilder (Standard: Unique Artwork Pfad)")
    parser.add_argument("--langs", type=str, default="en,de",
                       help="Komma-separierte Sprachcodes für Bilder (z.B. en,de). Standard: en,de")
    parser.add_argument("--bulk-types", type=str, default="all_cards",
                       help="Komma-separierte Bulk-Typen (all_cards, unique_artwork, default_cards, oracle_cards, rulings). Standard: all_cards")
    parser.add_argument("--download-images", action="store_true", default=True,
                       help="Bilder herunterladen (Standard: an)")
    parser.add_argument("--no-download-images", dest="download_images", action="store_false",
                       help="Bild-Download deaktivieren")
    parser.add_argument("--skip-db-import", action="store_true",
                       help="Überspringt DB-Import (nützlich wenn DB bereits befüllt und nur Bilder ergänzt werden)")
    parser.add_argument("--purge-without-images", action="store_true",
                       help="Entfernt nach Import alle Karten aus der DB, die keinen Eintrag in card_images haben")
    parser.add_argument("--image-version-order", type=str, default="normal",
                       help="Reihenfolge der Bild-Varianten (Standard: nur normal)")
    parser.add_argument("--skip-download", action="store_true",
                       help="Überspringt den Download und verwendet nur vorhandene JSON-Datei")
    parser.add_argument("--analyze-json", action="store_true",
                       help="Analysiert nur die JSON-Struktur ohne Verarbeitung")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Aktiviert detailliertes Debug-Logging")
    
    args = parser.parse_args()
    
    # Setze Log-Level basierend auf verbose Flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug-Logging aktiviert")
    
    # Verwende die übergebenen Argumente
    karten_db = args.db_path
    bulk_data_dir = args.bulk_data_dir
    
    logger.info("=== SCRYFALL KARTENDATEN UPDATE GESTARTET ===")
    logger.info(f"Datenbank-Pfad: {karten_db}")
    logger.info(f"Bulk-Data-Verzeichnis: {bulk_data_dir}")
    logger.info(f"Bilder-Ausgabe: {args.out_dir}")
    logger.info(f"Sprachen: {args.langs}")
    logger.info(f"Bulk-Typen: {args.bulk_types}")
    logger.info(f"Force Download: {args.force_download}")
    logger.info(f"Skip Download: {args.skip_download}")
    logger.info(f"Analyze JSON: {args.analyze_json}")
    
    try:
        start_time = datetime.now()
        
        # Schritt 1: Bulk-Katalog abrufen und URLs auflösen
        logger.info("--- Schritt 1: Bulk-Katalog abrufen ---")
        catalog = get_bulk_catalog()
        requested_types = [t.strip() for t in (args.bulk_types.split(",") if args.bulk_types else []) if t.strip()]
        bulk_map: Dict[str, Dict] = {}
        for t in requested_types:
            try:
                item = resolve_bulk_entry(catalog, t)
                bulk_map[t] = item
                logger.info(f"Bulk '{t}': {item.get('download_uri')} (size: {item.get('size','?')} bytes, updated: {item.get('updated_at','?')})")
            except Exception as e:
                logger.warning(f"Bulk-Typ '{t}' nicht verfügbar: {e}")
        
        # Schritt 2: Bulk-JSON herunterladen (prüft automatisch auf Updates)
        logger.info("--- Schritt 2: Bulk-JSON Download/Prüfung ---")
        os.makedirs(bulk_data_dir, exist_ok=True)
        
        if args.force_download:
            logger.info("Force-Download aktiviert - lösche vorhandene Bulk-Dateien für die gewählten Bulk-Typen")
            import glob
            for t in requested_types:
                pattern = f"{t.replace('_','-')}*.json"
                existing_files = glob.glob(os.path.join(bulk_data_dir, pattern))
                for old_file in existing_files:
                    logger.info(f"Lösche vorhandene Datei: {os.path.basename(old_file)}")
                    os.remove(old_file)
        
        downloaded_files: Dict[str, Optional[str]] = {t: None for t in requested_types}
        if not args.skip_download:
            for t, item in bulk_map.items():
                try:
                    downloaded_files[t] = download_bulk_json(item["download_uri"], bulk_data_dir)
                except Exception as e:
                    logger.error(f"Download für '{t}' fehlgeschlagen: {e}")
        else:
            logger.info("Download übersprungen - verwende vorhandene Dateien nach Namensmuster.")
            import glob
            for t in requested_types:
                pattern = f"{t.replace('_','-')}*.json"
                candidates = glob.glob(os.path.join(bulk_data_dir, pattern))
                if candidates:
                    downloaded_files[t] = sorted(candidates)[-1]
                    logger.info(f"Verwende vorhandene Datei für {t}: {os.path.basename(downloaded_files[t])}")
                else:
                    logger.error(f"Keine lokale Datei für {t} in {bulk_data_dir} gefunden.")
        
        # Wenn nur Analyse gewünscht ist
        if args.analyze_json:
            logger.info("--- JSON-Struktur-Analyse ---")
            for t, path in downloaded_files.items():
                if path:
                    logger.info(f"Analysiere {t}: {os.path.basename(path)}")
                    analyze_json_structure(path)
            return

        # Schritt 3: Karten laden und in DB schreiben
        logger.info("--- Schritt 3: Bulk-Import in Datenbank ---")
        # Unterstütze verschiedene Bulk-Typen
        cards_path = (downloaded_files.get("all_cards") or 
                     downloaded_files.get("unique_artwork") or 
                     downloaded_files.get("default_cards"))

        total_cards = 0
        total_images = 0
        langs = {s.strip() for s in (args.langs.split(',') if args.langs else []) if s.strip()}

        # Store initialisieren (Schema/Tabellen)
        store = SqliteEmbeddingStore(karten_db, emb_dim=1024)

        # Bild-Variantenreihenfolge aus CLI
        version_order = [v.strip() for v in (args.image_version_order.split(',') if args.image_version_order else []) if v.strip()]

        if cards_path:
            bulk_type = os.path.basename(cards_path).split('-')[0]
            cards = load_cards_from_file(cards_path)
            total_cards = len(cards)
            logger.info(f"{bulk_type.upper()} Cards: {total_cards} Karten geladen")

            # *** WICHTIG: Alle Karten in die SQLite-DB schreiben (alle Metadaten, alle Sprachen) ***
            if args.skip_db_import:
                logger.info("DB-Import übersprungen (--skip-db-import gesetzt)")
            else:
                logger.info("Schreibe ALLE Karten in Datenbank (alle Sprachen, alle Metadaten)...")
                upsert_cards(cards, db_path=karten_db)

            # Bild-Download nur wenn gewünscht
            if not args.download_images:
                logger.info("Bild-Download übersprungen (--no-download-images gesetzt)")
                total_images = 0
            else:
                # Vorab-Schätzung Bilder
                candidate_images = [c for c in cards if (not langs or (c.get('lang') in langs)) and get_image_uri(c, version_order)]
                est_count = len(candidate_images)
                logger.info(f"Vorab-Schätzung: {est_count} Bilder für Download (Sprachen: {sorted(langs) if langs else 'alle'})")

                # Bild-Download nur für EN/DE mit oracle_id-Ordner
                out_dir = Path(args.out_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                session = requests.Session()
                session.headers.update({"User-Agent": "TCG-Sorter/1.0"})

                downloaded_imgs = 0
                
                # Progress-Bar für Bild-Download
                with tqdm(total=est_count, desc="Download Bilder (EN/DE)", unit="Bilder") as pbar:
                    for card in cards:
                        # Nur EN/DE-Bilder
                        if langs and card.get("lang") not in langs:
                            continue

                        url = get_image_uri(card, version_order)
                        if not url:
                            continue

                        filename = build_filename(card)

                        # *** NEU: Ordner nach oracle_id ***
                        oracle_id = card.get("oracle_id") or "no_oracle"
                        oracle_dir = out_dir / oracle_id
                        out_path = oracle_dir / filename

                        try:
                            download_image(session, url, out_path)

                            rel = ensure_relative_path(out_path)
                            try:
                                store.get_or_create_image(
                                    scryfall_id=card.get("id"),
                                    oracle_id=card.get("oracle_id"),
                                    file_path=rel,
                                    source="scryfall",
                                    language=card.get("lang"),
                                    is_training=True,
                                )
                                downloaded_imgs += 1
                            except Exception as e:
                                logger.debug(f"card_images upsert fehlgeschlagen ({filename}): {e}")
                        except Exception as e:
                            logger.debug(f"Bild-Download fehlgeschlagen ({filename}): {e}")
                        
                        pbar.update(1)
                        
                total_images = downloaded_imgs
        else:
            logger.warning("Keine Bulk-Datei vorhanden – Bilder-Download und Kartenimport übersprungen.")
        
        # Finale Statistiken
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logger.info("=== SCRYFALL KARTENDATEN UPDATE ABGESCHLOSSEN ===")
        logger.info(f"Gesamtdauer: {total_duration:.1f} Sekunden ({total_duration/60:.1f} Minuten)")
        if cards_path:
            logger.info(f"Bulk JSON: {os.path.basename(cards_path)}")
        logger.info(f"DB: {karten_db}")
        if cards_path:
            logger.info(f"Karten importiert (alle Sprachen): {total_cards}")
            logger.info(f"Bilder heruntergeladen (EN/DE): {total_images}")
            logger.info(f"Bildpfad: {args.out_dir}")
            logger.info(f"Ordnerstruktur: <oracle_id>/<set>_<collector>_<lang>_<scryfall_id>.jpg")

        # Orphan-Purge: entferne karten-Einträge ohne zugehöriges Bild, wenn gewünscht
        if args.purge_without_images:
            try:
                with sqlite3.connect(karten_db) as conn:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        DELETE FROM karten
                        WHERE id NOT IN (
                            SELECT DISTINCT scryfall_id FROM card_images
                        )
                        """
                    )
                    removed = cur.rowcount if cur.rowcount is not None else 0
                    conn.commit()
                    logger.info(f"Orphan-Purge: {removed} Karten ohne Bilder entfernt.")
            except Exception as e:
                logger.warning(f"Orphan-Purge fehlgeschlagen: {e}")
        
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()