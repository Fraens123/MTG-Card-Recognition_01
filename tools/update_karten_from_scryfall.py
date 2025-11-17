import requests
import sqlite3
import json
import os
import argparse
import ijson
import logging
from datetime import datetime

# Konfiguration
KARTEN_DB = "tcg_database/database/karten.db"
SCRYFALL_BULK_DATA_API = "https://api.scryfall.com/bulk-data"

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

def get_latest_default_cards_url():
    """Ruft die aktuellste Default Cards URL von der Scryfall API ab."""
    logger = logging.getLogger(__name__)
    logger.info("Rufe aktuelle Bulk-Data-Informationen von Scryfall ab...")
    
    try:
        response = requests.get(SCRYFALL_BULK_DATA_API, timeout=30)
        response.raise_for_status()
        logger.info(f"Scryfall API erfolgreich kontaktiert. Status: {response.status_code}")
        
        bulk_data = response.json()
        logger.debug(f"Bulk-Data Antwort enthält {len(bulk_data.get('data', []))} Einträge")
        
        # Suche nach dem "default_cards" Eintrag
        for item in bulk_data["data"]:
            if item["type"] == "default_cards":
                url = item["download_uri"]
                file_size = item.get("size", "unbekannt")
                updated_at = item.get("updated_at", "unbekannt")
                logger.info(f"Default Cards URL gefunden: {url}")
                logger.info(f"Dateigröße: {file_size} Bytes, Letzte Aktualisierung: {updated_at}")
                return url
        
        logger.error("Kein 'default_cards' Eintrag in der Scryfall Bulk-Data API gefunden!")
        raise ValueError("Keine Default Cards URL in der Scryfall Bulk-Data API gefunden!")
        
    except requests.RequestException as e:
        logger.error(f"Fehler beim Abrufen der Scryfall API: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Fehler beim Parsen der Scryfall API Antwort: {e}")
        raise

def download_bulk_json(url: str, out_dir: str):
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
        
        with open(new_filepath, "wb") as f:
            downloaded = 0
            last_log_time = datetime.now()
            
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                
                # Zeige Fortschritt alle 10MB oder alle 30 Sekunden
                current_time = datetime.now()
                if (downloaded % (10 * 1024 * 1024) == 0 or 
                    (current_time - last_log_time).seconds >= 30) and total_size > 0:
                    progress = (downloaded / total_size) * 100
                    speed = downloaded / (current_time - start_time).total_seconds() / (1024*1024)
                    logger.info(f"Download-Fortschritt: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB) - Geschwindigkeit: {speed:.1f} MB/s")
                    last_log_time = current_time
        
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

def load_cards_from_file(file_path: str):
    logger = logging.getLogger(__name__)
    logger.info(f"Lese Karten aus {file_path} (streamweise verarbeitung für große Dateien)...")
    
    file_size = os.path.getsize(file_path)
    logger.info(f"Dateigröße: {file_size / (1024*1024):.1f} MB")
    
    if file_size > 50 * 1024 * 1024:  # Über 50MB - verwende ijson
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

def load_cards_streaming(file_path: str):
    """Lädt Karten streamweise aus einer großen JSON-Datei mit ijson."""
    logger = logging.getLogger(__name__)
    logger.info("Verwende ijson für streamweise JSON-Verarbeitung...")
    
    cards = []
    start_time = datetime.now()
    last_log_time = start_time
    
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
def upsert_cards(cards, db_path=None):
    logger = logging.getLogger(__name__)
    
    if db_path is None:
        db_path = KARTEN_DB
    
    logger.info(f"Starte Datenbankoperationen mit {len(cards)} Karten")
    logger.info(f"Datenbank-Pfad: {db_path}")
        
    # Stelle sicher, dass der database Ordner existiert
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Erstelle die Tabelle falls sie nicht existiert
        logger.info("Erstelle/überprüfe Datenbanktabelle...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS karten (
                id TEXT PRIMARY KEY,
                oracle_id TEXT,
                name TEXT,
                "set" TEXT,
                set_name TEXT,
                collector_number TEXT,
                type_line TEXT,
                oracle_text TEXT,
                mana_cost TEXT,
                cmc INTEGER,
                colors TEXT,
                color_identity TEXT,
                rarity TEXT,
                image_uris TEXT,
                legalities TEXT
            )
        """)
        
        # Prüfe aktuelle Anzahl Einträge in DB
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
        last_log_time = start_time
        
        for i in range(0, len(cards), batch_size):
            batch = cards[i:i + batch_size]
            batch_data = []
            batch_errors = 0
            
            for card in batch:
                # Prüfe ob card ein Dictionary ist
                if not isinstance(card, dict):
                    logger.warning(f"Karte ist kein Dictionary: {type(card)} - {str(card)[:100]}...")
                    batch_errors += 1
                    continue
                
                try:
                    batch_data.append((
                        card.get("id"),
                        card.get("oracle_id"),
                        card.get("name"),
                        card.get("set"),
                        card.get("set_name"),
                        card.get("collector_number"),
                        card.get("type_line") or "",
                        card.get("oracle_text") or "",
                        card.get("mana_cost") or "",
                        int(card.get("cmc") or 0),  # Konvertiere zu int für SQLite
                        ",".join(card.get("colors", [])),
                        ",".join(card.get("color_identity", [])),
                        card.get("rarity"),
                        json.dumps(card.get("image_uris")) if card.get("image_uris") else "",
                        json.dumps(card.get("legalities")) if card.get("legalities") else ""
                    ))
                except Exception as e:
                    logger.warning(f"Fehler bei Karte {card.get('id', 'unbekannt')}: {e}")
                    batch_errors += 1
            
            if batch_data:
                try:
                    # Führe Batch-Insert aus
                    cur.executemany("""
                        INSERT INTO karten (id, oracle_id, name, "set", set_name, collector_number, type_line, oracle_text, mana_cost, cmc, colors, color_identity, rarity, image_uris, legalities)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            oracle_id=excluded.oracle_id,
                            name=excluded.name,
                            "set"=excluded."set",
                            set_name=excluded.set_name,
                            collector_number=excluded.collector_number,
                            type_line=excluded.type_line,
                            oracle_text=excluded.oracle_text,
                            mana_cost=excluded.mana_cost,
                            cmc=excluded.cmc,
                            colors=excluded.colors,
                            color_identity=excluded.color_identity,
                            rarity=excluded.rarity,
                            image_uris=excluded.image_uris,
                            legalities=excluded.legalities
                    """, batch_data)
                    
                    # Zähle Inserts vs Updates (vereinfacht)
                    batch_inserted = len(batch_data)
                    inserted += batch_inserted
                    
                except Exception as e:
                    logger.error(f"Fehler beim Batch-Insert: {e}")
                    batch_errors += len(batch_data)
            
            count += len(batch_data)
            errors += batch_errors
            
            # Commit nach jedem Batch für bessere Performance
            conn.commit()
            
            # Fortschritts-Logging
            current_time = datetime.now()
            if count % 5000 == 0 or i + batch_size >= len(cards):
                duration_since_start = (current_time - start_time).total_seconds()
                duration_since_last = (current_time - last_log_time).total_seconds()
                cards_per_sec = 5000 / duration_since_last if duration_since_last > 0 else 0
                progress = ((i + batch_size) / len(cards)) * 100
                logger.info(f"Fortschritt: {progress:.1f}% - {count} Karten verarbeitet ({cards_per_sec:.1f} Karten/s)")
                last_log_time = current_time
        
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

def main():
    # Initialisiere Logging als erstes
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="Lädt Kartendaten von Scryfall herunter und speichert sie in der Datenbank")
    parser.add_argument("--force-download", action="store_true", 
                       help="Erzwingt den Download einer neuen JSON-Datei, auch wenn bereits eine existiert")
    parser.add_argument("--db-path", type=str, default=KARTEN_DB,
                       help=f"Pfad zur Datenbank (Standard: {KARTEN_DB})")
    parser.add_argument("--bulk-data-dir", type=str, default="tcg_database/bulk_data",
                       help="Verzeichnis für Bulk-Daten (Standard: tcg_database/bulk_data)")
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
    logger.info(f"Force Download: {args.force_download}")
    logger.info(f"Skip Download: {args.skip_download}")
    logger.info(f"Analyze JSON: {args.analyze_json}")
    
    try:
        start_time = datetime.now()
        
        # Schritt 1: Aktuelle Bulk-URL abrufen
        logger.info("--- Schritt 1: Bulk-URL abrufen ---")
        bulk_url = get_latest_default_cards_url()
        
        # Schritt 2: Bulk-JSON herunterladen (prüft automatisch auf Updates)
        logger.info("--- Schritt 2: Bulk-JSON Download/Prüfung ---")
        os.makedirs(bulk_data_dir, exist_ok=True)
        
        if args.force_download:
            logger.info("Force-Download aktiviert - lösche alle vorhandenen Dateien")
            import glob
            existing_files = glob.glob(os.path.join(bulk_data_dir, "default-cards*.json"))
            for old_file in existing_files:
                logger.info(f"Lösche vorhandene Datei: {os.path.basename(old_file)}")
                os.remove(old_file)
        
        if not args.skip_download:
            # Prüfe immer auf Updates, außer wenn explizit übersprungen
            json_file = download_bulk_json(bulk_url, bulk_data_dir)
        else:
            # Nur bei --skip-download: verwende vorhandene Datei
            logger.info("Download übersprungen - suche lokale Datei")
            import glob
            existing_files = glob.glob(os.path.join(bulk_data_dir, "default-cards*.json"))
            
            if existing_files:
                # Nimm die neueste Datei (sortiert nach Dateinamen = Zeitstempel)
                json_file = sorted(existing_files)[-1]
                logger.info(f"Verwende vorhandene Datei: {os.path.basename(json_file)}")
            else:
                logger.error(f"Keine JSON-Datei gefunden in {bulk_data_dir}!")
                logger.info("Tipp: Führe das Skript ohne --skip-download aus, um eine Datei herunterzuladen.")
                return
        
        # Wenn nur Analyse gewünscht ist
        if args.analyze_json:
            logger.info("--- JSON-Struktur-Analyse ---")
            analyze_json_structure(json_file)
            return
        
        # Schritt 3: Karten einlesen
        logger.info("--- Schritt 3: Karten einlesen ---")
        cards = load_cards_from_file(json_file)
        
        # Schritt 4: In DB eintragen
        logger.info("--- Schritt 4: Datenbank-Import ---")
        upsert_cards(cards, karten_db)
        
        # Finale Statistiken
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logger.info("=== SCRYFALL KARTENDATEN UPDATE ABGESCHLOSSEN ===")
        logger.info(f"Gesamtdauer: {total_duration:.1f} Sekunden ({total_duration/60:.1f} Minuten)")
        logger.info(f"JSON-Datei: {os.path.basename(json_file)}")
        logger.info(f"Datenbank: {karten_db}")
        
    except Exception as e:
        logger.error(f"Kritischer Fehler: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()