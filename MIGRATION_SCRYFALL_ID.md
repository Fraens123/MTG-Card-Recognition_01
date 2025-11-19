# Migration: oracle_id → scryfall_id als Primärschlüssel

## Zusammenfassung der Änderungen

Das Projekt wurde erfolgreich umgestellt, sodass **Scryfall-ID (id)** der Primärschlüssel für Bilder, Trainingsklassen und Embeddings ist, während **oracle_id** nur noch als Metadatum zur logischen Gruppierung verwendet wird.

## Geänderte Dateien

### 1. **src/core/sqlite_store.py** ✅
- **Schema**: Bereits korrekt - `karten.id` (PRIMARY KEY) = Scryfall-ID
- **UNIQUE Index**: Geändert von `(oracle_id, file_path)` zu `(scryfall_id, file_path)` für `card_images`
- **get_or_create_image()**: Verwendet jetzt `(scryfall_id, file_path)` als eindeutigen Schlüssel
- **load_embeddings_with_meta()**: Gruppiert jetzt nach `scryfall_id` (Print-ID) statt oracle_id
- **Kommentare**: Hinzugefügt um klarzustellen dass scryfall_id der Primärschlüssel ist

**Datenmodell:**
```sql
-- Karten-Tabelle (Scryfall Prints)
CREATE TABLE karten (
    id TEXT PRIMARY KEY,           -- Scryfall-ID (Print-ID)
    oracle_id TEXT,                 -- Logische Karte (mehrere Prints)
    name TEXT,
    ...
);

-- Bilder-Tabelle
CREATE TABLE card_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scryfall_id TEXT NOT NULL,     -- FK zu karten.id
    oracle_id TEXT,                 -- Metadatum
    file_path TEXT NOT NULL,
    UNIQUE(scryfall_id, file_path) -- Jeder Print kann nur einmal denselben Pfad haben
);

-- Embeddings-Tabelle
CREATE TABLE card_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scryfall_id TEXT NOT NULL,     -- FK zu karten.id
    oracle_id TEXT,                 -- Metadatum für Analysen
    emb BLOB NOT NULL,
    mode TEXT NOT NULL,
    ...
);
```

### 2. **tools/update_karten_from_scryfall.py** ✅
- **build_filename()**: Verwendet jetzt `card.get("id")` (Scryfall-ID) statt `card.get("oracle_id")`
- **Dateinamenformat**: `{SET}_{COLLECTOR}_{LANG}_{SCRYFALL_ID}.jpg`

**Beispiel:**
```python
# Vorher (FALSCH):
"M20_123_en_abc123oracle.jpg"  # oracle_id

# Nachher (RICHTIG):
"M20_123_en_xyz789scryfall.jpg"  # scryfall_id
```

### 3. **tools/download_scryfall_cards_from_CSV.py** ✅
- **build_filename()**: Identische Änderung wie oben
- Dateinamen enthalten jetzt die eindeutige Scryfall-ID

### 4. **src/datasets/card_datasets.py** ✅
- **parse_scryfall_filename()**: Bereits korrekt implementiert
  - Extrahiert Scryfall-ID aus Dateinamen (letzter Teil)
  - Gibt `(scryfall_id, set_code, collector_number, card_name)` zurück
- **CoarseDataset**: Verwendet `card_uuid = scryfall_id` als Trainingsklasse
- **TripletImageDataset**: Verwendet `card_uuid = scryfall_id` für Triplet-Sampling
- **Kommentare**: Verdeutlicht, dass card_uuid die Scryfall-ID ist

### 5. **src/training/export_embeddings.py** ✅
**Wichtigste Änderungen:**
- **Neue Hilfsfunktion**: `_get_oracle_id_from_db(store, scryfall_id)`
  - Holt die oracle_id aus der `karten`-Tabelle
  - Vermeidet `oracle_id = scryfall_id` Fehler
  
- **_prepare_card_tensors()**:
  - `card_dict["card_uuid"]` = Scryfall-ID (aus Dateinamen)
  - Keine oracle_id mehr im card_dict
  
- **main()**:
  ```python
  # Für jede Karte (cid = scryfall_id):
  oracle_id = _get_oracle_id_from_db(store, cid)  # Lade echte oracle_id aus DB
  
  # Beim Speichern:
  store.get_or_create_image(
      scryfall_id=cid,        # Print-ID
      oracle_id=oracle_id,    # Logische Karte
      ...
  )
  
  store.add_embedding(
      scryfall_id=cid,        # Print-ID
      oracle_id=oracle_id,    # Logische Karte
      ...
  )
  ```

### 6. **src/core/sqlite_embeddings.py** ✅
- **load_embeddings_grouped_by_oracle()**: Kommentar hinzugefügt
  - Diese Funktion bleibt für Oracle-basierte Analysen (z.B. Cluster-Spread)
  - **Wichtig**: Für normale Vektorsuche wird `load_embeddings_with_meta()` verwendet (gruppiert nach scryfall_id)

### 7. **src/core/card_database.py** ✅
- **SimpleCardDB.load_from_sqlite()**:
  - Lädt Embeddings gruppiert nach `scryfall_id`
  - Fügt `oracle_id` als Metadatum hinzu
  ```python
  base = {
      "card_uuid": cid,                    # scryfall_id
      "oracle_id": meta.get("oracle_id"),  # Metadatum
      "name": meta.get("name"),
      ...
  }
  ```

### 8. **src/recognize_cards.py** ✅
- Keine Änderungen nötig - bereits korrekt!
- Verwendet `card_uuid` (=scryfall_id) aus den Suchergebnissen
- `oracle_id` ist bereits in den Metadaten verfügbar

## Datenfluss: Bild → Embedding → scryfall_id → oracle_id

### 1. Training (CoarseDataset / TripletImageDataset)
```
Bilddatei: M20_123_en_abc123scryfall.jpg
    ↓ parse_scryfall_filename()
scryfall_id: "abc123scryfall"
    ↓ Dataset gruppiert nach scryfall_id
Trainingsklasse / Label: scryfall_id (jeder Print ist eine eigene Klasse)
```

### 2. Embedding Export
```
Für jedes Bild:
    ↓ Dateiname parsen
scryfall_id: "abc123scryfall"
    ↓ Aus Datenbank laden
oracle_id: "xyz789oracle" (aus karten.oracle_id WHERE karten.id = scryfall_id)
    ↓ Speichern in DB
card_images:    (scryfall_id="abc123scryfall", oracle_id="xyz789oracle", file_path=...)
card_embeddings: (scryfall_id="abc123scryfall", oracle_id="xyz789oracle", emb=...)
```

### 3. Vektorsuche (Matching)
```
Kamerabild
    ↓ CNN Encoder
Query-Embedding
    ↓ Ähnlichkeitssuche in card_embeddings (gruppiert nach scryfall_id)
Best Match: scryfall_id="abc123scryfall" (konkreter Print)
    ↓ Metadaten aus DB laden
{
    "card_uuid": "abc123scryfall",      # Scryfall-ID (gefundener Print)
    "oracle_id": "xyz789oracle",         # Oracle-ID (logische Karte)
    "name": "Lightning Bolt",
    "set_code": "M20",
    "similarity": 0.95
}
    ↓ Optional: Alle Prints dieser oracle_id laden
SELECT * FROM karten WHERE oracle_id = "xyz789oracle"
→ Zeigt alle Lightning Bolt Varianten (Alpha, Beta, M20, etc.)
```

## Wichtige Konzepte

### Scryfall-ID (id) = Print-ID
- **Eindeutiger Identifier** für einen konkreten Druck einer Karte
- Beispiel: Lightning Bolt aus M20 hat eine andere Scryfall-ID als aus Alpha
- **Verwendung**: 
  - Primärschlüssel in allen Tabellen
  - Trainingsklassen im CNN
  - Embedding-Zuordnung
  - Bilddateinamen

### Oracle-ID = Logische Karte
- **Gemeinsamer Identifier** für alle Drucke derselben logischen Karte
- Beispiel: Alle Lightning Bolt Varianten haben dieselbe oracle_id
- **Verwendung**:
  - Metadatum für Gruppierung
  - Anzeige von Varianten
  - Analysen (Cluster-Spread, etc.)

## Migration durchführen

Da Sie die Datenbank neu aufbauen, sind keine speziellen Migrationsskripte nötig:

```bash
# 1. Alte Datenbank löschen
rm tcg_database/database/karten.db

# 2. Bilder neu von Scryfall herunterladen (mit korrekten Dateinamen)
python tools/update_karten_from_scryfall.py --force-download

# 3. Embeddings neu exportieren
python -m src.training.export_embeddings --mode runtime

# 4. Erkennung testen
python -m src.recognize_cards
```

## Validierung

Nach der Migration sollten diese Aussagen zutreffen:

✅ Alle Dateinamen in `data/scryfall_images/` enden mit der Scryfall-ID
✅ `karten.id` enthält Scryfall-IDs (nicht oracle_ids)
✅ `card_images.scryfall_id` verweist auf `karten.id`
✅ `card_embeddings.scryfall_id` verweist auf `karten.id`
✅ `oracle_id` ist in allen Tabellen nur ein zusätzliches Feld
✅ Vektorsuche findet konkrete Prints (scryfall_id)
✅ Aus dem Match kann man zur oracle_id navigieren und alle Varianten finden

## Troubleshooting

### Problem: "oracle_id not found in database"
→ Stelle sicher, dass `update_karten_from_scryfall.py` die oracle_id korrekt in die `karten`-Tabelle schreibt

### Problem: "Duplicate key violation on card_images"
→ Der UNIQUE Index wurde von `(oracle_id, file_path)` zu `(scryfall_id, file_path)` geändert
→ Lösche die alte Datenbank und baue sie neu auf

### Problem: "No matches found"
→ Stelle sicher, dass Embeddings mit `--mode runtime` exportiert wurden
→ Prüfe ob `card_embeddings.scryfall_id` korrekt befüllt ist

## Nächste Schritte

1. ✅ Code ist umgestellt
2. ⏳ Datenbank löschen und neu aufbauen
3. ⏳ Bilder neu herunterladen mit korrekten Dateinamen
4. ⏳ Embeddings neu exportieren
5. ⏳ Training durchführen (optional, falls Modell neu trainiert werden soll)
6. ⏳ Erkennung testen

---

**Stand:** 19. November 2025
**Autor:** GitHub Copilot (Claude Sonnet 4.5)
