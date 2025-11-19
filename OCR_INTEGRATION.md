# OCR Integration - Zwei-Stufen-Erkennung

## Übersicht

Die OCR-Integration erweitert die MTG-Kartenerkennung um eine zweite Erkennungsstufe zur präzisen Auswahl der richtigen Print-Variante:

**Stufe 1 (CNN)**: Embedding-basierte Ähnlichkeitssuche findet die richtige Karte (oracle_id)  
**Stufe 2 (OCR)**: Textextraktion wählt den korrekten Print (scryfall_id) basierend auf Set, Collector Number und Sprache

## Architektur

### Module: `src/OCR/`

- **`ocr_result.py`**: `OcrResult` Dataclass mit extrahierten Text-Daten
- **`ocr_engine.py`**: `run_ocr_for_card_image()` führt Tesseract-OCR mit festen Pixel-Crops aus
- **`ocr_scoring.py`**: `select_print_with_ocr()` wählt besten Print via Fuzzy-Matching (rapidfuzz)

### Integration: `src/recognize_cards.py`

Die Funktion `search_camera_image()` wurde um OCR-Pipeline erweitert:

1. **CNN-Suche**: Top-1-Match via Cosine Similarity → `best_match_cnn`
2. **Oracle-Lookup**: `db.get_cards_by_oracle_id(oracle_id)` → alle Print-Varianten
3. **OCR-Ausführung**: `run_ocr_for_card_image(card_img)` → `OcrResult`
4. **Print-Auswahl**: `select_print_with_ocr(candidates, ocr_result)` → finaler Match

## OCR-Crop-Koordinaten (Fest, Pixel-basiert)

Alle Crops verwenden absolute Pixel-Werte für Pi-Camera-Bilder (4056x3040):

```python
# Kartenname
CROP_X_START = 100
CROP_X_END = 2850
CROP_START_Y = 50
CROP_HEIGHT = 200
LEFT_TRIM_PERCENT = 0.02  # 2% von links
RIGHT_TRIM_PERCENT = 0.08  # 8% von rechts

# Collector Number (unten links)
COLLECTOR_NUMBER_CROP = (100, 3600, 500, 80)  # x, y, width, height

# Set ID (unten links)
SETID_CROP = (100, 3700, 500, 80)
```

## Installation

### 1. Python-Dependencies

```powershell
pip install pytesseract rapidfuzz
```

### 2. Tesseract-OCR (Windows)

Download von: https://github.com/UB-Mannheim/tesseract/wiki

Nach Installation Pfad zur `tesseract.exe` in `ocr_engine.py` anpassen:

```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

## Verwendung

### Automatisch (Standard)

OCR ist per Default aktiviert:

```powershell
python src/recognize_cards.py
```

### OCR deaktivieren

Im Code `search_camera_image()` aufrufen mit:

```python
best_match, similarity, time = search_camera_image(
    model, transform, db, camera_img_path, 
    device, config, art_cfg, 
    use_ocr=False  # OCR ausschalten
)
```

## Debug-Output

OCR-Crops und Ergebnisse werden gespeichert in:

```
./debug/ocr/
    {filename}_name_crop.png
    {filename}_collector_crop.png
    {filename}_setid_crop.png
```

## OCR-Scoring-Logik

### `score_candidate_with_ocr()`

Berechnet Konfidenz-Score (0-100) basierend auf:

- **Name-Match**: `rapidfuzz.fuzz.ratio()` auf Kartennamen
- **Collector Number-Match**: Exakter String-Vergleich (case-insensitive)
- **Set ID-Match**: Exakter String-Vergleich (case-insensitive)

Gewichtung: 50% Name, 25% Collector, 25% Set ID

### `select_print_with_ocr()`

Wählt Print mit höchstem Score:

1. Scores für alle Kandidaten berechnen
2. Sortieren nach Score (absteigend)
3. Top-1 zurückgeben

## Beispiel-Ablauf

```
[CNN] Top-1: Lightning Bolt (Set: 2XM, #141) - cos=0.95
[OCR] Lade alle Prints für Oracle-ID: 0e048c8f...
[OCR] Gefunden: 47 Print-Varianten
[OCR] Erkannt: Name='Lightning Bolt', Collector=196, Set=M11
[OCR] OCR wählt: Lightning Bolt (Set: M11, #196)
```

## Vorteile

✅ **Präzise Print-Auswahl**: Unterscheidung von Sets, Collector Numbers, Sprachen  
✅ **Robust**: CNN findet korrekte Karte, OCR verfeinert Auswahl  
✅ **Schnell**: OCR nur bei >1 Print-Variante  
✅ **Debug-fähig**: Crop-Visualisierung und detailliertes Logging

## Limitierungen

⚠️ **Tesseract-Genauigkeit**: OCR kann bei schlechter Bildqualität versagen  
⚠️ **Fixed Crops**: Koordinaten sind auf 4056x3040 Pi-Camera-Auflösung optimiert  
⚠️ **Performance**: OCR-Ausführung erhöht Latenz um ~50-100ms pro Karte

## Nächste Schritte

- [ ] OCR-Genauigkeit testen mit Validation-Set
- [ ] Adaptive Crop-Koordinaten für variable Bildauflösungen
- [ ] Konfidenz-Schwellwert für OCR-Fallback zu CNN-Match
- [ ] Multi-Language-Support (Deutsch, Französisch, etc.)
