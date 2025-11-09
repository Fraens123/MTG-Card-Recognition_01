# MTG Card Recognition System

Ein offline Magic: The Gathering Kartenerkennungs-System mit CNN-basierten Embeddings und JSON-basierter Datenbank für schnelle Ähnlichkeitssuche.

## 🚀 Features

- **CNN-basierte Embeddings** mit ResNet50 + Triplet Loss
- **Offline-Betrieb** - keine externe API nötig
- **JSON-Database** mit Cosine Similarity Search
- **Kamera-ähnliche Augmentierung** für robustes Training
- **L2-normalisierte 256-D Vektoren** für optimale Performance
- **Automatische Bildgrößenerkennung** (MTG Aspect-Ratio)
- **Side-by-Side Visualisierung** der Erkennungsergebnisse

## 📁 Projektstruktur

```
CardScannerCNN_02/
├── src/cardscanner/           # Haupt-Bibliothek
│   ├── config.py             # [VERALTET] Ersetzt durch config.yaml
│   ├── model.py              # ResNet50 Encoder Model
│   ├── dataset.py            # Triplet Dataset für Training
│   ├── transforms.py         # Bild-Preprocessing Pipeline
│   ├── db.py                 # JSON-Database Interface
│   ├── embed_db.py           # Embedding-Generierung
│   ├── train_triplet.py      # CNN Training mit Triplet Loss
│   ├── augment_cards.py      # Kamera-ähnliche Augmentierung
│   ├── generate_embeddings.py # Embedding-Pipeline
│   ├── recognize_cards.py    # Kartenerkennung mit Visualisierung
│   └── service/              # FastAPI Service (Optional)
│       ├── main.py           # API Endpoints
│       ├── camera.py         # Pi Kamera Support
│       └── visualizer.py     # Match-Visualisierung
├── scripts/                  # SQL-Skripte (nicht verwendet)
│   ├── init_db.sql          # [VERALTET] PostgreSQL Setup
│   └── create_hnsw.sql      # [VERALTET] HNSW-Index
├── config.yaml              # 📋 Zentrale Konfigurationsdatei
├── requirements.txt          # Python Dependencies
├── data/                     # Daten-Verzeichnisse
│   ├── scryfall_images/     # Original Scryfall-Bilder (INPUT)
│   ├── scryfall_augmented/  # Augmentierte Trainingsbilder
│   ├── camera_images/       # Pi Camera Testbilder (INPUT)
│   └── cards.json           # JSON-Database mit Embeddings
├── models/                   # Trainierte Modelle
│   └── encoder_mtg_cards.pt # Gespeichertes CNN-Modell
└── output_matches/          # Erkennungs-Visualisierungen (OUTPUT)
```

## ⚙️ Technische Details

### Architektur
- **Backbone:** ResNet50 (ImageNet pre-trained)
- **Embedding-Dimension:** 256-D (konfigurierbar)
- **Loss:** Triplet Loss für metrisches Lernen  
- **Normalisierung:** L2-Normalisierung der Embeddings
- **Distanz-Metrik:** Cosine Similarity (Scikit-Learn)
- **Database:** JSON-Format für einfache Portabilität

### Training-Parameter
- **Batch-Size:** 8 (für große Bilder optimiert)
- **Learning-Rate:** 0.0001 (Adam Optimizer)
- **Early Stopping:** 5 Epochen ohne Verbesserung
- **Auto-Resize:** Automatische Erkennung der optimalen Bildgröße

### Augmentierung
- **Realistische Kamera-Bedingungen:** Belichtung, Kontrast, Blur, Rauschen
- **Geometrische Transformationen:** Rotation, perspektivische Verzerrung
- **Weißer Hintergrund:** Für Rotation/Perspektive (konfigurierbar)
- **Format-Erhaltung:** Original Scryfall-Format wird beibehalten

## 🛠️ Installation

### Voraussetzungen
- Python 3.10+
- CUDA-fähige GPU (für Training, optional)
- Ca. 2GB freier Speicher für Augmentierung

### Setup

1. **Repository klonen**
```bash
git clone https://github.com/Fraens123/MTG-Card-Recognition.git
cd MTG-Card-Recognition
```

2. **Virtual Environment erstellen**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Dependencies installieren**
```bash
pip install -r requirements.txt
```

4. **Verzeichnisse erstellen**
```bash
mkdir data\scryfall_images data\camera_images models output_matches
```

## � Quick Start (Play-Button freundlich)

### **Minimaler Workflow für VS Code:**

1. **📂 Daten vorbereiten**
   ```bash
   # Scryfall-Kartenbilder in data/scryfall_images/ kopieren
   # Camera-Testbilder in data/camera_images/ kopieren
   ```

2. **🎮 Scripts mit Play-Button ausführen (in dieser Reihenfolge):**
   - `src/cardscanner/augment_cards.py` ▶️ (Augmentierung)
   - `src/cardscanner/train_triplet.py` ▶️ (CNN Training)  
   - `src/cardscanner/generate_embeddings.py` ▶️ (Database erstellen)
   - `src/cardscanner/recognize_cards.py` ▶️ (Testen)

**Alle Scripts nutzen automatisch die Parameter aus `config.yaml` - kein Terminal nötig!**

### **SCHRITT 1: Scryfall-Bilder vorbereiten**

**Ziel:** Original MTG-Karten von Scryfall in das System laden

**Input:** Scryfall-Kartenbilder (JPG/PNG)  
**Output:** `data/scryfall_images/` gefüllt mit Originalkarten  

```bash
# Scryfall-Bilder manuell nach data/scryfall_images/ kopieren
# Format ist beliebig - keine Umbenennung nötig!
# Beispiele: war-65-rescuer-sphinx.png, soi-167-inner-struggle.png
```

**⚠️ Wichtig:** 
- Mindestens 5-10 verschiedene Karten für sinnvolles Training
- Bildformat wird automatisch erkannt und angepasst
- Dateiname kann beliebig sein

---

### **SCHRITT 2: Augmentierung ausführen**

**Skript:** `src/cardscanner/augment_cards.py`  
**Zweck:** Generiert kamera-ähnliche Varianten der Scryfall-Bilder für robustes Training

**Input:** `data/scryfall_images/` (Original-Bilder)  
**Output:** `data/scryfall_augmented/` (Augmentierte Trainingsbilder)

```bash
# Standard-Augmentierung (aus config.yaml)
python -m src.cardscanner.augment_cards

# Oder mit benutzerdefinierten Parametern
python -m src.cardscanner.augment_cards \
    --num_augmentations 5 \
    --brightness_min 0.7 \
    --brightness_max 1.3 \
    --background_color white
```

**Parameter:**
- `--num_augmentations`: Anzahl Varianten pro Bild (default: aus config.yaml)
- `--brightness_min/max`: Helligkeitsbereich (Belichtungssimulation)
- `--contrast_min/max`: Kontrastbereich 
- `--blur_max`: Maximale Bewegungsunschärfe
- `--noise_max`: Sensorrauschen-Intensität
- `--rotation_max`: Maximale Rotation in Grad
- `--background_color`: "white" oder "black" für Transformationen

**Was passiert:**
- Simuliert verschiedene Kamera-Bedingungen (Belichtung, Fokus, Winkel)
- Erstellt realistische Trainingsdaten ohne echte Kamera-Aufnahmen
- Behält Original-Format bei für Konsistenz

---

### **SCHRITT 3: CNN-Modell trainieren**

**Skript:** `src/cardscanner/train_triplet.py`  
**Zweck:** Trainiert das CNN-Modell mit Triplet Loss für Kartenrepräsentationen

**Input:** `data/scryfall_images/` + `data/scryfall_augmented/`  
**Output:** `models/encoder_mtg_cards.pt` (trainiertes Modell)

```bash
# Standard-Training (Parameter aus config.yaml)
python -m src.cardscanner.train_triplet

# Oder mit benutzerdefinierten Parametern
python -m src.cardscanner.train_triplet \
    --epochs 20 \
    --batch_size 4 \
    --embed_dim 512
```

**Parameter:**
- `--epochs`: Anzahl Trainingsdurchläufe (default: aus config.yaml)
- `--batch_size`: Bilder pro Batch (default: 8)
- `--embed_dim`: Embedding-Dimension (default: 256)
- `--learning_rate`: Lernrate (default: 0.0001)

**Was passiert:**
- Lädt alle Original- und augmentierten Bilder
- Erkennt automatisch optimale Bildgröße (MTG Aspect-Ratio)
- Trainiert ResNet50 mit Triplet Loss
- Early Stopping bei Stagnation
- Speichert bestes Modell automatisch

**⏱️ Dauer:** 5-30 Minuten je nach GPU und Datenmenge

---

### **SCHRITT 4: Embeddings generieren**

**Skript:** `src/cardscanner/generate_embeddings.py` ▶️ **Kann mit Play-Button gestartet werden**  
**Zweck:** Erstellt Embeddings für alle Scryfall-Bilder und speichert in JSON-Database

**Input:** `data/scryfall_images/` + trainiertes Modell  
**Output:** `data/cards.json` (Database mit Embeddings)

```bash
# Standard-Generierung (nutzt config.yaml Parameter)
python src/cardscanner/generate_embeddings.py

# Alternative: Als Python-Modul
python -m src.cardscanner.generate_embeddings
```

**Config-Parameter (in config.yaml):**
```yaml
database:
  embedding_mode: "original"    # Empfohlen: ~400 Embeddings, schneller
  # embedding_mode: "augmented" # Alternativ: ~1,600 Embeddings, mehr Variationen
```

**Was passiert:**
- Lädt trainiertes CNN-Modell
- Wählt Bildquelle basierend auf `embedding_mode`:
  - `"original"`: Nutzt `data/scryfall_images/` → ~400 Embeddings, schnellere Suche
  - `"augmented"`: Nutzt `data/scryfall_augmented/` → ~1,600 Embeddings, mehr Variationen  
- Speichert in JSON-Format mit automatischem Backup
- **Empfehlung:** `"original"` Modus zeigt identische Erkennungsqualität bei besserer Performance

**📊 Output-Format:**
```json
{
  "Rescuer Sphinx (war)": {
    "embedding": [0.123, -0.456, ...], // 256-D Vektor
    "card_uuid": "12345",
    "set_code": "war",
    "collector_number": "65",
    "image_path": "data/scryfall_images/war-65-rescuer-sphinx.png"
  }
}
```

---

### **SCHRITT 5: Kamera-Bilder vorbereiten** (optional)

**Ziel:** Echte Kamera-Aufnahmen zum Testen der Erkennung

**Input:** Pi Camera oder Handy-Fotos von MTG-Karten  
**Output:** `data/camera_images/` gefüllt mit Testbildern

```bash
# Kamera-Bilder manuell nach data/camera_images/ kopieren
# Format: beliebig (JPG, PNG)
# Beispiel: Inner Struggle_01.jpg, Rescuer Sphinx_01.jpg
```

**💡 Tipp:** 
- Verwenden Sie dieselben Karten wie in Scryfall-Images
- Verschiedene Winkel, Beleuchtung, Hintergründe testen
- Mindestens 1-2 Bilder pro bekannte Karte

---

### **SCHRITT 6: Kartenerkennung testen**

**Skript:** `src/cardscanner/recognize_cards.py` ▶️ **Kann mit Play-Button gestartet werden**  
**Zweck:** Testet die Erkennung mit visueller Ausgabe der besten Matches

**Input:** `data/camera_images/` + `data/cards.json`  
**Output:** `output_matches/` (Side-by-Side Vergleichsgrafiken)

```bash
# Einfachste Verwendung: Play-Button in VS Code drücken!
# → Nutzt automatisch alle Parameter aus config.yaml

# Oder manuell im Terminal:
python src/cardscanner/recognize_cards.py

# Mit benutzerdefinierten Parametern:
python src/cardscanner/recognize_cards.py --camera-dir data/camera_images --output-dir output_test

# Als Python-Modul:
python -m src.cardscanner.recognize_cards
```

**Parameter:**
- **Ohne Parameter:** Nutzt automatisch `config.yaml` (empfohlen für Play-Button)
- `--camera-dir`: Override Camera-Bildverzeichnis  
- `--output-dir`: Override Ausgabeverzeichnis
- `--model-path`: Override Modell-Pfad

**Was passiert:**
- ✅ **Play-Button-freundlich:** Prüft automatisch alle Abhängigkeiten
- Lädt alle Camera-Bilder aus `data/camera_images/`
- Berechnet Embeddings mit trainiertem Modell
- Sucht ähnlichste Karten in JSON-Database (Cosine Similarity)
- Erstellt Side-by-Side Vergleichsgrafiken
- Zeigt Similarity-Scores und Suchzeit an

**📊 Output:**
```
🎮 Play-Button Modus: Nutze Standard-Parameter aus config.yaml
🚀 MTG Card Similarity Search Testing
📷 Gefunden: 5 Camera-Bilder
[1/5] 🔍 Verarbeite: Inner Struggle_01.jpg
   🎯 Best Match: Inner Struggle (SOI) (0.9385)
   ⏱️ Search Time: 97.60ms
   💾 Vergleichsgrafik gespeichert: output_matches/Inner Struggle_01_comparison.png
```

**🔧 Troubleshooting:**
- **❌ Modell nicht gefunden:** → Zuerst `train_triplet.py` ausführen
- **❌ Database nicht gefunden:** → Zuerst `generate_embeddings.py` ausführen  
- **❌ Keine Camera-Bilder:** → Testbilder in `data/camera_images/` kopieren
- **❌ Camera-Verzeichnis nicht gefunden:** → Wird automatisch erstellt

---

## 🎯 Erkennungsqualität optimieren

### Training verbessern:
```bash
# Mehr Augmentierungen für bessere Generalisierung
python -m src.cardscanner.augment_cards --num_augmentations 10

# Längeres Training mit kleinerer Batch-Size
python -m src.cardscanner.train_triplet --epochs 50 --batch_size 4

# Höhere Embedding-Dimension für mehr Details
python -m src.cardscanner.train_triplet --embed_dim 512
```

### Neue Karten hinzufügen:
```bash
# 1. Neue Scryfall-Bilder in data/scryfall_images/ kopieren
# 2. Augmentierung wiederholen
python -m src.cardscanner.augment_cards
# 3. Modell neu trainieren
python -m src.cardscanner.train_triplet
# 4. Embeddings neu generieren
python -m src.cardscanner.generate_embeddings --backup
```

### Embedding-Qualität visualisieren (t-SNE)
Um den Embedding-Raum grafisch zu prüfen, kannst du das neue Tool unter `tools/visualization` verwenden. Beispielaufruf (PowerShell):

```powershell
.venv\Scripts\python.exe tools/visualization/tsne_embeddings.py `
    --input data/cards.json `
    --label-key set_code `
    --name-key name `
    --use-pca --pca-components 50 `
    --perplexity 35 --learning-rate 200 --n-iter 1500 `
    --output-plot tools/visualization/tsne_sets.png `
    --output-csv tools/visualization/tsne_sets.csv
```

Der Plot landet als PNG unter `tools/visualization/`, die CSV enthält zusätzlich pro Karte den Namen, das Label sowie den verwendeten Farbcode.

## 🔧 Konfiguration (config.yaml)

```yaml
# Database Configuration (Simple JSON-based)
database:
  path: "./data/cards.json"
  # Embedding Mode: "original" (empfohlen: schneller, weniger Speicher) oder "augmented" (mehr Variationen)
  embedding_mode: "original"
  
# Vector Configuration  
vector:
  dimension: 256

# Data Directories
data:
  scryfall_images: "./data/scryfall_images"
  scryfall_augmented: "./data/scryfall_augmented"
  camera_images: "./data/camera_images"
  output_dir: "./output_matches"

# Model Configuration
model:
  weights_path: "./models/encoder_mtg_cards.pt"
  embed_dim: 256
  
# Training Parameters
training:
  batch_size: 8
  learning_rate: 0.0001
  epochs: 50
  margin: 0.2
  early_stopping_patience: 5
  auto_detect_size: true  # Automatische Bildgrößenerkennung

# Augmentation Settings
augmentation:
  num_augmentations: 20
  brightness_min: 0.6
  brightness_max: 1.4
  contrast_min: 0.7
  contrast_max: 1.3
  blur_max: 2.0
  noise_max: 20.0
  rotation_max: 1.0
  perspective: 0.05
  shadow: 0.3
  background_color: "white"  # "white" oder "black"

# Hardware Settings
hardware:
  use_cuda: true
```

## 🔧 Troubleshooting

### Häufige Probleme:

**Import-Probleme beim Play-Button:**
Falls es zu Import-Problemen beim Play-Button kommt:
```bash
cd "C:\Users\Fraens\Documents\Fraens\Youtube\TCG Sorter\CNN-Test\CardScannerCNN_02"
& "./.venv/Scripts/python.exe" src/cardscanner/recognize_cards.py
```

**Unicode-Warnung bei Grafiken:**
Die Unicode-Warnungen (Glyph missing) bei der Grafik-Erstellung sind harmlos und beeinträchtigen die Funktionalität nicht. Die Grafiken werden korrekt erstellt.

**Training schlägt fehl:**
```bash
# GPU-Speicher zu wenig → Batch-Size reduzieren
python -m src.cardscanner.train_triplet --batch_size 2

# CUDA nicht verfügbar → CPU verwenden (langsam)
# config.yaml: hardware.use_cuda: false
```

**Schlechte Erkennung:**
```bash
# Mehr Augmentierungen generieren
python -m src.cardscanner.augment_cards --num_augmentations 15

# Länger trainieren
python -m src.cardscanner.train_triplet --epochs 100
```

**Speicherprobleme:**
```bash
# Weniger Augmentierungen
python -m src.cardscanner.augment_cards --num_augmentations 5

# Kleinere Embedding-Dimension
python -m src.cardscanner.train_triplet --embed_dim 128
```

## 📈 Performance-Metriken

**Typische Ergebnisse:**
- **Trainingszeit:** 10-20 Epochen für 10 Karten
- **Erkennungsgenauigkeit:** 85-95% bei guten Kamera-Bildern
- **Suchzeit:** ~100ms pro Bild
- **Speicherbedarf:** ~1-2GB für 100 Karten mit Augmentierung

---

**Entwickelt für offline Magic: The Gathering Kartensammlung** 🎴✨
