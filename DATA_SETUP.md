# Daten-Setup für MTG Card Recognition

## 📁 Ordnerstruktur

```
CardScannerCNN_02/
├── data/
│   ├── scryfall_images/          # Original Scryfall-Scans
│   ├── scryfall_augmented/       # Augmentierte Scryfall-Varianten  
│   └── camera_images/            # Echte Pi Cam Bilder (später)
├── models/                       # Trainierte Model-Weights
├── output_matches/               # Runtime Visualisierungen
└── scripts/
    └── augment_scryfall.py       # Scryfall-Augmentierung
```

## 🖼️ Bild-Formate

### Scryfall-Bilder (Original)
- **Ort:** `data/scryfall_images/`
- **Format:** `{set_code}_{collector_number}_{card_name}_{card_uuid}.jpg`
- **Beispiel:** `10E_31_Pacifism_686352f6.jpg`

### Augmentierte Scryfall-Bilder (Training)
- **Ort:** `data/scryfall_augmented/`
- **Format:** `{set_code}_{collector_number}_{card_name}_{card_uuid}_aug{N}.jpg`
- **Beispiel:** `10E_31_Pacifism_686352f6_aug1.jpg`

### Echte Kamera-Bilder (später)
- **Ort:** `data/camera_images/`
- **Format:** Beliebig (noch nicht verwendet)

## 🚀 Workflow

### 1. Scryfall-Bilder vorbereiten

**Format:** `10E_31_Pacifism_686352f6.jpg` ✅ (korrekt, nicht ändern!)

```bash
# 1. Ihre Scryfall-Bilder direkt nach data/scryfall_images/ kopieren
# 2. Keine Umbennung nötig - Format ist bereits korrekt!
```

### 2. Augmentierte Trainingsbilder generieren

```bash
# Aus Scryfall-Bildern augmentierte Trainingsbilder erstellen
python scripts/augment_scryfall.py --variants 3

# Mit spezifischen Ordnern
python scripts/augment_scryfall.py \
    --scryfall-dir data/scryfall_images \
    --augmented-dir data/scryfall_augmented \
    --variants 3

# Nur wenige Bilder für Tests
python scripts/augment_scryfall.py --variants 2 --max-images 100
```

### 3. Model trainieren

```bash
# Triplet-Loss Training mit Original + Augmentierungen
python -m src.cardscanner.train_triplet \
    --images-dir data/scryfall_images \
    --augmented-dir data/scryfall_augmented \
    --epochs 10 \
    --embed-dim 512 \
    --out models/encoder_resnet50_512.pt
```

### 4. Datenbank mit Embeddings füllen

```bash
# PostgreSQL + pgvector Setup (einmalig)
# Siehe README.md für DB-Setup

# Embeddings generieren und in DB laden
python -m src.cardscanner.embed_db \
    --images-dir data/scryfall_images \
    --model models/encoder_resnet50_512.pt
```

## 🎯 Augmentierung-Details

Das `augment_scryfall.py` Skript simuliert realistische Kamera-Bedingungen:

- **Perspektive:** Schräg fotografierte Karten
- **Rotation:** ±15° Drehung
- **Beleuchtung:** Helligkeit/Kontrast-Variationen
- **Unschärfe:** Bewegungs-/Fokus-Blur
- **Rauschen:** Kamera-Sensor-Noise
- **Farbtemperatur:** Warme/kalte Lichtquellen

## ⚙️ Konfiguration

In `.env` oder Environment-Variablen:

```env
# Daten-Pfade
SCRYFALL_IMAGES_DIR=./data/scryfall_images
SCRYFALL_AUGMENTED_DIR=./data/scryfall_augmented
CAMERA_IMAGES_DIR=./data/camera_images
OUTPUT_DIR=./output_matches

# Model-Parameter  
VECTOR_DIM=512
EF_SEARCH=32
TOP_K=10
DIST_THRESHOLD=0.3

# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/mtg
```

## 📊 Empfohlene Datenmengen

- **Scryfall-Bilder:** 10.000+ verschiedene Karten
- **Augmentierte Bilder:** 3x Varianten = 30.000+ Trainingsbilder  
- **Embedding-Dimension:** 512-D (guter Kompromiss)
- **Batch-Size Training:** 32-64 (je nach GPU)

## 🔍 Debugging

```bash
# Prüfe Dateinamen-Format
ls data/scryfall_images/ | head -5

# Prüfe augmentierte Bilder
ls data/scryfall_augmented/ | head -5

# Test einzelnes Bild
python -c "
from src.cardscanner.dataset import parse_scryfall_filename
print(parse_scryfall_filename('10E_31_Pacifism_686352f6.jpg'))
# Sollte ausgeben: ('686352f6', '10E', '31', 'Pacifism')
"
```