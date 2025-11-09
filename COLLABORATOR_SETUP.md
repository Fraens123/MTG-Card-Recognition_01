# Setup-Anweisungen für Collaborators

## 🚀 Schnellstart für neue Entwickler

### 1. Repository klonen
```bash
git clone https://github.com/Fraens123/MTG-Card-Recognition.git
cd MTG-Card-Recognition
```

### 2. Virtual Environment einrichten
```bash
# Python Virtual Environment erstellen
python -m venv venv

# Aktivieren
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. Konfiguration erstellen
```bash
# Konfigurationsdatei kopieren
cp config.example.yaml config.yaml

# config.yaml bearbeiten:
# - Database URL anpassen
# - Pfade für lokales System anpassen
```

### 5. Ordnerstruktur erstellen
```bash
# Erforderliche Ordner erstellen
mkdir -p data/scryfall_images
mkdir -p data/scryfall_augmented
mkdir -p data/camera_images
mkdir -p models
mkdir -p output_matches
```

### 6. PostgreSQL Setup (falls lokal entwickelt wird)
```bash
# PostgreSQL installieren und Datenbank erstellen
createdb mtg

# Extensions und Tabellen erstellen
psql -U postgres -d mtg -f scripts/init_db.sql
psql -U postgres -d mtg -f scripts/create_hnsw.sql
```

## 📁 Daten hinzufügen

### Scryfall-Bilder (für Training)
1. Scryfall-Bilder beschaffen im Format: `10E_31_Pacifism_686352f6.jpg`
2. In `data/scryfall_images/` ablegen
3. Augmentierungen generieren:
```bash
python scripts/augment_scryfall.py --variants 3
```

## 🧠 Model Training

```bash
# Training starten (nach dem Daten-Setup)
python -m src.cardscanner.train_triplet \
    --images-dir data/scryfall_images \
    --augmented-dir data/scryfall_augmented \
    --epochs 10 \
    --embed-dim 512 \
    --out models/encoder_resnet50_512.pt
```

## 🚀 Service testen

```bash
# API-Service starten
uvicorn src.cardscanner.service.main:app --reload --host 0.0.0.0 --port 8000

# In anderem Terminal testen:
curl -X POST "http://localhost:8000/health"
```

## 🛠️ Development Workflow

### Neue Features entwickeln
```bash
# Neuen Branch erstellen
git checkout -b feature/mein-feature

# Änderungen machen...
# Testen...

# Committen und pushen
git add .
git commit -m "Add: Beschreibung des Features"
git push origin feature/mein-feature

# Pull Request auf GitHub erstellen
```

### Updates vom main branch holen
```bash
git checkout main
git pull origin main
```

## 🐛 Häufige Probleme

### Virtual Environment
```bash
# Falls venv nicht funktioniert:
python -m pip install --user virtualenv
python -m virtualenv venv
```

### Dependencies
```bash
# Falls PyTorch Probleme macht:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Für CUDA (GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### PostgreSQL
```bash
# Connection testen:
psql -U postgres -h localhost -d mtg -c "SELECT version();"

# pgvector Extension prüfen:
psql -U postgres -d mtg -c "SELECT * FROM pg_extension WHERE extname='vector';"
```

## 📞 Support

Bei Problemen:
1. GitHub Issues erstellen
2. Logs in `output_matches/` prüfen
3. `git status` und `git log` ausführen
4. Konfiguration in `config.yaml` überprüfen

---

**Happy Coding! 🎉**