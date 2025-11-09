# Virtual Environment Setup

## Warum venv verwenden?

- **Isolation:** Projektspezifische Dependencies ohne Konflikte
- **Reproduzierbarkeit:** Exakte Package-Versionen für alle
- **Sicherheit:** Keine systemweiten Python-Änderungen
- **Portabilität:** Einfacher Transfer zwischen Entwicklung/Produktion

## Setup

### 1. Virtual Environment erstellen
```bash
python -m venv venv
```

### 2. Aktivieren
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 4. Nach der Arbeit deaktivieren
```bash
deactivate
```

## Entwicklung

```bash
# Immer erst aktivieren vor der Arbeit
venv\Scripts\activate

# Training
python -m src.cardscanner.train_triplet --epochs 10

# Service starten
uvicorn src.cardscanner.service.main:app --reload

# Deaktivieren wenn fertig
deactivate
```

## Raspberry Pi Deployment

```bash
# Auf dem Pi auch venv verwenden
python -m venv venv_pi
source venv_pi/bin/activate
pip install -r requirements.txt

# Service starten
uvicorn src.cardscanner.service.main:app --host 0.0.0.0 --port 8000
```