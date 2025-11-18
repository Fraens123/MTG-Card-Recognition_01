# TCG Sorter – Trainingspipeline

Leichtgewichtiger Workflow, um den Karten-Encoder zu trainieren und Embeddings zu exportieren. Der komplette Ablauf wird über `config.yaml` gesteuert und besteht aus drei Schritten (Coarse-Training → Fine-Training → Embedding-Export).

## Projektstruktur
```
CardScannerCNN_02/
├── src/
│   ├── core/
│   │   ├── config_utils.py       # Config laden + Merging
│   │   ├── augmentations.py      # CameraLikeAugmentor
│   │   ├── image_ops.py          # Crop- und Resize-Helfer
│   │   ├── embedding_utils.py    # build_card_embedding
│   │   └── model_builder.py      # CardEncoder + load/save
│   ├── datasets/
│   │   └── card_datasets.py      # Coarse- & Triplet-Datasets
│   └── training/
│       ├── train_coarse.py       # CE-Pretraining
│       ├── train_fine.py         # Triplet-Finetuning
│       └── export_embeddings.py  # JSON-Embedding-Export
├── config.yaml
├── requirements.txt
├── data/
│   ├── scryfall_images/
│   └── camera_images/
├── models/
└── output_matches/
```

## Setup
```bash
git clone <repo>
cd CardScannerCNN_02
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
mkdir -p data/scryfall_images data/camera_images models output_matches
```
Kartenbilder aus Scryfall nach `data/scryfall_images/`, Kameraaufnahmen nach `data/camera_images/` kopieren.

## Workflow
1. **Coarse-Training** (reine CE-Verluste auf Scryfall-Daten):
   ```bash
   python -m src.training.train_coarse --config config.yaml
   ```
2. **Fine-Training** (Triplet + CE mit Kamera-Augmentierung):
   ```bash
   python -m src.training.train_fine --config config.yaml
   ```
   Erwartet `models/encoder_coarse.pt` vom vorherigen Schritt.
3. **Embeddings exportieren** (JSON-Datenbank in `paths.embeddings_dir`):
   ```bash
   python -m src.training.export_embeddings --config config.yaml
   ```

## Recognition (optional)
Nach dem Export kannst du Kamera-Bilder gegen die Embedding-Datenbank matchen:
```bash
python -m src.recognize_cards --config config.yaml --camera-dir data/camera_images --output-dir output_matches
```
Das Skript lädt `models/encoder_fine.pt` und `database.path` aus der Config, schneidet die Kamera-ROI zu und sucht den besten Treffer per Cosine-Similarity.

## Konfiguration
`config.yaml` gliedert sich in die wichtigsten Abschnitte 
- `paths`: Scryfall-/Kamera-Verzeichnisse, Modelle, Embedding-Output
- `images`: Zielgrößen + Cropping für Full-Card-Bilder
- `encoder`: Backbone-Typ (`resnet18`/`resnet50`) und Embedding-Dimensionen
- `training.coarse` / `training.fine`: Batchgrößen, Learning-Rate, Epochen, Augmentierung

Alle Skripte lesen ausschließlich aus dieser Datei – Änderungen werden automatisch übernommen, ohne den Code anzufassen. Stelle nach jedem Trainingslauf sicher, dass das `models/`-Verzeichnis beschreibbar ist und genügend Speicherplatz für die Checkpoints vorhanden ist.

## Analyse & Visualisierung
Zur Qualitätsanalyse der Embeddings (Oracle-IDs) sowie zur visuellen Inspektion gibt es ein einheitliches Tool: `tools/visualization/oracle_cluster_viz.py`.

Standardverhalten:
- Es werden standardmäßig nur „Verdachtsfälle“ angezeigt (geflaggte Cluster/Points mit `overlap`, `spread` oder `metadata` gemäß Tabelle `oracle_quality`).
- Um alle Einträge zu sehen, verwende `--cluster-show-all` bzw. `--scatter-show-all`.

Beispiele (PowerShell):
```powershell
# Standard (nur Verdachtsfälle) – beide Ansichten: Cluster-Report + 2D-Scatter (PCA)
python -m tools.visualization.oracle_cluster_viz --config config.yaml --mode analysis --view both --scatter-method pca

# Standard (nur Verdachtsfälle) – nur Cluster-Report (HTML)
python -m tools.visualization.oracle_cluster_viz --config config.yaml --mode analysis --view cluster

# Standard (nur Verdachtsfälle) – nur Scatter (t-SNE)
python -m tools.visualization.oracle_cluster_viz --config config.yaml --mode analysis --view scatter --scatter-method tsne --perplexity 30

# ALLES anzeigen (beide Ansichten):
python -m tools.visualization.oracle_cluster_viz --config config.yaml --mode analysis --view both --cluster-show-all --scatter-show-all

# Nur Cluster vollständig anzeigen, Scatter weiter gefiltert lassen:
python -m tools.visualization.oracle_cluster_viz --config config.yaml --mode analysis --view both --cluster-show-all

# Nur Scatter vollständig anzeigen, Cluster weiter gefiltert lassen:
python -m tools.visualization.oracle_cluster_viz --config config.yaml --mode analysis --view both --scatter-show-all
```

Ergebnisse werden standardmäßig unter `debug/` mit Zeitstempel gespeichert. Im Scatter können Punkte angeklickt werden, um direkt in den passenden Cluster-Abschnitt des Cluster-Reports zu springen.

Hinweis: Die ehemaligen Skripte `render_oracle_clusters_html.py` und `scatter_oracle_clusters.py` wurden entfernt und im neuen, vereinheitlichten Skript `oracle_cluster_viz.py` zusammengeführt.
