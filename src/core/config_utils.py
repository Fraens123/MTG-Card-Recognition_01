from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Liest die YAML-Konfiguration und validiert leere Dateien."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config-Datei nicht gefunden: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not cfg:
        raise ValueError(f"Config-Datei ist leer: {cfg_path}")
    return cfg


def get_training_config(cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    Extrahiert die Trainingskonfiguration fuer coarse oder fine.
    Globale Werte wie Pfade, Encoder-Settings und Image-Specs werden automatisch gemischt.
    """
    training_cfg = cfg.get("training", {})
    key = mode.lower()
    if key not in training_cfg:
        raise ValueError(f"Unbekannter Trainingsmodus '{mode}'. Verfuegbare Modi: {list(training_cfg.keys())}")
    mode_cfg = copy.deepcopy(training_cfg[key])

    merged: Dict[str, Any] = {
        "paths": copy.deepcopy(cfg.get("paths", {})),
        "images": copy.deepcopy(cfg.get("images", {})),
        "encoder": copy.deepcopy(cfg.get("encoder", {})),
        "debug": copy.deepcopy(cfg.get("debug", {})),
    }
    merged.update(mode_cfg)
    merged["augment"] = copy.deepcopy(mode_cfg.get("augment", {}))
    return merged
