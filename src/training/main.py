"""
Pipeline-Starter fuer Training/Analyse.

Standard: config.train5k.yaml + Szenario "full".
Über --scenario laesst sich das YAML-Default ueberschreiben.
"""
import argparse
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import yaml

# Ensure project root on sys.path so module imports work when executed via absolute python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TCG-Sorter Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.train5k.yaml",  # Default explizit auf 5k-Config legen
        help="Pfad zur YAML-Konfiguration (Default: config.train5k.yaml)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Szenario-Name zum Ueberschreiben von pipeline.scenario in der Config",
    )
    return parser.parse_args()


def build_step_commands(config_path: str, cfg: dict) -> Dict[str, Dict[str, List[str]]]:
    """Mapping Step-Name -> Kommando + Beschreibung."""
    oracle_json = cfg.get("analysis", {}).get("oracle_summary_file", "debug/oracle_cluster_summary.json")
    return {
        "coarse": {
            "cmd": ["python", "-m", "src.training.train_coarse", "--config", config_path],
            "desc": "Coarse-Training",
        },
        "export_analysis_coarse": {
            "cmd": ["python", "-m", "src.training.export_embeddings", "--config", config_path, "--mode", "analysis"],
            "desc": "Embeddings (Analysis) aus Coarse-Modell exportieren",
        },
        "oracle_analysis_for_hardneg": {
            "cmd": [
                "python",
                "-m",
                "tools.analyze_oracle_clusters",
                "--config",
                config_path,
                "--out-json",
                oracle_json,
            ],
            "desc": "Oracle-Cluster-Analyse fuer Hard-Negatives",
        },
        "build_hard_negatives": {
            "cmd": [
                "python",
                "-m",
                "src.training.build_hard_negatives",
                "--config",
                config_path,
                "--groups",
                "suspect_overlap,suspect_cluster_spread",
                "--top-k",
                str(cfg.get("training", {}).get("fine", {}).get("hard_negatives", {}).get("top_k", 20)),
            ],
            "desc": "Hard-Negatives fuer problematische Karten erzeugen",
        },
        "fine": {
            "cmd": ["python", "-m", "src.training.train_fine", "--config", config_path],
            "desc": "Fine-Training (Triplet + CE)",
        },
        "export_runtime_fine": {
            "cmd": ["python", "-m", "src.training.export_embeddings", "--config", config_path, "--mode", "runtime"],
            "desc": "Embeddings (Runtime) aus Fine-Modell exportieren",
        },
        "export_analysis_fine": {
            "cmd": ["python", "-m", "src.training.export_embeddings", "--config", config_path, "--mode", "analysis"],
            "desc": "Embeddings (Analysis) aus Fine-Modell exportieren",
        },
        "analyze_embeddings": {
            "cmd": ["python", "-m", "tools.analyze_embeddings", "--config", config_path, "--mode", "analysis"],
            "desc": "Analyse der Embeddings (Q-Werte, Distanzen)",
        },
        "oracle_analysis_final": {
            "cmd": [
                "python",
                "-m",
                "tools.analyze_oracle_clusters",
                "--config",
                config_path,
                "--out-json",
                oracle_json,
            ],
            "desc": "Finale Oracle-Cluster-Analyse mit aktuellem Modell",
        },
        # Hier lassen sich zukuenftig weitere Steps einhaengen, z. B.:
        # "partial_fine": {...},
        # "rebuild_embeddings_only": {...},
    }


def run(cmd: List[str], description: str) -> None:
    """Fuehrt einen Schritt aus und bricht bei Fehler ab."""
    print(f"\n===== STARTE: {description} =====")
    print("COMMAND:", " ".join(cmd))
    start = time.perf_counter()
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"FEHLER in Schritt '{description}': {exc}")
        sys.exit(1)
    elapsed = time.perf_counter() - start
    print(f"===== FERTIG: {description} ===== (Dauer: {elapsed/60:.2f} min)")


def main() -> None:
    args = parse_args()
    config_path = args.config or "config.train5k.yaml"
    cfg = load_config(config_path)
    pipeline_start = time.perf_counter()

    # Kurze Übersicht wichtiger Parameter einmalig ausgeben
    train_coarse = cfg.get("training", {}).get("coarse", {})
    train_fine = cfg.get("training", {}).get("fine", {})
    paths = cfg.get("paths", {})
    hard_cfg = train_fine.get("hard_negatives", {})
    print(
        "[CFG] coarse: "
        f"bs={train_coarse.get('batch_size')} epochs={train_coarse.get('epochs')} "
        f"lr={train_coarse.get('lr')} workers={train_coarse.get('num_workers')} "
        f"cache={train_coarse.get('cache_images')} cache_size={train_coarse.get('cache_size')}"
    )
    print(
        "[CFG] fine: "
        f"bs={train_fine.get('batch_size')} epochs={train_fine.get('epochs')} "
        f"lr={train_fine.get('lr')} margin={train_fine.get('margin')} freeze={train_fine.get('freeze_ratio')} "
        f"workers={train_fine.get('num_workers')} "
        f"cache={train_fine.get('cache_images')} cache_size={train_fine.get('cache_size')} "
        f"triplet_w={train_fine.get('triplet_weight')} ce_w={train_fine.get('ce_weight')}"
    )
    print(
        f"[CFG] paths: models_dir={paths.get('models_dir')} embeddings_dir={paths.get('embeddings_dir')} "
        f"oracle_json={cfg.get('analysis', {}).get('oracle_summary_file')} "
        f"hard_negs_file={hard_cfg.get('file')} hard_topk={hard_cfg.get('top_k')}"
    )

    pipeline_cfg = cfg.get("pipeline", {}) if isinstance(cfg, dict) else {}
    scenario_name = args.scenario or pipeline_cfg.get("scenario") or "full"
    scenarios = pipeline_cfg.get("scenarios", {})
    scenario_steps = scenarios.get(scenario_name)
    if not scenario_steps:
        print(f"[WARN] Szenario '{scenario_name}' nicht gefunden. Abbruch.")
        sys.exit(1)

    for step in scenario_steps:
        step_config_path = config_path
        # Für den Coarse-Export forcieren wir das coarse-Modell per temporärer Config
        if step == "export_analysis_coarse":
            tmp_cfg = deepcopy(cfg)
            models_dir = Path(tmp_cfg.get("paths", {}).get("models_dir", "./models"))
            tmp_cfg.setdefault("embedding_export_analysis", {})
            tmp_cfg["embedding_export_analysis"]["model_path"] = str(models_dir / "encoder_coarse.pt")
            debug_dir = Path(tmp_cfg.get("paths", {}).get("debug_dir", "./debug"))
            debug_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = debug_dir / "tmp_config_coarse_export.yaml"
            with tmp_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(tmp_cfg, f, sort_keys=False, allow_unicode=True)
            step_config_path = str(tmp_path)

        step_map = build_step_commands(step_config_path, cfg)
        if step not in step_map:
            print(f"[WARN] Step '{step}' ist nicht bekannt und wird uebersprungen.")
            continue
        meta = step_map[step]
        run(meta["cmd"], meta["desc"])

    total = time.perf_counter() - pipeline_start
    print(f"\nPipeline abgeschlossen – Szenario '{scenario_name}' durchlaufen. Gesamt: {total/60:.2f} min")


if __name__ == "__main__":
    main()
