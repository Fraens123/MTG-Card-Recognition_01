from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def resolve_logdir(config: dict, override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    output_dir = Path(config.get("paths", {}).get("output_dir", "./output_matches"))
    return (output_dir / "runs").resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch TensorBoard pointing at output_matches/runs (or custom logdir)."
    )
    parser.add_argument("--config", default="config.yaml", help="Pfad zur YAML-Konfiguration.")
    parser.add_argument(
        "--logdir",
        default=None,
        help="Optional explizites Log-Verzeichnis (überschreibt config).",
    )
    parser.add_argument("--host", default="localhost", help="TensorBoard Host (default: localhost).")
    parser.add_argument("--port", default="6006", help="TensorBoard Port (default: 6006).")
    args = parser.parse_args()

    config = load_config(args.config)
    logdir = resolve_logdir(config, args.logdir)
    if not logdir.exists():
        raise FileNotFoundError(
            f"Log-Verzeichnis {logdir} existiert nicht. Erst Training laufen lassen?"
        )

    print(f"Starte TensorBoard auf {args.host}:{args.port}")
    print(f"Log-Verzeichnis: {logdir}")
    print("Abbrechen mit CTRL+C.")

    # Einige Installationen haben keinen tensorboard.__main__, daher direkt tensorboard.main.
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        str(logdir),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
