from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config_utils import load_config
from src.core.embedding_utils import l2_normalize
from src.core.sqlite_embeddings import (
    ensure_oracle_quality_table,
    load_embeddings_grouped_by_oracle,
)
import sqlite3


DEFAULT_SPREAD_MAX = 0.90
DEFAULT_OVERLAP = 0.60


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Oracle-Cluster-Analyse: Spreads, Overlaps und Metadatenkonflikte")
    p.add_argument("--config", type=str, default="config.yaml", help="Pfad zur YAML-Konfiguration")
    p.add_argument("--sqlite", type=str, default=None, help="Pfad zur SQLite-DB (ueberschreibt config.database.sqlite_path)")
    p.add_argument("--mode", type=str, default="analysis", choices=["analysis", "runtime"], help="Embedding-Mode")
    p.add_argument("--out-csv", type=str, default=None, help="Optionaler Pfad fuer CSV-Report")
    p.add_argument("--out-json", type=str, default=None, help="Optionaler Pfad fuer JSON-Top-Report")
    p.add_argument("--csv-delimiter", type=str, default=",", help="CSV-Delimiter (z. B. ';' fuer DE)")
    p.add_argument("--decimal-comma", action="store_true", help="CSV: Dezimal-Komma verwenden (DE)")
    p.add_argument("--print-top", type=int, default=20, help="Top-N verdächtige Einträge in der Konsole ausgeben (0=aus)")
    p.add_argument("--spread-max-thr", type=float, default=DEFAULT_SPREAD_MAX, help="Schwelle fuer intra_max_dist-Flag")
    p.add_argument("--overlap-thr", type=float, default=DEFAULT_OVERLAP, help="Schwelle fuer Nachbar-Overlap (Zentroid-Distanz)")
    return p.parse_args()


def _resolve_paths_and_dims(args: argparse.Namespace) -> Tuple[Path, int, dict]:
    cfg = load_config(args.config)
    sqlite_path = args.sqlite or cfg.get("database", {}).get("sqlite_path") or "tcg_database/database/karten.db"
    emb_dim = int(
        cfg.get("encoder", {}).get("emb_dim")
        or cfg.get("vector", {}).get("dimension")
        or cfg.get("model", {}).get("embed_dim", 1024)
    )
    return Path(sqlite_path), emb_dim, cfg


def _resolve_thresholds(args: argparse.Namespace, cfg: dict) -> Tuple[float, float]:
    # Config-Werte lesen
    cfg_thr = cfg.get("analysis", {}).get("thresholds", {}) if isinstance(cfg, dict) else {}
    cfg_spread = cfg_thr.get("spread_max")
    cfg_overlap = cfg_thr.get("overlap")

    spread = args.spread_max_thr
    overlap = args.overlap_thr

    # Wenn CLI-Werte den Defaults entsprechen, interpretieren wir das als "nicht explizit gesetzt"
    # und nutzen vorrangig config-Werte (falls vorhanden)
    if spread == DEFAULT_SPREAD_MAX and cfg_spread is not None:
        try:
            spread = float(cfg_spread)
        except Exception:
            pass
    if overlap == DEFAULT_OVERLAP and cfg_overlap is not None:
        try:
            overlap = float(cfg_overlap)
        except Exception:
            pass
    return float(spread), float(overlap)


def _compute_cluster_spreads(grouped: Dict[str, np.ndarray]) -> Tuple[Dict[str, Dict], float, float]:
    stats: Dict[str, Dict] = {}
    mean_values: List[float] = []
    for oid, arr in grouped.items():
        if arr.size == 0:
            continue
        # L2-normalisieren (Sicherheitsnetz)
        arr = arr.astype(np.float32, copy=False)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-12, None)
        c = np.mean(arr, axis=0)
        c = l2_normalize(c)
        dists = np.linalg.norm(arr - c[None, :], axis=1)
        intra_mean = float(np.mean(dists)) if dists.size else 0.0
        intra_max = float(np.max(dists)) if dists.size else 0.0
        stats[oid] = {
            "centroid": c,
            "intra_mean_dist": intra_mean,
            "intra_max_dist": intra_max,
        }
        mean_values.append(intra_mean)

    global_mean = float(np.mean(mean_values)) if mean_values else float("nan")
    global_std = float(np.std(mean_values)) if mean_values else float("nan")
    return stats, global_mean, global_std


def _nearest_neighbors(centroids: Dict[str, np.ndarray]) -> Dict[str, Tuple[str, float]]:
    keys = list(centroids.keys())
    if len(keys) < 2:
        return {k: (None, math.inf) for k in keys}
    C = np.stack([l2_normalize(centroids[k]) for k in keys], axis=0)  # [M, D]
    # Kosinus-Aehnlichkeit
    S = C @ C.T
    # Euclid. Distanz zwischen Einheitsvektoren: sqrt(2 - 2cos)
    D = np.sqrt(np.clip(2.0 - 2.0 * S, 0.0, 4.0))
    np.fill_diagonal(D, np.inf)
    nn_idx = np.argmin(D, axis=1)
    nn_dist = D[np.arange(len(keys)), nn_idx]
    out: Dict[str, Tuple[str, float]] = {}
    for i, k in enumerate(keys):
        out[k] = (keys[int(nn_idx[i])], float(nn_dist[i]))
    return out


def _has_internal_metadata_conflict(meta: Dict[str, Dict], oid: str) -> bool:
    m = meta.get(oid, {})
    # Kernfelder: Name, ManaCost, CMC, Colors, Color Identity
    return any(
        len(m.get(field, []) or []) > 1
        for field in ("names_all", "mana_costs_all", "cmcs_all", "colors_all", "color_identities_all")
    )


def _cross_metadata_conflict(meta_a: Dict, meta_b: Dict) -> bool:
    def _first(mv, default=None):
        if isinstance(mv, list):
            return mv[0] if mv else default
        return mv
    a = meta_a or {}
    b = meta_b or {}
    # Vergleiche repr. Werte: mana_cost, cmc, colors, color_identity
    return any(
        [
            (a.get("mana_cost") != b.get("mana_cost")),
            (a.get("cmc") != b.get("cmc")),
            (json.dumps(a.get("colors", []), sort_keys=True) != json.dumps(b.get("colors", []), sort_keys=True)),
            (
                json.dumps(a.get("color_identity", []), sort_keys=True)
                != json.dumps(b.get("color_identity", []), sort_keys=True)
            ),
        ]
    )


def _upsert_quality(
    sqlite_path: Path,
    rows: List[Dict],
) -> None:
    sql = (
        """
        INSERT INTO oracle_quality (
            oracle_id,
            suspect_overlap,
            suspect_cluster_spread,
            suspect_metadata_conflict,
            nearest_neighbor_oracle_id,
            nearest_neighbor_dist,
            intra_mean_dist,
            intra_max_dist,
            meta_info
        ) VALUES (
            :oracle_id,
            :suspect_overlap,
            :suspect_cluster_spread,
            :suspect_metadata_conflict,
            :nearest_neighbor_oracle_id,
            :nearest_neighbor_dist,
            :intra_mean_dist,
            :intra_max_dist,
            :meta_info
        )
        ON CONFLICT(oracle_id) DO UPDATE SET
            suspect_overlap = excluded.suspect_overlap,
            suspect_cluster_spread = excluded.suspect_cluster_spread,
            suspect_metadata_conflict = excluded.suspect_metadata_conflict,
            nearest_neighbor_oracle_id = excluded.nearest_neighbor_oracle_id,
            nearest_neighbor_dist = excluded.nearest_neighbor_dist,
            intra_mean_dist = excluded.intra_mean_dist,
            intra_max_dist = excluded.intra_max_dist,
            meta_info = excluded.meta_info
        """
    )
    conn = sqlite3.connect(str(sqlite_path))
    try:
        with conn:
            conn.executemany(sql, rows)
    finally:
        conn.close()


def main():
    args = parse_args()
    sqlite_path, emb_dim, cfg = _resolve_paths_and_dims(args)
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite-DB nicht gefunden: {sqlite_path}")

    ensure_oracle_quality_table(str(sqlite_path))

    grouped, meta = load_embeddings_grouped_by_oracle(str(sqlite_path), args.mode, emb_dim)
    if not grouped:
        raise SystemExit("Keine Embeddings fuer den angegebenen mode gefunden.")

    # Thresholds aus config (mit CLI-Override)
    spread_thr, overlap_thr = _resolve_thresholds(args, cfg)

    # A) Cluster-Spreads
    stats, global_mean, global_std = _compute_cluster_spreads(grouped)

    # B) Overlaps per Zentroid
    centroids = {oid: stats[oid]["centroid"] for oid in stats.keys()}
    neighbors = _nearest_neighbors(centroids)

    # C) Metadaten-Konflikte
    rows: List[Dict] = []
    for oid, arr in grouped.items():
        s = stats.get(oid, {})
        intra_mean = float(s.get("intra_mean_dist", 0.0))
        intra_max = float(s.get("intra_max_dist", 0.0))

        suspect_spread = 0
        if global_mean == global_mean and global_std == global_std:  # nicht NaN
            if intra_mean > (global_mean + 2.0 * global_std):
                suspect_spread = 1
        if intra_max > spread_thr:
            suspect_spread = 1

        neighbor_id, neighbor_dist = neighbors.get(oid, (None, float("inf")))
        suspect_overlap = 1 if (neighbor_id is not None and neighbor_dist < overlap_thr) else 0

        # Metadaten-Konflikte: intern
        meta_conflict = 1 if _has_internal_metadata_conflict(meta, oid) else 0
        # ggf. extern (bei Overlap) A vs B
        if suspect_overlap and neighbor_id:
            if _cross_metadata_conflict(meta.get(oid, {}), meta.get(neighbor_id, {})):
                meta_conflict = 1

        # meta_info JSON
        m = meta.get(oid, {})
        meta_info = {
            "names_all": m.get("names_all", []),
            "mana_costs_all": m.get("mana_costs_all", []),
            "cmcs_all": m.get("cmcs_all", []),
            "colors_all": m.get("colors_all", []),
            "color_identities_all": m.get("color_identities_all", []),
            "set_codes_all": m.get("set_codes_all", []),
            "set_names_all": m.get("set_names_all", []),
            "langs_all": m.get("langs_all", []),
            "thresholds": {
                "spread_max_thr": spread_thr,
                "overlap_thr": overlap_thr,
                "global_intra_mean": global_mean,
                "global_intra_std": global_std,
                "mode": args.mode,
            },
        }

        rows.append(
            {
                "oracle_id": oid,
                "suspect_overlap": int(suspect_overlap),
                "suspect_cluster_spread": int(suspect_spread),
                "suspect_metadata_conflict": int(meta_conflict),
                "nearest_neighbor_oracle_id": neighbor_id,
                "nearest_neighbor_dist": float(neighbor_dist if neighbor_id else float("inf")),
                "intra_mean_dist": intra_mean,
                "intra_max_dist": intra_max,
                "meta_info": json.dumps(meta_info, ensure_ascii=False),
            }
        )

    # Konsolenreport Top-N
    if args.print_top and args.print_top > 0:
        N = int(args.print_top)
        def _name(oid: str) -> str:
            m = meta.get(oid, {})
            n = m.get("name") or ""
            return n if n else oid

        # Spread: flag zuerst, dann nach intra_mean_dist bzw. intra_max_dist sortieren
        spread_sorted = sorted(
            rows,
            key=lambda r: (
                -int(r["suspect_cluster_spread"]),
                -float(r["intra_mean_dist"]),
                -float(r["intra_max_dist"]),
            ),
        )
        print("\n[TOP] Cluster-Spread (verdächtig zuerst):")
        for i, r in enumerate(spread_sorted[:N], 1):
            print(
                f" {i:2d}) { _name(r['oracle_id']) }  spread_flag={r['suspect_cluster_spread']}  "
                f"mean={r['intra_mean_dist']:.4f}  max={r['intra_max_dist']:.4f}"
            )

        # Overlap: nach kleinster Nachbar-Distanz
        overlap_sorted = sorted(
            rows,
            key=lambda r: (float(r["nearest_neighbor_dist"]) if math.isfinite(r["nearest_neighbor_dist"]) else 1e9),
        )
        print("\n[TOP] Overlap (kleinste Nachbar-Distanzen):")
        for i, r in enumerate(overlap_sorted[:N], 1):
            nb = r["nearest_neighbor_oracle_id"] or ""
            print(
                f" {i:2d}) { _name(r['oracle_id']) } ~ { _name(nb) }  dist="
                f"{r['nearest_neighbor_dist']:.4f}  overlap_flag={r['suspect_overlap']}"
            )

        # Metadaten-Konflikte
        meta_conf = [r for r in rows if r["suspect_metadata_conflict"]]
        print("\n[TOP] Metadaten-Konflikte:")
        if not meta_conf:
            print("  (keine)")
        else:
            meta_conf_sorted = sorted(
                meta_conf,
                key=lambda r: (
                    -int(r["suspect_overlap"]),
                    -float(r["intra_mean_dist"]),
                ),
            )
            for i, r in enumerate(meta_conf_sorted[:N], 1):
                nb = r["nearest_neighbor_oracle_id"] or ""
                print(
                    f" {i:2d}) { _name(r['oracle_id']) }  meta_conflict=1  "
                    f"overlap_flag={r['suspect_overlap']}  neighbor={_name(nb)}"
                )

    # Schreiben in oracle_quality (UPSERT)
    _upsert_quality(sqlite_path, rows)
    print(f"[OK] oracle_quality aktualisiert: {len(rows)} Eintraege.")

    # Optional/Default: CSV-Report
    out_csv_arg = args.out_csv
    if not out_csv_arg:
        debug_dir = cfg.get("paths", {}).get("debug_dir", "./debug") if isinstance(cfg, dict) else "./debug"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv_arg = str(Path(debug_dir) / f"oracle_cluster_report_{ts}.csv")
    if out_csv_arg:
        out_path = Path(out_csv_arg)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=args.csv_delimiter)
            def fmt(x: float) -> str:
                s = f"{x:.6f}"
                return s.replace(".", ",") if args.decimal_comma else s
            writer.writerow(
                [
                    "oracle_id",
                    "name",
                    "set_code",
                    "set_name",
                    "image_path",
                    "mana_cost",
                    "cmc",
                    "colors",
                    "suspect_overlap",
                    "suspect_cluster_spread",
                    "suspect_metadata_conflict",
                    "nearest_neighbor_oracle_id",
                    "nearest_neighbor_name",
                    "nearest_neighbor_image_path",
                    "nearest_neighbor_dist",
                    "intra_mean_dist",
                    "intra_max_dist",
                    "spread_max_thr",
                    "overlap_thr",
                    "global_intra_mean",
                    "global_intra_std",
                    "mode",
                ]
            )
            for r in rows:
                m = meta.get(r["oracle_id"], {})
                nb_id = r["nearest_neighbor_oracle_id"]
                nb_meta = meta.get(nb_id, {}) if nb_id else {}
                writer.writerow(
                    [
                        r["oracle_id"],
                        m.get("name") or "",
                        m.get("set_code") or "",
                        m.get("set_name") or "",
                        nb_meta and m.get("image_path") or m.get("image_path") or "",
                        m.get("mana_cost") or "",
                        (fmt(float(m.get("cmc"))) if m.get("cmc") is not None else ""),
                        json.dumps(m.get("colors", []), ensure_ascii=False),
                        r["suspect_overlap"],
                        r["suspect_cluster_spread"],
                        r["suspect_metadata_conflict"],
                        r["nearest_neighbor_oracle_id"] or "",
                        (nb_meta.get("name") or "") if nb_id else "",
                        (nb_meta.get("image_path") or "") if nb_id else "",
                        (fmt(r['nearest_neighbor_dist']) if math.isfinite(r["nearest_neighbor_dist"]) else ""),
                        fmt(r['intra_mean_dist']),
                        fmt(r['intra_max_dist']),
                        fmt(spread_thr),
                        fmt(overlap_thr),
                        (fmt(global_mean) if global_mean == global_mean else ""),
                        (fmt(global_std) if global_std == global_std else ""),
                        args.mode,
                    ]
                )
        print(f"[OK] CSV geschrieben: {out_path}")

    # Optional/Default: JSON-Top-Report
    out_json_arg = args.out_json
    if not out_json_arg:
        debug_dir = cfg.get("paths", {}).get("debug_dir", "./debug") if isinstance(cfg, dict) else "./debug"
        tsj = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json_arg = str(Path(debug_dir) / f"oracle_cluster_summary_{tsj}.json")
    if out_json_arg:
        # Erzeuge Top-Listen (entsprechen Konsole)
        N = max(1, int(args.print_top)) if args.print_top else 20
        def _name(oid: str) -> str:
            m = meta.get(oid, {})
            return m.get("name") or oid
        spread_sorted = sorted(
            rows,
            key=lambda r: (
                -int(r["suspect_cluster_spread"]),
                -float(r["intra_mean_dist"]),
                -float(r["intra_max_dist"]),
            ),
        )
        overlap_sorted = sorted(
            rows,
            key=lambda r: (float(r["nearest_neighbor_dist"]) if math.isfinite(r["nearest_neighbor_dist"]) else 1e9),
        )
        meta_conf_sorted = [r for r in rows if r["suspect_metadata_conflict"]]
        meta_conf_sorted = sorted(
            meta_conf_sorted,
            key=lambda r: (-int(r["suspect_overlap"]), -float(r["intra_mean_dist"]))
        )
        payload = {
            "mode": args.mode,
            "thresholds": {
                "spread_max_thr": spread_thr,
                "overlap_thr": overlap_thr,
                "global_intra_mean": global_mean,
                "global_intra_std": global_std,
            },
            "counts": {
                "oracle_ids": len(rows),
                "flags": {
                    "suspect_cluster_spread": int(sum(int(r["suspect_cluster_spread"]) for r in rows)),
                    "suspect_overlap": int(sum(int(r["suspect_overlap"]) for r in rows)),
                    "suspect_metadata_conflict": int(sum(int(r["suspect_metadata_conflict"]) for r in rows)),
                },
            },
            "top": {
                "spread": [
                    {
                        "oracle_id": r["oracle_id"],
                        "name": _name(r["oracle_id"]),
                        "intra_mean_dist": r["intra_mean_dist"],
                        "intra_max_dist": r["intra_max_dist"],
                        "flag": int(r["suspect_cluster_spread"]),
                    }
                    for r in spread_sorted[:N]
                ],
                "overlap": [
                    {
                        "oracle_id": r["oracle_id"],
                        "name": _name(r["oracle_id"]),
                        "neighbor_oracle_id": r["nearest_neighbor_oracle_id"],
                        "neighbor_name": _name(r["nearest_neighbor_oracle_id"]) if r["nearest_neighbor_oracle_id"] else None,
                        "neighbor_dist": r["nearest_neighbor_dist"],
                        "flag": int(r["suspect_overlap"]),
                    }
                    for r in overlap_sorted[:N]
                ],
                "metadata_conflicts": [
                    {
                        "oracle_id": r["oracle_id"],
                        "name": _name(r["oracle_id"]),
                        "neighbor_oracle_id": r["nearest_neighbor_oracle_id"],
                        "neighbor_name": _name(r["nearest_neighbor_oracle_id"]) if r["nearest_neighbor_oracle_id"] else None,
                        "overlap_flag": int(r["suspect_overlap"]),
                    }
                    for r in meta_conf_sorted[:N]
                ],
            },
        }
        json_path = Path(out_json_arg)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as jf:
            json.dump(payload, jf, ensure_ascii=False, indent=2)
        print(f"[OK] JSON geschrieben: {json_path}")


if __name__ == "__main__":
    main()
