#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_per_example_rows.py

Wrapper to run the existing scorer on each experiment under runs/responses,
collect per-example "rows" from the generated score JSONs (or generate them
if missing), and merge all rows into a single CSV for downstream analysis.

Usage:
  python3 scripts/export_per_example_rows.py
  python3 scripts/export_per_example_rows.py --responses_dir runs/responses --scores_dir runs/scores --out runs/scores/all_rows.csv

Notes:
  - This script calls the repository scorer `scripts/score_exam.py` for each
    experiment directory and requests `--dump_rows`. It requires Python 3.
  - It is conservative: if a score JSON with rows already exists it will reuse it.
  - Default output location is under `results/` (see --scores_dir / --out).
"""
from __future__ import annotations
import argparse
import subprocess
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import subprocess


def find_experiments(responses_dir: Path) -> List[Path]:
    if not responses_dir.exists():
        raise SystemExit(f"Responses dir not found: {responses_dir}")
    exps = [p for p in sorted(responses_dir.iterdir()) if p.is_dir() and not p.name.startswith('_')]
    return exps


def ensure_score_with_rows(exp_dir: Path, scores_dir: Path, force: bool = False) -> Optional[Path]:
    """
    Ensure a score JSON with rows exists for the given experiment directory.
    Returns path to the score JSON.
    """
    # Determine responses_arg (may be a file within nested model dir) and pick sensible out_name
    out_name_base = exp_dir.name
    out_path = scores_dir / f"score_{out_name_base}_with_rows.json"
    # If file exists and contains "rows", return it
    if out_path.exists() and not force:
        try:
            data = json.loads(out_path.read_text(encoding="utf-8"))
            if "rows" in data and isinstance(data["rows"], list):
                return out_path
        except Exception:
            pass

    # Otherwise call scorer to produce it
    scores_dir.mkdir(parents=True, exist_ok=True)

    # If there are jsonl response files anywhere under exp_dir, prefer the
    # most recent one and pass it directly to the scorer. This handles
    # nested provider/variant subdirectories.
    jsonl_files = list(exp_dir.rglob("*.jsonl"))
    if jsonl_files:
        jsonl_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        responses_arg = str(jsonl_files[0])
        print(f"Found jsonl under {exp_dir.name}: using {jsonl_files[0].name}")
        # if the selected jsonl is inside a per-model directory, use that dir name for out file
        try:
            sel_parent = jsonl_files[0].parent
            # if parent is nested under exp_dir (e.g., TASK_B_2-shot/<model>/requests*.jsonl)
            if sel_parent != exp_dir:
                out_name_base = sel_parent.name
        except Exception:
            pass
    else:
        responses_arg = str(exp_dir)

    out_path = scores_dir / f"score_{out_name_base}_with_rows.json"
    cmd = [
        "python3", "scripts/score_exam.py",
        "--responses", responses_arg,
        "--dump_rows",
        "--out", str(out_path),
    ]
    print(f"Running scorer for experiment {exp_dir.name} -> {out_path.name} (responses={responses_arg})")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: scorer failed for experiment {exp_dir.name}: {e}. Skipping.")
        return None
    if not out_path.exists():
        print(f"Warning: Expected score file not created for {exp_dir.name}: {out_path}. Skipping.")
        return None
    return out_path


def collect_rows(score_files: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sf in score_files:
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: failed to read {sf}: {e}")
            continue
        exp_name = sf.stem.replace("score_","").replace("_with_rows","")
        if "rows" in data and isinstance(data["rows"], list):
            for r in data["rows"]:
                r["_source_experiment"] = exp_name
                rows.append(r)
        else:
            # If rows not present, try load failure_examples as fallback
            print(f"Warning: no rows in {sf}; skipping")
    return rows


def write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    if not rows:
        print("No rows to write.")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Build header union
    fieldnames = [
        "uid", "task", "env_task", "group_id", "t", "variant", "temporal", "visual",
        "pred_norm", "gold", "parse_mode", "failure", "score", "correct",
        "b_components", "b_acc_threshold", "b_acc_at_threshold", "error",
        "_source_experiment"
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row = dict.fromkeys(fieldnames, "")
            for k in fieldnames:
                v = r.get(k)
                if k == "b_components" and isinstance(v, dict):
                    row[k] = json.dumps(v, ensure_ascii=False)
                else:
                    row[k] = v if v is not None else ""
            writer.writerow(row)
    print(f"Wrote {len(rows)} rows to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses_dir", default="runs/responses", help="Directory with experiments' response subdirs")
    ap.add_argument("--scores_dir", default="results/scores", help="Directory to read/write score JSONs (default: results/scores)")
    ap.add_argument("--out", default="results/all_rows.csv", help="Merged CSV output path (default: results/all_rows.csv)")
    ap.add_argument("--skip_existing", action="store_true", help="If set, skip running scorer for experiments that already have rows")
    ap.add_argument("--force", action="store_true", help="If set, recompute scores even if rows file exists")
    args = ap.parse_args()

    responses_dir = Path(args.responses_dir)
    scores_dir = Path(args.scores_dir)
    out_csv = Path(args.out)

    exps = find_experiments(responses_dir)
    print(f"Found {len(exps)} experiments under {responses_dir}")

    score_files: List[Path] = []
    for e in exps:
        out_path = scores_dir / f"score_{e.name}_with_rows.json"
        if out_path.exists() and not args.force:
            try:
                data = json.loads(out_path.read_text(encoding="utf-8"))
                if "rows" in data and isinstance(data["rows"], list):
                    print(f"Using existing rows file: {out_path.name}")
                    score_files.append(out_path)
                    if args.skip_existing:
                        continue
            except Exception:
                pass
        # ensure generation
        sf = ensure_score_with_rows(e, scores_dir, force=args.force)
        if sf is not None:
            score_files.append(sf)
        else:
            print(f"Skipped experiment {e.name} due to scoring error or missing outputs.")

    rows = collect_rows(score_files)
    write_csv(rows, out_csv)


if __name__ == "__main__":
    main()


