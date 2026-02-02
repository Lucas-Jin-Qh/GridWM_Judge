#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DoorKey Triplet Validator for GridWM-Judge.

Validates DoorKey triplets with strict compliance to:
- GridWM-Judge Project Overview Scientific Contract (Final SSOT)
- NoCue_spec.md
- TRIPLET_AUDIT_SPEC_REVIEWER.md

Usage:
    python validate_doorkey_triplets.py --triplets data_doorkey_vFinal/triplets.jsonl --root data_doorkey_vFinal --out-audit audit_final.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT

ALLOWED_INPUTS = {"frames", "mission", "terminated", "truncated"}
PROHIBITED_INPUTS = {"state_seq", "actions_id", "success", "reward", "cf_meta", "nocue_meta", "seed"}

def load_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                # Crash tolerance: stop at partial line
                print("Warning: Stopped loading at corrupt JSON line.")
                break
    return out

def _agent_traj(rec):
    return [(tuple(st["agent"]["pos"]), int(st["agent"]["dir"])) for st in rec["state_seq"]]

def validate_state_encoding(enc):
    if enc is None: return False
    arr = np.array(enc)
    if len(arr.shape) != 3 or arr.shape[2] != 3: return False
    if not np.issubdtype(arr.dtype, np.integer): return False
    H, W, _ = arr.shape
    for i in range(H):
        for j in range(W):
            obj_idx, _, state = map(int, arr[i, j])
            if obj_idx not in IDX_TO_OBJECT: return False
            name = IDX_TO_OBJECT[obj_idx]
            if name == "door" and state not in (0, 1, 2): return False
            if name not in ("door", "empty", "unseen") and state != 0: return False
    return True

def get_tile_mask(enc, target_name="key"):
    arr = np.array(enc)
    H, W, _ = arr.shape
    mask = np.zeros((H, W), dtype=bool)
    for i in range(H):
        for j in range(W):
            if IDX_TO_OBJECT.get(arr[i, j, 0]) == target_name:
                mask[i, j] = True
    return mask

def door_open_after_toggle(states, toggle_t):
    if toggle_t + 1 >= len(states): return False
    s = states[toggle_t + 1]["front_cell"].get("state", None)
    if s == 0: return True
    for o in states[toggle_t + 1].get("objects", []):
        if o.get("type") == "door" and o.get("state") == 0: return True
    return False

def validate_group(gid, full, nocue, cf, root: Path, audit_stats: Dict):
    # D1 Action Identity
    if full["actions_id"] != nocue["actions_id"] or full["actions_id"] != cf["actions_id"]:
        return False, "action_mismatch"

    for v_name, v in [("full", full), ("nocue", nocue), ("cf", cf)]:
        if len(v["frames"]) != len(v["actions_id"]) + 1: return False, f"{v_name}_len_mismatch"
        if len(v["state_seq"]) != len(v["actions_id"]) + 1: return False, f"{v_name}_state_len_mismatch"

        mif = set(v.get("model_input_fields", []))
        if mif != ALLOWED_INPUTS: return False, f"{v_name}_bad_contract_fields"
        if not all(k in v for k in mif): return False, f"{v_name}_missing_contract_data"
        if any(k in v.keys() and k in PROHIBITED_INPUTS for k in mif): return False, f"{v_name}_info_leak"

        for st in v["state_seq"]:
            if not validate_state_encoding(st.get("state_encoding")): return False, f"{v_name}_invalid_encoding"

    # Termination
    if full["terminated"] is not True: return False, "full_not_terminated"
    if cf["terminated"] is True: return False, "cf_terminated_error"
    if cf["truncated"] is True: return False, "cf_truncated_error"
    if nocue["terminated"] != full["terminated"]: return False, "nocue_term_mismatch"

    # Outcomes
    if not full["success"] or not nocue["success"] or cf["success"]: return False, "outcome_fail"
    if float(cf.get("reward", 0)) != 0.0: return False, "cf_reward_nonzero"

    # D6 Physics
    ft, nt, ct = _agent_traj(full), _agent_traj(nocue), _agent_traj(cf)
    if ft != nt or ft != ct: return False, "physics_drift"

    # E1 Semantics
    actions, states = full["actions_id"], full["state_seq"]
    pickup_t = next((t for t, a in enumerate(actions) if a == 3 and states[t]["front_cell"]["type"] == "key"), None)
    toggle_t = next((t for t, a in enumerate(actions) if a == 5 and states[t]["front_cell"]["type"] == "door"), None)
    if pickup_t is None or toggle_t is None or pickup_t > toggle_t: return False, "semantic_seq_fail"
    if states[toggle_t]["front_cell"].get("state") != 2: return False, "door_not_locked"
    if not door_open_after_toggle(states, toggle_t): return False, "door_not_open"

    # CF Observability
    for st in cf["state_seq"]:
        if "goal" in st["visible_types_gt"]: return False, "cf_goal_visible"
    cfm = cf.get("cf_meta", {})
    goal_to = tuple(cfm.get("goal_to", []))
    if goal_to in {pos for pos, _ in ft}: return False, "cf_goal_on_path"

    # NoCue Hard Gates
    meta = nocue.get("nocue_meta", {})
    masked = meta.get("window_steps", [])
    if not masked: return False, "empty_mask"

    if int(meta.get("masked_tiles_total", 10**9)) > int(meta.get("mask_budget_tiles", -1)):
        return False, "budget_exceeded"

    # Diff-in-Mask
    for idx in masked:
        fp_f, fp_n = root / full["frames"][idx], root / nocue["frames"][idx]
        if not fp_f.exists() or not fp_n.exists(): return False, "missing_png"

        # FIX: FD Safety here too
        with Image.open(fp_f) as imf, Image.open(fp_n) as imn:
            arr_f, arr_n = np.array(imf), np.array(imn)

        diff = np.any(arr_f != arr_n, axis=2)

        enc = full["state_seq"][idx]["state_encoding"]
        tile_mask_bool = get_tile_mask(enc, "key")
        Ht, Wt = tile_mask_bool.shape
        H, W = arr_f.shape[:2]
        th, tw = H // Ht, W // Wt
        pixel_mask = np.kron(tile_mask_bool, np.ones((th, tw), dtype=bool))[:H, :W]

        if np.any(diff & ~pixel_mask): return False, "diff_outside_target_tile"
        if not np.any(diff & pixel_mask): return False, "no_diff_inside_target"

    audit_stats["mask_counts"].append(len(masked))
    return True, "pass"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", type=str, required=True)
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out-audit", type=str, default="audit_report.json")
    args = ap.parse_args()

    data = load_jsonl(Path(args.triplets))
    groups = defaultdict(dict)
    for r in data: groups[r["group_id"]][r["variant"]] = r

    audit = {"passed": 0, "failed": 0, "reasons": defaultdict(int), "mask_counts": []}
    for gid, vs in groups.items():
        if not {"full", "nocue", "cf"}.issubset(vs.keys()):
            audit["failed"] += 1; audit["reasons"]["missing_variant"] += 1; continue
        ok, reason = validate_group(gid, vs["full"], vs["nocue"], vs["cf"], Path(args.root), audit)
        if ok: audit["passed"] += 1
        else: audit["failed"] += 1; audit["reasons"][reason] += 1; print(f"FAIL {gid}: {reason}")

    with open(args.out_audit, "w") as f: json.dump(audit, f, indent=2)
    print(f"Audit: {audit['passed']} Passed, {audit['failed']} Failed")

if __name__ == "__main__":
    main()
