#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiRoom Triplet Validator (Golden Release).

Strict Zero-Trust Validation.
Production-optimized:
- Reduced IO overhead during pixel audit.
- Full D7 & NoCue compliance checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
from PIL import Image

from minigrid.core.constants import OBJECT_TO_IDX, IDX_TO_OBJECT

ALLOWED_INPUTS = ["frames", "mission", "terminated", "truncated"]
MASK_RGB = (0, 0, 0)

def _die(msg: str) -> None:
    raise SystemExit(msg)

def _load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def _cue_tile_mask(state_encoding: List[List[List[int]]], cue_type: str, cue_color: str) -> np.ndarray:
    enc = np.asarray(state_encoding, dtype=np.uint8)  # (7,7,3)
    obj = OBJECT_TO_IDX.get(cue_type, None)
    col = None  # We don't use color for multiroom since targets are door/goal
    if obj is None:
        return np.zeros((7, 7), dtype=bool)
    if col is not None:
        return (enc[:, :, 0] == int(obj)) & (enc[:, :, 1] == int(col))
    return enc[:, :, 0] == int(obj)

def get_tile_mask(enc, target_names: Set[str]) -> np.ndarray:
    arr = np.array(enc, dtype=int)
    Ht, Wt = arr.shape[:2]
    mask = np.zeros((Ht, Wt), dtype=bool)
    for i in range(Ht):
        for j in range(Wt):
            oid = int(arr[i, j, 0])
            obj_name = IDX_TO_OBJECT.get(oid)
            if obj_name in target_names:
                mask[i, j] = True
    return mask

def _pixel_mask_from_tile_mask(tile_mask: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    if frame_h % 7 != 0 or frame_w % 7 != 0:
        _die(f"Frame shape not divisible by 7: {(frame_h, frame_w)}")
    th, tw = frame_h // 7, frame_w // 7
    return np.kron(tile_mask.astype(np.uint8), np.ones((th, tw), dtype=np.uint8)).astype(bool)

def validate_state_encoding(enc) -> bool:
    if enc is None:
        return False
    arr = np.array(enc)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False
    H, W, _ = arr.shape
    for i in range(H):
        for j in range(W):
            obj_i, col_i, st_i = int(arr[i, j, 0]), int(arr[i, j, 1]), int(arr[i, j, 2])
            if obj_i not in OBJECT_TO_IDX:
                return False
            # color and state validation minimal here
    return True

def _agent_traj(rec: Dict) -> List[Tuple[Tuple[int, int], int]]:
    traj = []
    for st in rec["state_seq"]:
        pos = tuple(st["agent"]["pos"])
        d = int(st["agent"]["dir"])
        traj.append((pos, d))
    return traj

def door_state_at(st: Dict, door_pos: Tuple[int, int]) -> Optional[int]:
    for o in st.get("objects", []):
        if o.get("type") == "door" and tuple(o.get("pos", [])) == door_pos:
            return int(o.get("state", -1))
    return None

def door_opens_after_action(states: List[Dict], t: int, door_pos: Tuple[int, int], horizon: int = 4) -> bool:
    for j in range(t + 1, min(t + 1 + horizon, len(states))):
        ds = door_state_at(states[j], door_pos)
        if ds == 0:
            return True
    return False

def check_door_semantics(full_rec: Dict) -> Tuple[bool, str]:
    actions = full_rec["actions_id"]
    states = full_rec["state_seq"]
    toggle_indices = [i for i, a in enumerate(actions) if a == 5]
    if len(toggle_indices) < 2:
        return False, "toggles_lt_2"
    for t in toggle_indices:
        fc = states[t]["front_cell"]
        if fc.get("type") != "door":
            return False, f"toggle_non_door_t{t}"
        door_pos = tuple(fc["pos"])
        if t + 1 >= len(states):
            return False, f"toggle_last_step_t{t}"
        post = states[t + 1]
        door_obj = next((o for o in post.get("objects", []) if tuple(o.get("pos", [])) == door_pos and o.get("type") == "door"), None)
        if door_obj is not None:
            if int(door_obj.get("state", 999)) != 0:
                return False, f"door_not_open_t{t}"
        else:
            fc2 = post["front_cell"]
            if tuple(fc2.get("pos")) == door_pos and fc2.get("type") == "door":
                if int(fc2.get("state", 999)) != 0:
                    return False, f"door_not_open_fc_t{t}"
            else:
                return False, f"door_obj_missing_t{t}"
    return True, "ok"

def _check_state_encoding_consistency(v: Dict, v_name: str) -> Tuple[bool, str, Optional[Tuple[int, int, int]]]:
    ref_shape = None
    for t, st in enumerate(v["state_seq"]):
        enc = st.get("state_encoding", [])
        arr = np.array(enc)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return False, f"bad_state_enc_rank_{v_name}_t{t}", None
        if ref_shape is None:
            ref_shape = arr.shape
        elif arr.shape != ref_shape:
            return False, f"state_enc_drift_{v_name}_t{t}", None
    return True, "ok", ref_shape

def validate_group(group: List[Dict], root: Path, audit_stats: Dict) -> Tuple[bool, str]:
    if len(group) != 3:
        return False, "group_len_not_3"
    full, nocue, cf = group[0], group[1], group[2]
    if [full.get("variant"), nocue.get("variant"), cf.get("variant")] != ["full", "nocue", "cf"]:
        return False, "bad_variant_order"
    gid = full.get("group_id")
    if gid is None or nocue.get("group_id") != gid or cf.get("group_id") != gid:
        return False, "group_id_mismatch"

    ref_shapes = []
    for v_name, v in [("full", full), ("nocue", nocue), ("cf", cf)]:
        mif = v.get("model_input_fields", None)
        if mif != ALLOWED_INPUTS:
            return False, f"bad_inputs_{v_name}"
        for k in mif:
            if k not in v:
                return False, f"missing_input_{k}_{v_name}"
        T = len(v.get("actions_id", []))
        if len(v.get("frames", [])) != T + 1:
            return False, f"frames_len_mismatch_{v_name}"
        if len(v.get("state_seq", [])) != T + 1:
            return False, f"state_seq_len_mismatch_{v_name}"
        ok, msg, ref_shape = _check_state_encoding_consistency(v, v_name)
        if not ok:
            return False, msg
        ref_shapes.append(ref_shape)

    if not (ref_shapes[0] == ref_shapes[1] == ref_shapes[2]):
        return False, "state_enc_shape_mismatch_across_variants"
    if not (full["actions_id"] == nocue["actions_id"] == cf["actions_id"]):
        return False, "action_mismatch"

    traj_f = _agent_traj(full)
    traj_n = _agent_traj(nocue)
    traj_c = _agent_traj(cf)
    if traj_f != traj_n:
        return False, "physics_mismatch_nocue"
    if traj_f != traj_c:
        return False, "physics_mismatch_cf"

    if not (full.get("success", False) and full.get("terminated", False)):
        return False, "full_not_success"
    if cf.get("success", False) or cf.get("terminated", False):
        return False, "cf_must_not_terminate"
    if float(cf.get("reward", 0.0)) != 0.0:
        return False, "cf_reward_nonzero"

    sem_ok, sem_msg = check_door_semantics(full)
    if not sem_ok:
        return False, f"semantic_{sem_msg}"

    for st in cf["state_seq"]:
        if "goal" in st.get("visible_types_gt", []):
            return False, "cf_goal_visible"

    meta = nocue.get("nocue_meta", None)
    if not isinstance(meta, dict):
        return False, "missing_nocue_meta"
    if not all(k in meta for k in ["targets", "window_policy", "window_steps", "mask_strength_target", "mask_strength_actual", "alignment_score", "alignment_threshold", "masked_frames", "mask_type", "physics_check_passed", "masked_tiles_total", "mask_budget_tiles"]):
        return False, "missing_nocue_meta_keys"
    if meta.get("window_policy") != "EARLY":
        return False, "bad_window_policy"
    if not isinstance(meta.get("targets"), list) or set(meta.get("targets")) != {"door", "goal"}:
        return False, "nocue_targets_bad"
    if int(meta.get("masked_frames")) != int(len(meta.get("window_steps", []))):
        return False, "nocue_mask_count_mismatch"
    if float(meta.get("mask_strength_actual")) > float(meta.get("mask_strength_threshold")):
        return False, "nocue_mask_strength_exceeded"
    if float(meta.get("alignment_score")) < float(meta.get("alignment_threshold")):
        return False, "nocue_alignment_fail"

    window = set(meta.get("window_steps", []))
    t_enter = int(meta.get("event_end_action_idx", max(window) + 1))
    for t in window:
        if t >= t_enter:
            return False, "mask_outside_window"
        if full["state_seq"][t]["front_cell"].get("type") in ("door", "goal"):
            return False, "mask_interaction"

    for t in range(len(full["frames"])):
        fp_f = root / full["frames"][t]
        fp_n = root / nocue["frames"][t]
        if not fp_f.exists() or not fp_n.exists():
            return False, "missing_png"

        im_f = _load_png(fp_f)
        im_n = _load_png(fp_n)
        if im_f.shape != im_n.shape:
            return False, "shape_mismatch"

        diff = np.any(im_f != im_n, axis=2)

        if t not in window:
            if np.any(diff):
                return False, f"diff_unmasked_t{t}"
        else:
            enc = full["state_seq"][t]["state_encoding"]
            tile_mask = get_tile_mask(enc, {"door", "goal"})
            H, W = im_f.shape[:2]
            th, tw = H // 7, W // 7
            pix_mask = np.kron(tile_mask.astype(np.uint8), np.ones((th, tw), dtype=np.uint8)).astype(bool)
            if np.any(diff & ~pix_mask):
                return False, f"leak_outside_target_t{t}"

    audit_stats["mask_counts"].append(len(window))
    audit_stats["mask_ratios"].append(float(meta.get("mask_strength_actual", 0)))
    audit_stats["toggles"].append(int(sum(1 for a in full["actions_id"] if a == 5)))
    return True, "pass"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-audit", default="audit.json")
    ap.add_argument("--fail-fast", action="store_true")
    args = ap.parse_args()

    data = []
    with open(args.triplets, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if len(data) % 3 != 0:
        _die("JSONL lines must be multiple of 3")

    groups = defaultdict(list)
    for r in data:
        groups[r["group_id"]].append(r)

    audit = {"passed": 0, "failed": 0, "reasons": defaultdict(int), "stats": {"mask_counts": [], "mask_ratios": [], "toggles": []}}
    root = Path(args.root)

    for gid, group in groups.items():
        try:
            ok, msg = validate_group(group, root, audit["stats"])
        except Exception as e:
            ok, msg = False, f"exception_{e}"
        if ok:
            audit["passed"] += 1
        else:
            audit["failed"] += 1
            audit["reasons"][msg] += 1

    summary = {
        "total": len(groups),
        "passed": audit["passed"],
        "failed": audit["failed"],
        "pass_rate": audit["passed"] / len(groups) if groups else 0,
        "avg_masks": np.mean(audit["stats"]["mask_counts"]) if audit["stats"]["mask_counts"] else 0,
        "failure_modes": dict(audit["reasons"])
    }
    print("\n=== Audit Report ===")
    print(json.dumps(summary, indent=2))
    with open(args.out_audit, "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
