#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LavaGap Triplet Validator.

Audits (hard gates):
- Contract: model_input_fields == ALLOWED_INPUTS and contains required keys
- Physics Identity: Full/NoCue/CF must have identical (agent_pos, agent_dir) sequence and same actions_id
- Outcome: Full success; CF fail (reward=0, not truncated); NoCue matches Full outcomes
- Semantic: Full crosses lava wall row via gap; never steps on lava
- NoCue (EARLY): pixel diffs only allowed on lava tiles within window_steps
- D5: must NOT mask frames where front_cell is lava
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from minigrid.core.constants import OBJECT_TO_IDX

ALLOWED_INPUTS = ["frames", "mission", "terminated", "truncated"]

def load_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(json.loads(s))
    return out

def _agent_traj(state_seq):
    return [(tuple(s["agent"]["pos"]), int(s["agent"]["dir"])) for s in state_seq]

def _semantic_audit_full(full_rec):
    states = full_rec["state_seq"]
    if not states:
        return "empty_traj"

    lavas = [tuple(o["pos"]) for o in states[0].get("objects", []) if o.get("type") == "lava"]
    if not lavas:
        return "no_lava_found"
    lava_set = set(lavas)

    ys = [p[1] for p in lavas]
    wall_y = max(set(ys), key=ys.count)

    agent_ys = []
    for s in states:
        apos = tuple(s["agent"]["pos"])
        if apos in lava_set:
            return "stepped_on_lava"
        agent_ys.append(int(apos[1]))

    if not (min(agent_ys) < wall_y < max(agent_ys)):
        return "did_not_cross_wall"
    return "OK"

def _expand_tile_mask_to_pixels(tile_mask: np.ndarray, H: int, W: int) -> np.ndarray:
    Ht, Wt = tile_mask.shape
    th, tw = H // Ht, W // Wt
    pix = np.kron(tile_mask, np.ones((th, tw), dtype=bool))
    return pix[:H, :W]  # crop safety

def validate_group(gid: str, recs: dict, root: Path):
    if set(recs.keys()) != {"full", "nocue", "cf"}:
        return False, "missing_variants"

    full, nc, cf = recs["full"], recs["nocue"], recs["cf"]

    # Contract fields
    for v in (full, nc, cf):
        mif = v.get("model_input_fields", None)
        if mif != ALLOWED_INPUTS:
            return False, "bad_model_input_fields"
        for k in ALLOWED_INPUTS:
            if k not in v:
                return False, f"missing_input_key_{k}"

        if v.get("variant") not in ("full", "nocue", "cf"):
            return False, "bad_variant"

    # Actions must match
    if full.get("actions_id") != nc.get("actions_id") or full.get("actions_id") != cf.get("actions_id"):
        return False, "action_mismatch"

    # Physics identity
    ft, nt, ct = _agent_traj(full["state_seq"]), _agent_traj(nc["state_seq"]), _agent_traj(cf["state_seq"])
    if ft != nt or ft != ct:
        return False, "physics_drift"

    # Outcomes
    if not full["success"] or full["reward"] <= 0 or full["truncated"]:
        return False, "full_failed"
    if cf["success"] or cf["reward"] != 0 or cf["truncated"]:
        return False, "cf_bad_outcome"
    if nc["success"] != full["success"] or nc["reward"] != full["reward"] or nc["truncated"] != full["truncated"]:
        return False, "nocue_outcome_mismatch"

    # Semantic audit (Full)
    sem = _semantic_audit_full(full)
    if sem != "OK":
        return False, f"semantic_{sem}"

    # NoCue meta
    nm = nc.get("nocue_meta", {})
    if nm.get("window_policy") != "EARLY":
        return False, "nocue_bad_window_policy"
    if nm.get("targets") != ["lava"]:
        return False, "nocue_bad_targets"
    w_steps = set(nm.get("window_steps", []))
    if not w_steps:
        return False, "nocue_no_masking"

    # Pixel audit (NoCue)
    lava_idx = int(OBJECT_TO_IDX["lava"])

    for t, (fp_rel, np_rel) in enumerate(zip(full["frames"], nc["frames"])):
        fp = root / fp_rel
        np_path = root / np_rel
        if not fp.exists() or not np_path.exists():
            return False, "missing_png"

        img_f = np.array(Image.open(fp))
        img_n = np.array(Image.open(np_path))
        if img_f.shape != img_n.shape:
            return False, "shape_mismatch"

        diff = np.any(img_f != img_n, axis=2)

        if t not in w_steps:
            if np.any(diff):
                return False, "diff_outside_window"
        else:
            # D5: front_cell must NOT be lava when masked
            if full["state_seq"][t]["front_cell"]["type"] == "lava":
                return False, "masked_interaction_frame_D5"

            enc = np.array(full["state_seq"][t]["state_encoding"], dtype=int)
            tile_mask = (enc[:, :, 0] == lava_idx)

            pixel_mask = _expand_tile_mask_to_pixels(tile_mask, img_f.shape[0], img_f.shape[1])

            if np.any(diff & ~pixel_mask):
                return False, "diff_outside_lava_tiles"
            if not np.any(diff & pixel_mask):
                return False, "no_diff_in_target_tiles"

    return True, "OK"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", required=True)
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    data = load_jsonl(Path(args.triplets))
    groups = defaultdict(dict)
    for r in data:
        groups[r["group_id"]][r["variant"]] = r

    ok_cnt = 0
    fail_cnt = 0

    for gid in sorted(groups.keys()):
        ok, msg = validate_group(gid, groups[gid], Path(args.root))
        if not ok:
            fail_cnt += 1
            print(f"[FAIL] {gid}: {msg}")
        else:
            ok_cnt += 1

    print(json.dumps({"passes": ok_cnt, "fails": fail_cnt}, indent=2))
    if fail_cnt == 0 and ok_cnt > 0:
        print("✅ ALL CHECKS PASSED.")
    else:
        print("❌ FAILED.")

if __name__ == "__main__":
    main()
