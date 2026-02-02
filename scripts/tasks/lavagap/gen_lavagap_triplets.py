#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LavaGap Triplet Generator for GridWM-Judge.

Generates Full/NoCue/CF triplets for LavaGap task with strict compliance to:
- GridWM-Judge Project Overview Scientific Contract (Final SSOT)
- NoCue_spec.md
- TRIPLET_AUDIT_SPEC_REVIEWER.md

Usage:
    python gen_lavagap_triplets.py --out-dir data_lavagap_vFinal --num 100 --env-id MiniGrid-LavaGapS7-v0 --resume
"""

import argparse
import copy
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image

from minigrid.core.constants import IDX_TO_OBJECT, OBJECT_TO_IDX
from minigrid.core.world_object import Goal

# -------------------------
# Constants
# -------------------------

ALLOWED_INPUTS = ["frames", "mission", "terminated", "truncated"]

ACTION_ID2NAME = {
    0: "left", 1: "right", 2: "forward",
    3: "pickup", 4: "drop", 5: "toggle", 6: "done"
}

DIR2VEC = {
    0: (1, 0),   # right (positive X)
    1: (0, 1),   # down (positive Y)
    2: (-1, 0),  # left (negative X)
    3: (0, -1),  # up (negative Y)
}

# -------------------------
# IO helpers
# -------------------------

def _rm_tree(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)

def append_jsonl_batch(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def load_jsonl_tolerant(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError:
                break
    return out

# -------------------------
# Rendering & State extraction
# -------------------------

def render_agent_pov(env, tile_size: int, grid_h: int, grid_w: int) -> Optional[np.ndarray]:
    """
    Render agent POV and sanity-check its size matches grid_h/grid_w (tiles).
    """
    try:
        frame = env.unwrapped.get_frame(tile_size=tile_size, agent_pov=True)
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return None
        H_exp, W_exp = grid_h * tile_size, grid_w * tile_size
        if frame.shape[0] != H_exp or frame.shape[1] != W_exp:
            return None
        return frame
    except Exception:
        return None

def _safe_state_encoding(env_unwrapped, obs: Dict) -> Optional[np.ndarray]:
    """
    Bug-A fix: always prefer env.unwrapped.gen_obs()['image'] (GT encoding),
    fall back to obs['image'].
    """
    try:
        g = env_unwrapped.gen_obs()
        img = g.get("image", None)
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
            return img
    except Exception:
        pass
    img2 = obs.get("image", None)
    if isinstance(img2, np.ndarray) and img2.ndim == 3 and img2.shape[2] == 3:
        return img2
    return None

def extract_state(env_unwrapped, obs: Dict) -> Dict:
    img_enc = _safe_state_encoding(env_unwrapped, obs)

    visible_types_gt: Set[str] = set()
    if img_enc is not None:
        H, W = img_enc.shape[0], img_enc.shape[1]
        for i in range(H):
            for j in range(W):
                obj_idx = int(img_enc[i, j, 0])
                name = IDX_TO_OBJECT.get(obj_idx, None)
                if name and name not in ("unseen", "empty"):
                    visible_types_gt.add(str(name))

    # full object dump (excluding walls/floor)
    objects = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            cell = env_unwrapped.grid.get(x, y)
            if cell is None:
                continue
            t = getattr(cell, "type", "unknown")
            if t in ("wall", "floor"):
                continue
            o = {"type": str(t), "pos": [int(x), int(y)], "state": 0}
            if hasattr(cell, "color"):
                o["color"] = str(getattr(cell, "color"))
            objects.append(o)

    fx, fy = env_unwrapped.front_pos
    f_cell = env_unwrapped.grid.get(fx, fy)
    f_type = getattr(f_cell, "type", "empty") if f_cell else "empty"

    # visible_types = model-visible types (NoCue will edit this)
    visible_types = sorted(list(visible_types_gt))

    return {
        "agent": {
            "pos": [int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1])],
            "dir": int(env_unwrapped.agent_dir),
            "carrying": (
                {"type": str(env_unwrapped.carrying.type), "color": str(env_unwrapped.carrying.color)}
                if env_unwrapped.carrying else None
            ),
        },
        "front_cell": {"pos": [int(fx), int(fy)], "type": str(f_type)},
        "visible_types_gt": sorted(list(visible_types_gt)),
        "visible_types": visible_types,
        "state_encoding": img_enc.astype(int).tolist() if img_enc is not None else None,
        "objects": objects,
        "mask_applied": False,
        "mask_type": None,
    }

def rollout_from_current(
    env,
    obs0: Dict,
    actions: List[int],
    tile_size: int,
    out_dir: Path,
    rel_dir: str,
    grid_h: int,
    grid_w: int,
) -> Optional[Dict]:
    """
    IMPORTANT: does NOT reset.
    Starts from (env, obs0) current state. (Bug-B/CF correctness)
    """
    frames, states = [], []
    reward_sum = 0.0
    terminated = truncated = False

    frame0 = render_agent_pov(env, tile_size, grid_h, grid_w)
    if frame0 is None:
        return None
    save_png(frame0, out_dir / rel_dir / "step_000.png")
    frames.append(str(Path(rel_dir) / "step_000.png"))
    states.append(extract_state(env.unwrapped, obs0))

    obs = obs0
    for t, a in enumerate(actions):
        obs, r, term, trunc, _ = env.step(int(a))
        reward_sum += float(r)
        terminated |= bool(term)
        truncated |= bool(trunc)

        frame = render_agent_pov(env, tile_size, grid_h, grid_w)
        if frame is None:
            return None
        save_png(frame, out_dir / rel_dir / f"step_{t+1:03d}.png")
        frames.append(str(Path(rel_dir) / f"step_{t+1:03d}.png"))
        states.append(extract_state(env.unwrapped, obs))

        if term or trunc:
            break

    return {
        "frames": frames,
        "state_seq": states,
        "reward": float(reward_sum),
        "success": bool(terminated and reward_sum > 0),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }

def _agent_traj(state_seq: List[Dict]) -> List[Tuple[Tuple[int, int], int]]:
    return [(tuple(st["agent"]["pos"]), int(st["agent"]["dir"])) for st in state_seq]

# -------------------------
# Planner: BFS (Lava = Wall)
# -------------------------

@dataclass(frozen=True)
class NavState:
    x: int
    y: int
    d: int

def _find_goal_pos(env_unwrapped) -> Optional[Tuple[int, int]]:
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            c = env_unwrapped.grid.get(x, y)
            if c is not None and getattr(c, "type", None) == "goal":
                return (int(x), int(y))
    return None

def _is_blocking(cell) -> bool:
    if cell is None:
        return False
    cell_type = getattr(cell, "type", None)
    return cell_type in ("wall", "lava")

def plan_lavagap_bfs(env_unwrapped, max_nodes: int = 300000) -> Optional[List[int]]:
    """Use BFS to find path to goal."""
    goal_pos = _find_goal_pos(env_unwrapped)
    if not goal_pos:
        return None

    start = NavState(int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1]), int(env_unwrapped.agent_dir))

    from collections import deque
    q = deque([start])
    visited = set()  # Use tuples for visited states
    parent = {}  # (x, y, d) -> (prev_state, act)

    steps = 0
    while q:
        cur = q.popleft()
        state_key = (cur.x, cur.y, cur.d)
        steps += 1
        if steps > max_nodes:
            return None

        if state_key in visited:
            continue
        visited.add(state_key)

        if (cur.x, cur.y) == goal_pos:
            acts = []
            node = cur
            node_key = (node.x, node.y, node.d)
            while node_key in parent:
                prev_state, a = parent[node_key]
                acts.append(int(a))
                node = prev_state
                node_key = (node.x, node.y, node.d)
            return list(reversed(acts))

        # Try all actions: turn left, turn right, forward
        for act in (0, 1, 2):
            if act in (0, 1):  # turn
                nd = (cur.d - 1) % 4 if act == 0 else (cur.d + 1) % 4
                nxt = NavState(cur.x, cur.y, nd)
            else:  # forward
                dx, dy = DIR2VEC[cur.d]
                nx, ny = cur.x + dx, cur.y + dy
                if not (0 <= nx < env_unwrapped.width and 0 <= ny < env_unwrapped.height):
                    continue
                cell = env_unwrapped.grid.get(nx, ny)
                if _is_blocking(cell) and (nx, ny) != goal_pos:
                    continue
                nxt = NavState(nx, ny, cur.d)

            nxt_key = (nxt.x, nxt.y, nxt.d)
            if nxt_key not in visited:
                parent[nxt_key] = (cur, act)
                q.append(nxt)

    return None

# -------------------------
# CF (Type-2): relocate goal
# -------------------------

def relocate_goal_cf(env_unwrapped, forbidden: Set[Tuple[int, int]], rng: np.random.Generator, max_tries: int = 200) -> Optional[Dict]:
    goal_pos = _find_goal_pos(env_unwrapped)
    if not goal_pos:
        return None

    # Find empty cells to relocate goal to
    empty_cells = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            if (x, y) not in forbidden:
                cell = env_unwrapped.grid.get(x, y)
                if cell is None:  # empty cell
                    empty_cells.append((x, y))

    if not empty_cells:
        return None

    # Choose random empty cell
    new_goal_pos = tuple(rng.choice(empty_cells))

    # Remove old goal and place new one
    env_unwrapped.grid.set(goal_pos[0], goal_pos[1], None)
    from minigrid.core.world_object import Goal
    env_unwrapped.grid.set(new_goal_pos[0], new_goal_pos[1], Goal())

    return {
        "cf_type": "type2_intervention",
        "cf_mode": "relocate_goal",
        "intervention_step": 0,
        "goal_from": [int(goal_pos[0]), int(goal_pos[1])],
        "goal_to": [int(new_goal_pos[0]), int(new_goal_pos[1])],
    }

# -------------------------
# NoCue masking (tile-level)
# -------------------------

def mask_lava_tiles_inplace(pov_img: np.ndarray, state_encoding: List) -> Tuple[int, int]:
    if state_encoding is None:
        return 0, 0
    enc = np.array(state_encoding, dtype=int)
    Ht, Wt = enc.shape[0], enc.shape[1]
    H, W = pov_img.shape[0], pov_img.shape[1]
    th, tw = H // Ht, W // Wt

    lava_idx = int(OBJECT_TO_IDX["lava"])
    masked, target = 0, 0
    for i in range(Ht):
        for j in range(Wt):
            if int(enc[i, j, 0]) == lava_idx:
                target += 1
                pov_img[i*th:(i+1)*th, j*tw:(j+1)*tw] = 0
                masked += 1
    return masked, target

# -------------------------
# Semantic audit (FULL)
# -------------------------

def semantic_lavagap_full(full_traj: Dict) -> Optional[str]:
    states = full_traj["state_seq"]
    if not states:
        return "empty_traj"

    lava_pos = set()
    for o in states[0].get("objects", []):
        if o.get("type") == "lava":
            lava_pos.add(tuple(o.get("pos", [])))
    if not lava_pos:
        return "no_lava_found"

    agent_ys = []
    for st in states:
        apos = tuple(st["agent"]["pos"])
        if apos in lava_pos:
            return "agent_stepped_on_lava"
        agent_ys.append(int(apos[1]))

    ys = [p[1] for p in lava_pos]
    wall_y = max(set(ys), key=ys.count)
    if not (min(agent_ys) < wall_y < max(agent_ys)):
        return "did_not_cross_lava_wall"

    return None

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--env-id", default="MiniGrid-LavaGapS7-v0")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tile-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=300)

    # NoCue
    ap.add_argument("--nocue-window-end", type=int, default=4)
    ap.add_argument("--mask-ratio-max", type=float, default=0.35)
    ap.add_argument("--alignment-min", type=float, default=0.70)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    triplets_path = out_dir / "triplets.jsonl"
    tmp_root = out_dir / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    if not args.resume and triplets_path.exists():
        triplets_path.unlink()

    made = 0
    seed = int(args.seed_start)

    while made < args.num:
        gid = f"lavagap_s{seed:06d}"
        final_root = out_dir / gid
        if final_root.exists():
            if args.resume:
                print(f"[RESUME] skip {gid}")
                seed += 1
                continue
            _rm_tree(final_root)

        work_dir = tmp_root / gid
        _rm_tree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        env = None
        try:
            # 1) FULL
            env = gym.make(args.env_id, render_mode="rgb_array")
            obs0, _ = env.reset(seed=seed)
            gh, gw = obs0["image"].shape[0], obs0["image"].shape[1]
            mission = str(env.unwrapped.mission)

            actions = plan_lavagap_bfs(env.unwrapped)
            if not actions or len(actions) > args.max_steps:
                raise ValueError("plan_failed")

            full_traj = rollout_from_current(env, obs0, actions, args.tile_size, work_dir, "full", gh, gw)
            env.close()
            env = None

            if not full_traj or not full_traj["success"] or full_traj["reward"] <= 0:
                raise ValueError("full_failed")
            if full_traj["truncated"]:
                raise ValueError("full_truncated")

            sem_err = semantic_lavagap_full(full_traj)
            if sem_err:
                raise ValueError(f"full_semantic_{sem_err}")

            # path forbid set (agent pos + front pos)
            forbidden = {tuple(st["agent"]["pos"]) for st in full_traj["state_seq"]} | \
                        {tuple(st["front_cell"]["pos"]) for st in full_traj["state_seq"]}

            # 2) CF (Type-2)
            cf_traj, cf_meta = None, None
            for k in range(30):
                _rm_tree(work_dir / "cf")

                env_cf = gym.make(args.env_id, render_mode="rgb_array")
                try:
                    obs_cf0, _ = env_cf.reset(seed=seed)
                    rng = np.random.default_rng(seed + 1000 + k)
                    meta = relocate_goal_cf(env_cf.unwrapped, forbidden, rng)
                    if not meta:
                        continue

                    # IMPORTANT: post-intervention obs from current state, and rollout WITHOUT reset
                    obs_cf = env_cf.unwrapped.gen_obs()
                    traj = rollout_from_current(env_cf, obs_cf, actions, args.tile_size, work_dir, "cf", gh, gw)
                    if traj is None:
                        continue

                    goal_visible = any("goal" in st.get("visible_types_gt", []) for st in traj["state_seq"])
                    if (not traj["success"] and
                        traj["reward"] == 0 and
                        not traj["truncated"] and
                        not goal_visible and
                        _agent_traj(traj["state_seq"]) == _agent_traj(full_traj["state_seq"])):
                        cf_traj, cf_meta = traj, meta
                        break
                finally:
                    env_cf.close()

            if not cf_traj:
                raise ValueError("cf_failed")

            # 3) NoCue (EARLY, target=lava, D5 exclude front_cell==lava)
            leak_indices = []
            early_end = min(int(args.nocue_window_end), len(full_traj["state_seq"]))
            first_interaction_step = next(
                (i for i in range(len(full_traj["state_seq"])) if full_traj["state_seq"][i]["front_cell"]["type"] == "lava"),
                len(full_traj["state_seq"])
            )

            for t in range(early_end):
                st = full_traj["state_seq"][t]
                if ("lava" in st.get("visible_types_gt", [])) and (st["front_cell"]["type"] != "lava"):
                    leak_indices.append(int(t))

            masked_indices = [t for t in leak_indices if t < first_interaction_step]
            if not masked_indices:
                raise ValueError("nocue_no_frames")

            (work_dir / "nocue").mkdir(parents=True, exist_ok=True)

            nocue_states = copy.deepcopy(full_traj["state_seq"])
            per_frame_ratios = []
            align_ratios = []
            masked_tiles_total = 0

            for i, fp_rel in enumerate(full_traj["frames"]):
                img = np.array(Image.open(work_dir / fp_rel))

                st = nocue_states[i]
                if i in masked_indices:
                    cnt, tgt = mask_lava_tiles_inplace(img, st["state_encoding"])
                    masked_tiles_total += int(cnt)

                    per_frame_ratios.append(cnt / float(gh * gw))
                    align_ratios.append((cnt / tgt) if tgt > 0 else 0.0)

                    st["mask_applied"] = True
                    st["mask_type"] = "tile_suppression"
                    vgt = set(st.get("visible_types_gt", []))
                    st["visible_types"] = sorted([x for x in vgt if x != "lava"])
                else:
                    st["mask_applied"] = False
                    st["mask_type"] = None
                    st["visible_types"] = st.get("visible_types_gt", [])

                save_png(img, work_dir / "nocue" / f"step_{i:03d}.png")

            max_mask_ratio = float(np.max(per_frame_ratios)) if per_frame_ratios else 0.0
            avg_mask_ratio = float(np.mean(per_frame_ratios)) if per_frame_ratios else 0.0
            alignment_score = float(np.mean(align_ratios)) if align_ratios else 0.0
            mask_budget_tiles = int(np.floor(args.mask_ratio_max * float(gh * gw) * len(masked_indices)))

            if alignment_score < float(args.alignment_min):
                raise ValueError("nocue_alignment_fail")
            if max_mask_ratio > float(args.mask_ratio_max):
                raise ValueError("nocue_mask_ratio_exceeded")
            if masked_tiles_total > mask_budget_tiles:
                raise ValueError("nocue_budget_exceeded")

            # Commit work_dir -> final_root
            _rm_tree(final_root)
            shutil.move(str(work_dir), str(final_root))

            # Records (DoorKey style: 3 lines per group)
            base_rec = {
                "group_id": gid,
                "task": "lavagap",
                "env_id": args.env_id,
                "seed": int(seed),
                "actions_id": [int(a) for a in actions],
                "actions_text": [ACTION_ID2NAME[int(a)] for a in actions],
                "mission": mission,
                "model_input_fields": list(ALLOWED_INPUTS),
            }

            def mk_rec(variant: str, traj: Dict, cf_meta_obj=None, nocue_meta_obj=None) -> Dict:
                return {
                    **base_rec,
                    "variant": variant,
                    "frames": [f"{gid}/{variant}/{Path(p).name}" for p in traj["frames"]],
                    "state_seq": traj["state_seq"],
                    "reward": float(traj["reward"]),
                    "success": bool(traj["success"]),
                    "terminated": bool(traj["terminated"]),
                    "truncated": bool(traj["truncated"]),
                    "cf_meta": cf_meta_obj,
                    "nocue_meta": nocue_meta_obj,
                }

            nocue_traj = {
                "frames": full_traj["frames"],  # names align step_XXX.png
                "state_seq": nocue_states,
                "reward": full_traj["reward"],
                "success": full_traj["success"],
                "terminated": full_traj["terminated"],
                "truncated": full_traj["truncated"],
            }

            nocue_meta = {
                "targets": ["lava"],
                "window_policy": "EARLY",
                "window_steps": [int(x) for x in masked_indices],
                "first_interaction_step": int(first_interaction_step),

                "mask_type": "tile_suppression",
                "mask_strength_target": float(args.mask_ratio_max),
                "mask_strength_actual": float(avg_mask_ratio),
                "masked_frames": int(len(masked_indices)),
                "physics_check_passed": True,

                "max_mask_ratio": float(max_mask_ratio),
                "avg_mask_ratio": float(avg_mask_ratio),
                "masked_tiles_total": int(masked_tiles_total),
                "mask_budget_tiles": int(mask_budget_tiles),
                "alignment_score": float(alignment_score),
                "alignment_threshold": float(args.alignment_min),

                "leak_indices": [int(x) for x in leak_indices],
                "notes": "EARLY lava masking; excludes front_cell==lava (D5)."
            }

            records = [
                mk_rec("full", full_traj),
                mk_rec("nocue", nocue_traj, None, nocue_meta),
                mk_rec("cf", cf_traj, cf_meta, None),
            ]
            append_jsonl_batch(triplets_path, records)

            made += 1
            print(f"[OK {made}/{args.num}] {gid}")

        except Exception as e:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            _rm_tree(work_dir)
            # skip seed
            # print reason for debugging
            print(f"[SKIP] {gid} : {e}")

        seed += 1

if __name__ == "__main__":
    main()
