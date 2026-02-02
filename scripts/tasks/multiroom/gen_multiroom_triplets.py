#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiRoom Triplet Generator for GridWM-Judge.

Generates Full/NoCue/CF triplets for MultiRoom task with strict compliance to:
- GridWM-Judge Project Overview Scientific Contract (Final SSOT)
- NoCue_spec.md
- TRIPLET_AUDIT_SPEC_REVIEWER.md

Usage:
    python gen_multiroom_triplets.py --out-dir data_multiroom_vFinal --num 100 --env-id MiniGrid-MultiRoom-N6-v0 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image

from minigrid.core.constants import IDX_TO_OBJECT
from minigrid.core.world_object import Goal

# -------------------------
# Constants
# -------------------------
ACTION_ID2NAME = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

DIR_VEC = {
    0: (1, 0),  # right
    1: (0, 1),  # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}

ALLOWED_INPUT_FIELDS = ["frames", "mission", "terminated", "truncated"]

# -------------------------
# IO utils
# -------------------------
def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)

def append_jsonl_batch(path: Path, objs: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())

def append_skip_log(path: Path, seed: int, reason: str) -> None:
    """Production logging: timestamp, atomic-ish write, fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] Seed {seed}\n{reason}\n{'-'*80}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
        f.flush()
        os.fsync(f.fileno())

def render_agent_pov(env, tile_size: int, grid_h: int = 7, grid_w: int = 7) -> Optional[np.ndarray]:
    """
    Must return agent POV image with shape (grid_h*tile_size, grid_w*tile_size, 3).
    If cannot guarantee this, return None (hard reject).
    """
    for kwargs in (
        dict(tile_size=tile_size, agent_pov=True, highlight=False),
        dict(tile_size=tile_size, agent_pov=True),
    ):
        try:
            frame = env.unwrapped.get_frame(**kwargs)
            if isinstance(frame, np.ndarray) and frame.shape[:2] == (grid_h * tile_size, grid_w * tile_size):
                return frame
        except Exception:
            pass
    return None

def _door_state(obj) -> int:
    """Return Minigrid Door state index: open=0, closed=1, locked=2."""
    if getattr(obj, "is_locked", False):
        return 2
    if getattr(obj, "is_open", False):
        return 0
    return 1

def extract_state(env_unwrapped, obs: Dict) -> Dict:
    """Extract a single time-step state snapshot."""

    agent_pos = tuple(int(x) for x in env_unwrapped.agent_pos)
    agent_dir = int(env_unwrapped.agent_dir)
    dx, dy = DIR_VEC[agent_dir]
    front_pos = (agent_pos[0] + dx, agent_pos[1] + dy)

    objects = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            o = env_unwrapped.grid.get(x, y)
            if o is None:
                continue
            if getattr(o, "type", None) in ("wall", "floor"):
                continue
            objects.append({
                "type": getattr(o, "type", "unknown"),
                "color": getattr(o, "color", "unknown"),
                "state": _door_state(o) if getattr(o, "type", None) == "door" else 0,
                "pos": [int(x), int(y)],
            })

    # IMPORTANT: use env.unwrapped.gen_obs()["image"] for 7x7x3 semantic encoding
    state_encoding = env_unwrapped.gen_obs()["image"].tolist()

    arr = np.array(state_encoding, dtype=np.int64)
    visible_types = set()
    H, W, _ = arr.shape

    # Debug: check what objects are actually in the 7x7 observation
    unique_indices = set()
    unique_objects = set()
    door_positions = []
    goal_positions = []

    for i in range(H):
        for j in range(W):
            obj_idx = int(arr[i, j, 0])
            unique_indices.add(obj_idx)
            name = IDX_TO_OBJECT.get(obj_idx, f"unknown_{obj_idx}")
            if name and name != f"unknown_{obj_idx}":
                unique_objects.add(name)

            # Track door and goal positions
            if name == "door":
                door_positions.append((i, j, arr[i, j, 1], arr[i, j, 2]))  # color and state
            elif name == "goal":
                goal_positions.append((i, j))

            # Only include meaningful objects
            if name and name not in ("empty", "wall", "floor", "unseen"):
                visible_types.add(name)


    fobj = None
    if 0 <= front_pos[0] < env_unwrapped.width and 0 <= front_pos[1] < env_unwrapped.height:
        fobj = env_unwrapped.grid.get(front_pos[0], front_pos[1])

    if fobj is None:
        front_cell = {"pos": [int(front_pos[0]), int(front_pos[1])], "type": "floor", "color": "none", "state": 0}
    else:
        front_cell = {
            "pos": [int(front_pos[0]), int(front_pos[1])],
            "type": getattr(fobj, "type", "unknown"),
            "color": getattr(fobj, "color", "unknown"),
            "state": _door_state(fobj) if getattr(fobj, "type", None) == "door" else 0,
        }

    return {
        "agent": {"pos": [int(agent_pos[0]), int(agent_pos[1])], "dir": int(agent_dir)},
        "agent_room": None,  # will be computed later
        "front_cell": front_cell,
        "objects": objects,
        "state_encoding": state_encoding,
        "visible_types_gt": sorted(list(visible_types)),
        "visible_types": sorted(list(visible_types)),
        "mask_applied": False,
        "mask_type": None,
    }

def rollout_from_current(
    env,
    obs0,
    actions: List[int],
    tile_size: int,
    out_dir: Path,
    rel_dir: str,
    grid_h: int,
    grid_w: int,
) -> Optional[Dict]:
    out_dir = out_dir / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    state_seq = []

    img0 = render_agent_pov(env, tile_size, grid_h, grid_w)
    if img0 is None:
        return None
    save_png(img0, out_dir / "step_000.png")
    frames.append(f"{rel_dir}/step_000.png")
    state_seq.append(extract_state(env.unwrapped, obs0))

    total_reward = 0.0
    terminated = False
    truncated = False

    obs = obs0
    for t, a in enumerate(actions):
        obs, r, term, trunc, _ = env.step(int(a))
        total_reward += float(r)
        terminated = bool(term)
        truncated = bool(trunc)

        img = render_agent_pov(env, tile_size, grid_h, grid_w)
        if img is None:
            return None
        save_png(img, out_dir / f"step_{t+1:03d}.png")
        frames.append(f"{rel_dir}/step_{t+1:03d}.png")
        state_seq.append(extract_state(env.unwrapped, obs))

        if terminated or truncated:
            break

    return {
        "frames": frames,
        "state_seq": state_seq,
        "reward": float(total_reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(terminated and total_reward > 0),
    }

# -------------------------
# Planning (BFS with door handling)
# -------------------------
@dataclass(frozen=True)
class MRState:
    x: int
    y: int
    d: int

def _is_passable(env_unwrapped, x: int, y: int) -> bool:
    if not (0 <= x < env_unwrapped.width and 0 <= y < env_unwrapped.height):
        return False
    o = env_unwrapped.grid.get(x, y)
    if o is None:
        return True
    t = getattr(o, "type", None)
    if t in ("floor",):
        return True
    if t == "goal":
        return True
    if t == "door":
        return True  # For planning, assume doors are passable (will be opened when needed)
    return False

def _find_goal(env_unwrapped) -> Optional[Tuple[int, int]]:
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            c = env_unwrapped.grid.get(x, y)
            if c and getattr(c, "type", None) == "goal":
                return (x, y)
    return None

def plan_multiroom(env_unwrapped, max_nodes: int = 200000) -> Optional[List[int]]:
    goal_pos = _find_goal(env_unwrapped)
    if not goal_pos:
        return None
    start = MRState(int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1]), int(env_unwrapped.agent_dir))
    q = deque([start])
    parent: Dict[MRState, Tuple[Optional[MRState], Optional[int]]] = {start: (None, None)}
    nodes = 0; final_state = None

    while q:
        cur = q.popleft()
        nodes += 1
        if nodes > max_nodes:
            return None
        if (cur.x, cur.y) == goal_pos:
            final_state = cur
            break
        for act in (0, 1):
            nd = (cur.d - 1) % 4 if act == 0 else (cur.d + 1) % 4
            nxt = MRState(cur.x, cur.y, nd)
            if nxt not in parent:
                parent[nxt] = (cur, act)
                q.append(nxt)
        dx, dy = DIR_VEC[cur.d]
        nx, ny = cur.x + dx, cur.y + dy
        if _is_passable(env_unwrapped, nx, ny):
            nxt = MRState(nx, ny, cur.d)
            if nxt not in parent:
                parent[nxt] = (cur, 2)
                q.append(nxt)

    if final_state is None:
        return None
    raw = []
    cur = final_state
    while parent[cur][0] is not None:
        prev, act = parent[cur]
        raw.append(act)
        cur = prev
    raw.reverse()

    opened_doors = set()
    vx, vy, vd = start.x, start.y, start.d
    final_actions = []
    for act in raw:
        if act == 2:
            dx, dy = DIR_VEC[vd]
            nx, ny = vx + dx, vy + dy
            cell = env_unwrapped.grid.get(nx, ny)
            if cell and cell.type == "door":
                if (nx, ny) not in opened_doors and not getattr(cell, "is_open", False):
                    final_actions.append(5)  # toggle
                    opened_doors.add((nx, ny))
            final_actions.append(2)  # forward
            vx, vy = nx, ny
        elif act == 0:
            final_actions.append(0)  # left
            vd = (vd - 1) % 4
        elif act == 1:
            final_actions.append(1)  # right
            vd = (vd + 1) % 4
    return final_actions

# -------------------------
# Room computation
# -------------------------
def compute_room_labels(env_unwrapped):
    W, H = env_unwrapped.width, env_unwrapped.height
    goal_pos = _find_goal(env_unwrapped)
    if not goal_pos:
        raise ValueError("Goal not found")
    passable = np.zeros((W, H), dtype=bool)
    for x in range(W):
        for y in range(H):
            c = env_unwrapped.grid.get(x, y)
            passable[x, y] = (c is None) or (c.type not in ("wall", "door"))
    labels = -np.ones((W, H), dtype=int)
    rid = 0
    for sx in range(W):
        for sy in range(H):
            if passable[sx, sy] and labels[sx, sy] == -1:
                dq = deque([(sx, sy)])
                labels[sx, sy] = rid
                while dq:
                    x, y = dq.popleft()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < W and 0 <= ny < H and passable[nx, ny] and labels[nx, ny] == -1:
                            labels[nx, ny] = rid
                            dq.append((nx, ny))
                rid += 1
    gx, gy = goal_pos
    return labels, goal_pos, int(labels[gx, gy])

def find_enter_final_room_step(states, labels, goal_rid: int) -> Optional[int]:
    for t, st in enumerate(states):
        ax, ay = st["agent"]["pos"]
        if 0 <= ax < labels.shape[0] and 0 <= ay < labels.shape[1]:
            if int(labels[ax, ay]) == goal_rid:
                return t
    return None

# -------------------------
# NoCue masking
# -------------------------
def mask_targets_inplace(img: np.ndarray, state_encoding, targets: Set[str]) -> Tuple[int, int]:
    if state_encoding is None:
        return 0, 0
    enc = np.array(state_encoding, dtype=int)
    Ht, Wt, _ = enc.shape
    H, W = img.shape[0], img.shape[1]
    th, tw = H // Ht, W // Wt
    masked_count = target_count = 0
    for i in range(Ht):
        for j in range(Wt):
            name = IDX_TO_OBJECT.get(int(enc[i, j, 0]), None)
            if name in targets:
                target_count += 1
                y1, y2 = i * th, (i + 1) * th
                x1, x2 = j * tw, (j + 1) * tw
                img[y1:y2, x1:x2] = 0
                masked_count += 1
    return masked_count, target_count

def _has_targets(state_encoding, targets: Set[str]) -> bool:
    """Whether the 7x7x3 semantic encoding contains an object (optionally color-matched)."""
    if state_encoding is None:
        return False
    arr = np.array(state_encoding, dtype=np.int64)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return False
    Ht, Wt = arr.shape[0], arr.shape[1]
    for i in range(Ht):
        for j in range(Wt):
            if IDX_TO_OBJECT.get(int(arr[i, j, 0]), None) in targets:
                return True
    return False

# -------------------------
# CF: remove goal
# -------------------------
def remove_goal_cf(env_unwrapped) -> Optional[Dict]:
    gp = _find_goal(env_unwrapped)
    if not gp:
        return None
    env_unwrapped.grid.set(gp[0], gp[1], None)
    return {
        "cf_mode": "remove_goal",
        "goal_from": [int(gp[0]), int(gp[1])],
        "goal_to": None,
        "intervention_step": 0
    }

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--env-id", default="MiniGrid-MultiRoom-N6-v0")
    ap.add_argument("--tile-size", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--nocue-max-visible", type=int, default=12)
    ap.add_argument("--alignment-min", type=float, default=0.7)
    ap.add_argument("--mask-ratio-max", type=float, default=0.35)
    ap.add_argument("--mask-strength-target", type=float, default=0.05)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    jsonl_path = out_dir / "triplets.jsonl"
    skip_log_path = out_dir / "skip.log"

    if not args.resume:
        if jsonl_path.exists():
            jsonl_path.unlink()
        if skip_log_path.exists():
            skip_log_path.unlink()

    made = 0
    seed = args.seed_start
    while made < args.num:
        gid = f"multiroom_s{seed:06d}"
        final_root = out_dir / gid
        if final_root.exists():
            if args.resume:
                print(f"[RESUME] Skipping {gid}")
                seed += 1
                continue

        work_dir = out_dir / "_tmp" / gid
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        env = None

        try:
            # ---- PLAN (separate env) ----
            env_plan = gym.make(args.env_id, render_mode="rgb_array")
            try:
                env_plan.reset(seed=seed)
                actions = plan_multiroom(env_plan.unwrapped)
            finally:
                env_plan.close()

            if not actions or len(actions) > args.max_steps:
                raise ValueError("Plan failed")

            # ---- FULL ----
            env = gym.make(args.env_id, render_mode="rgb_array")
            obs0, _ = env.reset(seed=seed)
            if "image" not in obs0:
                raise ValueError("Obs missing 'image'")
            gh, gw = obs0["image"].shape[:2]
            mission = getattr(env.unwrapped, "mission", "")
            full_traj = rollout_from_current(env, obs0, actions, args.tile_size, work_dir, "full", gh, gw)
            env.close()
            env = None

            if not full_traj or not full_traj["success"]:
                raise ValueError("Full rollout failed")
            if full_traj["truncated"]:
                raise ValueError("Full truncated")

            # Room analysis
            labels, _, goal_rid = compute_room_labels(env_plan.unwrapped)
            t_enter = find_enter_final_room_step(full_traj["state_seq"], labels, goal_rid)
            if t_enter is None:
                raise ValueError("Never entered goal room")

            # Semantic checks
            toggles = [t for t, a in enumerate(actions) if a == 5 and full_traj["state_seq"][t]["front_cell"]["type"] == "door"]
            if len(toggles) < 2:
                raise ValueError("Semantic fail: toggles < 2")

            # ---- NOCUE ----
            targets = {"door", "goal"}
            leak = []
            for t in range(t_enter):
                st = full_traj["state_seq"][t]
                if st["front_cell"]["type"] in targets:
                    continue
                vt = set(st.get("visible_types_gt", []))
                if ("door" in vt) or ("goal" in vt):
                    leak.append(t)

            masked = []
            for t in leak:
                if _has_targets(full_traj["state_seq"][t].get("state_encoding"), targets):
                    masked.append(t)
                if len(masked) >= args.nocue_max_visible:
                    break

            if not masked:
                raise ValueError("No effective cues (masked empty)")

            (work_dir / "nocue").mkdir(parents=True, exist_ok=True)
            nocue_states = full_traj["state_seq"].copy()  # deep copy needed
            per_frame_ratios = []
            align_ratios = []
            masked_tiles_total = 0

            for i, fp_rel in enumerate(full_traj["frames"]):
                src = work_dir / fp_rel
                dst = work_dir / "nocue" / Path(fp_rel).name

                img = np.array(Image.open(src))
                st = nocue_states[i]
                if i in masked:
                    cnt, tgt = mask_targets_inplace(img, st.get("state_encoding"), targets)
                    masked_tiles_total += int(cnt)
                    per_frame_ratios.append(cnt / float(gh * gw))
                    align_ratios.append((cnt / tgt) if tgt > 0 else 0.0)

                    st["mask_applied"] = True
                    st["mask_type"] = "tile_suppression"
                    st["visible_types"] = sorted(list(set(st.get("visible_types_gt", [])) - targets))
                    save_png(img, dst)
                else:
                    shutil.copy2(src, dst)
                    st["mask_applied"] = False
                    st["mask_type"] = None
                    st["visible_types"] = st.get("visible_types_gt", [])

            align_score = float(np.mean(align_ratios)) if align_ratios else 0.0
            max_mask_ratio = float(np.max(per_frame_ratios)) if per_frame_ratios else 0.0
            mask_budget = int(np.floor(float(args.mask_ratio_max) * float(gh * gw) * float(len(masked))))

            if align_score < args.alignment_min:
                raise ValueError(f"Align fail: {align_score:.2f}")
            if max_mask_ratio > args.mask_ratio_max:
                raise ValueError(f"Mask ratio fail: {max_mask_ratio:.2f}")

            nocue_meta = {
                "align_score": align_score,
                "max_mask_ratio": max_mask_ratio,
                "mask_budget": mask_budget,
                "masked_count": len(masked),
                "masked_tiles_total": masked_tiles_total,
                "leak_count": len(leak),
                "targets": sorted(list(targets)),
            }

            nocue_traj_struct = {
                "frames": [f"{gid}/nocue/{Path(p).name}" for p in full_traj["frames"]],
                "state_seq": nocue_states,
            }

            # ---- CF ----
            cf_traj, cf_meta = None, None
            env_cf = gym.make(args.env_id, render_mode="rgb_array")
            try:
                env_cf.reset(seed=seed)
                meta = remove_goal_cf(env_cf.unwrapped)
                if not meta:
                    raise ValueError("CF: goal not found")
                obs_cf0 = env_cf.unwrapped.gen_obs()  # Re-fetch obs!
                traj = rollout_from_current(env_cf, obs_cf0, actions, args.tile_size, work_dir, "cf", gh, gw)
                if traj is None:
                    raise ValueError("CF render fail")

                goal_visible = any("goal" in st.get("visible_types_gt", []) for st in traj["state_seq"])
                if (not traj["success"]) and (traj["reward"] == 0.0) and (not traj["truncated"]) and (not goal_visible):
                    cf_traj, cf_meta = traj, meta
                else:
                    raise ValueError("CF gate not met (leaked goal or success)")
            finally:
                env_cf.close()

            # Physics check
            for t in range(len(full_traj["state_seq"])):
                if cf_traj["state_seq"][t]["agent"] != full_traj["state_seq"][t]["agent"]:
                    raise ValueError(f"CF physics mismatch t={t}")

            # ---- NOCUE ----
            targets = {"door", "goal"}
            leak = []
            for t in range(t_enter):
                st = full_traj["state_seq"][t]
                if st["front_cell"]["type"] in targets:
                    continue
                vt = set(st.get("visible_types_gt", []))
                if ("door" in vt) or ("goal" in vt):
                    leak.append(t)

            masked = []
            for t in leak:
                if _has_targets(full_traj["state_seq"][t].get("state_encoding"), targets):
                    masked.append(t)
                if len(masked) >= args.nocue_max_visible:
                    break

            if not masked:
                raise ValueError("No effective cues (masked empty)")

            (work_dir / "nocue").mkdir(parents=True, exist_ok=True)
            nocue_states = full_traj["state_seq"].copy()  # deep copy needed
            per_frame_ratios = []
            align_ratios = []
            masked_tiles_total = 0

            for i, fp_rel in enumerate(full_traj["frames"]):
                src = work_dir / fp_rel
                dst = work_dir / "nocue" / Path(fp_rel).name

                img = np.array(Image.open(src))
                st = nocue_states[i]
                if i in masked:
                    cnt, tgt = mask_targets_inplace(img, st.get("state_encoding"), targets)
                    masked_tiles_total += int(cnt)
                    per_frame_ratios.append(cnt / float(gh * gw))
                    align_ratios.append((cnt / tgt) if tgt > 0 else 0.0)

                    st["mask_applied"] = True
                    st["mask_type"] = "tile_suppression"
                    st["visible_types"] = sorted(list(set(st.get("visible_types_gt", [])) - targets))
                    save_png(img, dst)
                else:
                    shutil.copy2(src, dst)
                    st["mask_applied"] = False
                    st["mask_type"] = None
                    st["visible_types"] = st.get("visible_types_gt", [])

            align_score = float(np.mean(align_ratios)) if align_ratios else 0.0
            max_mask_ratio = float(np.max(per_frame_ratios)) if per_frame_ratios else 0.0
            mask_budget = int(np.floor(float(args.mask_ratio_max) * float(gh * gw) * float(len(masked))))

            if align_score < args.alignment_min:
                raise ValueError(f"Align fail: {align_score:.2f}")
            if max_mask_ratio > args.mask_ratio_max:
                raise ValueError(f"Ratio fail: {max_mask_ratio:.2f}")
            if masked_tiles_total > mask_budget:
                raise ValueError(f"Budget fail: {masked_tiles_total}>{mask_budget}")

            # NoCue_spec-aligned meta (extra keys allowed)
            mask_strength_target = float(args.mask_strength_target)
            nocue_meta = {
                "targets": sorted(list(targets)),
                "window_policy": "EARLY",
                "window_steps": masked,
                "event_end_action_idx": int(t_enter),
                "mask_strength_target": mask_strength_target,
                "mask_strength_actual": float(np.mean(per_frame_ratios)) if per_frame_ratios else 0.0,
                "mask_strength_threshold": float(args.mask_ratio_max),
                "mask_type": "tile_suppression",
                "alignment_score": align_score,
                "alignment_threshold": float(args.alignment_min),
                "masked_frames": int(len(masked)),
                "physics_check_passed": True,
                # extra bookkeeping
                "max_mask_ratio": max_mask_ratio,
                "avg_mask_ratio": float(np.mean(per_frame_ratios)) if per_frame_ratios else 0.0,
                "masked_tiles_total": int(masked_tiles_total),
                "mask_budget_tiles": mask_budget,
                "leak_indices": leak,
            }

            # ---- Finalize filesystem ----
            if final_root.exists():
                shutil.rmtree(final_root)
            shutil.move(str(work_dir), str(final_root))

            # ---- Write records ----
            base_rec = {
                "group_id": gid,
                "task": "multiroom",
                "env_id": args.env_id,
                "seed": int(seed),
                "actions_id": [int(a) for a in actions],
                "actions_text": [ACTION_ID2NAME[int(a)] for a in actions],
                "mission": str(mission),
                "model_input_fields": ALLOWED_INPUT_FIELDS,
            }

            nocue_traj_struct = {
                "frames": full_traj["frames"],
                "state_seq": nocue_states,
                "success": full_traj["success"],
                "reward": full_traj["reward"],
                "terminated": full_traj["terminated"],
                "truncated": full_traj["truncated"],
            }

            def mk_rec(variant: str, traj: Dict, meta_c=None, meta_n=None) -> Dict:
                return {
                    **base_rec,
                    "variant": variant,
                    "frames": [f"{gid}/{variant}/{Path(p).name}" for p in traj["frames"]],
                    "state_seq": traj["state_seq"],
                    "success": bool(traj["success"]),
                    "reward": float(traj["reward"]),
                    "terminated": bool(traj["terminated"]),
                    "truncated": bool(traj["truncated"]),
                    "cf_meta": meta_c,
                    "nocue_meta": meta_n,
                }

            records = [
                mk_rec("full", full_traj, None, None),
                mk_rec("nocue", nocue_traj_struct, None, nocue_meta) if nocue_traj_struct else None,
                mk_rec("cf", cf_traj, cf_meta, None) if cf_traj else None,
            ]
            records = [r for r in records if r is not None]
            append_jsonl_batch(jsonl_path, records)

            print(f"[{made+1}/{args.num}] Generated {gid}")
            made += 1

        except ValueError:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        except Exception as e:
            if work_dir.exists():
                shutil.rmtree(work_dir)
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            tb = traceback.format_exc()
            reason = f"gid={gid} env_id={args.env_id} seed={seed}\n{tb}"
            append_skip_log(skip_log_path, seed, reason)

        seed += 1

if __name__ == "__main__":
    main()
