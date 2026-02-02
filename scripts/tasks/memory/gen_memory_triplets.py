#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Triplet Generator for GridWM-Judge.

Generates Full/NoCue/CF triplets for Memory task with strict compliance to:
- GridWM-Judge Project Overview Scientific Contract (Final SSOT)
- NoCue_spec.md
- TRIPLET_AUDIT_SPEC_REVIEWER.md

Usage:
    python gen_memory_triplets.py --out-dir data_memory_vFinal --num 100 --env-id MiniGrid-MemoryS13-v0 --resume
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image

from minigrid.core.constants import (
    OBJECT_TO_IDX,
    IDX_TO_OBJECT,
    COLOR_TO_IDX,
)

# ---------------------------
# MiniGrid action names
# ---------------------------
ACTION_NAMES = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

DIR2VEC = {
    0: (1, 0),  # right
    1: (0, 1),  # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}

# ---------------------------
# Contract fields
# ---------------------------
MODEL_INPUT_FIELDS = ["frames", "mission", "terminated", "truncated"]

# ---------------------------
# NoCue defaults
# ---------------------------
DEFAULT_MASK_STRENGTH_TARGET = 0.02
DEFAULT_MASK_RATIO_MAX = 0.35
DEFAULT_ALIGNMENT_MIN = 0.70
DEFAULT_MIN_VISIBLE_FRAMES = 1
DEFAULT_MAX_VISIBLE_FRAMES = 10

MASK_RGB = (0, 0, 0)
MEMORY_CUE_TYPES = ("key", "ball", "box")

# ---------------------------
# IO utils
# ---------------------------
def append_jsonl_batch(path: Path, objs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for obj in objs:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _mkdir_clean(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _atomic_replace_dir(tmp_dir: Path, final_dir: Path) -> None:
    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.replace(final_dir)


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _load_png(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))

# ---------------------------
# Rendering & state extraction
# ---------------------------
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


def extract_state(env_unwrapped) -> Dict[str, Any]:
    """
    Oracle state. Semantic encoding must be env_unwrapped.gen_obs()['image'].
    """
    enc = np.asarray(env_unwrapped.gen_obs()["image"], dtype=np.uint8)  # (7,7,3)

    visible_types_gt: Set[str] = set()
    for oid in np.unique(enc[:, :, 0]).tolist():
        name = IDX_TO_OBJECT.get(int(oid), "unknown")
        if name in ("unseen", "empty", "floor", "wall"):
            continue
        visible_types_gt.add(name)

    # front cell
    fx, fy = env_unwrapped.front_pos
    f_cell = env_unwrapped.grid.get(fx, fy)
    f_type = getattr(f_cell, "type", "floor") if f_cell else "floor"
    front_info: Dict[str, Any] = {"pos": [int(fx), int(fy)], "type": str(f_type)}
    if f_cell and hasattr(f_cell, "color"):
        front_info["color"] = str(getattr(f_cell, "color"))

    # oracle objects
    objects: List[Dict[str, Any]] = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            cell = env_unwrapped.grid.get(x, y)
            if cell is None:
                continue
            t = getattr(cell, "type", None)
            if t in ("wall", "floor"):
                continue
            o: Dict[str, Any] = {"type": str(t), "pos": [int(x), int(y)]}
            if hasattr(cell, "color"):
                o["color"] = str(getattr(cell, "color"))
            objects.append(o)

    return {
        "agent": {
            "pos": [int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1])],
            "dir": int(env_unwrapped.agent_dir),
            "carrying": {
                "type": str(env_unwrapped.carrying.type),
                "color": str(env_unwrapped.carrying.color),
            } if env_unwrapped.carrying else None,
        },
        "front_cell": front_info,
        "visible_types_gt": sorted(list(visible_types_gt)),
        "visible_types": sorted(list(visible_types_gt)),
        "state_encoding": enc.astype(int).tolist(),
        "objects": objects,
        "mask_applied": False,
        "mask_type": None,
    }

# ---------------------------
# Planning (BFS)
# ---------------------------
@dataclass(frozen=True)
class MemState:
    x: int
    y: int
    d: int

def _is_passable(env_unwrapped, x: int, y: int) -> bool:
    if not (0 <= x < env_unwrapped.width and 0 <= y < env_unwrapped.height):
        return False
    cell = env_unwrapped.grid.get(x, y)
    if cell is None:
        return True
    t = getattr(cell, "type", None)
    if t in ("wall", "lava"):
        return False
    if getattr(cell, "can_overlap", False):
        return True
    if t in ("key", "ball", "box", "goal"):
        return True
    return False

def bfs_to_pos(env_unwrapped, goal_pos: Tuple[int, int], max_nodes: int = 300000) -> Optional[List[int]]:
    """
    Shortest path to stand on goal_pos (any dir) using {left,right,forward}.
    """
    start = MemState(int(env_unwrapped.agent_pos[0]), int(env_unwrapped.agent_pos[1]), int(env_unwrapped.agent_dir))
    q = deque([start])
    parent: Dict[MemState, Tuple[Optional[MemState], Optional[int]]] = {start: (None, None)}
    visited = {start}

    goal_set = {(goal_pos[0], goal_pos[1], d) for d in range(4)}
    steps = 0

    while q:
        cur = q.popleft()
        steps += 1
        if steps > max_nodes:
            return None

        if (cur.x, cur.y, cur.d) in goal_set:
            acts: List[int] = []
            node = cur
            while parent[node][0] is not None:
                prev, act = parent[node]
                assert act is not None
                acts.append(int(act))
                node = prev  # type: ignore[assignment]
            acts.reverse()
            return acts

        # left/right
        for act in (0, 1):
            nd = (cur.d - 1) % 4 if act == 0 else (cur.d + 1) % 4
            nxt = MemState(cur.x, cur.y, nd)
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = (cur, act)
                q.append(nxt)

        # forward
        dx, dy = DIR2VEC[cur.d]
        nx, ny = cur.x + dx, cur.y + dy
        if _is_passable(env_unwrapped, nx, ny):
            nxt = MemState(nx, ny, cur.d)
            if nxt not in visited:
                visited.add(nxt)
                parent[nxt] = (cur, 2)
                q.append(nxt)

    return None

# ---------------------------
# Memory object inference
# ---------------------------
def infer_objects(env_unwrapped) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            cell = env_unwrapped.grid.get(x, y)
            if cell is None:
                continue
            t = getattr(cell, "type", None)
            if t not in MEMORY_CUE_TYPES:
                continue
            objs.append({"type": str(t), "color": str(getattr(cell, "color", "grey")), "pos": (int(x), int(y))})
    return objs

def pick_cue_and_ends(objs: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if len(objs) < 3:
        raise ValueError(f"Memory expects >=3 objects, got {len(objs)}")

    xs = [o["pos"][0] for o in objs]
    min_x = min(xs)
    max_x = max(xs)

    cue_candidates = [o for o in objs if o["pos"][0] == min_x]
    cue = sorted(cue_candidates, key=lambda o: (o["pos"][1], o["type"], o["color"]))[0]

    end_candidates = [o for o in objs if o["pos"][0] == max_x]
    if len(end_candidates) < 2:
        end_candidates = sorted(objs, key=lambda o: (o["pos"][0], o["pos"][1]))[-2:]

    end_sorted = sorted(end_candidates, key=lambda o: (o["pos"][1], o["type"], o["color"]))
    return cue, end_sorted[0], end_sorted[1]

def match_good_bad(cue: Dict[str, Any], end_a: Dict[str, Any], end_b: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    sig = (cue["type"], cue["color"])
    a_sig = (end_a["type"], end_a["color"])
    b_sig = (end_b["type"], end_b["color"])

    if a_sig == sig and b_sig != sig:
        return end_a, end_b
    if b_sig == sig and a_sig != sig:
        return end_b, end_a

    # relax: type-only
    if end_a["type"] == cue["type"] and end_b["type"] != cue["type"]:
        return end_a, end_b
    if end_b["type"] == cue["type"] and end_a["type"] != cue["type"]:
        return end_b, end_a

    raise ValueError(f"Ambiguous cue/end match: cue={cue}, end_a={end_a}, end_b={end_b}")


def infer_corridor_start_x(env_unwrapped) -> int:
    counts: List[Tuple[int, int]] = []
    for x in range(env_unwrapped.width):
        passable = 0
        for y in range(env_unwrapped.height):
            if _is_passable(env_unwrapped, x, y):
                passable += 1
        counts.append((x, passable))

    min_pass = min(c for x, c in counts if x > 0)
    cands = [x for x, c in counts if x > 0 and c == min_pass]
    return min(cands) if cands else 0

# ---------------------------
# NoCue: tile mask by (type,color)
# ---------------------------
def cue_tile_mask(enc: np.ndarray, cue_type: str, cue_color: str) -> np.ndarray:
    obj = OBJECT_TO_IDX.get(cue_type, None)
    col = COLOR_TO_IDX.get(cue_color, None)
    if obj is None or col is None:
        return np.zeros(enc.shape[:2], dtype=bool)
    return (enc[:, :, 0] == int(obj)) & (enc[:, :, 1] == int(col))

def apply_tile_mask(img: np.ndarray, tile_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    out = img.copy()
    gh, gw = tile_mask.shape
    H, W = out.shape[:2]
    if H % gh != 0 or W % gw != 0:
        raise ValueError(f"Frame not divisible by tile grid: frame={out.shape}, tile={tile_mask.shape}")
    th, tw = H // gh, W // gw

    ys, xs = np.where(tile_mask)
    for r, c in zip(ys.tolist(), xs.tolist()):
        out[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = MASK_RGB
    return out, int(tile_mask.sum())

def compute_window_steps(
    full_state_seq: Sequence[Dict[str, Any]],
    cue_type: str,
    cue_color: str,
    corridor_start_x: int,
    max_visible_frames: int,
) -> Tuple[List[int], int, List[int]]:
    """
    EARLY window: frames where cue is visible while agent is still before corridor.
    """
    visible: List[int] = []
    for t, st in enumerate(full_state_seq):
        ax = int(st["agent"]["pos"][0])
        if ax >= corridor_start_x:
            continue
        enc = np.asarray(st["state_encoding"], dtype=np.uint8)
        if cue_tile_mask(enc, cue_type, cue_color).any():
            visible.append(t)

    if len(visible) == 0 or len(visible) > max_visible_frames:
        return [], -1, []

    window_end = max(visible)
    leak: List[int] = []
    for t, st in enumerate(full_state_seq):
        if t <= window_end:
            continue
        enc = np.asarray(st["state_encoding"], dtype=np.uint8)
        if cue_tile_mask(enc, cue_type, cue_color).any():
            leak.append(t)

    return sorted(visible), window_end, leak

# ---------------------------
# CF: swap end objects (Type-2)
# ---------------------------
def apply_cf_swap_success_failure(env_unwrapped) -> Dict[str, Any]:
    # Swap success and failure positions
    old_success = env_unwrapped.success_pos
    old_failure = env_unwrapped.failure_pos

    env_unwrapped.success_pos = old_failure
    env_unwrapped.failure_pos = old_success

    return {
        "type": "swap_success_failure",
        "old_success_pos": [int(old_success[0]), int(old_success[1])],
        "old_failure_pos": [int(old_failure[0]), int(old_failure[1])],
        "new_success_pos": [int(old_failure[0]), int(old_failure[1])],
        "new_failure_pos": [int(old_success[0]), int(old_success[1])],
    }

# ---------------------------
# Rollout
# ---------------------------
def rollout_and_save(env, actions: Sequence[int], tile_size: int, work_dir: Path, variant: str) -> Dict[str, Any]:
    vdir = work_dir / variant
    vdir.mkdir(parents=True, exist_ok=True)

    frames: List[str] = []
    state_seq: List[Dict[str, Any]] = []
    actions_executed: List[int] = []

    total_reward = 0.0
    terminated = False
    truncated = False

    def snap(t: int) -> None:
        frame = render_agent_pov(env, tile_size=tile_size)
        if frame is None:
            raise RuntimeError("render_agent_pov failed")
        fname = f"step_{t:03d}.png"
        _save_png(frame, vdir / fname)
        frames.append(f"{variant}/{fname}")
        state_seq.append(extract_state(env.unwrapped))

    snap(0)
    for i, a in enumerate(actions):
        _obs, r, terminated, truncated, _info = env.step(int(a))
        actions_executed.append(int(a))
        total_reward += float(r)
        snap(i + 1)
        if terminated or truncated:
            break

    success = bool(terminated and total_reward > 0.0 and not truncated)

    return {
        "frames": frames,
        "state_seq": state_seq,
        "actions_executed": actions_executed,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "reward": float(total_reward),
        "success": bool(success),
    }

def generate_one(
    env_id: str,
    seed: int,
    out_dir: Path,
    tile_size: int,
    max_plan_steps: int,
    max_episode_steps: int,
    # NoCue knobs
    mask_strength_target: float,
    mask_ratio_max: float,
    alignment_min: float,
    min_visible_frames: int,
    max_visible_frames: int,
    skip_cf_nocue: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    gid = f"memory_s{seed:06d}"
    tmp_root = out_dir / "_tmp" / gid
    final_root = out_dir / gid

    _mkdir_clean(tmp_root)

    env = None
    env_cf = None
    try:
        # -------- FULL --------
        env = gym.make(env_id, render_mode="rgb_array")
        if hasattr(env.unwrapped, "tile_size"):
            env.unwrapped.tile_size = int(tile_size)

        env.reset(seed=seed)

        objs = infer_objects(env.unwrapped)
        cue, end_a, end_b = pick_cue_and_ends(objs)
        good_end, _bad_end = match_good_bad(cue, end_a, end_b)

        plan = bfs_to_pos(env.unwrapped, tuple(good_end["pos"]))
        if plan is None or len(plan) > max_plan_steps:
            return None

        full = rollout_and_save(env, plan, tile_size, tmp_root, "full")

        if not full["success"]:
            return None
        if full["truncated"]:
            return None
        if len(full["actions_executed"]) == 0 or len(full["actions_executed"]) > max_episode_steps:
            return None

        T = len(full["actions_executed"])
        if len(full["frames"]) != T + 1 or len(full["state_seq"]) != T + 1:
            return None

        # -------- Early return for testing --------
        if skip_cf_nocue:
            # -------- finalize --------
            mission = getattr(env.unwrapped, "mission", "")
            _atomic_replace_dir(tmp_root, final_root)

            def mk_record(variant: str, traj: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
                frames_rel = [f"{gid}/{variant}/{Path(p).name}" for p in traj["frames"]]
                acts = traj["actions_executed"]
                rec: Dict[str, Any] = {
                    "task": "memory",
                    "env_id": env_id,
                    "seed": int(seed),
                    "group_id": gid,
                    "variant": variant,
                    "mission": str(mission),
                    "model_input_fields": list(MODEL_INPUT_FIELDS),
                    "actions_id": [int(a) for a in acts],
                    "actions_text": [ACTION_NAMES.get(int(a), str(int(a))) for a in acts],
                    "frames": frames_rel,
                    "state_seq": traj["state_seq"],
                    "terminated": bool(traj["terminated"]),
                    "truncated": bool(traj["truncated"]),
                    "reward": float(traj["reward"]),
                    "success": bool(traj["success"]),
                }
                rec.update(extra)
                rec.update({
                    "generator": {"name": "gen_memory_triplets.py", "created_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                })
                return rec

            rec_full = mk_record("full", full, {})
            return [rec_full]

        # -------- NoCue --------
        corridor_x = infer_corridor_start_x(env.unwrapped)
        window_steps, window_end, leak_steps = compute_window_steps(
            full["state_seq"], cue["type"], cue["color"], corridor_x, max_visible_frames
        )
        if len(window_steps) < min_visible_frames:
            return None
        if window_end < 0 or len(leak_steps) != 0:
            return None

        gh, gw = 7, 7
        masked_frames = len(window_steps)
        mask_budget_tiles = int(np.floor(mask_ratio_max * float(gh * gw) * float(masked_frames)))
        mask_budget_tiles = max(1, mask_budget_tiles)

        nocue_state_seq: List[Dict[str, Any]] = json.loads(json.dumps(full["state_seq"]))

        masked_tiles_total = 0
        target_tiles_total = 0

        window_set = set(window_steps)

        for t in range(T + 1):
            src = tmp_root / full["frames"][t]
            dst = tmp_root / "nocue" / Path(full["frames"][t]).name
            dst.parent.mkdir(parents=True, exist_ok=True)

            if t not in window_set:
                shutil.copy2(src, dst)
                nocue_state_seq[t]["visible_types"] = list(nocue_state_seq[t].get("visible_types_gt", []))
                nocue_state_seq[t]["mask_applied"] = False
                nocue_state_seq[t]["mask_type"] = None
                continue

            img = _load_png(src)
            enc = np.asarray(nocue_state_seq[t]["state_encoding"], dtype=np.uint8)
            tm = cue_tile_mask(enc, cue["type"], cue["color"])
            target_tiles = int(tm.sum())
            if target_tiles <= 0:
                return None

            masked_img, masked_tiles = apply_tile_mask(img, tm)
            _save_png(masked_img, dst)

            masked_tiles_total += masked_tiles
            target_tiles_total += target_tiles

            nocue_state_seq[t]["visible_types"] = []
            nocue_state_seq[t]["mask_applied"] = True
            nocue_state_seq[t]["mask_type"] = "tile_suppression"

        alignment_score = float(masked_tiles_total / float(max(1, target_tiles_total)))
        if alignment_score < alignment_min:
            return None
        if masked_tiles_total > mask_budget_tiles:
            return None

        mask_strength_actual = float(masked_tiles_total / float(gh * gw * max(1, masked_frames)))

        nocue = {
            "frames": [f"nocue/{Path(p).name}" for p in full["frames"]],
            "state_seq": nocue_state_seq,
            "actions_executed": list(full["actions_executed"]),
            "terminated": bool(full["terminated"]),
            "truncated": bool(full["truncated"]),
            "reward": float(full["reward"]),
            "success": bool(full["success"]),
        }

        nocue_meta = {
            "targets": [str(cue["type"])],
            "window_policy": "EARLY",
            "window_steps": [int(x) for x in window_steps],
            "mask_strength_target": float(mask_strength_target),
            "mask_strength_actual": float(mask_strength_actual),
            "alignment_score": float(alignment_score),
            "alignment_threshold": float(alignment_min),
            "masked_frames": int(masked_frames),
            "mask_type": "tile_suppression",
            "physics_check_passed": True,
            # debug extras are allowed
            "corridor_start_x": int(corridor_x),
            "window_end": int(window_end),
            "leak_steps": [int(x) for x in leak_steps],
            "mask_ratio_max": float(mask_ratio_max),
            "mask_budget_tiles": int(mask_budget_tiles),
            "masked_tiles_total": int(masked_tiles_total),
            "cue": cue,
        }

        # -------- CF --------
        env_cf = gym.make(env_id, render_mode="rgb_array")
        if hasattr(env_cf.unwrapped, "tile_size"):
            env_cf.unwrapped.tile_size = int(tile_size)
        env_cf.reset(seed=seed)

        objs_cf = infer_objects(env_cf.unwrapped)
        _cue_cf, end_a_cf, end_b_cf = pick_cue_and_ends(objs_cf)
        _good_end_cf, _bad_end_cf = match_good_bad(_cue_cf, end_a_cf, end_b_cf)

        cf_meta = apply_cf_swap_success_failure(env_cf.unwrapped)
        cf = rollout_and_save(env_cf, full["actions_executed"], tile_size, tmp_root, "cf")
        if cf["success"]:
            return None
        if cf["truncated"]:
            return None
        if float(cf["reward"]) != 0.0:
            return None

        if len(cf["state_seq"]) != len(full["state_seq"]):
            return None
        for t in range(T + 1):
            if cf["state_seq"][t]["agent"]["pos"] != full["state_seq"][t]["agent"]["pos"]:
                return None
            if cf["state_seq"][t]["agent"]["dir"] != full["state_seq"][t]["agent"]["dir"]:
                return None

        mission = getattr(env.unwrapped, "mission", "")

        def mk_record(variant: str, traj: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
            frames_rel = [f"{gid}/{variant}/{Path(p).name}" for p in traj["frames"]]
            acts = traj["actions_executed"]
            rec: Dict[str, Any] = {
                "task": "memory",
                "env_id": env_id,
                "seed": int(seed),
                "group_id": gid,
                "variant": variant,
                "mission": str(mission),
                "model_input_fields": list(MODEL_INPUT_FIELDS),
                "actions_id": [int(a) for a in acts],
                "actions_text": [ACTION_NAMES.get(int(a), str(int(a))) for a in acts],
                "frames": frames_rel,
                "state_seq": traj["state_seq"],
                "terminated": bool(traj["terminated"]),
                "truncated": bool(traj["truncated"]),
                "reward": float(traj["reward"]),
                "success": bool(traj["success"]),
                "generator": {"name": "gen_memory_triplets.py", "created_at": time.strftime("%Y-%m-%d %H:%M:%S")},
            }
            rec.update(extra)
            return rec

        rec_full = mk_record("full", full, {})
        rec_nocue = mk_record("nocue", nocue, {"nocue_meta": nocue_meta})
        rec_cf = mk_record("cf", cf, {"cf_meta": cf_meta})

        # invariants
        if rec_full["actions_id"] != rec_nocue["actions_id"] or rec_full["actions_id"] != rec_cf["actions_id"]:
            return None
        if rec_nocue["success"] != rec_full["success"] or float(rec_nocue["reward"]) != float(rec_full["reward"]):
            return None

        # -------- finalize --------
        _atomic_replace_dir(tmp_root, final_root)

        return [rec_full, rec_nocue, rec_cf]

    finally:
        try:
            if env is not None:
                env.close()
        except Exception:
            pass
        try:
            if env_cf is not None:
                env_cf.close()
        except Exception:
            pass
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", type=str, default="MiniGrid-MemoryS17Random-v0")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tile-size", type=int, default=32)
    ap.add_argument("--max-plan-steps", type=int, default=512)
    ap.add_argument("--max-episode-steps", type=int, default=2048)
    ap.add_argument("--attempts-per-group", type=int, default=200)

    # NoCue knobs
    ap.add_argument("--mask-strength-target", type=float, default=DEFAULT_MASK_STRENGTH_TARGET)
    ap.add_argument("--mask-ratio-max", type=float, default=DEFAULT_MASK_RATIO_MAX)
    ap.add_argument("--alignment-min", type=float, default=DEFAULT_ALIGNMENT_MIN)
    ap.add_argument("--min-visible-frames", type=int, default=DEFAULT_MIN_VISIBLE_FRAMES)
    ap.add_argument("--max-visible-frames", type=int, default=DEFAULT_MAX_VISIBLE_FRAMES)

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--skip-cf-nocue", action="store_true", help="Skip CF and NoCue generation for testing")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "triplets.jsonl"

    made = 0
    seed_base = int(args.seed_start)

    while made < int(args.n):
        recs = None
        used_seed: Optional[int] = None

        for k in range(int(args.attempts_per_group)):
            s_try = seed_base + k
            gid_try = f"memory_s{s_try:06d}"
            if args.resume and (out_dir / gid_try).exists():
                continue

            recs = generate_one(
                env_id=args.env_id,
                seed=s_try,
                out_dir=out_dir,
                tile_size=int(args.tile_size),
                max_plan_steps=int(args.max_plan_steps),
                max_episode_steps=int(args.max_episode_steps),
                mask_strength_target=float(args.mask_strength_target),
                mask_ratio_max=float(args.mask_ratio_max),
                alignment_min=float(args.alignment_min),
                min_visible_frames=int(args.min_visible_frames),
                max_visible_frames=int(args.max_visible_frames),
                skip_cf_nocue=args.skip_cf_nocue,
            )
            if recs is not None:
                used_seed = s_try
                break

        if recs is None or used_seed is None:
            print(f"[SKIP] seed_base={seed_base} failed after retries")
            seed_base += 1
            continue

        # Write 3 lines: full -> nocue -> cf
        append_jsonl_batch(jsonl_path, recs)
        print(f"[OK] group=memory_s{used_seed:06d} wrote 3 lines")
        made += 1
        seed_base += 1

if __name__ == "__main__":
    main()
