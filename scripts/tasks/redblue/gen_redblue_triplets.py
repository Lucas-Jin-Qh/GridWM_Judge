#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedBlueDoor Triplet Generator for GridWM-Judge.

Generates Full/NoCue/CF triplets for RedBlueDoor task with strict compliance to:
- GridWM-Judge Project Overview Scientific Contract (Final SSOT)
- NoCue_spec.md
- TRIPLET_AUDIT_SPEC_REVIEWER.md

Usage:
    python gen_redblue_triplets.py --out-dir data_redblue_vFinal --num 100 --env-id MiniGrid-RedBlueDoors-8x8-v0 --resume

SSOT Compliance:
- NoCue Spec v1.0 (EARLY window, remove evidence of red door early, preserve interaction evidence)
- Triplet Audit v2.0 (action identity, physics invariance, strict gates)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import gymnasium as gym

# Bug D Fix: Use official constants
from minigrid.core.constants import (
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
)
from minigrid.core.world_object import Wall
from minigrid.wrappers import RGBImgPartialObsWrapper


# -----------------------------
# CLI / Logging / IO helpers
# -----------------------------

ACTION_ID2NAME = {
    0: "left",
    1: "right",
    2: "forward",
    3: "pickup",
    4: "drop",
    5: "toggle",
    6: "done",
}

DIR2VEC = {
    0: (1, 0),   # right
    1: (0, 1),   # down
    2: (-1, 0),  # left
    3: (0, -1),  # up
}

VEC2DIR = {v: k for k, v in DIR2VEC.items()}


def setup_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def append_jsonl_batch(path: Path, records: List[Dict[str, Any]]) -> None:
    """
    Append multiple json lines and fsync for crash-safety.
    """
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# -----------------------------
# Rendering & State Extraction
# -----------------------------

def render_agent_pov(env: gym.Env, tile_size: int, grid_h: int, grid_w: int) -> np.ndarray:
    """
    Render agent POV RGB frame aligned to the 7x7 observation grid.
    """
    # Assuming env is wrapped with RGBImgPartialObsWrapper
    # We use get_frame from the unwrapped env or the wrapper if accessible
    # Standard MiniGrid: env.get_frame(...) returns the full env render or POV.
    # RGBImgPartialObsWrapper step() returns the POV image directly in obs['image'].
    # To get it manually without stepping:
    frame = env.unwrapped.get_frame(tile_size=tile_size, agent_pov=True)

    if not isinstance(frame, np.ndarray):
        frame = np.array(frame, dtype=np.uint8)
    expected_h = grid_h * tile_size
    expected_w = grid_w * tile_size
    # Basic shape check
    assert frame.shape[0] == expected_h and frame.shape[1] == expected_w, (
        f"render shape mismatch: got {frame.shape[:2]} expected {(expected_h, expected_w)}"
    )
    return frame.astype(np.uint8)


def _door_state(obj: Any) -> int:
    # 0=open, 1=closed, 2=locked; unknown -> -1
    if obj is None or getattr(obj, "type", None) != "door":
        return -1
    st = getattr(obj, "state", None)
    if st is None:
        if getattr(obj, "is_open", False):
            return 0
        return 1
    return int(st)


def extract_state(env_unwrapped: Any) -> Dict[str, Any]:
    """
    Bug A fix: always use env_unwrapped.gen_obs()['image'] as semantic encoding.
    """
    obs = env_unwrapped.gen_obs()
    enc = obs["image"]  # (7,7,3) semantic grid

    ax, ay = env_unwrapped.agent_pos
    ad = int(env_unwrapped.agent_dir)

    fx, fy = env_unwrapped.front_pos
    front_obj = None
    if 0 <= fx < env_unwrapped.width and 0 <= fy < env_unwrapped.height:
        front_obj = env_unwrapped.grid.get(fx, fy)

    front_cell = {
        "pos": [int(fx), int(fy)],
        "type": getattr(front_obj, "type", "wall" if front_obj is not None else "floor"),
        "color": getattr(front_obj, "color", None),
        "state": _door_state(front_obj),
    }

    # list non-trivial objects
    objects = []
    grid = env_unwrapped.grid
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            o = grid.get(x, y)
            if o is None:
                continue
            ot = getattr(o, "type", None)
            if ot in ("wall", "floor"):
                continue
            obj_rec = {"type": ot, "pos": [int(x), int(y)]}
            if hasattr(o, "color"):
                obj_rec["color"] = getattr(o, "color", None)
            if ot == "door":
                obj_rec["state"] = _door_state(o)
            objects.append(obj_rec)

    # visible types (GT) from semantic encoding
    visible_types_gt = []
    seen = set()
    for i in range(enc.shape[0]):
        for j in range(enc.shape[1]):
            oid = int(enc[i, j, 0])
            oname = IDX_TO_OBJECT.get(oid, "unknown")
            if oname in ("unseen", "empty", "floor", "wall"):
                continue
            if oname == "door":
                cid = int(enc[i, j, 1])
                cname = IDX_TO_COLOR.get(cid, None)
                if cname == "red":
                    tag = "red_door"
                elif cname == "blue":
                    tag = "blue_door"
                else:
                    tag = "door"
                if tag not in seen:
                    seen.add(tag)
                    visible_types_gt.append(tag)
            else:
                if oname not in seen:
                    seen.add(oname)
                    visible_types_gt.append(oname)

    return {
        "step": int(env_unwrapped.step_count),
        "agent_pos": [int(ax), int(ay)],
        "agent_dir": ad,
        "front_cell": front_cell,
        "objects": objects,
        "state_encoding": enc.tolist(),
        "visible_types_gt": visible_types_gt,
        "visible_types": visible_types_gt,  # may be overwritten in NoCue
        "mask_applied": False,
    }


# -----------------------------
# Planning (BFS on interaction states)
# -----------------------------

@dataclass(frozen=True)
class DoorPos:
    color: str
    pos: Tuple[int, int]
    open0: bool


def find_red_blue_doors(env_unwrapped: Any) -> Tuple[DoorPos, DoorPos]:
    red = None
    blue = None
    grid = env_unwrapped.grid
    for x in range(env_unwrapped.width):
        for y in range(env_unwrapped.height):
            o = grid.get(x, y)
            if o is None or getattr(o, "type", None) != "door":
                continue
            c = getattr(o, "color", None)
            st = _door_state(o)
            open0 = (st == 0)
            if c == "red":
                red = DoorPos("red", (x, y), open0)
            elif c == "blue":
                blue = DoorPos("blue", (x, y), open0)
    if red is None or blue is None:
        raise RuntimeError("Red/Blue doors not found in grid.")
    return red, blue


@dataclass(frozen=True)
class RBState:
    ax: int
    ay: int
    ad: int
    red_open: bool
    blue_open: bool


def _front_pos(ax: int, ay: int, ad: int) -> Tuple[int, int]:
    dx, dy = DIR2VEC[ad]
    return ax + dx, ay + dy


def _cell_passable(env_unwrapped: Any, x: int, y: int, red_pos: Tuple[int, int], blue_pos: Tuple[int, int]) -> bool:
    if not (0 <= x < env_unwrapped.width and 0 <= y < env_unwrapped.height):
        return False
    # Bug C Fix: Treat doors as obstacles for movement (we toggle them from adjacent)
    if (x, y) == red_pos or (x, y) == blue_pos:
        return False
    o = env_unwrapped.grid.get(x, y)
    if o is None:
        return True
    if getattr(o, "type", None) == "wall":
        return False
    # conservative: block any other object
    return False


def bfs_plan_red_then_blue(env_unwrapped: Any, max_nodes: int = 20000) -> List[int]:
    """
    Plan with BFS over (pos,dir,door_states).
    Strategy: Red Open -> Blue Open.
    """
    red, blue = find_red_blue_doors(env_unwrapped)
    red_pos, blue_pos = red.pos, blue.pos

    sx, sy = map(int, env_unwrapped.agent_pos)
    sd = int(env_unwrapped.agent_dir)

    start = RBState(sx, sy, sd, red.open0, blue.open0)

    q = deque([start])
    parent: Dict[RBState, Tuple[Optional[RBState], Optional[int]]] = {start: (None, None)}
    expanded = 0

    def is_goal(s: RBState) -> bool:
        return s.red_open and s.blue_open

    while q:
        s = q.popleft()
        expanded += 1
        if expanded > max_nodes:
            break
        if is_goal(s):
            actions_rev: List[int] = []
            cur = s
            while True:
                prev, act = parent[cur]
                if prev is None:
                    break
                assert act is not None
                actions_rev.append(act)
                cur = prev
            return list(reversed(actions_rev))

        # Actions: left, right, forward
        # left
        ns = RBState(s.ax, s.ay, (s.ad - 1) % 4, s.red_open, s.blue_open)
        if ns not in parent:
            parent[ns] = (s, 0)
            q.append(ns)

        # right
        ns = RBState(s.ax, s.ay, (s.ad + 1) % 4, s.red_open, s.blue_open)
        if ns not in parent:
            parent[ns] = (s, 1)
            q.append(ns)

        # forward
        nx, ny = _front_pos(s.ax, s.ay, s.ad)
        if _cell_passable(env_unwrapped, nx, ny, red_pos, blue_pos):
            ns = RBState(nx, ny, s.ad, s.red_open, s.blue_open)
            if ns not in parent:
                parent[ns] = (s, 2)
                q.append(ns)

        # toggle (Interaction Logic)
        fx, fy = _front_pos(s.ax, s.ay, s.ad)
        # 1. Open Red
        if (fx, fy) == red_pos and (not s.red_open):
            ns = RBState(s.ax, s.ay, s.ad, True, s.blue_open)
            if ns not in parent:
                parent[ns] = (s, 5)
                q.append(ns)
        # 2. Open Blue (only if Red is open)
        elif (fx, fy) == blue_pos and s.red_open and (not s.blue_open):
            ns = RBState(s.ax, s.ay, s.ad, s.red_open, True)
            if ns not in parent:
                parent[ns] = (s, 5)
                q.append(ns)

    raise RuntimeError("BFS failed to find a valid plan.")


# -----------------------------
# Rollout
# -----------------------------

def rollout_from_current(
    env: gym.Env,
    obs0: Dict[str, Any],
    actions: List[int],
    tile_size: int,
    out_dir: Path,
    rel_dir: str,
) -> Dict[str, Any]:
    """
    Execute actions from current env state, saving frames.
    """
    # obs0['image'] from wrapper is (H,W,3) RGB
    grid_h = int(obs0["image"].shape[0]) // tile_size
    grid_w = int(obs0["image"].shape[1]) // tile_size

    frames: List[str] = []
    state_seq: List[Dict[str, Any]] = []

    # step 0
    img0 = render_agent_pov(env, tile_size, grid_h, grid_w)
    fp0 = out_dir / rel_dir / "step_000.png"
    save_png(img0, fp0)
    frames.append(str(Path(rel_dir) / "step_000.png"))
    state_seq.append(extract_state(env.unwrapped))

    total_reward = 0.0
    terminated = False
    truncated = False

    for t, a in enumerate(actions, start=1):
        obs, r, term, trunc, _ = env.step(a)
        total_reward += float(r)
        terminated = bool(term)
        truncated = bool(trunc)

        img = render_agent_pov(env, tile_size, grid_h, grid_w)
        fp = out_dir / rel_dir / f"step_{t:03d}.png"
        save_png(img, fp)
        frames.append(str(Path(rel_dir) / f"step_{t:03d}.png"))
        state_seq.append(extract_state(env.unwrapped))

        if terminated or truncated:
            break

    return {
        "frames": frames,
        "state_seq": state_seq,
        "actions_id": actions[: len(frames) - 1],
        "actions_text": [ACTION_ID2NAME[x] for x in actions[: len(frames) - 1]],
        "terminated": terminated,
        "truncated": truncated,
        "reward": float(total_reward),
        "success": bool(terminated and total_reward > 0.0),
        "mission": obs0.get("mission", ""),
    }


# -----------------------------
# NoCue Logic
# -----------------------------

def is_red_door_visible(state_encoding: np.ndarray) -> bool:
    door_id = OBJECT_TO_IDX["door"]
    red_id = COLOR_TO_IDX["red"]
    return bool(np.any((state_encoding[:, :, 0] == door_id) & (state_encoding[:, :, 1] == red_id)))


def mask_red_door_tiles_inplace(pov_img: np.ndarray, state_encoding: np.ndarray) -> Tuple[int, int]:
    """
    Black-out tiles where semantic encoding indicates red door.
    """
    H, W, _ = pov_img.shape
    gh, gw = state_encoding.shape[0], state_encoding.shape[1]
    tile_h = H // gh
    tile_w = W // gw

    door_id = OBJECT_TO_IDX["door"]
    red_id = COLOR_TO_IDX["red"]

    target_tiles = 0
    masked_tiles = 0

    for i in range(gh):
        for j in range(gw):
            if int(state_encoding[i, j, 0]) == door_id and int(state_encoding[i, j, 1]) == red_id:
                target_tiles += 1
                y1, y2 = i * tile_h, (i + 1) * tile_h
                x1, x2 = j * tile_w, (j + 1) * tile_w
                # Hard blackout
                pov_img[y1:y2, x1:x2] = 0
                masked_tiles += 1

    return masked_tiles, target_tiles


def build_nocue_from_full(
    full_traj: Dict[str, Any],
    work_dir: Path,
    tile_size: int,
    nocue_max_visible: int,
    alignment_min: float,
    mask_ratio_max: float,
    mask_strength_target: float,
) -> Optional[Dict[str, Any]]:
    state_seq = full_traj["state_seq"]
    actions = full_traj["actions_id"]

    # 1. T_interact: First toggle Red
    t_interact = None
    for t, a in enumerate(actions):
        if a != 5: continue
        fc = state_seq[t]["front_cell"]
        if fc.get("type") == "door" and fc.get("color") == "red":
            t_interact = t
            break
    if t_interact is None: return None

    # 2. First Seen
    t_first_seen = None
    for t in range(len(state_seq)):
        enc = np.array(state_seq[t]["state_encoding"], dtype=np.int64)
        if is_red_door_visible(enc):
            t_first_seen = t
            break
    if t_first_seen is None: return None

    # 3. Window: EARLY (t <= first_seen)
    window_end = min(t_first_seen, t_interact - 1)
    if window_end < 0: return None

    # 4. Filter leak (Interaction Evidence)
    leak_indices: List[int] = []
    for t in range(window_end + 1):
        enc = np.array(state_seq[t]["state_encoding"], dtype=np.int64)
        if not is_red_door_visible(enc): continue

        fc = state_seq[t]["front_cell"]
        # Interaction Evidence Preservation: Skip if front_cell is Red Door
        if fc.get("type") == "door" and fc.get("color") == "red": continue

        leak_indices.append(t)

    if not leak_indices: return None

    # 5. Mask
    masked_indices = leak_indices[: max(0, int(nocue_max_visible))]
    if not masked_indices: return None

    (work_dir / "nocue").mkdir(parents=True, exist_ok=True)
    per_frame_mask_ratio = []
    per_frame_alignment = []

    for t in range(len(full_traj["frames"])):
        src = work_dir / full_traj["frames"][t]
        dst = work_dir / "nocue" / Path(full_traj["frames"][t]).name

        img = np.array(Image.open(src), dtype=np.uint8)
        if t in masked_indices:
            enc = np.array(state_seq[t]["state_encoding"], dtype=np.int64)
            masked_tiles, target_tiles = mask_red_door_tiles_inplace(img, enc)

            gh, gw = enc.shape[0], enc.shape[1]
            ratio = float(masked_tiles) / float(max(1, gh * gw))
            align = float(masked_tiles) / float(max(1, target_tiles))

            per_frame_mask_ratio.append(ratio)
            per_frame_alignment.append(align)

            # Metadata update
            state_seq[t]["mask_applied"] = True
            state_seq[t]["visible_types"] = [] # Evidence removed

        save_png(img, dst)

    if not per_frame_alignment: return None

    mask_strength_actual = float(np.mean(per_frame_mask_ratio))
    alignment_score = float(np.mean(per_frame_alignment))

    if alignment_score < alignment_min: return None
    if mask_strength_actual > mask_ratio_max: return None

    nocue_meta = {
        "targets": ["red_door"],
        "window_policy": "EARLY",
        "t_first_seen": int(t_first_seen),
        "t_interact": int(t_interact),
        "window_steps": masked_indices,
        "masked_indices": masked_indices,
        "leak_indices": leak_indices,
        "mask_type": "tile_blackout",
        "mask_strength_actual": mask_strength_actual,
        "alignment_score": alignment_score,
        "physics_check_passed": True,
    }
    return {"nocue_meta": nocue_meta}


# -----------------------------
# CF (Type-2 Intervention)
# -----------------------------

def apply_cf_blue_door_to_wall(env_unwrapped: Any) -> Dict[str, Any]:
    red, blue = find_red_blue_doors(env_unwrapped)
    bx, by = blue.pos
    env_unwrapped.grid.set(bx, by, Wall())
    return {
        "cf_mode": "blue_door_to_wall",
        "intervention_type": "Type-2",
        "blue_door_pos": [int(bx), int(by)],
    }


# -----------------------------
# Record Packaging
# -----------------------------

def mk_rec(
    gid: str,
    task: str,
    env_id: str,
    seed: int,
    variant: str,
    traj: Dict[str, Any],
    cf_meta: Optional[Dict[str, Any]] = None,
    nocue_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    allowed_input_fields = {"frames", "mission", "terminated", "truncated"}
    rec: Dict[str, Any] = {
        "group_id": gid,
        "task": task,
        "variant": variant,
        "env_id": env_id,
        "seed": int(seed),
        "frames": [f"{gid}/{variant}/{Path(p).name}" for p in traj["frames"]],
        "actions_id": traj["actions_id"],
        "actions_text": traj["actions_text"],
        "mission": traj.get("mission", ""),
        "terminated": bool(traj["terminated"]),
        "truncated": bool(traj["truncated"]),
        "success": bool(traj["success"]),
        "reward": float(traj["reward"]),
        "state_seq": traj["state_seq"],
        "model_input_fields": sorted(list(allowed_input_fields)),
    }
    if cf_meta: rec["cf_meta"] = cf_meta
    if nocue_meta: rec["nocue_meta"] = nocue_meta
    return rec


# -----------------------------
# Main
# -----------------------------

def generate_one(
    env_id: str,
    seed: int,
    out_dir: Path,
    tile_size: int,
    max_nodes: int,
    nocue_max_visible: int,
    alignment_min: float,
    mask_ratio_max: float,
    mask_strength_target: float,
) -> Optional[List[Dict[str, Any]]]:
    gid = f"redblue_s{seed:06d}"
    final_dir = out_dir / gid
    if final_dir.exists(): return None

    tmp_root = out_dir / "_tmp"
    work_dir = tmp_root / gid
    if work_dir.exists(): shutil.rmtree(work_dir, ignore_errors=True)
    ensure_dir(work_dir)

    env = None
    try:
        # Full
        env = gym.make(env_id)
        env = RGBImgPartialObsWrapper(env, tile_size=tile_size)

        # Bug B Fix: Plan after reset
        obs0, _ = env.reset(seed=seed)
        scan_prefix = [1, 1, 1, 1] # Deterministic scan
        plan_actions = bfs_plan_red_then_blue(env.unwrapped, max_nodes=max_nodes)
        actions = scan_prefix + plan_actions

        full_traj = rollout_from_current(
            env=env,
            obs0=obs0,
            actions=actions,
            tile_size=tile_size,
            out_dir=work_dir,
            rel_dir="full",
        )

        if full_traj["truncated"]: return None
        if not full_traj["success"]: return None
        if not full_traj["terminated"]: return None

    except Exception:
        return None
    finally:
        if env: env.close()

    # NoCue
    try:
        ensure_dir(work_dir / "full")
        nocue_pack = build_nocue_from_full(
            full_traj=full_traj,
            work_dir=work_dir,
            tile_size=tile_size,
            nocue_max_visible=nocue_max_visible,
            alignment_min=alignment_min,
            mask_ratio_max=mask_ratio_max,
            mask_strength_target=mask_strength_target,
        )
        if not nocue_pack: return None
        nocue_meta = nocue_pack["nocue_meta"]
    except Exception:
        return None

    # CF
    env_cf = None
    try:
        env_cf = gym.make(env_id)
        env_cf = RGBImgPartialObsWrapper(env_cf, tile_size=tile_size)
        obs_cf0, _ = env_cf.reset(seed=seed)

        cf_meta = apply_cf_blue_door_to_wall(env_cf.unwrapped)

        cf_traj = rollout_from_current(
            env=env_cf,
            obs0=obs_cf0,
            actions=full_traj["actions_id"],
            tile_size=tile_size,
            out_dir=work_dir,
            rel_dir="cf",
        )

        # CF Gates: Type-2
        if cf_traj["truncated"]: return None
        if cf_traj["success"]: return None
        if cf_traj["terminated"]: return None # Wall toggle shouldn't terminate
        if cf_traj["reward"] != 0.0: return None

        # Physics Invariance
        if len(cf_traj["state_seq"]) != len(full_traj["state_seq"]): return None
        for t in range(len(full_traj["state_seq"])):
            sf = full_traj["state_seq"][t]
            sc = cf_traj["state_seq"][t]
            if sf["agent_pos"] != sc["agent_pos"] or sf["agent_dir"] != sc["agent_dir"]:
                return None

    except Exception:
        return None
    finally:
        if env_cf: env_cf.close()

    # Finalize
    ensure_dir(final_dir.parent)
    if final_dir.exists(): shutil.rmtree(final_dir)
    shutil.move(str(work_dir), str(final_dir))

    full_rec = mk_rec(gid, "redblue", env_id, seed, "full", full_traj)
    # Fix paths for jsonl (point to final)
    full_rec["frames"] = [f"{gid}/full/{Path(p).name}" for p in full_traj["frames"]]

    nocue_rec = mk_rec(gid, "redblue", env_id, seed, "nocue", full_traj, nocue_meta=nocue_meta)
    nocue_rec["frames"] = [f"{gid}/nocue/{Path(p).name}" for p in full_traj["frames"]]

    cf_rec = mk_rec(gid, "redblue", env_id, seed, "cf", cf_traj, cf_meta=cf_meta)
    cf_rec["frames"] = [f"{gid}/cf/{Path(p).name}" for p in cf_traj["frames"]]

    return [full_rec, nocue_rec, cf_rec]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="MiniGrid-RedBlueDoors-8x8-v0")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--tile_size", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    setup_logger(args.verbose)

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "_tmp")
    jsonl_path = out_dir / "triplets.jsonl"

    target = args.n
    n_ok = 0
    seed = args.seed_start

    while n_ok < target:
        recs = generate_one(
            env_id=args.env_id,
            seed=seed,
            out_dir=out_dir,
            tile_size=args.tile_size,
            max_nodes=20000,
            nocue_max_visible=5,
            alignment_min=0.70,
            mask_ratio_max=0.35,
            mask_strength_target=0.05
        )
        if recs:
            append_jsonl_batch(jsonl_path, recs)
            n_ok += 1
            if n_ok % 10 == 0:
                logging.info(f"[OK] {n_ok}/{target} (last seed {seed})")
        seed += 1

    logging.info(f"Done. {jsonl_path}")

if __name__ == "__main__":
    main()
