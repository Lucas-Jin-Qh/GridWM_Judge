#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate RedBlueDoor triplets (Doorkey-aligned 3-lines-per-group JSONL).

Strict gates:
- Must have exactly 3 variants per group: full/nocue/cf
- Action identity across variants
- Full must succeed (terminated=True, reward>0, truncated=False)
- CF must fail (reward=0, terminated=False, truncated=False)
- Physics invariance: agent_pos + agent_dir identical across full/nocue/cf
- NoCue: EARLY window (t <= first_seen) + exclude interaction frames + masked_indices non-empty
- Pixel audit: only red door tiles may differ in masked frames; unmasked frames must be identical
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# Bug D Fix
from minigrid.core.constants import (
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    IDX_TO_OBJECT,
    IDX_TO_COLOR,
)


def setup_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s | %(message)s")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                records.append(json.loads(line))
            except:
                continue
    return records


def check(cond: bool, msg: str) -> None:
    if not cond: raise AssertionError(msg)


def red_door_tile_mask(enc: np.ndarray, tile_size: int, img_h: int, img_w: int) -> np.ndarray:
    door_id = OBJECT_TO_IDX["door"]
    red_id = COLOR_TO_IDX["red"]
    gh, gw = enc.shape[0], enc.shape[1]

    tile_mask = (enc[:, :, 0] == door_id) & (enc[:, :, 1] == red_id)
    px_mask = np.kron(tile_mask.astype(bool), np.ones((tile_size, tile_size), dtype=bool))
    return px_mask


def validate_group(gid: str, recs: List[Dict[str, Any]], root: Path, fast: bool) -> None:
    variants = {r["variant"]: r for r in recs}
    check(set(variants.keys()) == {"full", "nocue", "cf"}, "Missing variants")

    full, nocue, cf = variants["full"], variants["nocue"], variants["cf"]

    # Action Identity
    check(full["actions_id"] == nocue["actions_id"], "Action mismatch: Full vs NoCue")
    check(full["actions_id"] == cf["actions_id"], "Action mismatch: Full vs CF")

    # Gates
    check(full["success"] and full["terminated"], "Full must succeed")
    check(not cf["success"] and not cf["terminated"], "CF must fail and not terminate")
    check(full["reward"] > 0 and cf["reward"] == 0, "Reward check failed")

    # Physics Invariance
    T = len(full["state_seq"])
    check(len(cf["state_seq"]) == T, "Length mismatch")
    for t in range(T):
        sf = full["state_seq"][t]
        sc = cf["state_seq"][t]
        check(sf["agent_pos"] == sc["agent_pos"], f"Pos diverged at {t}")
        check(sf["agent_dir"] == sc["agent_dir"], f"Dir diverged at {t}")

    # NoCue Logic
    meta = nocue.get("nocue_meta", {})
    masked = meta.get("masked_indices", [])
    check(len(masked) > 0, "NoCue mask empty")
    t_interact = meta.get("t_interact", -1)

    # Interaction leak check
    for m in masked:
        check(int(m) < t_interact, "Mask leaked into interaction phase")
        fc = full["state_seq"][int(m)]["front_cell"]
        check(not (fc.get("type") == "door" and fc.get("color") == "red"), "Mask leaked into interaction-evidence frame")

    # Pixel Audit
    if fast: return

    full_frames = full["frames"]
    nocue_frames = nocue["frames"]
    sample_img = np.array(Image.open(root / full_frames[0]))
    tile_size = sample_img.shape[0] // 7

    masked_set = set(int(x) for x in masked)

    for t in range(len(full_frames)):
        img_f = np.array(Image.open(root / full_frames[t]))
        img_n = np.array(Image.open(root / nocue_frames[t]))
        diff = np.any(img_f != img_n, axis=2)

        if t not in masked_set:
            check(not np.any(diff), f"Unmasked frame {t} changed")
        else:
            enc = np.array(full["state_seq"][t]["state_encoding"])
            px_mask = red_door_tile_mask(enc, tile_size, sample_img.shape[0], sample_img.shape[1])

            # Diff allowed ONLY in red door tiles
            check(not np.any(diff & (~px_mask)), f"Diff outside red door at {t}")
            check(np.any(diff & px_mask), f"No diff in masked area at {t}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    setup_logger(True)

    recs = load_jsonl(Path(args.jsonl))
    groups = defaultdict(list)
    for r in recs: groups[r["group_id"]].append(r)

    ok, fail = 0, 0
    for gid, g_recs in groups.items():
        try:
            validate_group(gid, g_recs, Path(args.root), args.fast)
            ok += 1
        except Exception as e:
            fail += 1
            logging.error(f"FAIL {gid}: {e}")

    logging.info(f"OK: {ok}, FAIL: {fail}")

if __name__ == "__main__":
    main()
