#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_exam.py - GridWM-Judge SSOT Scorer

Evaluates VLM performance on GridWM-Judge benchmark with capability-preserving scoring.
Directly scores raw inference outputs against ground truth, implementing robust answer
extraction to distinguish format noise from capability limitations.

Design Principles:
- SSOT: Scoring based ONLY on exam_dir (ground truth) + raw inference outputs
- Capability-Fidelity: Recover from harmless formatting noise (strict vs recoverable)
- Transparency: Report both strict and recoverable performance metrics
- Completeness: Every UID gets a score or documented failure reason

Usage:
    python score_exam.py  # Auto-discover latest experiment
    python score_exam.py --responses runs/responses/experiment_dir/
    python score_exam.py --b_acc_threshold 0.85 --b_weights "agent_pos=0.4,front_cell=0.3"

Output: JSON report with micro/macro scores, per-task breakdown, failure analysis
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import defaultdict, Counter

API_ERROR_PREFIX = "__GW_ERR__:"


# -------------------------
# Shared utilities (aligned with run_inference.py)
# -------------------------

def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Load JSONL file with error handling."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def parse_exam_id(uid: str) -> Dict[str, Any]:
    """
    Parse GridWM-Judge exam_id format:
      A.<env_task>.<group_id>.t<step>
      B.<env_task>.<group_id>.t<frame>
      C.<env_task>.<group_id>.<variant>.<temporal>[.<visual>]
    """
    out: Dict[str, Any] = {}
    if not uid or not isinstance(uid, str):
        return out
    parts = uid.split(".")
    if len(parts) < 2:
        return out
    task = parts[0]
    out["task"] = task

    if task in ("A", "B"):
        if len(parts) >= 4:
            out["env_task"] = parts[1]
            out["group_id"] = parts[2]
            tpart = parts[3]
            if tpart.startswith("t") and tpart[1:].isdigit():
                out["t"] = int(tpart[1:])
        return out

    if task == "C":
        if len(parts) >= 5:
            out["env_task"] = parts[1]
            out["group_id"] = parts[2]
            out["variant"] = parts[3]
            out["temporal"] = parts[4]
            if len(parts) >= 6:
                out["visual"] = parts[5]
        return out

    return out


# -------------------------
# Ground Truth Loading (SSOT)
# -------------------------

@dataclass
class Gold:
    uid: str
    task: str
    env_task: Optional[str]
    group_id: Optional[str]
    t: Optional[int] = None
    variant: Optional[str] = None
    temporal: Optional[str] = None
    visual: Optional[str] = None

    # Task-specific ground truth
    a_answer: Optional[str] = None          # "A"/"B"/"C"/"D"
    a_label: Optional[int] = None           # 0..3
    b_answer_json: Optional[Dict[str, Any]] = None
    c_answer: Optional[str] = None          # "Success"/"Fail"
    c_label: Optional[int] = None           # optional


def load_gold_from_exam_dir(exam_dir: Path) -> Dict[str, Gold]:
    """Load ground truth from exam directory (build_exam.py output)."""
    gold: Dict[str, Gold] = {}

    # Task A
    p = exam_dir / "task_a_exam.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            uid = r.get("exam_id")
            if not uid:
                continue
            info = parse_exam_id(uid)
            g = Gold(
                uid=uid,
                task="A",
                env_task=info.get("env_task"),
                group_id=info.get("group_id"),
                t=info.get("t"),
                a_answer=r.get("answer"),
                a_label=r.get("label"),
            )
            # Normalize answer format
            if isinstance(g.a_answer, str):
                g.a_answer = g.a_answer.strip().upper()
            if g.a_answer not in ("A", "B", "C", "D"):
                if isinstance(g.a_label, int):
                    letters = {0: "A", 1: "B", 2: "C", 3: "D"}
                    g.a_answer = letters.get(g.a_label)
            gold[uid] = g

    # Task B
    p = exam_dir / "task_b_exam.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            uid = r.get("exam_id")
            if not uid:
                continue
            info = parse_exam_id(uid)
            g = Gold(
                uid=uid,
                task="B",
                env_task=info.get("env_task"),
                group_id=info.get("group_id"),
                t=info.get("t"),
                b_answer_json=r.get("answer_json"),
            )
            gold[uid] = g

    # Task C
    p = exam_dir / "task_c_exam.jsonl"
    if p.exists():
        for r in load_jsonl(p):
            uid = r.get("exam_id")
            if not uid:
                continue
            info = parse_exam_id(uid)
            g = Gold(
                uid=uid,
                task="C",
                env_task=info.get("env_task"),
                group_id=info.get("group_id"),
                variant=r.get("variant"),
                temporal=info.get("temporal"),
                visual=r.get("visual") or info.get("visual"),
                c_answer=r.get("answer"),
                c_label=r.get("label"),
            )
            if isinstance(g.c_answer, str):
                g.c_answer = g.c_answer.strip()
            gold[uid] = g

    return gold


# -------------------------
# Response Loading
# -------------------------

@dataclass
class Resp:
    uid: str
    pred: str
    raw: str
    ok: bool
    error: Optional[str] = None


def discover_latest_responses(responses_dir: Path) -> Path:
    """Auto-discover the most recent experiment results."""
    if not responses_dir.exists():
        raise SystemExit(f"Responses directory not found: {responses_dir}")

    # Find experiment directories (exclude _requests_from_exam)
    exp_dirs = [d for d in responses_dir.iterdir()
               if d.is_dir() and not d.name.startswith('_')]
    if not exp_dirs:
        raise SystemExit(f"No experiment directories found in {responses_dir}")

    # Find the most recent experiment
    exp_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest_exp = exp_dirs[0]

    # Find response files in the latest experiment
    resp_files = list(latest_exp.glob("*.jsonl"))
    if not resp_files:
        raise SystemExit(f"No response files found in {latest_exp}")

    # Return the most recent response file
    resp_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return resp_files[0]


def load_responses(responses_path: Path) -> Tuple[Dict[str, Resp], Dict[str, Any]]:
    """
    Load responses from file or auto-discover from directory.

    Accepts:
      - Single JSONL file
      - Directory containing *.jsonl (sharded outputs supported)
    """
    files: List[Path] = []
    if responses_path.is_dir():
        files = sorted([p for p in responses_path.glob("*.jsonl") if p.is_file()])
    else:
        files = [responses_path]

    out: Dict[str, Resp] = {}
    duplicates = 0
    total_lines = 0
    for fp in files:
        for r in load_jsonl(fp):
            total_lines += 1
            uid = r.get("uid")
            if not uid:
                continue
            if uid in out:
                duplicates += 1
                continue

            raw = r.get("raw", r.get("pred", ""))
            pred = r.get("pred", "")
            meta = r.get("meta", {})

            # Enhanced error detection
            ok = meta.get("ok", True)
            error = meta.get("error")

            # Check for API errors in raw text
            if isinstance(raw, str) and raw.startswith(API_ERROR_PREFIX):
                ok = False
                error = raw

            out[uid] = Resp(
                uid=uid,
                pred=str(pred),
                raw=str(raw),
                ok=ok,
                error=str(error) if error else None,
            )

    telemetry = {
        "n_files": len(files),
        "files": [str(p) for p in files],
        "total_lines": total_lines,
        "n_uids": len(out),
        "duplicates_dropped": duplicates,
    }
    return out, telemetry


# -------------------------
# Task A/C Parsing (Capability-Fidelity)
# -------------------------

_A_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
_C_RE = re.compile(r"\b(success(?:ful(?:ly)?)?|fail(?:ure|ed)?)\b", re.IGNORECASE)

def parse_task_a_answer(text: str) -> Tuple[Optional[str], str]:
    """
    Extract A/B/C/D answer with capability-preserving parsing.

    Returns (answer_letter, mode):
      mode = strict | recoverable | fail
    """
    t = (text or "").strip()
    if not t:
        return None, "fail"
    # Strict: exactly one char
    if len(t) == 1 and t.upper() in ("A","B","C","D"):
        return t.upper(), "strict"
    # Recoverable: find anywhere in text
    m = _A_LETTER_RE.search(t)
    if m:
        return m.group(1).upper(), "recoverable"
    return None, "fail"

def parse_task_c_answer(text: str) -> Tuple[Optional[str], str]:
    """
    Extract Success/Fail answer with morphological variants.

    Capability-fidelity: accept common variants (successful/successfully/failed/failure)
    to avoid penalizing models for superficial wording differences.
    """
    t = (text or "").strip()
    if not t:
        return None, "fail"

    tl = t.lower()
    # Strict matches
    if tl in ("success", "fail"):
        return t[:1].upper() + t[1:].lower(), "strict"

    # Recoverable: morphological variants
    m = _C_RE.search(t)
    if m:
        w = m.group(1).lower()
        if w.startswith("success"):
            return "Success", "recoverable"
        if w.startswith("fail"):
            return "Fail", "recoverable"
    return None, "fail"


# -------------------------
# Task B Parsing + Scoring (Perception IoU)
# -------------------------

def _strip_code_fence(s: str) -> str:
    t = (s or "").strip()
    if "```" not in t:
        return t
    parts = t.split("```")
    if len(parts) >= 3:
        return parts[1].strip()
    return t.replace("```", "").strip()

def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract the most likely JSON object substring.
    - If whole string is JSON, return it.
    - Else take substring between first '{' and last '}'.
    """
    t = (text or "").strip()
    if not t:
        return None
    t = _strip_code_fence(t)
    # Quick path
    if t.startswith("{") and t.endswith("}"):
        return t
    l = t.find("{")
    r = t.rfind("}")
    if 0 <= l < r:
        return t[l:r+1]
    return None

def _min_fix_json_commas(s: str) -> str:
    # Remove trailing commas before } or ]
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s

def parse_task_b_json(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Parse Task B JSON with recovery logic.

    Returns (obj, mode):
      mode = strict | recoverable | fail_json | fail_type
    """
    t = (text or "").strip()
    if not t:
        return None, "fail_json"

    cand = _extract_json_object(t)
    if cand is None:
        return None, "fail_json"

    # Strict attempt
    try:
        obj = json.loads(cand)
        if isinstance(obj, dict):
            return obj, ("strict" if cand.strip() == t.strip() else "recoverable")
        return None, "fail_type"
    except Exception:
        # Minimal recovery: trailing commas
        try:
            obj = json.loads(_min_fix_json_commas(cand))
            if isinstance(obj, dict):
                return obj, "recoverable"
            return None, "fail_type"
        except Exception:
            return None, "fail_json"

def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    if isinstance(x, str) and x.strip().lstrip("-").isdigit():
        return int(x.strip())
    return None

def _to_pos(x: Any) -> Optional[Tuple[int,int]]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        a = _to_int(x[0]); b = _to_int(x[1])
        if a is not None and b is not None:
            return (a, b)
    return None

def _canon_state(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (int, float, bool)):
        return str(int(x)) if isinstance(x, bool) or (isinstance(x, float) and float(x).is_integer()) else str(x)
    if isinstance(x, str):
        return x.strip()
    return str(x)

def _canon_obj(o: Any) -> Optional[Tuple[str, Tuple[int,int], Optional[str], Optional[str]]]:
    """
    Canonical object tuple: (type, pos, color, state)
    Returns None if critical fields missing.
    """
    if not isinstance(o, dict):
        return None
    typ = o.get("type")
    pos = _to_pos(o.get("pos"))
    if not isinstance(typ, str) or pos is None:
        return None
    color = o.get("color")
    color = color.strip() if isinstance(color, str) else None
    state = _canon_state(o.get("state"))
    return (typ.strip(), pos, color, state)

def _canon_front(o: Any) -> Optional[Tuple[Tuple[int,int], str, Optional[str]]]:
    if not isinstance(o, dict):
        return None
    pos = _to_pos(o.get("pos"))
    typ = o.get("type")
    if pos is None or not isinstance(typ, str):
        return None
    state = _canon_state(o.get("state"))
    return (pos, typ.strip(), state)

def _canon_carry(x: Any) -> Optional[Tuple[str, Optional[str]]]:
    """
    carrying: object|null
      - None => None
      - dict => (type, color)
      - str => (str, None)
    """
    if x is None:
        return None
    if isinstance(x, dict):
        typ = x.get("type")
        if not isinstance(typ, str):
            return ("<unknown>", None)
        color = x.get("color")
        color = color.strip() if isinstance(color, str) else None
        return (typ.strip(), color)
    if isinstance(x, str):
        return (x.strip(), None)
    return ("<unknown>", None)

def normalize_task_b(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize Task B JSON to canonical format."""
    out: Dict[str, Any] = {}

    agent = obj.get("agent")
    if isinstance(agent, dict):
        out["agent_pos"] = _to_pos(agent.get("pos"))
        out["agent_dir"] = _to_int(agent.get("dir"))
        out["carrying"] = _canon_carry(agent.get("carrying"))
    else:
        out["agent_pos"] = None
        out["agent_dir"] = None
        out["carrying"] = None

    fc = obj.get("front_cell")
    out["front_cell"] = _canon_front(fc)

    objs = obj.get("objects")
    s = set()
    if isinstance(objs, list):
        for o in objs:
            co = _canon_obj(o)
            if co is not None:
                s.add(co)
    out["objects_set"] = s

    return out

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a and b:
        return 0.0
    if a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 0.0

def score_task_b(
    pred_obj: Dict[str, Any],
    gold_obj: Dict[str, Any],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, Any], List[str]]:
    """
    Score Task B with weighted perception IoU.

    Returns:
      total_score in [0,1],
      component dict,
      failure_tags (schema/partial indicators)
    """
    p = normalize_task_b(pred_obj or {})
    g = normalize_task_b(gold_obj or {})

    failures: List[str] = []

    # Component scores
    agent_pos = 1.0 if (p["agent_pos"] is not None and g["agent_pos"] is not None and p["agent_pos"] == g["agent_pos"]) else 0.0
    if p["agent_pos"] is None:
        failures.append("B_missing_agent_pos")

    agent_dir = 1.0 if (p["agent_dir"] is not None and g["agent_dir"] is not None and p["agent_dir"] == g["agent_dir"]) else 0.0
    if p["agent_dir"] is None:
        failures.append("B_missing_agent_dir")

    carrying = 1.0 if (p["carrying"] == g["carrying"]) else 0.0
    if "carrying" not in (pred_obj.get("agent") or {}):
        failures.append("B_missing_carrying")

    front_cell = 1.0 if (p["front_cell"] is not None and g["front_cell"] is not None and p["front_cell"] == g["front_cell"]) else 0.0
    if p["front_cell"] is None:
        failures.append("B_missing_front_cell")

    objects = jaccard(p["objects_set"], g["objects_set"])
    if not isinstance(pred_obj.get("objects"), list):
        failures.append("B_missing_objects")

    comp = {
        "agent_pos": agent_pos,
        "agent_dir": agent_dir,
        "carrying": carrying,
        "front_cell": front_cell,
        "objects_jaccard": objects,
        "n_gold_objects": len(g["objects_set"]),
        "n_pred_objects": len(p["objects_set"]),
    }

    # Weighted sum
    wsum = sum(max(0.0, float(v)) for v in weights.values())
    if wsum <= 0:
        # Safe fallback weights
        weights = {"agent_pos": 0.30, "agent_dir": 0.10, "carrying": 0.15,
                  "front_cell": 0.25, "objects_jaccard": 0.20}
        wsum = sum(weights.values())

    total = (
        weights.get("agent_pos", 0.0) * agent_pos +
        weights.get("agent_dir", 0.0) * agent_dir +
        weights.get("carrying", 0.0) * carrying +
        weights.get("front_cell", 0.0) * front_cell +
        weights.get("objects_jaccard", 0.0) * objects
    ) / wsum

    # Mark partial schema if missing fields
    if failures:
        failures.append("B_partial_schema")

    return float(total), comp, failures


# -------------------------
# Scoring Logic
# -------------------------

def _default_weights_b() -> Dict[str, float]:
    """Default weights optimized for GridWM-Judge Task B."""
    return {
        "agent_pos": 0.30,    # Most critical for spatial reasoning
        "agent_dir": 0.10,    # Important for navigation
        "carrying": 0.15,     # Object manipulation state
        "front_cell": 0.25,   # Immediate environment perception
        "objects_jaccard": 0.20,  # Overall scene understanding
    }

def _parse_weights_b(s: Optional[str]) -> Dict[str, float]:
    """Parse custom weights string like 'agent_pos=0.4,objects_jaccard=0.3'."""
    if not s:
        return _default_weights_b()
    out = _default_weights_b()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except Exception:
            pass
    return out

def score_one(uid: str, g: Gold, r: Optional[Resp], weights_b: Dict[str, float], b_acc_threshold: float) -> Dict[str, Any]:
    """
    Score single example with detailed breakdown.

    Returns per-example row with score, correct, failure_code, parse_mode, details...
    """
    row: Dict[str, Any] = {
        "uid": uid,
        "task": g.task,
        "env_task": g.env_task,
        "group_id": g.group_id,
    }
    if g.task == "A":
        row["t"] = g.t
    if g.task == "C":
        row["variant"] = g.variant
        row["temporal"] = g.temporal
        if g.visual is not None:
            row["visual"] = g.visual

    # Missing response
    if r is None:
        row.update({"score": 0.0, "correct": False, "failure": "missing_response", "parse_mode": "fail"})
        return row

    # API error
    if not r.ok or (r.error is not None):
        row.update({"score": 0.0, "correct": False, "failure": "api_error", "parse_mode": "fail", "error": r.error})
        return row

    text = r.raw if r.raw else r.pred

    if g.task == "A":
        ans, mode = parse_task_a_answer(text)
        row["parse_mode"] = mode
        if ans is None:
            row.update({"score": 0.0, "correct": False, "failure": "A_no_letter"})
            return row
        gold_ans = g.a_answer
        if gold_ans is None:
            row.update({"score": 0.0, "correct": False, "failure": "A_missing_gold"})
            return row
        correct = (ans == gold_ans)
        row.update({"score": 1.0 if correct else 0.0, "correct": correct, "pred_norm": ans, "gold": gold_ans})
        if not correct:
            row["failure"] = "A_wrong"
        return row

    if g.task == "C":
        ans, mode = parse_task_c_answer(text)
        row["parse_mode"] = mode
        if ans is None:
            row.update({"score": 0.0, "correct": False, "failure": "C_no_label"})
            return row
        gold_ans = g.c_answer
        if isinstance(gold_ans, str):
            gold_ans = gold_ans.strip().lower()
        gold_norm = "Success" if gold_ans == "success" else ("Fail" if gold_ans == "fail" else None)
        if gold_norm is None:
            row.update({"score": 0.0, "correct": False, "failure": "C_missing_gold"})
            return row
        correct = (ans == gold_norm)
        row.update({"score": 1.0 if correct else 0.0, "correct": correct, "pred_norm": ans, "gold": gold_norm})
        if not correct:
            row["failure"] = "C_wrong"
        return row

    if g.task == "B":
        obj, mode = parse_task_b_json(text)
        row["parse_mode"] = mode
        if obj is None:
            row.update({"score": 0.0, "correct": False, "failure": "B_json_parse_error"})
            return row
        if g.b_answer_json is None:
            row.update({"score": 0.0, "correct": False, "failure": "B_missing_gold"})
            return row

        score, comp, failures = score_task_b(obj, g.b_answer_json, weights_b)
        correct = (score >= b_acc_threshold)
        row.update({
            "score": score,
            "correct": correct,
            "b_components": comp,
            "b_acc_threshold": b_acc_threshold,
            "b_acc_at_threshold": correct,
        })
        if failures:
            # Primary failure bucket
            primary = next((f for f in failures if f != "B_partial_schema"), failures[-1])
            row["failure"] = primary
            row["failure_tags"] = failures
        return row

    row.update({"score": 0.0, "correct": False, "failure": "unknown_task", "parse_mode": "fail"})
    return row


def _agg_stats(rows: List[Dict[str, Any]], task: Optional[str] = None) -> Dict[str, Any]:
    """Aggregate statistics for rows, optionally filtered by task."""
    if task is not None:
        rows = [r for r in rows if r.get("task") == task]
    n = len(rows)
    if n == 0:
        return {"n": 0}

    scores = [float(r.get("score", 0.0)) for r in rows]
    corrects = [bool(r.get("correct", False)) for r in rows]
    parse_modes = Counter([r.get("parse_mode", "fail") for r in rows])
    failures = Counter([r.get("failure", "none") for r in rows if r.get("failure")])

    api_err = sum(1 for r in rows if r.get("failure") == "api_error")
    missing = sum(1 for r in rows if r.get("failure") == "missing_response")

    return {
        "n": n,
        "mean_score": sum(scores)/n,
        "acc": sum(corrects)/n,
        "parse_mode": dict(parse_modes),
        "failure_hist": dict(failures),
        "api_error_rate": api_err/n,
        "missing_rate": missing/n,
    }


def main():
    ap = argparse.ArgumentParser(description="GridWM-Judge SSOT Scorer")
    ap.add_argument("--exam_dir", default="datasets/exams",
                   help="Directory containing task_*_exam.jsonl (build_exam.py output)")
    ap.add_argument("--responses", default="runs/responses",
                   help="Responses jsonl file OR directory (auto-discovers latest experiment)")
    ap.add_argument("--out", default="runs/scores/score_report.json",
                   help="Output score report JSON path")
    ap.add_argument("--dump_rows", action="store_true",
                   help="Include per-example rows in output JSON (can be large)")
    ap.add_argument("--max_bad_examples", type=int, default=50,
                   help="Store up to N failure examples per failure type")
    ap.add_argument("--b_acc_threshold", type=float, default=0.90,
                   help="Task B: score>=threshold counts as correct for acc metric")
    ap.add_argument("--b_weights", type=str, default=None,
                   help="Task B component weights, e.g. agent_pos=0.25,objects_jaccard=0.35,...")
    args = ap.parse_args()

    exam_dir = Path(args.exam_dir)
    responses_path = Path(args.responses)

    # Load ground truth (SSOT)
    print(f"📚 Loading ground truth from {exam_dir}")
    gold = load_gold_from_exam_dir(exam_dir)
    print(f"   Found {len(gold)} ground truth examples")

    # Load responses
    if responses_path.is_dir():
        responses_path = discover_latest_responses(responses_path)
        print(f"🤖 Using latest experiment: {responses_path.parent.name}")

    print(f"📊 Loading responses from {responses_path}")
    responses, resp_tel = load_responses(responses_path)
    print(f"   Found {len(responses)} response examples")

    # Setup scoring parameters
    weights_b = _parse_weights_b(args.b_weights)

    # Score every gold UID (SSOT: gold is canonical set)
    print("🧮 Scoring examples...")
    rows: List[Dict[str, Any]] = []
    for uid, g in gold.items():
        r = responses.get(uid)
        rows.append(score_one(uid, g, r, weights_b, args.b_acc_threshold))

    # Coverage analysis
    n_gold = len(gold)
    n_resp = len(responses)
    missing_uids = [r["uid"] for r in rows if r.get("failure") == "missing_response"]
    extra_uids = sorted([uid for uid in responses.keys() if uid not in gold])

    # Aggregate statistics
    per_task = {t: _agg_stats(rows, task=t) for t in ("A","B","C")}
    per_env_task: Dict[str, Dict[str, Any]] = {}
    groups = defaultdict(list)
    for r in rows:
        key = f'{r.get("task")}:{r.get("env_task")}'
        groups[key].append(r)
    for k, rs in groups.items():
        per_env_task[k] = _agg_stats(rs)

    # Global failure histogram
    global_fail_hist = Counter([r.get("failure","none") for r in rows if r.get("failure")])

    # Failure examples for debugging
    failure_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        f = r.get("failure")
        if not f:
            continue
        if len(failure_examples[f]) < args.max_bad_examples:
            ex = {k: r.get(k) for k in ("uid","task","env_task","parse_mode","failure","pred_norm","gold","error")}
            if r.get("task") == "B":
                ex["b_score"] = r.get("score")
                ex["b_components"] = r.get("b_components")
            failure_examples[f].append(ex)

    # Build final report
    report: Dict[str, Any] = {
        "ssot": {
            "gold_source": str(exam_dir),
            "scoring_policy": {
                "A_C": "accuracy with strict vs recoverable parsing",
                "B": {
                    "metric": "weighted perception IoU",
                    "weights": weights_b,
                    "acc_threshold": args.b_acc_threshold,
                },
            },
        },
        "coverage": {
            "n_gold": n_gold,
            "n_response_uids": n_resp,
            "n_scored": len(rows),
            "missing_responses": len(missing_uids),
            "extra_responses_not_in_gold": len(extra_uids),
            "responses_telemetry": resp_tel,
        },
        "overall_micro": _agg_stats(rows),
        "per_task": per_task,
        "per_env_task": per_env_task,
        "failure_hist": dict(global_fail_hist),
        "failure_examples": dict(failure_examples),
    }

    if extra_uids:
        report["coverage"]["extra_uids_sample"] = extra_uids[:50]

    if args.dump_rows:
        report["rows"] = rows

    # Save report
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summary
    print("\n" + "="*60)
    print("🎯 SCORING COMPLETE")
    print("="*60)
    overall = report["overall_micro"]
    print(f"📊 Overall: {overall['n']} examples")
    print(f"   Mean Score: {overall['mean_score']:.3f}")
    print(f"   Accuracy: {overall['acc']:.3f}")
    print(f"   API Error Rate: {overall['api_error_rate']:.1%}")
    print(f"   Missing Rate: {overall['missing_rate']:.1%}")
    print(f"📋 Strict vs Recoverable: {overall['parse_mode']}")

    print(f"\n🔍 Per-Task Breakdown:")
    for task in ("A", "B", "C"):
        if task in per_task:
            tstats = per_task[task]
            print(f"   {task}: acc={tstats['acc']:.3f}, n={tstats['n']}")

    print(f"\n💾 Report saved: {out_path}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
