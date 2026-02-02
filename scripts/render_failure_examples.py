#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render a small PNG poster for the first N failure examples in results/figs/failure_examples.jsonl.
Outputs: results/figs/failure_example_1.png ... failure_example_N.png
"""
from __future__ import annotations
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

IN_FILE = Path("results/figs/failure_examples.jsonl")
OUT_DIR = Path("results/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
N = 12

def load_examples(p, n):
    ex = []
    if not p.exists():
        return ex
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                ex.append(json.loads(line))
            except Exception:
                continue
    return ex

def render_example(e, out_path):
    w, h = 1200, 600
    img = Image.new("RGB", (w, h), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    lines = [
        f"uid: {e.get('uid')}",
        f"task: {e.get('task')}  env_task: {e.get('env_task')}",
        f"pred_norm: {e.get('pred_norm')}    gold: {e.get('gold')}",
        f"parse_mode: {e.get('parse_mode')}  failure: {e.get('failure')}",
        f"raw (truncated): {str(e.get('raw',''))[:400]}",
    ]
    y = 20
    for l in lines:
        draw.text((20,y), l, fill=(0,0,0), font=font)
        y += 28
    img.save(out_path)

def main():
    exs = load_examples(IN_FILE, N)
    for i, e in enumerate(exs, start=1):
        out = OUT_DIR / f"failure_example_{i}.png"
        render_example(e, out)
    print(f"Rendered {len(exs)} failure example images to {OUT_DIR}")

if __name__ == "__main__":
    main()


