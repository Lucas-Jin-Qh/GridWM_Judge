#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_inference.py - GridWM-Judge Universal VLM Inference Gateway

This script provides a unified, capability-preserving interface for running inference
on GridWM-Judge exam questions across multiple VLM backends. Designed with scientific
standards to prevent engineering artifacts from distorting capability assessments.

Key Features:
- Backend Agnostic: OpenAI-compatible (SiliconFlow/OpenAI/Zhizengzeng), Gemini, Local models
- Capability Preservation: Task-aware token budgets, PNG fidelity, robust error handling
- Real-time Progress: Live progress updates with ETA and error tracking
- Experiment Isolation: Automatic directory structure for clean result management
- Zero-Trust Security: Advanced secret masking and API parameter validation
- Scientific Standards: Sentinel errors, metadata preservation, reproducible execution
- Provider Compatibility: Unified API key resolution (ZZZ_API_KEY + legacy support)

Supported Backends:
  - openai_compatible: OpenAI, SiliconFlow, Zhizengzeng, custom OpenAI-compatible APIs
      * Task-aware token budgets (A:512, B:2048, C:256)
      * PNG-first image encoding for fidelity preservation
      * CamelCase token key normalization (maxOutputTokens -> max_output_tokens)
      * Dedicated __GW_ERR__: sentinel to avoid model output collision
      * Zhizengzeng unified endpoint (all models via api.zhizengzeng.com/v1)
      * API key compatibility: ZZZ_API_KEY (official) + ZHIZENGZENG_API_KEY (legacy)
  - gemini: Google Gemini API + Zhizengzeng Gemini
      * Secure parameter merging and validation
  - local: Local model inference (Qwen2.5-VL, InternVL2.5, LLaVA-NeXT-Video)
      * Device mapping and dtype configuration
      * Torch-based inference with proper memory management

Progress Reporting:
- Real-time updates every N items and T seconds
- Includes processing rate, ETA, error count, and current UID
- Immediate stdout flushing (no buffering issues)

Experiment Management:
- Automatic experiment ID generation: {backend}_{provider}_{model}_{protocol}_shard{shard_id}
- Isolated directory structure prevents result conflicts
- Resume capability with processed UID tracking

Supported Providers (OpenAI-compatible):
- openai: Official OpenAI API
- siliconflow: SiliconFlow API (CN optimized)
- zhizengzeng*: Zhizengzeng unified platform (ChatGPT, Claude, Grok, etc.)
- zhizengzeng_qwen: Zhizengzeng Qwen VL models (qwen-vl-plus, qwen-vl-max, etc.)
- closeai: CloseAI proxy
- custom: Custom base URL

Author: GridWM-Judge Team
Version: 7.6.12 (Zhizengzeng-Qwen-VL-Compatible Edition)
Date: 2025-01-14
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from PIL import Image

# Torch is optional unless you use a local backend. Keep imports lazy so
# `python scripts/run_inference.py --help` works without PyTorch installed.
torch = None  # type: ignore[assignment]

def _require_torch():
    global torch
    if torch is not None:
        return torch
    try:
        import torch as _torch  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyTorch is required for local backends. Install torch, or use a remote backend "
            "(e.g., --backend openai_compatible / gemini)."
        ) from e
    torch = _torch
    return _torch


# -------------------------
# Constants
# -------------------------

# Dedicated sentinel for gateway errors to avoid collision with model outputs
API_ERROR_PREFIX = "__GW_ERR__:"

_RE_TASK_FROM_UID = re.compile(r"^(?P<task>[ABC])\.")


# -------------------------
# Utilities
# -------------------------

def stable_hash64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def parse_dtype(s: str):
    if s in (None, "", "auto"): return "auto"
    s = s.lower()
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    if s in ("fp16", "float16"): return torch.float16
    if s in ("fp32", "float32"): return torch.float32
    raise ValueError(f"Unknown dtype: {s}")

def safe_load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()

def encode_image_base64(path: str, max_side: int, quality: int, prefer_png: bool = True) -> Tuple[str, str]:
    """
    Returns (mime_type, b64).
    - prefer_png=True: try PNG to avoid JPEG artifacts (esp. storyboard / grid images)
    - max_side<=0: no resize
    """
    p = Path(path)
    with Image.open(p) as im0:
        im = im0.convert("RGB")
        if max_side and max_side > 0:
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = BytesIO()
        # Keep PNG if asked; otherwise JPEG for smaller payload.
        if prefer_png:
            im.save(buf, format="PNG", optimize=True)
            return ("image/png", base64.b64encode(buf.getvalue()).decode("utf-8"))
        else:
            im.save(buf, format="JPEG", quality=int(quality), optimize=True)
            return ("image/jpeg", base64.b64encode(buf.getvalue()).decode("utf-8"))

def _infer_task(uid: str) -> Optional[str]:
    if not uid:
        return None
    m = _RE_TASK_FROM_UID.match(uid)
    return m.group("task") if m else None

def _task_max_tokens(uid: str, default_max: int, task_overrides: Dict[str, int]) -> int:
    """
    Task-aware token budget to avoid truncation causing fake failures.
    - A: 512 tokens (atomic transitions)
    - B: 2048 tokens (JSON structures)
    - C: 256 tokens (success/failure judgment)
    """
    t = _infer_task(uid)
    if not t:
        return int(default_max)
    if t in task_overrides:
        return int(task_overrides[t])
    return int(default_max)

def fuse_images(paths: List[str], out_path: Path, mode: str, quality: int) -> str:
    imgs = [safe_load_rgb(p) for p in paths]
    ws, hs = zip(*[im.size for im in imgs])
    if mode == "h":
        W, H = sum(ws), max(hs)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        x = 0
        for im in imgs: canvas.paste(im, (x, 0)); x += im.size[0]
    else:
        W, H = max(ws), sum(hs)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        y = 0
        for im in imgs: canvas.paste(im, (0, y)); y += im.size[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=quality)
    return str(out_path)

def _need(name: str, v: Optional[str]) -> str:
    if not v:
        raise ValueError(f"Missing {name}")
    return v

def _pick_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None

def _mask_any(x: Any) -> Any:
    """
    Recursively mask sensitive data in dicts/lists.
    - Hard-mask high-risk blobs: 'messages', 'contents'
    - Mask secrets with precise rules (handles snake_case, kebab-case, camelCase)
    """
    SECRET_EXACT = {
        "api_key", "apikey", "access_token", "refresh_token", "id_token",
        "authorization", "auth", "secret", "password", "x-api-key", "x-goog-api-key"
    }
    # Allow these for observability (case-insensitive checking below)
    ALLOW_OBSERVABILITY = {
        "max_tokens", "max_completion_tokens", "max_output_tokens",
        "maxtokens", "maxcompletiontokens", "maxoutputtokens"
    }

    def is_secret_key(key_str: str) -> bool:
        lk = key_str.lower()
        clean_lk = lk.replace("_", "").replace("-", "")

        if clean_lk in ALLOW_OBSERVABILITY:
            return False

        if lk in ("messages", "contents"):
            return True
        if lk in SECRET_EXACT:
            return True

        # Heuristics for variations: *_token, *_key, *-token, *-key
        if lk.endswith("token") or lk.endswith("key"):
            # Check if it looks like a known safe token param
            if clean_lk in ALLOW_OBSERVABILITY:
                return False
            return True # Default to secret if it ends in token/key and not whitelisted

        if "authorization" in lk:
            return True
        return False

    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if is_secret_key(str(k)):
                # Distinguish content vs secret masking
                if str(k).lower() in ("messages", "contents"):
                    out[k] = "***MASKED_CONTENT***"
                else:
                    out[k] = "***MASKED_SECRET***"
            else:
                out[k] = _mask_any(v)
        return out
    if isinstance(x, list):
        return [_mask_any(v) for v in x]
    return x

def is_o_series_model(model: str) -> bool:
    """
    Robust heuristic for OpenAI o-series.
    Handles namespaced model ids like 'openai/o1-mini', 'zzz/o3-mini', 'models/o1-preview'.
    """
    m = (model or "").strip().lower()
    m = m.split("/")[-1]
    m = m.split(":", 1)[0]
    return m.startswith(("o1", "o3", "o4", "o5")) or (len(m) >= 2 and m[0] == "o" and m[1].isdigit())

def sanitize_openai_extra_body(model: str, extra_body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fail-safe: prevent extra_body from sabotaging evaluation or breaking models.
    1. Canonicalization: camelCase tokens -> snake_case
    2. Integrity: forbid overriding core fields (linear scan)
    3. O-series / Standard logic: remap & clean
    """
    eb = dict(extra_body or {})
    modified_logs = []

    # --- 1. Canonicalization (Camel -> Snake) ---
    # Map common variations to standard snake_case keys
    # This catches maxOutputTokens, MaxTokens, etc.
    TOKEN_MAP = {
        "maxoutputtokens": "max_output_tokens",
        "maxcompletiontokens": "max_completion_tokens",
        "maxtokens": "max_tokens"
    }

    # Snapshot keys to allow modification
    original_keys = list(eb.keys())
    for k in original_keys:
        lk = k.lower().replace("_", "").replace("-", "")
        if lk in TOKEN_MAP:
            target = TOKEN_MAP[lk]
            if target != k:
                # Move value to standard key
                eb[target] = eb.pop(k)
                modified_logs.append(f"normalize: {k} -> {target}")

    # --- 2. Linear Scan Integrity Check ---
    # Scan ALL keys to catch 'streamOptions', 'Stream', etc.
    forbid = {"messages", "model", "stream", "stream_options", "streamoptions"}
    for k in eb.keys():
        if str(k).lower() in forbid:
             raise ValueError(f"extra_body must NOT contain '{k}' (would override/break constructed request).")

    # --- 3. Model-Specific Logic ---
    if is_o_series_model(model):
        # Remap aliases -> max_completion_tokens
        if "max_completion_tokens" not in eb:
            if "max_tokens" in eb:
                eb["max_completion_tokens"] = eb["max_tokens"]
                modified_logs.append("remap: max_tokens -> max_completion_tokens")
            elif "max_output_tokens" in eb:
                eb["max_completion_tokens"] = eb["max_output_tokens"]
                modified_logs.append("remap: max_output_tokens -> max_completion_tokens")

        # Drop incompatible aliases
        for k in ("max_tokens", "max_output_tokens"):
            if k in eb:
                eb.pop(k, None)
                modified_logs.append(f"drop: {k} (incompatible with o-series)")

        if "temperature" in eb:
            eb.pop("temperature", None)
            modified_logs.append("drop: temperature (fail-safe for o-series)")

    else:
        # Standard models
        # 1. Remap max_output_tokens -> max_tokens
        if "max_output_tokens" in eb and "max_tokens" not in eb:
            eb["max_tokens"] = eb["max_output_tokens"]
            modified_logs.append("remap: max_output_tokens -> max_tokens")

        # 2. Remap max_completion_tokens -> max_tokens (Prevent o-series config fail)
        if "max_completion_tokens" in eb and "max_tokens" not in eb:
            eb["max_tokens"] = eb["max_completion_tokens"]
            modified_logs.append("remap: max_completion_tokens -> max_tokens")

        # 3. Drop aliases unconditionally
        for k in ("max_output_tokens", "max_completion_tokens"):
            if k in eb:
                eb.pop(k, None)
                modified_logs.append(f"drop: {k} (standard model cleanup)")

    if modified_logs:
        print(f"🛡️ [Auto-Sanitize] Model='{model}': {', '.join(modified_logs)}", flush=True)

    return eb

def validate_backend_provider(backend: str, provider: str):
    allowed = {
        "openai_compatible": {
            "custom", "openai", "siliconflow", "closeai",
            # Unified Zhizengzeng providers (all use OpenAI-compatible interface)
            "zhizengzeng", "zhizengzeng_xai", "zhizengzeng_claude", "zhizengzeng_grok", "zhizengzeng_qwen"
        },
        "gemini": {"custom", "gemini", "zhizengzeng_gemini"},
        "qwen2.5-vl": None,
        "internvl2.5": None,
        "llava-next-video": None,
    "llava": None,
    "llava-next": None,
    }

    valid_set = allowed.get(backend)
    if valid_set is None:
        return

    if provider not in valid_set:
        raise ValueError(
            f"❌ Invalid provider '{provider}' for backend '{backend}'.\n"
            f"   Allowed providers: {sorted(list(valid_set))}"
        )


# --- Runners ---
class Runner:
    def generate(self, images: List[str], prompt: str, max_tokens_override: Optional[int] = None) -> str: raise NotImplementedError

class OpenAICompatibleRunner(Runner):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        max_side: int,
        quality: int,
        max_retries: int,
        backoff_base: float,
        seed: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        from openai import OpenAI
        # Some OpenAI-compatible providers are sensitive to trailing slashes.
        self.client = OpenAI(api_key=api_key, base_url=(base_url or "").rstrip("/"), timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_side = max_side
        self.quality = quality
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.seed = seed
        self.extra_body = sanitize_openai_extra_body(model, extra_body)

    def _is_retryable(self, e: Exception) -> bool:
        msg = str(e).lower()
        if "rate limit" in msg or "429" in msg or "timeout" in msg:
            return True
        if "502" in msg or "503" in msg or "504" in msg:
            return True
        return False

    def generate(self, images: List[str], prompt: str, max_tokens_override: Optional[int] = None) -> Dict[str, Any]:
        # SiliconFlow's multimodal examples follow OpenAI's content schema with image_url parts.
        # Put images first then text to match the most common OpenAI-compatible VLM convention.
        content = []
        for p in images:
            mime, b64 = encode_image_base64(
                p,
                max_side=self.max_side,
                quality=self.quality,
                prefer_png=getattr(self, "prefer_png", True),
            )
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        content.append({"type": "text", "text": prompt})

        last_err = None
        for i in range(self.max_retries + 1):
            try:
                req_kwargs = {}
                if self.seed is not None:
                    req_kwargs["seed"] = self.seed

                # Token limit logic (Sanitized)
                token_keys = ("max_completion_tokens", "max_tokens")
                if not any(k in self.extra_body for k in token_keys):
                    budget = int(max_tokens_override) if max_tokens_override is not None else int(self.max_tokens)
                    if is_o_series_model(self.model):
                        req_kwargs["max_completion_tokens"] = budget
                    else:
                        req_kwargs["max_tokens"] = budget

                # Temperature logic (Sanitized)
                if "temperature" not in self.extra_body and (not is_o_series_model(self.model)):
                    req_kwargs["temperature"] = float(self.temperature)

                if self.extra_body:
                    req_kwargs["extra_body"] = self.extra_body

                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    **req_kwargs,
                )

                # Null-Safety
                if not getattr(resp, "choices", None):
                    return {"text": f"{API_ERROR_PREFIX} empty choices (gateway error)", "finish_reason": "error", "usage": None}

                ch0 = resp.choices[0]
                msg = getattr(ch0, "message", None)
                text = getattr(msg, "content", None) if msg else None
                finish_reason = getattr(ch0, "finish_reason", "unknown")
                usage_raw = getattr(resp, "usage", None)
                # Convert CompletionUsage to dict for JSON serialization
                usage = None
                if usage_raw:
                    usage = {
                        "prompt_tokens": getattr(usage_raw, "prompt_tokens", None),
                        "completion_tokens": getattr(usage_raw, "completion_tokens", None),
                        "total_tokens": getattr(usage_raw, "total_tokens", None),
                    }

                if not text:
                    return {"text": f"{API_ERROR_PREFIX} empty message.content (finish_reason={finish_reason})", "finish_reason": finish_reason, "usage": usage}
                return {"text": text, "finish_reason": finish_reason, "usage": usage}

            except Exception as e:
                last_err = e
                if i < self.max_retries and self._is_retryable(e):
                    time.sleep(self.backoff_base * (2**i))
                    continue
                return {"text": f"{API_ERROR_PREFIX} {str(e)}", "finish_reason": "error", "usage": None}
        return {"text": f"{API_ERROR_PREFIX} {str(last_err)}", "finish_reason": "error", "usage": None}


class GeminiRunner(Runner):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int,
        temperature: float,
        timeout: float,
        max_side: int,
        quality: int,
        max_retries: int,
        backoff_base: float,
        extra_body: Optional[Dict[str, Any]] = None,
    ):
        self.model = self._canon_model(model)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") + "/"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_side = max_side
        self.quality = quality
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.extra_body = extra_body or {}

    @staticmethod
    def _canon_model(m: str) -> str:
        m = (m or "").strip()
        if m.startswith("models/"):
            m = m[len("models/"):]
        if ":generatecontent" in m.lower():
            m = m.split(":", 1)[0]
        return m

    def _is_retryable(self, status: Optional[int], err_text: str) -> bool:
        t = (err_text or "").lower()
        if status in (429, 500, 502, 503, 504):
            return True
        if "rate" in t and "limit" in t:
            return True
        if "timeout" in t:
            return True
        if "connection" in t or "reset" in t:
            return True
        return False

    def generate(self, images: List[str], prompt: str, max_tokens_override: Optional[int] = None) -> str:
        import requests

        url = f"{self.base_url}v1beta/models/{self.model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "X-goog-api-key": self.api_key,
        }

        parts = []
        for p in images:
            b64 = encode_image_base64_jpeg(p, max_side=self.max_side, quality=self.quality)
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b64}})
        parts.append({"text": prompt})

        # Use max_tokens_override if provided, otherwise use default max_tokens
        max_tokens = max_tokens_override if max_tokens_override is not None else self.max_tokens

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "temperature": float(self.temperature),
                "maxOutputTokens": int(max_tokens),
            },
        }

        if self.extra_body:
            eb = dict(self.extra_body)
            # Integrity Check
            if "contents" in eb:
                raise ValueError("extra_body must NOT contain 'contents' (would override constructed prompt/images).")

            gc = eb.pop("generationConfig", None)
            if isinstance(gc, dict):
                payload["generationConfig"].update(gc)

            payload.update(eb)

        last_err = None
        for i in range(self.max_retries + 1):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if r.status_code >= 400:
                    last_err = f"HTTP {r.status_code}: {r.text}"
                    if i < self.max_retries and self._is_retryable(r.status_code, r.text):
                        time.sleep(self.backoff_base * (2**i))
                        continue
                    return f"{API_ERROR_PREFIX} {last_err}"

                data = r.json()
                cands = data.get("candidates") or []
                if not cands:
                    return f"{API_ERROR_PREFIX} empty candidates"

                content = (cands[0].get("content") or {})
                out_parts = content.get("parts") or []
                texts = [p["text"] for p in out_parts if isinstance(p, dict) and "text" in p]
                return "".join(texts).strip() if texts else f"{API_ERROR_PREFIX} empty text"
            except Exception as e:
                last_err = str(e)
                if i < self.max_retries and self._is_retryable(None, last_err):
                    time.sleep(self.backoff_base * (2**i))
                    continue
                return f"{API_ERROR_PREFIX} {last_err}"
        return f"{API_ERROR_PREFIX} {str(last_err)}"


class Qwen25VLRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info

        # Set up attention implementation
        attn_config = {}
        if use_flash_attention:
            attn_config["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            **attn_config
        )
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256*28*28, max_pixels=1280*28*28)
        self.process_vision_info = process_vision_info
        self.cfg = {"max_new_tokens": max_new_tokens}
        if temperature > 0: self.cfg.update({"do_sample": True, "temperature": temperature, "top_p": 0.9})
        else: self.cfg.update({"do_sample": False})

    def generate(self, images: List[str], prompt: str, max_tokens_override: Optional[int] = None) -> str:
        content = [{"type": "image", "image": p} for p in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Create generation config with potential token override
        gen_cfg = dict(self.cfg)
        if max_tokens_override is not None:
            gen_cfg["max_new_tokens"] = max_tokens_override

        with torch.no_grad():
            ids = self.model.generate(**inputs, **gen_cfg)
        return self.processor.batch_decode([out[len(inp):] for inp, out in zip(inputs["input_ids"], ids)], skip_special_tokens=True)[0]

class InternVLRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Set up attention implementation
        attn_config = {}
        if use_flash_attention:
            attn_config["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            **attn_config
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.cfg = {"max_new_tokens": max_new_tokens}
        if temperature > 0: self.cfg.update({"do_sample": True, "temperature": temperature})
        else: self.cfg.update({"do_sample": False})

    def generate(self, images: List[str], prompt: str, max_tokens_override: Optional[int] = None) -> str:
        # Handle multi-image scenarios for GridWM-Judge tasks
        if len(images) == 1:
            # Single image (Task B)
            img = safe_load_rgb(images[0])
        else:
            # Multi-image fusion (Task A: 5 images, Task C: storyboard)
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                # Smart fusion strategy based on image count
                if len(images) <= 3:
                    mode = "h"  # Horizontal fusion for few images (Task A)
                else:
                    mode = "v"  # Vertical fusion for many images (Task C)

                # Use high quality for fusion to preserve details
                fused_path = fuse_images(images, Path(tmp_path), mode, quality=95)
                img = safe_load_rgb(fused_path)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        # Resize image to square for InternVL compatibility (448x448 as per model config)
        img = img.resize((448, 448), Image.Resampling.LANCZOS)

        # Create generation config with potential token override
        gen_cfg = dict(self.cfg)
        if max_tokens_override is not None:
            gen_cfg["max_new_tokens"] = max_tokens_override

        # For InternVL, use direct generation instead of chat method
        try:
            import numpy as np
            import torch
            from transformers import AutoTokenizer

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model.config._name_or_path, trust_remote_code=True)

            # Prepare image
            img_array = np.array(img)
            pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            pixel_values = pixel_values.to(dtype=self.model.dtype, device=self.model.device)

            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Add pixel values to inputs
            inputs['pixel_values'] = pixel_values
            inputs['image_flags'] = torch.tensor([[1]], dtype=torch.long, device=self.model.device)

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_cfg)

            # Decode response
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            return response.strip()
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"InternVL generation failed: {str(e)}") from e

class LlavaRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        import numpy as np

        # Set up attention implementation
        attn_config = {}
        if use_flash_attention:
            attn_config["attn_implementation"] = "flash_attention_2"

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            **attn_config
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.np = np
        self.cfg = {"max_new_tokens": max_new_tokens}
        if temperature > 0: self.cfg.update({"do_sample": True, "temperature": temperature})
        else: self.cfg.update({"do_sample": False})

    def generate(self, images: List[str], prompt: str, max_tokens_override: Optional[int] = None) -> str:
        try:
            # For single image, use standard LLaVA format
            if len(images) == 1:
                image = safe_load_rgb(images[0])
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(text=text, images=image, return_tensors="pt")
            else:
                # For multiple images, concatenate them horizontally
                import tempfile
                import os
                from pathlib import Path

                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    fused_path = fuse_images(images, Path(tmp_path), "h", quality=95)
                    image = safe_load_rgb(fused_path)

                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self.processor(text=text, images=image, return_tensors="pt")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Remove image_sizes if present (compatibility issue)
            if 'image_sizes' in inputs:
                del inputs['image_sizes']

            # Create generation config with potential token override
            gen_cfg = dict(self.cfg)
            if max_tokens_override is not None:
                gen_cfg["max_new_tokens"] = max_tokens_override

            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_cfg)
            return self.processor.batch_decode(out, skip_special_tokens=True)[0]
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"LLaVA generation failed: {str(e)}") from e

class LlavaNextRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
        import numpy as np

        # Set up attention implementation for maximum speed
        attn_config = {}
        if use_flash_attention:
            try:
                # Check if flash_attn is available
                import flash_attn
                attn_config["attn_implementation"] = "flash_attention_2"
                print("🎯 Using Flash Attention 2 for accelerated inference", flush=True)
            except ImportError:
                print("⚠️ Flash Attention 2 not available, falling back to eager attention", flush=True)
                attn_config["attn_implementation"] = "eager"

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            **attn_config
        )
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.np = np

        # Optimized generation config
        self.cfg = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        # Remove None values
        self.cfg = {k: v for k, v in self.cfg.items() if v is not None}

        # Set batch processing capability
        self.max_batch_size = 4  # Conservative batch size for stability

    def _prepare_batch_inputs(self, batch_requests: List[Tuple[List[str], str, Optional[int]]]) -> Tuple[Dict, List]:
        """
        Prepare batched inputs for multiple requests.
        Returns: (batched_inputs, original_indices)
        """
        batch_images = []
        batch_texts = []
        batch_indices = []

        for idx, (images, prompt, _) in enumerate(batch_requests):
            # High-quality image processing
            if len(images) == 1:
                image = safe_load_rgb(images[0])
            else:
                # Lossless PNG fusion for multiple images
                import tempfile
                import os
                from pathlib import Path

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    fused_path = fuse_images(images, Path(tmp_path), "h", quality=100)
                    image = safe_load_rgb(fused_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            batch_images.append(image)
            batch_indices.append(idx)

            # Create conversation
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            batch_texts.append(text)

        # Batch process
        inputs = self.processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        return inputs, batch_indices

    def generate_batch(self, batch_requests: List[Tuple[List[str], str, Optional[int]]]) -> List[str]:
        """
        Generate responses for a batch of requests.
        Each request is (images, prompt, max_tokens_override)
        """
        if len(batch_requests) == 0:
            return []

        try:
            # Prepare batched inputs
            inputs, batch_indices = self._prepare_batch_inputs(batch_requests)

            # Get max tokens for the batch
            max_tokens = max((req[2] if req[2] is not None else self.cfg["max_new_tokens"])
                           for req in batch_requests)

            # Create generation config
            gen_cfg = dict(self.cfg)
            gen_cfg["max_new_tokens"] = max_tokens

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_cfg)

            # Decode each response
            responses = []
            for i in range(len(batch_requests)):
                response = self.processor.decode(outputs[i], skip_special_tokens=True)
                responses.append(response.strip())

            return responses

        except Exception as e:
            # Fallback to individual processing if batch fails
            print(f"⚠️ Batch processing failed ({e}), falling back to individual processing", flush=True)
            return [self.generate(images, prompt, max_tokens) for images, prompt, max_tokens in batch_requests]

    def generate(self, images: List[str], prompt: str, max_tokens_override: Optional[int] = None) -> str:
        try:
            # High-quality image processing (no lossy compression)
            if len(images) == 1:
                # Load image with original quality and size
                image = safe_load_rgb(images[0])
                # Keep original size for maximum quality (LLaVA-NeXT handles resizing internally)
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = self.processor(text=text, images=image, return_tensors="pt")
            else:
                # For multiple images, use PNG for lossless fusion
                import tempfile
                import os
                from pathlib import Path

                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    # Use PNG quality=100 for lossless fusion
                    fused_path = fuse_images(images, Path(tmp_path), "h", quality=100)
                    image = safe_load_rgb(fused_path)

                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self.processor(text=text, images=image, return_tensors="pt")
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Create generation config with potential token override
            gen_cfg = dict(self.cfg)
            if max_tokens_override is not None:
                gen_cfg["max_new_tokens"] = max_tokens_override

            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_cfg)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"LLaVA-NeXT generation failed: {str(e)}") from e

class LlavaNextVideoRunner(Runner):
    def __init__(self, model_id, device_map, dtype, max_new_tokens, temperature, use_flash_attention=False):
        from transformers import LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor
        import numpy as np

        # Set up attention implementation
        attn_config = {}
        if use_flash_attention:
            attn_config["attn_implementation"] = "flash_attention_2"

        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            **attn_config
        )
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        self.np = np
        self.cfg = {"max_new_tokens": max_new_tokens}
        if temperature > 0: self.cfg.update({"do_sample": True, "temperature": temperature})
        else: self.cfg.update({"do_sample": False})

    def generate(self, images: List[str], prompt: str, max_tokens_override: Optional[int] = None) -> str:
        frames = [self.np.array(safe_load_rgb(p)) for p in images]
        video = self.np.stack(frames, axis=0)
        conversation = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(text=text, videos=video, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Create generation config with potential token override
        gen_cfg = dict(self.cfg)
        if max_tokens_override is not None:
            gen_cfg["max_new_tokens"] = max_tokens_override

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_cfg)
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]

def resolve_openai_provider(provider, base_url, api_key):
    if provider == "openai":
        key = api_key or _pick_env("OPENAI_API_KEY")
        return (base_url or "https://api.openai.com/v1"), _need("OPENAI_API_KEY", key)
    if provider == "siliconflow":
        key = api_key or _pick_env("SILICONFLOW_API_KEY")
        # SiliconFlow provides an OpenAI-compatible Chat Completions endpoint.
        # Default CN endpoint: https://api.siliconflow.cn/v1
        # Allow overrides via --base_url or env SILICONFLOW_BASE_URL (useful for non-CN endpoints).
        sf_base = base_url or _pick_env("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"
        return sf_base.rstrip("/"), _need("SILICONFLOW_API_KEY", key)
    if provider == "closeai":
        key = api_key or _pick_env("CLOSEAI_API_KEY")
        return "https://api.openai-proxy.org/v1", _need("CLOSEAI_API_KEY", key)
    if provider == "zhizengzeng_qwen":
        # Zhizengzeng Qwen VL models - uses Alibaba base URL
        key = api_key or _pick_env("ZZZ_API_KEY") or _pick_env("ZHIZENGZENG_API_KEY")
        return "https://api.zhizengzeng.com/alibaba", _need("ZZZ_API_KEY or ZHIZENGZENG_API_KEY", key)
    if provider.startswith("zhizengzeng"):
        # Unified Zhizengzeng endpoint - all models via OpenAI-compatible interface
        # Support both official ZZZ_API_KEY and legacy ZHIZENGZENG_API_KEY for compatibility
        key = api_key or _pick_env("ZZZ_API_KEY") or _pick_env("ZHIZENGZENG_API_KEY")
        return "https://api.zhizengzeng.com/v1", _need("ZZZ_API_KEY or ZHIZENGZENG_API_KEY", key)
    if provider == "custom":
        if not base_url: raise ValueError("provider=custom requires --base_url")
        key = api_key or _pick_env("OPENAI_API_KEY")
        return base_url, _need("custom api_key (or OPENAI_API_KEY)", key)
    return base_url, api_key

def resolve_gemini_provider(provider, base_url, api_key):
    if provider == "gemini":
        key = api_key or _pick_env("GEMINI_API_KEY")
        return (base_url or "https://generativelanguage.googleapis.com/"), _need("GEMINI_API_KEY", key)
    if provider == "zhizengzeng_gemini":
        # Zhizengzeng Gemini API uses Google-compatible format but different base_url
        # Official Google: https://generativelanguage.googleapis.com/
        # Zhizengzeng: https://api.zhizengzeng.com/google/
        key = api_key or _pick_env("ZZZ_API_KEY") or _pick_env("ZHIZENGZENG_GEMINI_API_KEY")
        return "https://api.zhizengzeng.com/google/", _need("ZZZ_API_KEY or ZHIZENGZENG_GEMINI_API_KEY", key)
    if provider == "custom":
        if not base_url: raise ValueError("provider=custom requires --base_url")
        key = api_key or _pick_env("GEMINI_API_KEY")
        return base_url, _need("custom api_key (or GEMINI_API_KEY)", key)
    return base_url, api_key

def run(args):
    validate_backend_provider(args.backend, args.provider)

    # Resolve requests path (support exam mode)
    if hasattr(args, 'exam_dir') and args.exam_dir:
        gen_dir = Path(args.responses_dir) / "_requests_from_exam"
        out_files = generate_requests_from_exam(Path(args.exam_dir), gen_dir, getattr(args, 'exam_task', 'all'))
        exam_task = getattr(args, 'exam_task', 'all')
        if exam_task == "all":
            combined = gen_dir / "requests_exam_all.jsonl"
            with combined.open("w") as fout:
                for f in out_files: shutil.copyfileobj(f.open("r"), fout)
            req_path = combined
        else: req_path = out_files[0]
        if getattr(args, 'build_exam_requests_only', False): return
    else:
        req_path = Path(args.requests)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    extra_body = {}
    if args.extra_body:
        with open(args.extra_body, "r", encoding="utf-8") as f:
            extra_body = json.load(f)
        # Deep secure masking
        print(f"[info] Loaded extra_body (masked): {_mask_any(extra_body)}", flush=True)

    protocol_tag = f"fused_{args.force_fuse}" if args.force_fuse else "native"
    model_tag = args.model_tag or args.model.replace("/", "_").replace(":", "_")[:50]  # Limit length
    backend_tag = args.backend.replace("_", "").replace("-", "")
    provider_tag = args.provider.replace("_", "").replace("-", "")

    # Create experiment identifier based on key parameters
    experiment_id = f"{backend_tag}_{provider_tag}_{model_tag}"
    if args.num_shards > 1:
        experiment_id += f"_shard{args.shard_id}"
    if args.force_fuse:
        experiment_id += f"_{protocol_tag}"

    resp_dir = Path(args.responses_dir) / experiment_id
    resp_dir.mkdir(parents=True, exist_ok=True)
    resp_path = resp_dir / f"{req_path.stem}.jsonl"

    dtype = None

    if args.backend == "openai_compatible":
        url, key = resolve_openai_provider(args.provider, args.base_url, args.api_key)
        runner = OpenAICompatibleRunner(
            model=args.model,
            api_key=key,
            base_url=url,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout=args.api_timeout,
            max_side=args.api_image_max_side,
            quality=args.api_jpeg_quality,
            max_retries=args.api_max_retries,
            backoff_base=args.api_backoff_base,
            seed=args.seed,
            extra_body=extra_body,
        )
        # Set PNG preference for image fidelity
        if hasattr(runner, "__dict__"):
            runner.prefer_png = bool(getattr(args, "api_prefer_png", True))

    elif args.backend == "gemini":
        url, key = resolve_gemini_provider(args.provider, args.base_url, args.api_key)
        runner = GeminiRunner(
            model=args.model,
            api_key=key,
            base_url=url,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout=args.api_timeout,
            max_side=args.api_image_max_side,
            quality=args.api_jpeg_quality,
            max_retries=args.api_max_retries,
            backoff_base=args.api_backoff_base,
            extra_body=extra_body,
        )

    elif args.backend == "qwen2.5-vl":
        _require_torch()
        dtype = parse_dtype(args.dtype)
        runner = Qwen25VLRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    elif args.backend == "internvl2.5":
        _require_torch()
        dtype = parse_dtype(args.dtype)
        runner = InternVLRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    elif args.backend == "llava-next-video":
        _require_torch()
        dtype = parse_dtype(args.dtype)
        runner = LlavaNextVideoRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    elif args.backend == "llava":
        _require_torch()
        dtype = parse_dtype(args.dtype)
        runner = LlavaRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    elif args.backend == "llava-next":
        _require_torch()
        dtype = parse_dtype(args.dtype)
        runner = LlavaNextRunner(args.model, args.device_map, dtype, args.max_new_tokens, args.temperature, args.use_flash_attention)

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # -------------------------
    # Progress configuration
    # -------------------------
    task_overrides = {"A": args.max_tokens_A, "B": args.max_tokens_B, "C": args.max_tokens_C}
    t_start = time.time()
    n_err = 0

    processed = set()
    if args.resume and resp_path.exists():
        for r in load_jsonl(resp_path): processed.add(r["uid"])
        print(f"🔄 Resuming {resp_path}: {len(processed)} done.", flush=True)

    # Optional total counting for better ETA (fast enough for ~1-10k lines)
    args._total_requests = 0
    if args.progress_total:
        try:
            with req_path.open("r", encoding="utf-8") as fcnt:
                for _ in fcnt:
                    args._total_requests += 1
        except Exception:
            args._total_requests = 0

    n_do = 0
    # Progress tracking
    start_time = time.time()
    last_progress_time = start_time
    processed_count = len(processed)
    total_processed = processed_count
    last_progress_count = processed_count  # Track items since last progress report

    print(f"🔄 Starting inference on shard {args.shard_id} (resume: {processed_count} already done)", flush=True)

    # Check if runner supports batch processing
    use_batch = hasattr(runner, 'generate_batch') and hasattr(runner, 'max_batch_size')

    with req_path.open("r", encoding="utf-8") as fin, resp_path.open("a", encoding="utf-8") as fout:
        if use_batch:
            # Batch processing mode
            batch_requests = []
            batch_metadata = []

            for line in fin:
                if not line.strip(): continue
                q = json.loads(line)
                uid = q["uid"]

                if args.num_shards > 1:
                    if (stable_hash64(uid) % args.num_shards) != args.shard_id: continue
                if uid in processed: continue

                images = q["images"]
                prompt = q["prompt"]
                fused = False

                # P0-2: Hard Protocol Enforcement & Prompt Injection
                if args.force_fuse:
                    if len(images) <= 1:
                        raise ValueError(f"Protocol Violation: --force_fuse set but found single image for {uid}. Task C Storyboard should run WITHOUT force_fuse.")

                    fused = True
                    fuse_out = tmp_dir / f"{uid.replace(':','_').replace('/','_')}.jpg"
                    images = [fuse_images(images, fuse_out, args.force_fuse, args.api_jpeg_quality)]

                    # Dynamic Prompt Replacement
                    if args.force_fuse == "h" and q.get("prompt_fused_h"):
                        prompt = q["prompt_fused_h"]
                    elif args.force_fuse == "v" and q.get("prompt_fused_v"):
                        prompt = q["prompt_fused_v"]
                    else:
                        layout = "horizontal" if args.force_fuse == "h" else "vertical"
                        prompt = f"NOTE: The provided image is a {layout} concatenation.\n\n{prompt}"

                budget = _task_max_tokens(uid, args.max_new_tokens, task_overrides)
                batch_requests.append((images, prompt, budget))
                batch_metadata.append((q, fused))

                # Process batch when full or at end
                if len(batch_requests) >= runner.max_batch_size:
                    t0 = time.time()
                    try:
                        results = runner.generate_batch(batch_requests)
                        for i, result in enumerate(results):
                            q, fused = batch_metadata[i]
                            uid = q["uid"]

                            if isinstance(result, dict):
                                pred = result["text"]
                                finish_reason = result.get("finish_reason")
                                usage = result.get("usage")
                            else:
                                pred = result
                                finish_reason = None
                                usage = None

                            err = None
                            ok = True

                            # Write result
                            fout.write(
                                json.dumps(
                                    {
                                        "uid": uid,
                                        "pred": pred,
                                        "raw": pred,
                                        "meta": {
                                            "ok": ok,
                                            "error": err,
                                            "shard": args.shard_id,
                                            "latency_ms": (time.time() - t0) * 1000.0 / len(batch_requests),
                                            "fused": fused,
                                            "protocol": protocol_tag,
                                            "backend": args.backend,
                                            "provider": args.provider,
                                            "model": args.model,
                                            "finish_reason": finish_reason,
                                            "usage": usage,
                                            "exam": q.get("exam"),
                                        },
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            fout.flush()
                            n_do += 1
                            total_processed += 1

                    except Exception as e:
                        # Fallback to individual processing for failed batch
                        pass

                    # Progress reporting for batch processing
                    current_time = time.time()
                    items_since_last_report = total_processed - last_progress_count
                    time_since_last_report = current_time - last_progress_time

                    should_report = False
                    if args.progress_interval_s > 0 and time_since_last_report >= args.progress_interval_s:
                        should_report = True
                    if args.progress_every > 0 and items_since_last_report >= args.progress_every:
                        should_report = True

                    if should_report:
                        elapsed = current_time - start_time
                        rate = total_processed / elapsed if elapsed > 0 else 0
                        print(f"[progress] shard {args.shard_id}: {total_processed} done, {rate:.2f}/sec, elapsed: {elapsed:.1f}s", flush=True)
                        last_progress_time = current_time
                        last_progress_count = total_processed
                        print(f"⚠️ Batch processing failed, falling back to individual processing", flush=True)
                        for i, (images, prompt, budget) in enumerate(batch_requests):
                            q, fused = batch_metadata[i]
                            uid = q["uid"]
                            t0 = time.time()
                            err = None
                            try:
                                result = runner.generate(images, prompt, max_tokens_override=budget)
                                if isinstance(result, dict):
                                    pred = result["text"]
                                    finish_reason = result.get("finish_reason")
                                    usage = result.get("usage")
                                else:
                                    pred = result
                                    finish_reason = None
                                    usage = None
                            except Exception as e:
                                pred = f"{API_ERROR_PREFIX} {str(e)}"
                                err = str(e)
                                finish_reason = "error"
                                usage = None

                            ok = (err is None)
                            raw = pred
                            if not ok:
                                pred = ""

                            fout.write(
                                json.dumps(
                                    {
                                        "uid": uid,
                                        "pred": pred,
                                        "raw": raw,
                                        "meta": {
                                            "ok": ok,
                                            "error": err,
                                            "shard": args.shard_id,
                                            "latency_ms": (time.time() - t0) * 1000.0,
                                            "fused": fused,
                                            "protocol": protocol_tag,
                                            "backend": args.backend,
                                            "provider": args.provider,
                                            "model": args.model,
                                            "finish_reason": finish_reason,
                                            "usage": usage,
                                            "exam": q.get("exam"),
                                        },
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            fout.flush()
                            n_do += 1
                            total_processed += 1

                    batch_requests = []
                    batch_metadata = []

            # Process remaining batch
            if batch_requests:
                t0 = time.time()
                try:
                    results = runner.generate_batch(batch_requests)
                    for i, result in enumerate(results):
                        q, fused = batch_metadata[i]
                        uid = q["uid"]

                        if isinstance(result, dict):
                            pred = result["text"]
                            finish_reason = result.get("finish_reason")
                            usage = result.get("usage")
                        else:
                            pred = result
                            finish_reason = None
                            usage = None

                        err = None
                        ok = True

                        fout.write(
                            json.dumps(
                                {
                                    "uid": uid,
                                    "pred": pred,
                                    "raw": pred,
                                    "meta": {
                                        "ok": ok,
                                        "error": err,
                                        "shard": args.shard_id,
                                        "latency_ms": (time.time() - t0) * 1000.0 / len(batch_requests),
                                        "fused": fused,
                                        "protocol": protocol_tag,
                                        "backend": args.backend,
                                        "provider": args.provider,
                                        "model": args.model,
                                        "finish_reason": finish_reason,
                                        "usage": usage,
                                        "exam": q.get("exam"),
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        fout.flush()
                        n_do += 1
                        total_processed += 1

                except Exception as e:
                    # Fallback to individual processing
                    print(f"⚠️ Final batch processing failed, falling back to individual processing", flush=True)

                # Progress reporting for batch mode
                current_time = time.time()
                items_since_last_report = total_processed - last_progress_count
                time_since_last_report = current_time - last_progress_time

                should_report = False
                if args.progress_interval_s > 0 and time_since_last_report >= args.progress_interval_s:
                    should_report = True
                if args.progress_every > 0 and items_since_last_report >= args.progress_every:
                    should_report = True

                if should_report:
                    elapsed = current_time - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"[progress] shard {args.shard_id}: {total_processed} done, {rate:.2f}/sec, elapsed: {elapsed:.1f}s", flush=True)
                    last_progress_time = current_time
                    last_progress_count = total_processed
                    for i, (images, prompt, budget) in enumerate(batch_requests):
                        q, fused = batch_metadata[i]
                        uid = q["uid"]
                        t0 = time.time()
                        err = None
                        try:
                            result = runner.generate(images, prompt, max_tokens_override=budget)
                            if isinstance(result, dict):
                                pred = result["text"]
                                finish_reason = result.get("finish_reason")
                                usage = result.get("usage")
                            else:
                                pred = result
                                finish_reason = None
                                usage = None
                        except Exception as e:
                            pred = f"{API_ERROR_PREFIX} {str(e)}"
                            err = str(e)
                            finish_reason = "error"
                            usage = None

                        ok = (err is None)
                        raw = pred
                        if not ok:
                            pred = ""

                        fout.write(
                            json.dumps(
                                {
                                    "uid": uid,
                                    "pred": pred,
                                    "raw": raw,
                                    "meta": {
                                        "ok": ok,
                                        "error": err,
                                        "shard": args.shard_id,
                                        "latency_ms": (time.time() - t0) * 1000.0,
                                        "fused": fused,
                                        "protocol": protocol_tag,
                                        "backend": args.backend,
                                        "provider": args.provider,
                                        "model": args.model,
                                        "finish_reason": finish_reason,
                                        "usage": usage,
                                        "exam": q.get("exam"),
                                    },
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        fout.flush()
                        n_do += 1
                        total_processed += 1
        else:
            # Original single-item processing mode
            for line in fin:
                if not line.strip(): continue
                q = json.loads(line)
                uid = q["uid"]

                if args.num_shards > 1:
                    if (stable_hash64(uid) % args.num_shards) != args.shard_id: continue
                if uid in processed: continue

                images = q["images"]
                prompt = q["prompt"]
                fused = False

                # P0-2: Hard Protocol Enforcement & Prompt Injection
                if args.force_fuse:
                    if len(images) <= 1:
                        # Single-image + --force_fuse is a protocol violation; fail fast to avoid contaminating results.
                        raise ValueError(f"Protocol Violation: --force_fuse set but found single image for {uid}. Task C Storyboard should run WITHOUT force_fuse.")

                    fused = True
                    fuse_out = tmp_dir / f"{uid.replace(':','_').replace('/','_')}.jpg"
                    images = [fuse_images(images, fuse_out, args.force_fuse, args.api_jpeg_quality)]

                    # Dynamic Prompt Replacement
                    if args.force_fuse == "h" and q.get("prompt_fused_h"):
                        prompt = q["prompt_fused_h"]
                    elif args.force_fuse == "v" and q.get("prompt_fused_v"):
                        prompt = q["prompt_fused_v"]
                    else:
                        layout = "horizontal" if args.force_fuse == "h" else "vertical"
                        prompt = f"NOTE: The provided image is a {layout} concatenation.\n\n{prompt}"

                if args.min_interval_ms > 0 and n_do > 0:
                    time.sleep(args.min_interval_ms / 1000.0)

                t0 = time.time()
                err = None
                try:
                    # Task-aware budget to avoid truncation -> fake JSON failures
                    budget = _task_max_tokens(uid, args.max_new_tokens, task_overrides)
                    result = runner.generate(images, prompt, max_tokens_override=budget)
                    if isinstance(result, dict):
                        pred = result["text"]
                        finish_reason = result.get("finish_reason")
                        usage = result.get("usage")
                    else:
                        # Fallback for backward compatibility
                        pred = result
                        finish_reason = None
                        usage = None
                except Exception as e:
                    pred = f"{API_ERROR_PREFIX} {str(e)}"
                    err = str(e)
                    finish_reason = "error"
                    usage = None

                ok = (err is None)
                raw = pred
                if not ok:
                    pred = ""

                fout.write(
                    json.dumps(
                        {
                            "uid": uid,
                            "pred": pred,
                            "raw": raw,
                            "meta": {
                                "ok": ok,
                                "error": err,
                                "shard": args.shard_id,
                                "latency_ms": (time.time() - t0) * 1000.0,
                                "fused": fused,
                                "protocol": protocol_tag,
                                "backend": args.backend,
                                "provider": args.provider,
                                "model": args.model,
                                "finish_reason": finish_reason,
                                "usage": usage,
                                "exam": q.get("exam"),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fout.flush()
                n_do += 1
                total_processed += 1

                # Progress reporting - check both time interval and item count
                current_time = time.time()
                items_since_last_report = total_processed - last_progress_count
                time_since_last_report = current_time - last_progress_time

                should_report = False
                if args.progress_interval_s > 0 and time_since_last_report >= args.progress_interval_s:
                    should_report = True
                if args.progress_every > 0 and items_since_last_report >= args.progress_every:
                    should_report = True

                if should_report:
                    elapsed = current_time - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    print(f"[progress] shard {args.shard_id}: {total_processed} done, {rate:.2f}/sec, elapsed: {elapsed:.1f}s", flush=True)
                    last_progress_time = current_time
                    last_progress_count = total_processed

            images = q["images"]
            prompt = q["prompt"]
            fused = False

            # P0-2: Hard Protocol Enforcement & Prompt Injection
            if args.force_fuse:
                if len(images) <= 1:
                    # Single-image + --force_fuse is a protocol violation; fail fast to avoid contaminating results.
                    raise ValueError(f"Protocol Violation: --force_fuse set but found single image for {uid}. Task C Storyboard should run WITHOUT force_fuse.")

                fused = True
                fuse_out = tmp_dir / f"{uid.replace(':','_').replace('/','_')}.jpg"
                images = [fuse_images(images, fuse_out, args.force_fuse, args.api_jpeg_quality)]

                # Dynamic Prompt Replacement
                if args.force_fuse == "h" and q.get("prompt_fused_h"):
                    prompt = q["prompt_fused_h"]
                elif args.force_fuse == "v" and q.get("prompt_fused_v"):
                    prompt = q["prompt_fused_v"]
                else:
                    layout = "horizontal" if args.force_fuse == "h" else "vertical"
                    prompt = f"NOTE: The provided image is a {layout} concatenation.\n\n{prompt}"

            if args.min_interval_ms > 0 and n_do > 0:
                time.sleep(args.min_interval_ms / 1000.0)

            t0 = time.time()
            err = None
            try:
                # Task-aware budget to avoid truncation -> fake JSON failures
                budget = _task_max_tokens(uid, args.max_new_tokens, task_overrides)
                result = runner.generate(images, prompt, max_tokens_override=budget)
                if isinstance(result, dict):
                    pred = result["text"]
                    finish_reason = result.get("finish_reason")
                    usage = result.get("usage")
                else:
                    # Fallback for backward compatibility
                    pred = result
                    finish_reason = None
                    usage = None
            except Exception as e:
                pred = f"{API_ERROR_PREFIX} {str(e)}"
                err = str(e)
                finish_reason = "error"
                usage = None

            # API_ERROR Propagation (Gap C & Sentinel)
            # Use dedicated sentinel constant for reliability
            if isinstance(pred, str) and pred.startswith(API_ERROR_PREFIX) and err is None:
                err = pred

            # Data Integrity (Dataset Poisoning Protection)
            ok = (err is None)
            raw = pred
            if not ok:
                pred = ""  # Prevent poisoning
                n_err += 1

            fout.write(
                json.dumps(
                    {
                        "uid": uid,
                        "pred": pred,
                        "raw": raw,
                        "meta": {
                            "ok": ok,
                            "error": err,
                            "shard": args.shard_id,
                            "latency_ms": (time.time() - t0) * 1000.0,
                            "fused": fused,
                            "protocol": protocol_tag,
                            "backend": args.backend,
                            "provider": args.provider,
                            "model": args.model,
                            "finish_reason": finish_reason,
                            "usage": usage,
                            "exam": q.get("exam"),  # Exam metadata resurrection
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            fout.flush()
            n_do += 1
            total_processed += 1

            # Progress reporting - check both time interval and item count
            current_time = time.time()
            items_since_last_report = total_processed - last_progress_count
            time_since_last_report = current_time - last_progress_time

            should_report = False
            if args.progress_interval_s > 0 and time_since_last_report >= args.progress_interval_s:
                should_report = True
            if args.progress_every > 0 and items_since_last_report >= args.progress_every:
                should_report = True

            if should_report:
                elapsed = current_time - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(f"[progress] shard {args.shard_id}: {total_processed} done, {rate:.2f}/sec, elapsed: {elapsed:.1f}s", flush=True)
                last_progress_time = current_time
                last_progress_count = total_processed

    # Final progress report
    final_time = time.time()
    total_elapsed = final_time - start_time
    final_rate = total_processed / total_elapsed if total_elapsed > 0 else 0
    print(f"[OK] shard {args.shard_id} completed: {total_processed} total, {final_rate:.2f}/sec, total time: {total_elapsed:.1f}s", flush=True)


# -------------------------
# Utilities
# -------------------------

def stable_hash64(s: str) -> int:
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little")

def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f: json.dump(obj, f, indent=2)

def parse_dtype(s: str):
    if s in (None, "", "auto"): return "auto"
    s = s.lower()
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    if s in ("fp16", "float16"): return torch.float16
    if s in ("fp32", "float32"): return torch.float32
    raise ValueError(f"Unknown dtype: {s}")

def safe_load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB").copy()

def encode_image_base64_jpeg(path: str, max_side: int, quality: int) -> str:
    img = safe_load_rgb(path)
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def fuse_images(paths: List[str], out_path: Path, mode: str, quality: int) -> str:
    imgs = [safe_load_rgb(p) for p in paths]
    ws, hs = zip(*[im.size for im in imgs])
    if mode == "h":
        W, H = sum(ws), max(hs)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        x = 0
        for im in imgs:
            canvas.paste(im, (x, 0))
            x += im.size[0]
    else:
        W, H = max(ws), sum(hs)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        y = 0
        for im in imgs:
            canvas.paste(im, (0, y))
            y += im.size[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=quality)
    return str(out_path)

def _need(name: str, v: Optional[str]) -> str:
    if not v: raise ValueError(f"Missing {name}")
    return v

def _pick_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v: return v
    return None

def _mask_any(x: Any) -> Any:
    SECRET_EXACT = {"api_key", "apikey", "access_token", "authorization", "auth", "secret", "password"}
    def is_secret_key(key_str: str) -> bool:
        lk = key_str.lower().replace("_", "").replace("-", "")
        return lk in SECRET_EXACT or lk.endswith("token") or lk.endswith("key")

    if isinstance(x, dict):
        return {k: ("***MASKED***" if is_secret_key(str(k)) else _mask_any(v)) for k, v in x.items()}
    if isinstance(x, list): return [_mask_any(v) for v in x]
    return x

# -------------------------
# Exam helpers (Metadata Resurrection)
# -------------------------

def parse_exam_id(uid: str) -> Dict[str, Any]:
    """
    Parse the canonical exam UID format:
      Task A/B: <Task>.<env_task>.<group_id>.t<step>
      Task C:   C.<env_task>.<group_id>.<variant>.<temporal>[.<visual>]
    """
    out = {}
    if not uid: return out
    parts = uid.split(".")
    if len(parts) < 2: return out

    task = parts[0]
    out["task"] = task
    if task in ("A", "B") and len(parts) >= 4:
        out["env_task"], out["group_id"] = parts[1], parts[2]
        tpart = parts[3]
        if tpart.startswith("t") and tpart[1:].isdigit(): out["t"] = int(tpart[1:])
    elif task == "C" and len(parts) >= 5:
        out["env_task"], out["group_id"] = parts[1], parts[2]
        out["variant"], out["temporal"] = parts[3], parts[4]
        if len(parts) >= 6:
            out["visual"] = parts[5]
    return out

# -------------------------
# Exam -> Requests Logic
# -------------------------

def generate_requests_from_exam(exam_dir: Path, out_requests_dir: Path, exam_task: str = "all", image_path_mode: str = "absolute") -> List[Path]:
    exam_dir = exam_dir.resolve()
    out_requests_dir.mkdir(parents=True, exist_ok=True)
    wanted = {"A", "B", "C"} if exam_task == "all" else {exam_task}
    telemetry = {"exam_dir": str(exam_dir), "tasks": sorted(list(wanted)), "counts": {}, "skips": defaultdict(int)}

    def img_full_path(rel: str) -> str:
        p = (exam_dir / rel).resolve()
        return str(p) if image_path_mode == "absolute" else str(p.relative_to(exam_dir))

    generated_files = []

    # Task A
    if "A" in wanted:
        reqs = []
        for it in load_jsonl(exam_dir / "task_a_exam.jsonl") if (exam_dir / "task_a_exam.jsonl").exists() else []:
            uid = it.get("exam_id")
            rel = it.get("image")
            if not uid or not rel: continue
            info = parse_exam_id(uid)
            prompt = (it.get("prompt", "") + "\nAnswer with ONLY the correct letter (A/B/C/D).").strip()
            reqs.append({"uid": uid, "images": [img_full_path(rel)], "prompt": prompt, "exam": info})

        if reqs:
            out = out_requests_dir / "requests_A.jsonl"
            with out.open("w") as f: [f.write(json.dumps(r)+"\n") for r in reqs]
            generated_files.append(out); telemetry["counts"]["A"] = len(reqs)

    # Task B (Structure JSON)
    if "B" in wanted:
        reqs = []
        b_prompt = """Return ONLY a JSON object with agent, front_cell, and objects keys.

GRID: 23x22 (x=0-22, y=0-21). [0,0] top-left, x→right, y↓down. Agent POV: faces UP in image.
Directions: 0=north↑, 1=east→, 2=south↓, 3=west←.

VISUAL DIRECTION CUES:
- If agent faces WALL/DOOR ahead: likely dir=0 (north) or dir=2 (south)
- If agent faces OPEN space ahead: check surroundings to determine direction
- Count grid cells: 23 across × 22 down
- Agent position anywhere in [0-22, 0-21]

Direction→front_cell: dir=0→[x,y-1]; dir=1→[x+1,y]; dir=2→[x,y+1]; dir=3→[x-1,y].

IMPORTANT: Return ONLY raw JSON, no markdown, no code blocks, no explanations.

Example 1 (agent facing south):
{
  "agent": {"pos": [5, 8], "dir": 2, "carrying": null},
  "front_cell": {"pos": [5, 9], "type": "floor", "state": 0},
  "objects": [
    {"type": "door", "pos": [3, 2], "color": "red", "state": 1},
    {"type": "goal", "pos": [7, 10], "color": "green", "state": 0}
  ]
}

Example 2 (agent facing east):
{
  "agent": {"pos": [12, 4], "dir": 1, "carrying": {"type": "key", "color": "blue"}},
  "front_cell": {"pos": [13, 4], "type": "wall", "state": 0},
  "objects": [
    {"type": "door", "pos": [8, 6], "color": "yellow", "state": 0},
    {"type": "key", "pos": [2, 9], "color": "red", "state": 0}
  ]
}"""
        for it in load_jsonl(exam_dir / "task_b_exam.jsonl") if (exam_dir / "task_b_exam.jsonl").exists() else []:
            uid = it.get("exam_id")
            rel = it.get("image")
            if not uid or not rel: continue
            info = parse_exam_id(uid)
            reqs.append({"uid": uid, "images": [img_full_path(rel)], "prompt": b_prompt, "exam": info})

        if reqs:
            out = out_requests_dir / "requests_B.jsonl"
            with out.open("w") as f: [f.write(json.dumps(r)+"\n") for r in reqs]
            generated_files.append(out); telemetry["counts"]["B"] = len(reqs)

    # Task C ( storyboard Success/Fail )
    if "C" in wanted:
        reqs = []
        for it in load_jsonl(exam_dir / "task_c_exam.jsonl") if (exam_dir / "task_c_exam.jsonl").exists() else []:
            uid = it.get("exam_id")
            rel = it.get("image")
            if not uid or not rel: continue
            info = parse_exam_id(uid)
            prompt = (it.get("prompt", "Evaluate success.") + "\nAnswer with ONLY: Success or Fail (no explanations).").strip()
            reqs.append({"uid": uid, "images": [img_full_path(rel)], "prompt": prompt, "exam": info})

        if reqs:
            out = out_requests_dir / "requests_C.jsonl"
            with out.open("w") as f: [f.write(json.dumps(r)+"\n") for r in reqs]
            generated_files.append(out); telemetry["counts"]["C"] = len(reqs)

    write_json(out_requests_dir / "requests_gen_manifest.json", telemetry)
    return generated_files

# -------------------------
# Provider Resolvers (CRITICAL FIX)
# -------------------------

# -------------------------
# Runners (OpenAI/Gemini/Local)
# -------------------------

# -------------------------
# Main Run Loop
# -------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Universal VLM Inference Gateway")

    # Core paths
    p.add_argument("--exam_dir", help="Path to exam directory (enables exam mode)")
    p.add_argument("--exam_task", default="all", choices=["all", "A", "B", "C"], help="Which exam tasks to process")
    p.add_argument("--requests", help="Path to requests JSONL file (alternative to exam_dir)")
    p.add_argument("--responses_dir", required=True, help="Directory to save responses")

    # Backend configuration
    p.add_argument("--backend", default="openai_compatible", choices=["openai_compatible", "gemini", "qwen2.5-vl", "internvl2.5", "llava-next-video", "llava", "llava-next"])
    p.add_argument("--provider", default="siliconflow", help="Provider for the backend (openai, siliconflow, zhizengzeng*, custom, etc.)")
    p.add_argument("--model", required=True, help="Model identifier")
    p.add_argument("--api_key", help="API key (can also use environment variables)")

    # Generation parameters
    p.add_argument("--max_new_tokens", type=int, default=512, help="Default max tokens")
    p.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")

    # Task-aware token budgets (capability preservation)
    p.add_argument("--max_tokens_A", type=int, default=512, help="Max tokens for Task A")
    p.add_argument("--max_tokens_B", type=int, default=2048, help="Max tokens for Task B (JSON)")
    p.add_argument("--max_tokens_C", type=int, default=256, help="Max tokens for Task C")

    # Image fidelity
    p.add_argument("--api_prefer_png", action="store_true", help="Prefer PNG for images (recommended)")
    p.add_argument("--api_image_max_side", type=int, default=2048, help="Max image side (<=0 disables resize)")
    p.add_argument("--api_jpeg_quality", type=int, default=85, help="JPEG quality when using JPEG")

    # Progress and logging
    p.add_argument("--progress_every", type=int, default=50, help="Progress update every N items")
    p.add_argument("--progress_interval_s", type=float, default=10.0, help="Heartbeat interval seconds")
    p.add_argument("--progress_show_uid", action="store_true", help="Show current UID in progress")
    p.add_argument("--progress_total", action="store_true", help="Count total items for ETA")

    # Exam mode specific
    p.add_argument("--build_exam_requests_only", action="store_true", help="Only generate requests, don't run inference")

    # Other options
    p.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    p.add_argument("--shard_id", type=int, default=0, help="Current shard ID")
    p.add_argument("--resume", action="store_true", help="Resume from existing responses")
    p.add_argument("--tmp_dir", default="./tmp", help="Temporary directory")
    p.add_argument("--dtype", default="auto", help="Data type for local models")
    p.add_argument("--device_map", default="auto", help="Device mapping for local models")
    p.add_argument("--use_flash_attention", action="store_true", help="Use Flash Attention 2 for acceleration (requires flash-attn)")
    p.add_argument("--seed", type=int, help="Random seed")

    # API settings
    p.add_argument("--api_timeout", type=float, default=60.0, help="API timeout seconds")
    p.add_argument("--api_max_retries", type=int, default=3, help="Max API retries")
    p.add_argument("--api_backoff_base", type=float, default=2.0, help="Retry backoff base")
    p.add_argument("--min_interval_ms", type=int, default=0, help="Min interval between requests")

    # Advanced options
    p.add_argument("--force_fuse", choices=["h", "v"], help="Force image fusion mode")
    p.add_argument("--base_url", help="Custom base URL")
    p.add_argument("--model_tag", help="Custom model tag for response directory")
    p.add_argument("--extra_body", help="Path to JSON file with extra request body")

    args = p.parse_args()
    run(args)
