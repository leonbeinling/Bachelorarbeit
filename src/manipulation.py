import os
import re
import json
import random
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    __version__ as hf_version,
)

# ==== Project paths =====================================================================
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_PROMPT_FILE = "/home/beinling/qwen-reasoning/prompts/5vorManPrompts.json"
OUTPUT_ROOT = os.path.join(PROJ_ROOT, "outputs", "4_manip")

# ==== Default configuration =============================================================
DEFAULTS = {
    "seed": 42,
    "model_id": "Qwen/Qwen3-32B",
    "revision": "9216db5781bf21249d130ec9da846c4624c16137",  # None = latest
    "device": "auto",                # "cuda" | "cpu" | "auto"
    "dtype": "bfloat16",             # "float16" | "bfloat16" | "float32"
    "max_new_tokens": 15000,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 0,
    "do_sample": True,
    "repetition_penalty": 1.0,
    "step_tokens": 256,              # chunk size for auto-continue
    "max_swaps": None,               # None = unbounded
    "detection_scope": "all",        # "all" (incl. <think>) | "visible" (after </think>)
}

# ==== Marker catalog ====================================================================
problem_markers = [
    "user wants",
    "okay, let's see",
    "okay, i need to come up",
]

idea_markers = [
    "maybe a",
    "maybe as",
    "maybe using",
    "what about",
    "another idea",
    "let me think",
    "another thought",
    "for example",
    "how about",
    "maybe for",
    "could be used",
    "maybe something",
]

eval_markers = [
    "that",
    "that's",
    "that's a",
    "might be",
    "let me check",
    "make sure",
    "maybe not",
    "but that's",
    "ensure",
    "might not be",
    "that's kind of",
]

answer_markers = [
    "i think i have enough",
    "let me count",
    "i have enough",
    "let me list them out",
    "final list",
    "let me finalize",
    "let me organize them",
]

MARKERS: Dict[str, List[str]] = {
    "PROBLEM": problem_markers,
    "IDEATE": idea_markers,
    "EVALUATE": eval_markers,
    "ANSWER": answer_markers,
}

# short, clean replacements for Evaluate
EVAL_REPLACEMENTS_POOL = ["that", "that's", "that's a", "might be", "let me check", "make sure"]

# Chat/think markers
ASSISTANT_START = "<|im_start|>assistant"
ASSISTANT_END   = "<|im_end|>"
USER_START      = "<|im_start|>user"
THINK_RE        = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# ==== Utils =============================================================================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def now_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_dump_json(path: str, payload: Dict[str, Any]):
    """Serialize first to catch errors early, then write."""
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def load_prompts(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "prompts" in data:
        return data["prompts"]
    if isinstance(data, list):
        return data
    raise ValueError("Unexpected prompt file format. Expected list or { 'prompts': [...] }")

def pick_device(pref: str = "auto") -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref

def str_dtype_to_torch(dtype: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32

def _stamp_marker_for_storage(text: str, pos: int, marker: str) -> str:
    """Insert marker into text for storage only (not used at runtime)."""
    return text[:pos] + marker + text[pos:]

# ==== Chat template (Qwen3) =============================================================
def build_chat_prompt_user_only(tokenizer, raw_prompt: str) -> str:
    """Return chat string up to '<|im_start|>assistant\\n' (no assistant text)."""
    messages = [{"role": "user", "content": raw_prompt.strip()}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )

def build_chat_prompt_with_assistant_prefix(tokenizer, raw_prompt: str, assistant_prefix: str) -> str:
    """Return user + assistant start + given assistant prefix."""
    user_only = build_chat_prompt_user_only(tokenizer, raw_prompt)
    return f"{user_only}{assistant_prefix}"

# ==== Model loading =====================================================================
def load_model_and_tokenizer(model_id: str, revision: Optional[str], device: str, dtype: str):
    torch_dtype = str_dtype_to_torch(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, use_fast=True, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto" if device in ("auto", "cuda") else None,
    ).eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.config, "eos_token_id", None) is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    return model, tokenizer

# ==== Helper: get the first assistant turn ==============================================
def extract_first_assistant_turn(full_text: str) -> str:
    s = full_text.find(ASSISTANT_START)
    if s == -1:
        return full_text.strip()
    s += len(ASSISTANT_START)
    if s < len(full_text) and full_text[s] == "\n":
        s += 1
    e = full_text.find(ASSISTANT_END, s)
    if e == -1:
        e = len(full_text)
    return full_text[s:e].strip()

# ==== One-shot generate with usage ======================================================
def _decode_and_reason(full_ids, tokenizer, max_new_tokens, prompt_tokens) -> Tuple[str, Dict[str, Any]]:
    """Decode ids, extract first turn, and stop info (for logging only)."""
    total_tokens = int(full_ids.shape[1])
    generated_tokens = total_tokens - prompt_tokens
    decoded = tokenizer.decode(full_ids[0], skip_special_tokens=False)
    assistant_text = extract_first_assistant_turn(decoded)

    low = decoded.lower()
    hit_role_assistant_end = (ASSISTANT_END.lower() in low)
    hit_role_user_start = (USER_START.lower() in low)

    last_tok = int(full_ids[0, -1].item())
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    hit_eos_token = (eos_id is not None and last_tok == eos_id) or (pad_id is not None and last_tok == pad_id)

    budget_exhausted = (generated_tokens >= max_new_tokens)

    if hit_eos_token:
        reason = "eos_token"
    elif budget_exhausted:
        reason = "budget_exhausted"
    elif hit_role_user_start:
        reason = "role_user_start_observed"
    elif hit_role_assistant_end:
        reason = "role_assistant_end_observed"
    else:
        reason = "unknown_or_done"

    usage = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": int(generated_tokens),
        "total_tokens": int(total_tokens),
        "stop_reason": reason,
        "stop_flags": {
            "hit_role_assistant_end_observed": bool(hit_role_assistant_end),
            "hit_role_user_start_observed": bool(hit_role_user_start),
            "hit_eos_token": bool(hit_eos_token),
            "budget_exhausted": bool(budget_exhausted),
        }
    }
    return assistant_text, usage

def generate_text(model, tokenizer, chat_formatted_prompt: str, gen_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Generate once and return assistant text + token/stop info."""
    inputs = tokenizer(chat_formatted_prompt, return_tensors="pt", padding=False, truncation=False).to(model.device)
    prompt_tokens = int(inputs.input_ids.shape[1])

    generation_config = GenerationConfig(
        max_new_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        top_k=gen_cfg["top_k"],
        do_sample=gen_cfg["do_sample"],
        repetition_penalty=gen_cfg["repetition_penalty"],
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.no_grad():
        out_ids = model.generate(**inputs, generation_config=generation_config)

    return _decode_and_reason(out_ids, tokenizer, gen_cfg["max_new_tokens"], prompt_tokens)

# ==== Marker regex & counting ============================================================
def compile_marker_regexes() -> Dict[str, List[re.Pattern]]:
    compiled = {}
    for phase, lst in MARKERS.items():
        ordered = sorted(lst, key=len, reverse=True)  # longer phrases first
        pats = [re.compile(rf"\b{re.escape(m)}(?!\w)", re.IGNORECASE) for m in ordered]
        compiled[phase] = pats
    return compiled

def find_triggers(text: str, patterns: List[re.Pattern]) -> List[Tuple[int, int, str]]:
    hits = []
    for pat in patterns:
        for m in pat.finditer(text):
            hits.append((m.start(), m.end(), m.group(0)))
    hits.sort(key=lambda t: t[0])
    return hits

def count_phase_triggers(text: str, compiled: Dict[str, List[re.Pattern]]) -> Dict[str, int]:
    counts = {}
    for phase, pats in compiled.items():
        total = 0
        for p in pats:
            total += len(list(p.finditer(text)))
        counts[phase] = total
    return counts

def visible_segment_view(assistant_text: str) -> Tuple[str, int]:
    """Return (visible_text_after_</think>, offset_in_assistant)."""
    m = THINK_RE.search(assistant_text)
    if m:
        return assistant_text[m.end():], m.end()
    return assistant_text, 0

def scope_view(assistant_text: str, scope: str) -> Tuple[str, int]:
    """Scope: 'all' | 'visible'."""
    if scope == "visible":
        return visible_segment_view(assistant_text)
    return assistant_text, 0

# ==== Multi-swap LIVE (every 2nd IDEATE) ================================================
def manipulated_multi_swap_live(
    model,
    tokenizer,
    raw_prompt: str,
    baseline_assistant_text: str,
    compiled_markers: Dict[str, List[re.Pattern]],
    eval_replacements_pool: List[str],
    max_new_tokens: int,
    step_tokens: int,
    gen_cfg: Dict[str, Any],
    detection_scope: str = "all",
    max_swaps: Optional[int] = 8,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    1) Find the 2nd IDEATE (by scope) in the baseline.
    2) Iterate LIVE: on each even IDEATE hit (2,4,6,...) replace the match with an EVAL token,
       cut the tail, and restart from this new prefix.
    3) Do NOT insert phase markers at runtime. Markers are stamped only for storage later.
    """

    cfg_base = dict(
        do_sample=gen_cfg.get("do_sample", True),
        temperature=gen_cfg.get("temperature", 0.7),
        top_p=gen_cfg.get("top_p", 0.95),
        top_k=gen_cfg.get("top_k", 0),
        repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    view0, off0 = scope_view(baseline_assistant_text, detection_scope)
    hits0 = find_triggers(view0, compiled_markers["IDEATE"])

    swaps_meta: List[Dict[str, Any]] = []
    performed_any = False

    events: List[Dict[str, Any]] = []
    generated_tokens_total = 0
    stop_reason = "unknown_or_done"
    stop_flags = {
        "hit_role_assistant_end_observed": False,
        "hit_role_user_start_observed": False,
        "hit_eos_token": False,
        "budget_exhausted": False,
        "no_progress": False,
    }

    prefix_before_first_swap = baseline_assistant_text

    # No 2nd IDEATE in scope → no manipulation; just continue generating.
    if len(hits0) < 2:
        before_seg = baseline_assistant_text
        composed = build_chat_prompt_with_assistant_prefix(tokenizer, raw_prompt, baseline_assistant_text)
        inputs = tokenizer(composed, return_tensors="pt", padding=False, truncation=False).to(model.device)

        with torch.inference_mode():
            out_ids = model.generate(**inputs, max_new_tokens=min(step_tokens, max_new_tokens), **cfg_base)
        produced_now = int(out_ids.shape[1] - inputs.input_ids.shape[1])
        generated_tokens_total += max(0, produced_now)
        events.append({"type": "generate", "produced": produced_now})
        budget_left = max_new_tokens - produced_now

        while budget_left > 0:
            prev = int(out_ids.shape[1])
            with torch.inference_mode():
                out_ids = model.generate(input_ids=out_ids, max_new_tokens=min(step_tokens, budget_left), **cfg_base)
            produced = int(out_ids.shape[1] - prev)
            if produced <= 0:
                stop_reason = "no_progress"
                stop_flags["no_progress"] = True
                events.append({"type": "no_progress"})
                break
            generated_tokens_total += produced
            budget_left -= produced
            events.append({"type": "generate", "produced": produced})

            low_txt = tokenizer.decode(out_ids[0], skip_special_tokens=False).lower()
            if ASSISTANT_END.lower() in low_txt:
                events.append({"type": "marker_observed", "marker": "ASSISTANT_END"})
                stop_flags["hit_role_assistant_end_observed"] = True
            if USER_START.lower() in low_txt:
                events.append({"type": "marker_observed", "marker": "USER_START"})
                stop_flags["hit_role_user_start_observed"] = True

            last_tok = int(out_ids[0, -1].item())
            if (tokenizer.eos_token_id is not None and last_tok == tokenizer.eos_token_id) or \
               (tokenizer.pad_token_id is not None and last_tok == tokenizer.pad_token_id):
                stop_reason = "eos_token"
                stop_flags["hit_eos_token"] = True
                events.append({"type": "eos_token"})
                break

        after_full = tokenizer.decode(out_ids[0], skip_special_tokens=False)
        after_seg = extract_first_assistant_turn(after_full)
        if budget_left <= 0 and stop_reason == "unknown_or_done":
            stop_reason = "budget_exhausted"
            stop_flags["budget_exhausted"] = True
            events.append({"type": "budget_exhausted"})

        token_usage = {
            "generated_tokens": generated_tokens_total,
            "max_new_tokens": max_new_tokens,
            "budget_left": max(0, budget_left),
            "stop_reason": stop_reason,
            "stop_flags": stop_flags,
            "events": events[-10:],
        }

        return before_seg, after_seg, {
            "performed": False,
            "reason": "no_second_ideate_in_scope",
            "scope": detection_scope,
            "swaps": [],
            "token_usage": token_usage,
        }

    # 1) keep baseline up to the 2nd IDEATE
    s2, e2, matched2 = hits0[1]
    a_start2 = off0 + s2
    a_end2   = off0 + e2
    prefix_before_first_swap = baseline_assistant_text[:a_start2]

    # first manipulated prefix (RUNTIME: no marker)
    repl0 = random.choice(eval_replacements_pool)
    assistant_prefix = prefix_before_first_swap + repl0
    swaps_meta.append({
        "ideate_index": 2,
        "original_trigger": matched2,
        "chosen_replacement": repl0,
        "assistant_cut_start": int(a_start2),
        "assistant_cut_end": int(a_end2),
        "storage_insert": {"pos": len(prefix_before_first_swap), "marker": "⟦IDEATE→EVALUATE⟧ "}
    })
    performed_any = True

    target_index = 4
    swaps_done = 1

    # start with user + current assistant prefix
    composed = build_chat_prompt_with_assistant_prefix(tokenizer, raw_prompt, assistant_prefix)
    inputs = tokenizer(composed, return_tensors="pt", padding=False, truncation=False).to(model.device)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=min(step_tokens, max_new_tokens), **cfg_base)
    produced_now = int(out_ids.shape[1] - inputs.input_ids.shape[1])
    generated_tokens_total += max(0, produced_now)
    events.append({"type": "generate", "produced": produced_now})
    budget_left = max_new_tokens - produced_now

    def extract_assistant_text_from_ids(ids) -> str:
        return extract_first_assistant_turn(tokenizer.decode(ids[0], skip_special_tokens=False))

    while budget_left > 0:
        decoded_assistant = extract_assistant_text_from_ids(out_ids)
        view, off = scope_view(decoded_assistant, detection_scope)
        hits = find_triggers(view, compiled_markers["IDEATE"])

        can_swap_more = (max_swaps is None) or (swaps_done < max_swaps)
        if len(hits) >= target_index and can_swap_more:
            s, e, matched = hits[target_index - 1]
            a_s = off + s
            a_e = off + e

            new_repl = random.choice(EVAL_REPLACEMENTS_POOL)
            edited_assistant = decoded_assistant[:a_s] + new_repl  # runtime without marker

            swaps_meta.append({
                "ideate_index": target_index,
                "original_trigger": matched,
                "chosen_replacement": new_repl,
                "assistant_cut_start": int(a_s),
                "assistant_cut_end": int(a_e),
                "storage_insert": {"pos": int(a_s), "marker": "⟦IDEATE→EVALUATE⟧ "}
            })
            swaps_done += 1
            target_index += 2
            events.append({"type": "swap_restart", "at_char": int(a_s), "replacement": new_repl})

            composed = build_chat_prompt_with_assistant_prefix(tokenizer, raw_prompt, edited_assistant)
            new_inputs = tokenizer(composed, return_tensors="pt", padding=False, truncation=False).to(model.device)
            with torch.inference_mode():
                out_ids = model.generate(**new_inputs, max_new_tokens=min(step_tokens, budget_left), **cfg_base)
            produced = int(out_ids.shape[1] - new_inputs.input_ids.shape[1])
            generated_tokens_total += max(0, produced)
            budget_left -= max(0, produced)
            events.append({"type": "generate", "produced": produced})

            last_tok = int(out_ids[0, -1].item())
            if (tokenizer.eos_token_id is not None and last_tok == tokenizer.eos_token_id) or \
               (tokenizer.pad_token_id is not None and last_tok == tokenizer.pad_token_id):
                stop_reason = "eos_token"
                stop_flags["hit_eos_token"] = True
                events.append({"type": "eos_token"})
                break

            continue

        # otherwise continue generating
        prev_len = int(out_ids.shape[1])
        with torch.inference_mode():
            out_ids = model.generate(input_ids=out_ids, max_new_tokens=min(step_tokens, budget_left), **cfg_base)
        produced = int(out_ids.shape[1] - prev_len)
        if produced <= 0:
            stop_reason = "no_progress"
            stop_flags["no_progress"] = True
            events.append({"type": "no_progress"})
            break
        generated_tokens_total += produced
        budget_left -= produced
        events.append({"type": "generate", "produced": produced})

        low_txt = tokenizer.decode(out_ids[0], skip_special_tokens=False).lower()
        if ASSISTANT_END.lower() in low_txt:
            events.append({"type": "marker_observed", "marker": "ASSISTANT_END"})
            stop_flags["hit_role_assistant_end_observed"] = True
        if USER_START.lower() in low_txt:
            events.append({"type": "marker_observed", "marker": "USER_START"})
            stop_flags["hit_role_user_start_observed"] = True

        last_tok = int(out_ids[0, -1].item())
        if (tokenizer.eos_token_id is not None and last_tok == tokenizer.eos_token_id) or \
           (tokenizer.pad_token_id is not None and last_tok == tokenizer.pad_token_id):
            stop_reason = "eos_token"
            stop_flags["hit_eos_token"] = True
            events.append({"type": "eos_token"})
            break

    if budget_left <= 0 and stop_reason == "unknown_or_done":
        stop_reason = "budget_exhausted"
        stop_flags["budget_exhausted"] = True
        events.append({"type": "budget_exhausted"})

    # finalize
    final_full = tokenizer.decode(out_ids[0], skip_special_tokens=False)
    after_seg = extract_first_assistant_turn(final_full)
    before_seg = prefix_before_first_swap

    if not after_seg.startswith(before_seg):
        def norm(x: str) -> str:
            return re.sub(r"\s+", " ", x).strip()
        if not norm(after_seg).startswith(norm(before_seg)):
            pass  # relaxed consistency check

    token_usage = {
        "generated_tokens": generated_tokens_total,
        "max_new_tokens": max_new_tokens,
        "budget_left": max(0, budget_left),
        "stop_reason": stop_reason,
        "stop_flags": stop_flags,
        "events": events[-10:],
    }

    return before_seg, after_seg, {
        "performed": performed_any,
        "scope": detection_scope,
        "max_swaps": max_swaps if max_swaps is not None else "unbounded",
        "swaps": swaps_meta,          # contains storage_insert positions
        "token_usage": token_usage,
    }

# ==== Main run ==========================================================================
def run(
    prompt_file: str,
    out_root: str,
    model_id: str,
    revision: Optional[str],
    device: str,
    seed: int,
    gen_cfg: Dict[str, Any]
):
    set_all_seeds(seed)
    device = pick_device(device)
    ensure_dir(out_root)

    ts_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, ts_dir)
    ensure_dir(out_dir)

    print(f"[INFO] Loading model: {model_id} (rev={revision}) on {device} …")
    model, tokenizer = load_model_and_tokenizer(model_id, revision, device, gen_cfg.get("dtype", "bfloat16"))
    compiled = compile_marker_regexes()

    prompts = load_prompts(prompt_file)
    print(f"[INFO] {len(prompts)} prompts loaded from {prompt_file}")

    master_meta = {
        "timestamp_utc": now_utc_ts(),
        "hf_version": hf_version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "model_id": model_id,
        "revision": revision,
        "device": str(device),
        "seed": seed,
        "gen_config": {k: gen_cfg[k] for k in ["max_new_tokens","temperature","top_p","top_k","do_sample","repetition_penalty"]},
        "output_dir": out_dir,
        "markers": MARKERS,
        "eval_replacement_pool": EVAL_REPLACEMENTS_POOL,
        "detection_scope": DEFAULTS["detection_scope"],
        "max_swaps": DEFAULTS["max_swaps"] if DEFAULTS["max_swaps"] is not None else "unbounded"
    }
    safe_dump_json(os.path.join(out_dir, "_meta.json"), master_meta)

    for i, item in enumerate(prompts, start=1):
        task_id = item.get("id") or item.get("task_id") or f"task_{i:02d}"
        raw_prompt = item.get("prompt") or item.get("text") or item.get("instruction")
        if not raw_prompt:
            print(f"[WARN] Prompt {task_id} has no 'prompt' field – skipped.")
            continue

        task_dir = os.path.join(out_dir, f"{i:02d}_{task_id}")
        ensure_dir(task_dir)

        # ===== Baseline =====
        user_only_prompt = build_chat_prompt_user_only(tokenizer, raw_prompt)
        baseline_text, baseline_usage = generate_text(model, tokenizer, user_only_prompt, gen_cfg)
        baseline_counts_all = count_phase_triggers(baseline_text, compiled)

        baseline_payload = {
            "timestamp_utc": now_utc_ts(),
            "run_type": "baseline",
            "task_id": task_id,
            "prompt": raw_prompt,
            "composed_prompt": user_only_prompt,
            "output_text": baseline_text,
            "trigger_counts_all_phases": baseline_counts_all,
            "token_usage": baseline_usage
        }
        safe_dump_json(os.path.join(task_dir, "baseline.json"), baseline_payload)

        # ===== Manipulated: multi-swap LIVE (every 2nd IDEATE) =====
        before_seg, after_seg, live_swap_meta = manipulated_multi_swap_live(
            model=model,
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            baseline_assistant_text=baseline_text,
            compiled_markers=compiled,
            eval_replacements_pool=EVAL_REPLACEMENTS_POOL,
            max_new_tokens=gen_cfg["max_new_tokens"],
            step_tokens=min(DEFAULTS["step_tokens"], gen_cfg["max_new_tokens"]),
            gen_cfg=gen_cfg,
            detection_scope=DEFAULTS["detection_scope"],
            max_swaps=DEFAULTS["max_swaps"],
        )

        counts_before = count_phase_triggers(before_seg, compiled)
        counts_after  = count_phase_triggers(after_seg, compiled)

        # Move token_usage to root level
        manip_token_usage = live_swap_meta.pop("token_usage", {})

        # STORAGE-ONLY: inject markers at recorded positions (descending order)
        after_with_markers = after_seg
        ins_list = [m.get("storage_insert") for m in live_swap_meta.get("swaps", []) if m.get("storage_insert")]
        for ins in sorted(ins_list, key=lambda x: x["pos"], reverse=True):
            after_with_markers = _stamp_marker_for_storage(after_with_markers, ins["pos"], ins["marker"])

        manipulated_payload = {
            "timestamp_utc": now_utc_ts(),
            "run_type": "manipulated",
            "task_id": task_id,
            "prompt": raw_prompt,
            "composed_prompt": build_chat_prompt_user_only(tokenizer, raw_prompt),
            "output_text_before_swap": before_seg,          # runtime text (no markers)
            "output_text_after_swap":  after_with_markers,  # storage text (with markers)
            "live_swap_meta": live_swap_meta,               # contains swaps + positions
            "trigger_counts_before_all_phases": counts_before,
            "trigger_counts_after_all_phases":  counts_after,  # counts on runtime text
            "token_usage": manip_token_usage
        }
        safe_dump_json(os.path.join(task_dir, "manipulated.json"), manipulated_payload)

        # Preview file (compact readable view)
        preview_lines = []
        preview_lines.append("===== BEFORE (excerpt up to 1.2k) =====")
        preview_lines.append(before_seg[:1200])
        preview_lines.append("")
        preview_lines.append("===== AFTER  (excerpt up to 1.2k, with markers) =====")
        preview_lines.append(after_with_markers[:1200])
        preview_lines.append("")
        preview_lines.append("===== LIVE SWAP META =====")
        preview_lines.append(json.dumps(live_swap_meta, ensure_ascii=False, indent=2))
        if live_swap_meta.get("swaps"):
            first_cut = live_swap_meta["swaps"][0]["assistant_cut_start"]
            start = max(0, int(first_cut) - 80)
            end   = int(first_cut) + 80
            preview_lines.append("")
            preview_lines.append("===== AFTER CONTEXT AROUND FIRST SWAP (±80, runtime text) =====")
            preview_lines.append(after_seg[start:end])
        with open(os.path.join(task_dir, "preview.txt"), "w", encoding="utf-8") as pf:
            pf.write("\n".join(preview_lines))

        print(f"[OK] {task_id}: baseline + manipulated saved under {task_dir}")

    print(f"\n[DONE] All results in: {out_dir}")

# ==== Entry point =======================================================================
def main():
    gen_cfg = dict(
        max_new_tokens=DEFAULTS["max_new_tokens"],
        temperature=DEFAULTS["temperature"],
        top_p=DEFAULTS["top_p"],
        top_k=DEFAULTS["top_k"],
        do_sample=DEFAULTS["do_sample"],
        repetition_penalty=DEFAULTS["repetition_penalty"],
        dtype=DEFAULTS["dtype"],
    )

    run(
        prompt_file=DEFAULT_PROMPT_FILE,
        out_root=OUTPUT_ROOT,
        model_id=DEFAULTS["model_id"],
        revision=DEFAULTS["revision"],
        device=DEFAULTS["device"],
        seed=DEFAULTS["seed"],
        gen_cfg=gen_cfg,
    )

if __name__ == "__main__":
    main()
