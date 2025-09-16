import os
import re
import json
import random
import hashlib
import platform
from datetime import datetime, timezone

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, __version__ as hf_version
from transformers import StoppingCriteria, StoppingCriteriaList  # for in-generate stopping

# =========================
# Config block
# =========================
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEBUG_MODE = True       # If True, process only NUM_TEST_PROMPTS
NUM_TEST_PROMPTS = 5

SEED = 42
MODEL_ID = "Qwen/Qwen3-32B"
REVISION = "9216db5781bf21249d130ec9da846c4624c16137"   # commit hash 2025-07-26
PROMPT_FILE = os.path.join(BASE, "prompts", "all_prompts.json")
OUTPUT_DIR  = os.path.join(BASE, "outputs", "raw")
# Generation parameters — deterministic by default (do_sample=False)
MAX_NEW_TOKENS = 2048    # 1st stage; Auto-Continue uses smaller chunks
DO_SAMPLE = False
TEMPERATURE = 1.0
TOP_P = 1.0
REPETITION_PENALTY = 1.0

# Auto-Continue settings
CHUNK_TOKENS = 512       # step size when continuing
HARD_CAP_TOKENS = 11000  # absolute cap of tokens per item
CTX_SAFETY = 64          # margin to avoid hitting the context limit

# =========================
# Backends (reproducibility & perf)
# =========================
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# More deterministic matmul (no TF32)
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = False
if hasattr(torch, "set_float32_matmul_precision"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# =========================
# Helper functions
# =========================
def file_sha256(path: str) -> str:
    """Return the SHA-256 hash of a file (for provenance)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_json(obj, path: str):
    """Write JSON atomically to avoid partial/corrupted files."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

# Chat markers we care about
ASSISTANT_START = "<|im_start|>assistant"
ASSISTANT_END   = "<|im_end|>"
USER_START      = "<|im_start|>user"

THINK_RE    = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
THINK_OPEN  = "<think>"
THINK_CLOSE = "</think>"

def _to_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]

def _first_assistant_turn(text: str) -> str:
    """
    Extract the *first* assistant turn from raw_output.
    Returns only its content (without <|im_start|>assistant and <|im_end|>).
    Fallback: if markers are missing, cut heuristically at the next role switch/EOS string.
    """
    s = text.find(ASSISTANT_START)
    if s == -1:
        # No structured chat block; return the full text
        return text
    s += len(ASSISTANT_START)
    if s < len(text) and text[s] == "\n":
        s += 1

    e = text.find(ASSISTANT_END, s)
    if e == -1:
        # Safety: cut at the next role marker or visible EOS string
        for marker in (USER_START, "<|endoftext|>"):
            m = text.find(marker, s)
            if m != -1:
                e = m
                break
    if e == -1:
        e = len(text)
    return text[s:e]

def extract_reasoning_and_answer(text: str):
    """
    1) Isolate the first assistant turn.
    2) Inside it: <think>…</think> is the reasoning, the following part is the final answer.
    3) Trim visible terminators/role switches if they appear inside the turn text.
    """
    segment = _first_assistant_turn(text)

    m = THINK_RE.search(segment)
    if m:
        cot = (m.group(1) or "").strip()
        post = (segment[m.end():] or "")
    else:
        cot = "N/A"
        post = segment

    # Cut at the first terminator/role switch if present in the turn text
    for marker in (ASSISTANT_END, "<|endoftext|>", USER_START):
        i = post.find(marker)
        if i != -1:
            post = post[:i]
            break

    answer = post.strip() if post.strip() else "N/A"
    return (cot if cot else "N/A"), answer

# =========================
# Stopping Criteria (stop already inside generate())
# =========================
class StopOnChatMarkers(StoppingCriteria):
    def __init__(self, tokenizer, markers):
        self.tokenizer = tokenizer
        self.markers = [m.lower() for m in markers]
        self._buf_start = 0  # decode from here

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        new_tokens = input_ids[0, self._buf_start:]
        if new_tokens.numel() == 0:
            return False
        tail_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False).lower()
        self._buf_start = input_ids.shape[1]
        return any(m in tail_text for m in self.markers)

# =========================
# Setup & load
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading tokenizer & model ({'DEBUG' if DEBUG_MODE else 'FULL'} MODE)…")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    revision=REVISION
)
# Fallback padding: use EOS if PAD is missing
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    revision=REVISION,
    device_map="auto",        # automatic device placement (can be sharded)
    torch_dtype=torch.float16
).eval()

GEN_KW = dict(
    do_sample=DO_SAMPLE,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    repetition_penalty=REPETITION_PENALTY,
    max_new_tokens=MAX_NEW_TOKENS,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# --- End-of-Sequence token(s) ----------------------------
EOS_IDS = []
for eid in _to_list(getattr(model.config, "eos_token_id", None)):
    if isinstance(eid, int) and eid >= 0:
        EOS_IDS.append(eid)
for eid in _to_list(getattr(tokenizer, "eos_token_id", None)):
    if isinstance(eid, int) and eid >= 0 and eid not in EOS_IDS:
        EOS_IDS.append(eid)

def generation_should_stop(prev_len: int, all_ids: torch.Tensor) -> bool:
    """
    Stop when:
      a) any EOS token appears in the newly generated tail, OR
      b) the tail text contains a role marker that starts/ends a turn:
         - ASSISTANT_END ("<|im_end|>")  → end of assistant turn
         - USER_START ("<|im_start|>user") → start of a *new* user turn
    This ends generation *before* a second human block, even if markers are produced as plain text.
    """
    # a) EOS token in the new token IDs?
    if EOS_IDS:
        tail_ids = all_ids[0, prev_len:]
        for t in tail_ids:
            if int(t.item()) in EOS_IDS:
                return True

    # b) Role markers in the *text* of the new tail?
    tail_text = tokenizer.decode(all_ids[0, prev_len:], skip_special_tokens=False)
    tail_low = tail_text.lower()
    if (ASSISTANT_END.lower() in tail_low) or (USER_START.lower() in tail_low):
        return True

    return False

# =========================
# Prompts & manifest
# =========================
with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    prompt_data = json.load(f)

if DEBUG_MODE:
    prompt_data = prompt_data[:NUM_TEST_PROMPTS]
    print(f"DEBUG_MODE: processing only {len(prompt_data)} prompt(s)")

prompt_sha = file_sha256(PROMPT_FILE)
gpu = {
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
}

run_manifest = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": MODEL_ID,
    "revision": REVISION,
    "transformers_version": hf_version,
    "torch_version": torch.__version__,
    "python": platform.python_version(),
    "gpu": gpu,
    "seed": SEED,
    "prompt_file": PROMPT_FILE,
    "prompt_sha256": prompt_sha,
    "generation_kwargs": GEN_KW,
    "debug_mode": DEBUG_MODE
}
atomic_write_json(run_manifest, os.path.join(OUTPUT_DIR, "_run_manifest.json"))

# =========================
# Batch processing (with Auto-Continue)
# =========================
# Determine context window (fallback to tokenizer limit)
CTX = getattr(model.config, "max_position_embeddings", None)
if CTX is None or CTX == float("inf"):
    CTX = getattr(tokenizer, "model_max_length", 8192)

# StoppingCriteria instance (requires tokenizer)
STOP_CRITERIA = StoppingCriteriaList([
    StopOnChatMarkers(tokenizer, markers=[ASSISTANT_END, USER_START])
])

for item in prompt_data:
    prompt = item["prompt"]
    task_id = item["id"]
    task_type = item.get("task_type", "N/A")

    print(f"\nProcessing {task_id} ({task_type})…")

    # Build a fresh, stateless chat input for each prompt
    messages = [{"role": "user", "content": prompt}]
    composed_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True   # Qwen3: explicit <think> mode
    )

    # Do not truncate the input silently
    inputs = tokenizer(
        composed_prompt,
        return_tensors="pt",
        padding=False,
        truncation=False
    ).to(model.device)

    # --- Stage 1: CTX-safe first generation (short burst)
    input_len = inputs.input_ids.shape[1]
    first_cap = min(
        CHUNK_TOKENS,
        MAX_NEW_TOKENS,
        HARD_CAP_TOKENS,
        max(32, CTX - input_len - CTX_SAFETY)
    )
    first_kw = dict(GEN_KW, max_new_tokens=first_cap)

    try:
        with torch.inference_mode():
            out = model.generate(**inputs, **first_kw, stopping_criteria=STOP_CRITERIA)
    except torch.cuda.OutOfMemoryError:
        print("WARN: CUDA OOM — reducing max_new_tokens and retrying…")
        torch.cuda.empty_cache()
        fallback_kw = dict(first_kw)
        fallback_kw["max_new_tokens"] = max(256, first_kw["max_new_tokens"] // 2)
        with torch.inference_mode():
            out = model.generate(**inputs, **fallback_kw, stopping_criteria=STOP_CRITERIA)

    all_ids = out  # (1, seq_len) = prompt + generated
    continue_steps = 0
    prompt_len = inputs.input_ids.shape[1]
    hard_cap_left = HARD_CAP_TOKENS - (all_ids.shape[1] - prompt_len)

    # --- Stage 2: Auto-Continue in chunks until done or cap reached
    while True:
        # If near CTX limit, keep the full prompt and trim only the generated tail
        if all_ids.shape[1] >= (CTX - CTX_SAFETY):
            gen_len = all_ids.shape[1] - prompt_len
            keep_gen = max(32, min(CHUNK_TOKENS, CTX - CTX_SAFETY - prompt_len, gen_len))
            all_ids = torch.cat([all_ids[:, :prompt_len], all_ids[:, -keep_gen:]], dim=1)

        if hard_cap_left <= 0:
            break

        step_kw = dict(GEN_KW)
        step_kw["max_new_tokens"] = max(32, min(CHUNK_TOKENS, hard_cap_left))

        prev_len = all_ids.shape[1]  # length before this step
        with torch.inference_mode():
            all_ids = model.generate(input_ids=all_ids, **step_kw, stopping_criteria=STOP_CRITERIA)
        added = max(0, all_ids.shape[1] - prev_len)  # count actual new tokens
        hard_cap_left -= added
        continue_steps += 1

        # Stop on EOS or when a role marker appears in the new tail
        if generation_should_stop(prev_len, all_ids):
            break

    # --- Stage 3: Final decode & write result
    # Use RAW decode so </think> is preserved for extraction.
    decoded_raw = tokenizer.decode(all_ids[0], skip_special_tokens=False)
    reasoning, answer = extract_reasoning_and_answer(decoded_raw)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type,
        "task_id": task_id,
        "prompt": prompt,
        "composed_prompt": composed_prompt,
        "raw_output": decoded_raw,  # keep raw to include <think> tags
        "reasoning_trace": reasoning,
        "final_answer": answer,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "seed": SEED,
        "generation_kwargs": GEN_KW,
        "auto_continue": {
            "enabled": True,
            "continue_steps": continue_steps,
            "new_tokens_total": (all_ids.shape[1] - prompt_len),
            "hard_cap_left": hard_cap_left,
            "ctx_limit": CTX,
            "ctx_safety": CTX_SAFETY,
            "chunk_tokens": CHUNK_TOKENS
        }
    }

    outpath = os.path.join(OUTPUT_DIR, f"{task_id}.json")
    atomic_write_json(result, outpath)
    print(f"Saved: {outpath}")

print("\nAll prompts processed.")
