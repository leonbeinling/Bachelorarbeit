import os, json, unicodedata, re, hashlib
from collections import Counter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo  # DST-aware local time

DEBUG = True

# Run metadata
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUN_VERSION = "seg-v2025-09-06-hard-only"
RUN_TZ = "Europe/Berlin"
RUN_TS_UTC = datetime.now(timezone.utc).isoformat()
RUN_TS_LOCAL = datetime.now(ZoneInfo(RUN_TZ)).isoformat()

input_dir = os.path.join(BASE, "outputs", "raw")
output_dir = os.path.join(BASE, "outputs", "structured_signals")
os.makedirs(output_dir, exist_ok=True)

# --- Utils -------------------------------------------------------------------

def merge_segments(segs):
    out = []
    for s in segs:
        if out and s["phase"] == out[-1]["phase"]:
            out[-1]["end_char"] = s["end_char"]
            out[-1]["text"] += s["text"]
        else:
            out.append(s)
    return out

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’","'").replace("“",'"').replace("”",'"')
    s = re.sub(r"[ \t]+"," ", s)
    s = re.sub(r"\n{3,}","\n\n", s)
    return s.strip()

def _norm_key(s: str) -> str:
    return " ".join(s.split()).lower()

def _compile_alt(phrases):
    ordered = sorted(phrases, key=len, reverse=True)
    pat = r"(?<!\w)(?:" + "|".join(re.escape(p) for p in ordered) + r")(?!\w)"
    return re.compile(pat, re.IGNORECASE)

def _catalog_hash(lists):
    blob = "||".join(["|".join(sorted(set(x))) for x in lists])
    return hashlib.md5(blob.encode("utf-8")).hexdigest()

def count_marker_usage(text):
    usage = {"CLARIFY": {}, "IDEATE": {}, "EVALUATE": {}, "FINALIZE": {}}
    for m in MARKER_RE.finditer(text):
        t = m.group(0).lower()
        phase = NORM2PHASE.get(_norm_key(t))
        if phase:
            usage[phase][t] = usage[phase].get(t, 0) + 1
    return usage

# Sentence boundary detection
SENT_BOUNDARY_RE = re.compile(r'([.!?]["\')\]]*\s+|\n+)', re.MULTILINE)

def sentence_start_idx(text: str, pos: int) -> int:
    """
    Return the start index of the sentence that contains position `pos`.
    We look back to the last sentence boundary (. ! ? + optional closing quotes/brackets + space, or newline).
    If none is found, return 0.
    """
    start = 0
    for m in SENT_BOUNDARY_RE.finditer(text, 0, pos):
        start = m.end()
    return start

# --- Clause boundary (comma rule + one-word carryover) -----------------------

def clause_start_idx(text: str, pos: int) -> int:
    """
    Start index when a trigger at `pos` fires:

    - Finde den aktuellen Satzanfang `ss`.
    - Finde das letzte Komma `last_comma` zwischen `ss` und `pos`.
    - Wenn es kein Komma gibt -> Segment startet am Satzanfang `ss`.
    - Wenn es ein Komma gibt:
        * Bestimme die Spanne des Halbsatzes VOR dem Komma:
          von `prev_boundary` (vorheriges Komma im Satz oder `ss`) bis `last_comma`.
        * Wenn in dieser Spanne (getrimmt; Wörter via \b\w+\b) GENAU 1–2 Wörter stehen,
          sollen diese Wörter MIT der neuen Phase markiert werden -> Segmentstart bei
          `prev_boundary` (inkl. der 1–2 Wörter vor dem Komma).
        * Andernfalls startet das neue Segment NACH dem Komma (zur vorherigen Klasse gehört der
          gesamte Teil vor dem Komma).

    Führende Leerzeichen nach der gewählten Startposition werden übersprungen.
    """
    ss = sentence_start_idx(text, pos)

    # Letztes Komma vor dem Trigger innerhalb des Satzes
    last_comma = text.rfind(",", ss, pos)
    if last_comma == -1:
        return ss

    # Vorherige Grenze: vorangehendes Komma oder Satzanfang
    prev_comma = text.rfind(",", ss, last_comma)
    prev_boundary = (prev_comma + 1) if prev_comma != -1 else ss

    # Spanne vor dem Komma analysieren
    pre_span = text[prev_boundary:last_comma]
    # Wörter zählen (alphanumerische Tokens)
    words = re.findall(r"\b\w+\b", pre_span.strip())

    if 1 <= len(words) <= 2:
        # 1–2 Wörter -> diese Wörter + Komma sollen zur neuen Phase gehören
        start = prev_boundary
    else:
        # >2 Wörter -> Start nach dem Komma (vorherige Klasse behält den Vorspann)
        start = last_comma + 1

    # Führende Leerzeichen überspringen
    while start < pos and start < len(text) and text[start] == " ":
        start += 1

    return start

# --- Marker catalog ----------------------------------------------------------

problem_markers = [
    "the task is", "we are asked to", "the question is", "question:",
    "our objective is", "i am supposed to", "it seems the goal is", 
    "the aim is to", "the goal is to", "okay, i need to come up",
    "okay, let's see", "okay, so", "user wants", 
    "let's brainstorm", "the request is", "what is being asked", 
    "what needs to be done", "the problem states that",
    "the problem is", "we are given", "we're given", "you are given", 
    "according to the prompt", "according to the question",
    "the prompt says", "restating the problem", "restate the problem", 
    "let me restate the problem", "let me restate the question",
    "from the prompt", "from the question", "given the problem", "task:", 
    "objective:", "goal:", "given:", "prompt:"
]

idea_markers = [
    "what if", "how about", "what about", "one idea is", "another idea", 
    "here's an idea", "let's consider", "imagine if", "alternatively", 
    "one possibility is", "another possibility", "an alternative would be", 
    "suppose we", "perhaps use", "could it be", "could be used", "another thought",
    "we could", "we might", "we could also", "we might also", "let me think", 
    "we can also", "another option", "consider using", "it might work to", 
    "perhaps we could", "maybe using", "perhaps using", "possible approaches", 
    "possible approach", "how might we", "maybe use", "maybe something", 
    "maybe as", "maybe for", "maybe a", "for example", "calculate", "compute"
]

eval_markers = [
    "let me check", "let's check", "check for uniqueness", "evaluate",
    "not sure if", "filter out", "double-check", "double check", "compare", 
    "verify", "narrowing down", "prioritize", "eliminate duplicates", 
    "select the most", "weighing options", "sanity check", "check units", "pros",
    "cons", "trade-offs", "tradeoffs", "trade offs", "validate", "cross-check", 
    "rank", "score", "sort by", "edge cases", "make sure", "ensure", "confirm", 
    "check that", "check if", "check whether", "on second thought", "maybe not", 
    "hold on", "hang on", "categorize them", "let me categorize", "group them",
    "let me group them", "out of scope", "that's kind of", "but that's", "maybe a bit",
    "might be", "might be not", "that's a", "might not be", "that", "that's"
]

answer_markers = [
    "final answer", "in conclusion", "to conclude", "in summary", "the answer is", 
    "final response", "my final answer", "final list", "solution:", "final solution", 
    "here is the list", "here's the list", "i'll compile the list", "compiled list", 
    "let me organize them", "time to list them", "let me list them out", 
    "let me list them", "let me count", "organize them into a numbered list", 
    "i think i have enough", "i have enough", "that should be enough", 
    "let's wrap up", "wrapping up", "let me finalize"
]

problem_markers = [m.lower() for m in problem_markers]
idea_markers    = [m.lower() for m in idea_markers]
eval_markers    = [m.lower() for m in eval_markers]
answer_markers  = [m.lower() for m in answer_markers]

PROBLEM_RE = _compile_alt(problem_markers)
IDEA_RE    = _compile_alt(idea_markers)
EVAL_RE    = _compile_alt(eval_markers)
ANSWER_RE  = _compile_alt(answer_markers)

MARKER2PHASE = {}
for m in problem_markers: MARKER2PHASE[m] = "CLARIFY"
for m in idea_markers:    MARKER2PHASE[m] = "IDEATE"
for m in eval_markers:    MARKER2PHASE[m] = "EVALUATE"
for m in answer_markers:  MARKER2PHASE[m] = "FINALIZE"

NORM2PHASE = {_norm_key(k): v for k, v in MARKER2PHASE.items()}
MARKER_RE   = _compile_alt(list(MARKER2PHASE.keys()))
FINALIZE_RE = ANSWER_RE
CAT_HASH    = _catalog_hash([problem_markers, idea_markers, eval_markers, answer_markers])

# --- Core --------------------------------------------------------------------

def segment_by_triggers(reasoning: str):
    raw = normalize_text(reasoning)
    segs = []
    cur_phase = "CLARIFY"
    cur_marker_txt = None
    cur_marker_start = None
    cur_marker_end = None
    idx = 0

    for m in MARKER_RE.finditer(raw):
        pos = m.start()
        matched_txt  = m.group(0)
        matched_norm = _norm_key(matched_txt)
        new_phase    = NORM2PHASE.get(matched_norm, cur_phase)

        if new_phase != cur_phase:
            # Phase change: snap boundary back to start of clause (with one-word carryover) or sentence start.
            ss = clause_start_idx(raw, pos)
            if ss < idx:
                ss = idx  # never go back into already emitted text

            prev_chunk = raw[idx:ss]
            if prev_chunk.strip():
                segs.append({
                    "phase": cur_phase,
                    "start_marker": cur_marker_txt,
                    "start_marker_start_char": cur_marker_start,
                    "start_marker_end_char": cur_marker_end,
                    "start_char": idx,
                    "end_char": ss,
                    "text": prev_chunk
                })

            # Update marker metadata to the new marker and switch phase.
            cur_marker_txt   = matched_txt.lower()
            cur_marker_start = m.start()
            cur_marker_end   = m.end()
            cur_phase = new_phase
            idx = ss  # new segment begins at clause/sentence start (with optional one-word carryover)
        else:
            # Same phase: emit up to the marker, then continue from marker
            chunk = raw[idx:pos]
            if chunk.strip():
                segs.append({
                    "phase": cur_phase,
                    "start_marker": cur_marker_txt,
                    "start_marker_start_char": cur_marker_start,
                    "start_marker_end_char": cur_marker_end,
                    "start_char": idx,
                    "end_char": pos,
                    "text": chunk
                })
            cur_marker_txt   = matched_txt.lower()
            cur_marker_start = m.start()
            cur_marker_end   = m.end()
            idx = m.start()

    tail = raw[idx:]
    if tail.strip():
        segs.append({
            "phase": "FINALIZE" if FINALIZE_RE.search(tail) else cur_phase,
            "start_marker": cur_marker_txt,
            "start_marker_start_char": cur_marker_start,
            "start_marker_end_char": cur_marker_end,
            "start_char": idx,
            "end_char": len(raw),
            "text": tail
        })

    cleaned = merge_segments(segs)
    return {"normalized_trace": raw, "segments_char": cleaned}

# --- Pruning / Reduction -----------------------------------------------------

DROP_KEYS_HARD = {
    "raw_output", "composed_prompt", "trace", "think", "cot", "rationale"
}

def prune_record(d: dict):
    """Remove redundant fields; keep essential metadata (incl. seed/kwargs)."""
    for k in list(DROP_KEYS_HARD):
        d.pop(k, None)

    if "reasoning_trace" in d and isinstance(d["reasoning_trace"], str) and d["reasoning_trace"]:
        rt = d.pop("reasoning_trace")
        d["reasoning_hash"] = "sha256:" + hashlib.sha256(rt.encode("utf-8")).hexdigest()

    sps = d.get("segmented_phases_signals")
    if isinstance(sps, dict) and "normalized_trace" in sps:
        txt = sps.pop("normalized_trace", "")
        if txt:
            sps["source_hash"] = "sha256:" + hashlib.sha256(txt.encode("utf-8")).hexdigest()
            sps["trace_preview"] = txt[:200]

    return d

# --- I/O ---------------------------------------------------------------------

files = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
if DEBUG:
    print(f"[seg] input_dir={input_dir} files_found={len(files)} output_dir={output_dir}")
    print(f"[seg] run_version={RUN_VERSION} utc={RUN_TS_UTC} local={RUN_TS_LOCAL} tz={RUN_TZ} catalog_hash={CAT_HASH}")

for filename in files:
    src = os.path.join(input_dir, filename)
    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    reasoning = (
        data.get("reasoning_trace") or
        data.get("raw_output") or
        data.get("trace") or
        data.get("reasoning") or
        data.get("think") or
        data.get("cot") or
        data.get("rationale") or
        ""
    )

    seg = segment_by_triggers(reasoning) if reasoning else {"normalized_trace": "", "segments_char": []}
    data["segmented_phases_signals"] = seg

    usage = count_marker_usage(seg.get("normalized_trace", ""))

    data.setdefault("segmentation_meta", {})
    data["segmentation_meta"].update({
        "version": RUN_VERSION,
        "run_utc": RUN_TS_UTC,
        "run_local": RUN_TS_LOCAL,
        "run_tz": RUN_TZ,
        "catalog_hash": CAT_HASH,
        "marker_counts": {
            "clarify": len(problem_markers),
            "ideate": len(idea_markers),
            "evaluate": len(eval_markers),
            "finalize": len(answer_markers),
        },
        "marker_usage": usage
    })

    # Write both UTC and local timestamps
    data["timestamp_utc"] = RUN_TS_UTC
    data["timestamp_local"] = RUN_TS_LOCAL
    data["timestamp_tz"] = RUN_TZ
    data.pop("timestamp_utc_prev", None)

    data = prune_record(data)

    dst = os.path.join(output_dir, filename)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    if DEBUG:
        seg_for_dbg = data.get("segmented_phases_signals", {})
        phases = [s["phase"] for s in seg_for_dbg.get("segments_char", [])]
        cnt = Counter(phases)
        print(f"[seg] {filename}: phases={dict(cnt)} -> {dst}")
