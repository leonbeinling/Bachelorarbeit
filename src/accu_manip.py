#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, glob
from typing import List, Dict, Any, Optional, Tuple, Iterable
from collections import defaultdict

# ---------- CONFIG ----------
CONFIG = {
    "BASE": None,  # project root = parent of this file if None
    "IN_LABELED_ROOT": None,  # <BASE>/outputs/5_eval_manip/labeled_segments if None
    "OUT_ACC_ROOT": None,     # <BASE>/outputs/5_eval_manip/accuracy if None

    "LABELED_FILE_GLOBS": ["**/*.jsonl", "**/*.json"],

    "PHASES": ["Clarify", "Ideate", "Evaluate", "Finalize"],

    "SIGNALS": [
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
    ],

    "SIGNAL_TO_PHASE": {
        "that": "Evaluate",
        "that's": "Evaluate",
        "that's a": "Evaluate",
        "might be": "Evaluate",
        "let me check": "Evaluate",
        "make sure": "Evaluate",
        "maybe not": "Evaluate",
        "but that's": "Evaluate",
        "ensure": "Evaluate",
        "might not be": "Evaluate",
        "that's kind of": "Evaluate",
    },
    "DEFAULT_INTENDED_PHASE": "Evaluate",

    # fields in segment items
    "TEXT_KEYS": ["text", "sentence", "content", "raw"],
    "GOLD_KEYS": ["gold_labels", "gold", "gold_label", "gold_phase", "label_gold", "label"],
    "MARKER_KEYS": ["marker", "marker_label", "marker_phase", "marker_tag"],

    "ITEM_LIST_KEYS": ["segments", "sentences", "items", "data", "labeled", "per_sentence"],
}
# ---------------------------

# ---------- PATHS ----------
def _derive_paths() -> Tuple[str, str, str]:
    base = CONFIG["BASE"] or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    in_root = CONFIG["IN_LABELED_ROOT"] or os.path.join(base, "outputs", "5_eval_manip", "labeled_segments")
    out_root = CONFIG["OUT_ACC_ROOT"] or os.path.join(base, "outputs", "5_eval_manip", "accuracy")
    return base, in_root, out_root

# ---------- PHASE NORMALIZATION ----------
_CANON = {p.lower(): p for p in CONFIG["PHASES"]}
_PHASE_ALIASES = {
    "answer": "Finalize",
    "final": "Finalize",
    "finalise": "Finalize",
    "finalize": "Finalize",
    "evaluation": "Evaluate",
    "evaluate": "Evaluate",
    "ideation": "Ideate",
    "idea": "Ideate",
    "clarification": "Clarify",
    "clarify": "Clarify",
}

def norm_phase(x: Optional[str]) -> Optional[str]:
    if not isinstance(x, str):
        return None
    s = x.strip().lower()
    if not s:
        return None
    if s in _CANON: return _CANON[s]
    if s in _PHASE_ALIASES: return _PHASE_ALIASES[s]
    t = s.split()[0]
    if t in _CANON: return _CANON[t]
    if t in _PHASE_ALIASES: return _PHASE_ALIASES[t]
    return x.strip()

# Parse tags like "⟦IDEATE→EVALUATE⟧" or "⟦EVALUATE⟧"
_ARROW_RE = re.compile(r"\s*(?:⟦|[\[\(])\s*([A-Za-z]+)(?:\s*(?:→|->)\s*([A-Za-z]+))?\s*(?:⟧|[\]\)])\s*")

def parse_tag_phase(tag: Optional[str]) -> Optional[str]:
    if not isinstance(tag, str):
        return None
    m = _ARROW_RE.search(tag)
    if not m:
        return norm_phase(tag)  # fallback
    rhs = m.group(2) or m.group(1)
    return norm_phase(rhs)

def normalize_tag(tag: Optional[str]) -> Optional[str]:
    if not isinstance(tag, str):
        return None
    t = tag.strip().upper()
    t = t.replace("⟦", "[").replace("⟧", "]")
    t = t.replace("→", "->")
    t = re.sub(r"\s+", "", t)
    return t

# ---------- IO HELPERS ----------
def _get_first_str(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str):
            s = v.strip()
            if s:
                return s
    return None

def _iter_labeled_files(in_root: str) -> Iterable[str]:
    for pat in CONFIG["LABELED_FILE_GLOBS"]:
        for fp in glob.iglob(os.path.join(in_root, pat), recursive=True):
            if os.path.isfile(fp) and (fp.endswith(".jsonl") or fp.endswith(".json")):
                yield fp

# Robust JSONL loader (pretty-printed objects ok)
def _load_jsonl(fp: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(fp, "r", encoding="utf-8") as f:
        data = f.read()
    dec = json.JSONDecoder()
    i, n = 0, len(data)
    while True:
        while i < n and data[i].isspace():
            i += 1
        if i >= n:
            break
        try:
            obj, j = dec.raw_decode(data, i)
            if isinstance(obj, dict):
                rows.append(obj)
            i = j
        except json.JSONDecodeError:
            nl = data.find("\n", i)
            if nl == -1:
                break
            i = nl + 1
    return rows

def _load_json(fp: str) -> List[Dict[str, Any]]:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return obj
    if isinstance(obj, dict):
        for k in CONFIG["ITEM_LIST_KEYS"]:
            v = obj.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []

def load_examples(fp: str) -> List[Dict[str, Optional[str]]]:
    items = _load_jsonl(fp) if fp.endswith(".jsonl") else _load_json(fp)
    out: List[Dict[str, Optional[str]]] = []
    for it in items:
        text   = _get_first_str(it, CONFIG["TEXT_KEYS"])
        gold   = _get_first_str(it, CONFIG["GOLD_KEYS"])
        marker = _get_first_str(it, CONFIG["MARKER_KEYS"])
        out.append({"text": text, "gold": gold, "marker": marker})
    return out

# ---------- SIGNALS ----------
def _compile_signal_regexes(signals: List[str]) -> List[Tuple[str, re.Pattern]]:
    ordered = sorted(signals, key=lambda s: len(s), reverse=True)
    regs: List[Tuple[str, re.Pattern]] = []
    for s in ordered:
        pat = r"^\s*" + re.escape(s) + r"(?:\b|[\.:,;!?\-\u2014])"
        regs.append((s, re.compile(pat, flags=re.IGNORECASE)))
    return regs

_SIGNAL_REGEXES = _compile_signal_regexes(CONFIG["SIGNALS"])

def detect_signal(text: Optional[str]) -> Optional[str]:
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.strip().lstrip('"\''"“”‘’")
    for sig, rx in _SIGNAL_REGEXES:
        if rx.search(t):
            return sig
    return None

def intended_phase_for_signal(sig: Optional[str]) -> Optional[str]:
    if not sig:
        return None
    return norm_phase(CONFIG["SIGNAL_TO_PHASE"].get(sig.lower(), CONFIG["DEFAULT_INTENDED_PHASE"]))

# ---------- METRICS ----------
def compute_marker_vs_gold(examples: List[Dict[str, Optional[str]]]) -> Dict[str, Any]:
    total = 0
    correct = 0
    per_signal = defaultdict(lambda: {"matched": 0, "correct": 0, "wrong": 0})
    for ex in examples:
        g_tag = normalize_tag(ex.get("gold"))
        m_tag = normalize_tag(ex.get("marker"))
        if not g_tag or not m_tag:
            continue
        total += 1
        ok = (g_tag == m_tag)
        if ok:
            correct += 1
        sig = detect_signal(ex.get("text"))
        if sig:
            per_signal[sig]["matched"] += 1
            if ok: per_signal[sig]["correct"] += 1
            else:  per_signal[sig]["wrong"] += 1

    acc = (correct / total) if total > 0 else None
    per_signal_out = []
    for sig, d in sorted(per_signal.items(), key=lambda kv: kv[0]):
        sig_acc = (d["correct"] / d["matched"]) if d["matched"] > 0 else None
        per_signal_out.append({
            "signal": sig,
            "matched": d["matched"],
            "correct": d["correct"],
            "wrong": d["wrong"],
            "accuracy_marker_vs_gold": sig_acc
        })
    return {
        "with_marker": total,
        "marker_correct": correct,
        "marker_wrong": (total - correct) if total else 0,
        "marker_vs_gold_accuracy": acc,
        "per_signal_marker": per_signal_out
    }

def compute_signal_vs_gold(examples: List[Dict[str, Optional[str]]]) -> Dict[str, Any]:
    with_sig = 0
    correct = 0
    per_signal = defaultdict(lambda: {"matched": 0, "correct": 0, "wrong": 0})
    for ex in examples:
        sig = detect_signal(ex.get("text"))
        if not sig:
            continue
        gold_phase = parse_tag_phase(ex.get("gold"))
        pred_phase = intended_phase_for_signal(sig)
        if gold_phase is None or pred_phase is None:
            continue
        with_sig += 1
        ok = (gold_phase == pred_phase)
        per_signal[sig]["matched"] += 1
        if ok:
            correct += 1
            per_signal[sig]["correct"] += 1
        else:
            per_signal[sig]["wrong"] += 1

    acc = (correct / with_sig) if with_sig > 0 else None
    per_signal_out = []
    for sig, d in sorted(per_signal.items(), key=lambda kv: kv[0]):
        sig_acc = (d["correct"] / d["matched"]) if d["matched"] > 0 else None
        per_signal_out.append({
            "signal": sig,
            "intended_phase": intended_phase_for_signal(sig),
            "matched": d["matched"],
            "correct": d["correct"],
            "wrong": d["wrong"],
            "accuracy_signal_vs_gold": sig_acc
        })
    return {
        "with_signal": with_sig,
        "signal_correct": correct,
        "signal_wrong": (with_sig - correct) if with_sig else 0,
        "signal_vs_gold_accuracy": acc,
        "per_signal": per_signal_out
    }

# ---------- MAIN ----------
def main():
    base, in_root, out_root = _derive_paths()
    if not os.path.isdir(in_root):
        raise SystemExit(f"Labeled segments folder not found: {in_root}")
    os.makedirs(out_root, exist_ok=True)

    files = sorted(_iter_labeled_files(in_root))
    if not files:
        raise SystemExit(f"No labeled files under: {in_root}")

    stats = {
        "files_processed": [],
        "sentences_total": 0,
        "with_marker": 0,
        "marker_correct": 0,
        "marker_wrong": 0,
        "marker_vs_gold_accuracy": None,
        "with_signal": 0,
        "signal_correct": 0,
        "signal_wrong": 0,
        "signal_vs_gold_accuracy": None,
        "per_signal_marker": [],
        "per_signal_signal": [],
        "per_file": []
    }

    agg_sig_marker = defaultdict(lambda: {"matched": 0, "correct": 0, "wrong": 0})
    agg_sig_signal = defaultdict(lambda: {"matched": 0, "correct": 0, "wrong": 0})

    for fp in files:
        exs = load_examples(fp)
        if not exs:
            print(f"[SKIP] {fp} -> no examples")
            continue

        mvg = compute_marker_vs_gold(exs)
        svg = compute_signal_vs_gold(exs)

        stats["files_processed"].append(os.path.relpath(fp, base))
        stats["sentences_total"] += len(exs)

        stats["with_marker"] += mvg["with_marker"]
        stats["marker_correct"] += mvg["marker_correct"]
        stats["marker_wrong"] += mvg["marker_wrong"]

        stats["with_signal"] += svg["with_signal"]
        stats["signal_correct"] += svg["signal_correct"]
        stats["signal_wrong"] += svg["signal_wrong"]

        for row in mvg["per_signal_marker"]:
            s = row["signal"]
            agg_sig_marker[s]["matched"] += row["matched"]
            agg_sig_marker[s]["correct"] += row["correct"]
            agg_sig_marker[s]["wrong"] += row["wrong"]

        for row in svg["per_signal"]:
            s = row["signal"]
            agg_sig_signal[s]["matched"] += row["matched"]
            agg_sig_signal[s]["correct"] += row["correct"]
            agg_sig_signal[s]["wrong"] += row["wrong"]

        stats["per_file"].append({
            "file": os.path.relpath(fp, base),
            "sentences_total": len(exs),
            "with_marker": mvg["with_marker"],
            "marker_vs_gold_accuracy": mvg["marker_vs_gold_accuracy"],
            "with_signal": svg["with_signal"],
            "signal_vs_gold_accuracy": svg["signal_vs_gold_accuracy"],
        })

    if stats["with_marker"] > 0:
        stats["marker_vs_gold_accuracy"] = stats["marker_correct"] / stats["with_marker"]
    if stats["with_signal"] > 0:
        stats["signal_vs_gold_accuracy"] = stats["signal_correct"] / stats["with_signal"]

    out_marker = []
    for sig in sorted(agg_sig_marker.keys()):
        d = agg_sig_marker[sig]
        acc = (d["correct"] / d["matched"]) if d["matched"] > 0 else None
        out_marker.append({
            "signal": sig,
            "matched": d["matched"],
            "correct": d["correct"],
            "wrong": d["wrong"],
            "accuracy_marker_vs_gold": acc
        })
    stats["per_signal_marker"] = out_marker

    out_signal = []
    for sig in sorted(agg_sig_signal.keys()):
        d = agg_sig_signal[sig]
        acc = (d["correct"] / d["matched"]) if d["matched"] > 0 else None
        out_signal.append({
            "signal": sig,
            "intended_phase": intended_phase_for_signal(sig),
            "matched": d["matched"],
            "correct": d["correct"],
            "wrong": d["wrong"],
            "accuracy_signal_vs_gold": acc
        })
    stats["per_signal_signal"] = out_signal

    out_json = os.path.join(out_root, "accuracy_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[OK] accuracy summary -> {out_json}")

if __name__ == "__main__":
    main()
