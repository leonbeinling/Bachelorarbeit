import argparse
import datetime as dt
import json
import os
import re
import sys
from glob import glob
from typing import List, Dict, Any, Optional

# --- Defaults ------------------------------------------------------------------
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_INPUT_ROOT = os.path.join(BASE, "outputs", "4_manip")
DEFAULT_OUTPUT_ROOT = os.path.join(BASE, "outputs", "5_eval_manip", "segments")
MARKER_RE = re.compile(r"⟦([A-Z]+)→([A-Z]+)⟧")
PROTECTED_SIBLINGS = {"duplicate", "flexibility", "fluency", "originality"}

# --- FS helpers ----------------------------------------------------------------
def discover_manipulated_files(input_root: str) -> List[str]:
    return sorted(glob(os.path.join(input_root, "*", "*", "manipulated.json")))

def ensure_new_run_dir(output_root: str) -> str:
    # Keep siblings untouched
    parent = os.path.abspath(os.path.join(output_root, ".."))
    os.makedirs(parent, exist_ok=True)
    for name in PROTECTED_SIBLINGS:
        p = os.path.join(parent, name)
        if os.path.exists(p) and not os.path.isdir(p):
            raise RuntimeError(f"Protected sibling occupied by file: {p}")
    os.makedirs(output_root, exist_ok=True)
    runstamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, runstamp)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir

def mirror_subpath_under_run(manip_path: str, input_root: str, run_dir: str) -> str:
    # Recreate "<timestamp>/<test>" under the new run
    rel = os.path.relpath(os.path.dirname(manip_path), input_root)
    out_dir = os.path.join(run_dir, rel)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# --- IO ------------------------------------------------------------------------
def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        sys.stderr.write(f"[WARN] Failed to load JSON: {path} ({e})\n")
        return None

def write_pretty_blocks(path: str, rows: List[Dict[str, Any]]) -> None:
    # One pretty JSON object per block, separated by a blank line
    with open(path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            # enforce key order
            obj = {
                "task_id": row["task_id"],
                "source_path": row["source_path"],
                "segment_index": row["segment_index"],
                "marker": row["marker"],
                "gold_labels": row.get("gold_labels", None),
                "text": row["text"],
            }
            f.write(json.dumps(obj, ensure_ascii=False, indent=2))
            f.write("\n\n")  # blank line between segments

# --- Text → Segments -----------------------------------------------------------
def extract_segments_from_text(text: str) -> List[Dict[str, Any]]:
    # Cut from end-of-marker to start-of-next-marker
    segments = []
    matches = list(MARKER_RE.finditer(text))
    if not matches:
        return segments
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append({
            "marker": m.group(0),
            "start": start,
            "end": end,
            "text": text[start:end].strip()
        })
    return segments

# --- Main ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Export pretty phase-change segments for gold evaluation.")
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)

    run_dir = ensure_new_run_dir(output_root)
    print(f"[INFO] Output run dir: {run_dir}")

    files = discover_manipulated_files(input_root)
    if not files:
        print(f"[WARN] No manipulated.json found in {input_root}")
        return

    summary = {
        "run_dir": run_dir,
        "input_root": input_root,
        "found_files": len(files),
        "processed": 0,
        "skipped_no_text": 0,
        "skipped_no_markers": 0,
        "outputs": []
    }

    for i, manip_path in enumerate(files, 1):
        data = load_json(manip_path)
        if not data:
            summary["skipped_no_text"] += 1
            continue

        # Prefer swapped text
        text = (data.get("output_text_after_swap") or "").strip()
        if not text:
            summary["skipped_no_text"] += 1
            continue

        segs = extract_segments_from_text(text)
        if not segs:
            summary["skipped_no_markers"] += 1
            continue

        rows = []
        task_id = data.get("task_id")
        for idx, s in enumerate(segs):
            rows.append({
                "task_id": task_id,
                "source_path": manip_path,
                "segment_index": idx,
                "marker": s["marker"],
                "gold_labels": None,  # you will fill this later
                "text": s["text"]
            })

        out_dir = mirror_subpath_under_run(manip_path, input_root, run_dir)
        out_path = os.path.join(out_dir, "segments.jsonl")  # pretty, multi-line blocks
        write_pretty_blocks(out_path, rows)

        summary["processed"] += 1
        summary["outputs"].append({
            "source": manip_path,
            "segments_written": len(rows),
            "output_path": out_path
        })
        print(f"[OK] ({i}/{len(files)}) -> {out_path}")

    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("[DONE] Summary written.")

if __name__ == "__main__":
    main()
