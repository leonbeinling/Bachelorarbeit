import os, re, json, glob
import pandas as pd
from collections import Counter

# ---------- CONFIG ----------
CONFIG = {
    "RUN_DIR": "/home/beinling/qwen-reasoning/outputs/4_manip/20250916_040418",
    "OUT_ROOT": None,
    "AUT_DIR_GLOB": "*_aut_*",
    "BASELINE_NAME": "baseline.json",
    "MANIPULATED_NAME": "manipulated.json",
    "ANSWER_FIELDS_BASE": ["final_answer", "raw_output", "output_text", "text"],
    "ANSWER_FIELDS_MANIP": ["output_text_after_swap", "final_answer", "raw_output", "output_text", "text"],
    "NUMBERED_PATTERN": r"^\s*(\d+)\s*[.)]\s+(.*\S)\s*$",
    "COUNT_MODE": "last",
    "SUMMARY_CSV": "_summary_fluency_simple.csv",
    "SUMMARY_JSON": "_summary_fluency_simple.json",
    "REPORT_SUFFIX": "_fluency_simple",
    "WRITE_ITEMS_CSV": True,
    "STRICT_AUT_FOLDERS": True,
    # Soft-overlap threshold (token Jaccard)
    "SOFT_JACCARD_THR": 0.65,
    # Max pairs to store per task for readability (example list only; counting unaffected)
    "SOFT_MAX_PAIRS": None,
}
# ----------------------------

NUMBERED_RE = re.compile(CONFIG["NUMBERED_PATTERN"])
PROMPT_FIELDS = ["prompt", "composed_prompt"]

def _norm(s: str) -> str:
    """String norm for exact duplicates."""
    t = s.lower().strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w ]+", "", t)
    return t

def _token_set(s: str) -> set:
    """Token set for soft overlap."""
    t = re.sub(r"[^\w ]+", " ", s.lower())
    return set(w for w in t.split() if w)

def read_answer_text(d: dict, file_kind: str) -> str:
    """Pick answer field by file kind."""
    fields = CONFIG["ANSWER_FIELDS_MANIP"] if file_kind == "manipulated" else CONFIG["ANSWER_FIELDS_BASE"]
    for k in fields:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def read_prompt(d: dict) -> str:
    """Get prompt with fallback."""
    for k in PROMPT_FIELDS:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def find_numbered_matches(text: str):
    """Return (idx, num, item) for numbered lines."""
    lines = text.splitlines()
    out = []
    for i, ln in enumerate(lines):
        m = NUMBERED_RE.match(ln)
        if m:
            out.append((i, int(m.group(1)), m.group(2).strip()))
    return out

def extract_items(text: str):
    """Extract numbered items per COUNT_MODE."""
    matches = find_numbered_matches(text)
    if not matches:
        return []
    if CONFIG["COUNT_MODE"] == "all":
        return [it for (_i, _n, it) in matches if it]
    last_one = None
    for idx in range(len(matches) - 1, -1, -1):
        if matches[idx][1] == 1:
            last_one = idx; break
    if last_one is None:
        window = matches[-10:]
        return [it for (_i, _n, it) in window if it]
    return [it for (_i, _n, it) in matches[last_one:] if it]

def make_report(input_path: str, outdir: str, file_kind: str):
    """Write per-file fluency report."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_id = data.get("task_id") or os.path.splitext(os.path.basename(input_path))[0]
    text = read_answer_text(data, file_kind)
    items = extract_items(text)
    fluency = len(items)
    prompt = read_prompt(data)

    os.makedirs(outdir, exist_ok=True)
    base = f"{task_id}_{file_kind}{CONFIG['REPORT_SUFFIX']}"
    out_json = os.path.join(outdir, f"{base}.json")
    out_csv  = os.path.join(outdir, f"{base}.csv")

    report = {
        "task_id": task_id,
        "file_kind": file_kind,
        "count_mode": CONFIG["COUNT_MODE"],
        "fluency": fluency,
        "items": items,
        "prompt": prompt,
        "source_file": os.path.abspath(input_path),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if CONFIG["WRITE_ITEMS_CSV"]:
        pd.DataFrame({"item": items}).to_csv(out_csv, index=False)

    print(f"[OK] {file_kind:<11} | {task_id:<12} | Fluency: {fluency:>4}")
    return {
        "aut_folder": os.path.basename(os.path.dirname(input_path)),
        "task_id": task_id,
        "file_kind": file_kind,
        "fluency": fluency,
        "items": items,  # used for duplicate metric
        "report_json": out_json,
        "report_csv": out_csv if CONFIG["WRITE_ITEMS_CSV"] else None,
    }

def find_aut_pairs(run_dir: str):
    """Find baseline/manipulated paths per AUT folder."""
    pairs = []
    for d in sorted(glob.glob(os.path.join(run_dir, CONFIG["AUT_DIR_GLOB"]))):
        base = os.path.join(d, CONFIG["BASELINE_NAME"])
        mani = os.path.join(d, CONFIG["MANIPULATED_NAME"])
        if os.path.isfile(base) or os.path.isfile(mani):
            pairs.append({
                "dir": d,
                "baseline": base if os.path.isfile(base) else None,
                "manipulated": mani if os.path.isfile(mani) else None
            })
    return pairs

# ---------- DUPLICATE + OVERLAP ----------
def _dup_stats(items: list) -> dict:
    """Internal duplicates inside one list."""
    norms = [_norm(x) for x in items]
    cnt = Counter(norms)
    dup_list = sorted([k for k, c in cnt.items() if c > 1])
    dup_count = sum(c - 1 for c in cnt.values() if c > 1)
    return {"unique": len(cnt), "dups_total": dup_count, "duplicates": dup_list}

def _overlap_stats(base_items: list, mani_items: list) -> dict:
    """Exact overlap between baseline and manipulated."""
    bs = { _norm(x): x for x in base_items }
    ms = { _norm(x): x for x in mani_items }
    inter_keys = sorted(set(bs.keys()) & set(ms.keys()))
    union_keys = set(bs.keys()) | set(ms.keys())
    return {
        "intersection_count": len(inter_keys),
        "jaccard": (len(inter_keys) / len(union_keys)) if union_keys else 0.0,
        "intersection_examples": [bs[k] for k in inter_keys][:20]
    }

def _soft_overlap_stats(base_items: list, mani_items: list, thr: float, cap: int) -> dict:
    """Token-Jaccard soft overlap (greedy best match per baseline idea).
    Counts ALL qualifying overlaps; optionally caps only the number of stored example pairs."""
    bs = list(dict.fromkeys(base_items))  # keep order, unique
    ms = list(dict.fromkeys(mani_items))
    ms_tokens = [(_token_set(x), x) for x in ms]

    pairs = []
    count_all = 0  # counts all matches >= thr

    for vb in bs:
        tb = _token_set(vb)
        if not tb:
            continue
        best = 0.0; match = None
        for tm, vm in ms_tokens:
            inter = len(tb & tm); union = len(tb | tm) or 1
            j = inter / union
            if j > best:
                best, match = j, vm
        if best >= thr and match is not None:
            count_all += 1
            # Only cap how many example pairs we STORE (not how many we COUNT)
            if cap is None or len(pairs) < int(cap):
                pairs.append({"baseline": vb, "manipulated": match, "jaccard": round(best, 3)})

    return {"soft_intersection_count": count_all, "soft_pairs": pairs}

def write_duplicate_summary(summary_rows: list, proj_root: str, run_name: str):
    """Write duplicates.json with exact + soft overlaps."""
    dup_root = os.path.join(proj_root, "outputs", "5_eval_manip", "duplicate", run_name)
    os.makedirs(dup_root, exist_ok=True)
    per_task = []
    glob_intersections = Counter()
    totals = {
        "tasks": 0,
        "baseline_dups_total": 0,
        "manipulated_dups_total": 0,
        "intersections_total": 0,
        "soft_intersections_total": 0,
    }

    # task_id -> items per kind
    by_task = {}
    for r in summary_rows:
        key = r["task_id"]
        by_task.setdefault(key, {})[r["file_kind"]] = r.get("items", [])

    for task_id, d in sorted(by_task.items()):
        base_items = d.get("baseline", []) or []
        mani_items = d.get("manipulated", []) or []

        base_stats = _dup_stats(base_items)
        mani_stats = _dup_stats(mani_items)
        ov = _overlap_stats(base_items, mani_items)
        sov = _soft_overlap_stats(base_items, mani_items, CONFIG["SOFT_JACCARD_THR"], CONFIG["SOFT_MAX_PAIRS"])

        # collect exact overlaps globally
        for s in ov["intersection_examples"]:
            glob_intersections[_norm(s)] += 1

        per_task.append({
            "task_id": task_id,
            "baseline": {
                "n_items": len(base_items),
                "unique": base_stats["unique"],
                "dups_total": base_stats["dups_total"],
                "duplicates": base_stats["duplicates"]
            },
            "manipulated": {
                "n_items": len(mani_items),
                "unique": mani_stats["unique"],
                "dups_total": mani_stats["dups_total"],
                "duplicates": mani_stats["duplicates"]
            },
            "overlap": ov,
            "soft_overlap": sov,
        })

        totals["tasks"] += 1
        totals["baseline_dups_total"] += base_stats["dups_total"]
        totals["manipulated_dups_total"] += mani_stats["dups_total"]
        totals["intersections_total"] += ov["intersection_count"]
        totals["soft_intersections_total"] += sov["soft_intersection_count"]

    top_overlap_examples = []
    for k, c in glob_intersections.most_common(50):
        top_overlap_examples.append({"idea_norm": k, "count": c})

    out = {
        "totals": totals,
        "per_task": per_task,
        "top_overlap_examples": top_overlap_examples,
        "params": {
            "soft_jaccard_thr": CONFIG["SOFT_JACCARD_THR"],
            "soft_max_pairs": CONFIG["SOFT_MAX_PAIRS"],
        }
    }
    with open(os.path.join(dup_root, "duplicates.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[INFO] duplicate summary -> {os.path.join(dup_root, 'duplicates.json')}")
# -------------------------------------

def main():
    run_dir = os.path.abspath(CONFIG["RUN_DIR"])
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run folder not found: {run_dir}")

    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    run_name  = os.path.basename(run_dir.rstrip("/"))
    out_root  = os.path.join(proj_root, "outputs", "5_eval_manip", "fluency", run_name) \
                if not CONFIG["OUT_ROOT"] else os.path.abspath(CONFIG["OUT_ROOT"])
    os.makedirs(out_root, exist_ok=True)

    pairs = find_aut_pairs(run_dir)
    if not pairs:
        msg = f"No AUT folders found under: {run_dir}"
        if CONFIG["STRICT_AUT_FOLDERS"]:
            raise SystemExit(msg)
        print("[INFO]", msg)
        return

    summary = []
    for p in pairs:
        aut_dir = p["dir"]
        out_dir_aut = os.path.join(out_root, os.path.basename(aut_dir))
        os.makedirs(out_dir_aut, exist_ok=True)

        for kind in ("baseline", "manipulated"):
            fp = p.get(kind)
            if not fp:
                print(f"[SKIP] {kind} missing in {aut_dir}")
                continue
            try:
                rep = make_report(fp, out_dir_aut, file_kind=kind)
                summary.append(rep)
            except Exception as e:
                print(f"[SKIP] {kind} in {aut_dir} -> {e}")

    if summary:
        # fluency summaries
        sum_csv  = os.path.join(out_root, CONFIG["SUMMARY_CSV"])
        sum_json = os.path.join(out_root, CONFIG["SUMMARY_JSON"])
        pd.DataFrame(summary).to_csv(sum_csv, index=False)
        with open(sum_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nGlobal summary: {sum_csv}")

        # duplicate + overlap summary
        write_duplicate_summary(summary, proj_root, run_name)
    else:
        print("[INFO] No reports created.")

if __name__ == "__main__":
    main()
