import os, re, json, glob, math, unicodedata
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# -------- CONFIG --------
CONFIG = {
    "BASE": "/home/beinling/qwen-reasoning",
    "RUN_DIR_4MANIP": "/home/beinling/qwen-reasoning/outputs/4_manip/20250916_040418",
    "IN_FLUENCY_ROOT": None,   # derives from RUN_DIR_4MANIP if None
    "OUT_ORIG_ROOT": None,     # derives from RUN_DIR_4MANIP if None
    "FLUENCY_GLOBS": ["*/*_fluency_simple.json", "*/*_fluency.json"],
    "ORIG_MODEL_NAME": os.getenv("ORIG_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "COMPUTE_CUE_DISTANCE": os.getenv("ORIG_COMPUTE_CUE", "1") not in ("0", "false", "False"),
    "SUMMARY_CSV": "_summary_originality.csv",
    "SUMMARY_JSON": "_summary_originality.json",
}
# ------------------------

_model = None
def _get_model():
    """Load encoder once."""
    global _model
    if _model is None:
        _model = SentenceTransformer(CONFIG["ORIG_MODEL_NAME"])
    return _model

# -------- text helpers --------
def normalize_text(s: str) -> str:
    """Light unicode and whitespace cleanup."""
    s = s.replace("\\n", "\n")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("“", "\"").replace("”", "\"")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

def norm_for_dedup(s: str) -> str:
    """Lowercased, alnum+space only, single-spaced."""
    base = s.lower()
    base = re.sub(r"[^a-z0-9 ]+", " ", base)
    base = re.sub(r"\s+", " ", base).strip()
    return base

# -------- object extraction --------
# Matches your AUT pattern and a safe fallback.
OBJ_PATTERNS = [
    re.compile(
        r"uses\s+(?:as\s+many\s+unusual\s+and\s+creative\s+)?as\s+possible\s+for\s+(?:an?\s+|the\s+)?([^\.!\?]+?)\s*(?:[\.!\?]|\s+Provide\b|$)",
        flags=re.IGNORECASE
    ),
    re.compile(
        r"\bfor\s+(?:an?\s+|the\s+)?([^\.!\?]+?)\s*(?:[\.!\?]|$)",
        flags=re.IGNORECASE
    ),
]

def extract_object_from_prompt(prompt: str) -> str:
    """Extract AUT object from prompt; returns '' if not found."""
    if not isinstance(prompt, str) or not prompt.strip():
        return ""
    p = prompt.strip()
    for rx in OBJ_PATTERNS:
        m = rx.search(p)
        if m:
            obj = m.group(1).strip()
            obj = re.sub(r"\s+", " ", obj)
            obj = re.sub(r"[,:;]\s*$", "", obj)
            obj = re.sub(r"\bProvide the final answer.*$", "", obj, flags=re.IGNORECASE).strip()
            return obj
    return ""

# -------- I/O over fluency --------
def list_fluency_files(in_root: str) -> List[str]:
    """Collect fluency jsons under in_root."""
    out = []
    for pat in CONFIG["FLUENCY_GLOBS"]:
        out.extend(glob.glob(os.path.join(in_root, pat)))
    return sorted(set(p for p in out if os.path.isfile(p)))

def load_fluency(path: str) -> Tuple[str, Dict[str, Any], List[str], str, str]:
    """Return (task_id, meta, ideas, file_kind, aut_folder)."""
    with open(path, "r", encoding="utf-8") as f:
        flu = json.load(f)
    task_id = flu.get("task_id") or os.path.splitext(os.path.basename(path))[0].replace("_fluency", "")
    ideas = flu.get("items") or flu.get("unique_ideas") or []
    ideas = [str(x).strip() for x in ideas if isinstance(x, (str, bytes)) and str(x).strip()]
    file_kind = flu.get("file_kind") or ("manipulated" if "manipulated" in os.path.basename(path) else "baseline")
    aut_folder = os.path.basename(os.path.dirname(path))
    return task_id, flu, ideas, file_kind, aut_folder

# -------- pool building (within object) --------
def build_object_pools(fluency_paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Build per-object pools of unique idea strings across all tasks."""
    obj_to_map: Dict[str, Dict[str, str]] = {}
    for p in fluency_paths:
        _, flu, ideas, _, _ = load_fluency(p)
        obj = (flu.get("object") or "").strip()
        if not obj:
            obj = extract_object_from_prompt(normalize_text(flu.get("prompt", "")))
        if not obj:
            continue
        bucket = obj_to_map.setdefault(obj, {})
        for idea in ideas:
            k = norm_for_dedup(idea)
            if k and k not in bucket:
                bucket[k] = idea

    pools: Dict[str, Dict[str, Any]] = {}
    model = _get_model()
    for obj, d in obj_to_map.items():
        strings = list(d.values())
        norms   = list(d.keys())
        emb = model.encode(strings, normalize_embeddings=True, convert_to_numpy=True) if strings else np.zeros((0, 384), dtype=float)
        cue_emb = None
        if CONFIG["COMPUTE_CUE_DISTANCE"] and obj:
            cue_emb = model.encode([obj], normalize_embeddings=True, convert_to_numpy=True)[0]
        pools[obj] = {"strings": strings, "norms": norms, "emb": emb, "cue_emb": cue_emb}
    return pools

# -------- core computations --------
def nearest_neighbor_max_sim(
    idea_vec: np.ndarray,
    pool_emb: np.ndarray,
    pool_norms: List[str],
    idea_norm: str
) -> Tuple[Optional[float], Optional[int]]:
    """Max cosine similarity in pool, excluding exact duplicate."""
    if pool_emb.size == 0:
        return None, None
    sims = pool_emb @ idea_vec
    if idea_norm in pool_norms:
        dup = pool_norms.index(idea_norm)
        sims = sims.copy()
        sims[dup] = -np.inf
    idx = int(np.argmax(sims))
    s_max = float(sims[idx])
    if math.isfinite(s_max):
        return s_max, idx
    return None, None

def compute_originality_for_task(
    object_name: str,
    ideas: List[str],
    pools: Dict[str, Dict[str, Any]],
    model: SentenceTransformer
) -> Dict[str, Any]:
    """Compute rarity (1 - max NN sim) and optional cue distance."""
    if not object_name or not ideas:
        return {"orig_rare": [], "orig_cue": [], "orig_mean": None, "nn_sim": [], "nn_idea": [], "pool_size": 0}

    pool = pools.get(object_name, {"strings": [], "norms": [], "emb": np.zeros((0, 1)), "cue_emb": None})
    pool_emb  = pool["emb"]
    pool_norm = pool["norms"]
    cue_emb   = pool["cue_emb"]

    vecs = model.encode(ideas, normalize_embeddings=True, convert_to_numpy=True)
    norms = [norm_for_dedup(s) for s in ideas]

    orig_rare: List[Optional[float]] = []
    nn_sim:   List[Optional[float]] = []
    nn_idea:  List[Optional[str]]   = []
    for v, nrm in zip(vecs, norms):
        s_max, idx = nearest_neighbor_max_sim(v, pool_emb, pool_norm, nrm)
        if s_max is None:
            orig_rare.append(None); nn_sim.append(None); nn_idea.append(None)
        else:
            orig_rare.append(float(1.0 - s_max))
            nn_sim.append(float(s_max))
            nn_idea.append(pool["strings"][idx] if idx is not None else None)

    orig_cue: List[Optional[float]] = []
    if CONFIG["COMPUTE_CUE_DISTANCE"] and cue_emb is not None:
        for v in vecs:
            orig_cue.append(float(1.0 - float(np.dot(v, cue_emb))))
    else:
        orig_cue = [None] * len(ideas)

    valid = [x for x in orig_rare if x is not None]
    orig_mean = float(np.mean(valid)) if valid else None

    return {"orig_rare": orig_rare, "orig_cue": orig_cue, "orig_mean": orig_mean,
            "nn_sim": nn_sim, "nn_idea": nn_idea, "pool_size": int(pool_emb.shape[0])}

# -------- per-file report --------
def make_report_for_fluency_file(
    fluency_path: str,
    pools: Dict[str, Dict[str, Any]],
    out_root: str
) -> Dict[str, Any]:
    """Compute and write per-file originality report."""
    task_id, flu, ideas, file_kind, aut_folder = load_fluency(fluency_path)
    prompt = flu.get("prompt", "")
    object_name = (flu.get("object") or "").strip() or extract_object_from_prompt(prompt)

    model = _get_model()
    res = compute_originality_for_task(object_name, ideas, pools, model)

    out_dir = os.path.join(out_root, aut_folder)
    os.makedirs(out_dir, exist_ok=True)

    base = f"{task_id}_{file_kind}_originality"
    out_json = os.path.join(out_dir, f"{base}.json")
    out_csv  = os.path.join(out_dir, f"{base}.csv")

    report = {
        "task_id": task_id,
        "file_kind": file_kind,
        "aut_folder": aut_folder,
        "task_type": flu.get("task_type", ""),
        "object": object_name,
        "prompt": prompt,
        "encoder_model": CONFIG["ORIG_MODEL_NAME"],
        "n_ideas": len(ideas),
        "pool_size": res["pool_size"],
        "originality_mean": res["orig_mean"],
        "idea_texts": ideas,
        "idea_orig_rare": res["orig_rare"],
        "idea_nn_sim": res["nn_sim"],
        "idea_nn_idea": res["nn_idea"],
        "idea_orig_cue": res["orig_cue"],
        "source_fluency": os.path.abspath(fluency_path),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "idx": list(range(len(ideas))),
        "idea": ideas,
        "orig_rare": report["idea_orig_rare"],
        "nn_sim": report["idea_nn_sim"],
        "nn_idea": report["idea_nn_idea"],
        "orig_cue": report["idea_orig_cue"],
    }).to_csv(out_csv, index=False)

    print(f"[OK] {file_kind:<11} | {task_id:<10} | ideas:{len(ideas):>3} | pool:{res['pool_size']:>3} | orig:{report['originality_mean']}")
    return {
        "task_id": task_id,
        "file_kind": file_kind,
        "aut_folder": aut_folder,
        "n_ideas": len(ideas),
        "pool_size": res["pool_size"],
        "originality_mean": report["originality_mean"],
        "report_json": out_json,
        "report_csv": out_csv
    }

# -------- summary --------
def add_within_object_zscores(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add z-scores per object over originality_mean."""
    df = pd.DataFrame(rows)
    if df.empty or "object" not in df.columns:
        return rows
    objs = []
    for r in rows:
        with open(r["report_json"], "r", encoding="utf-8") as f:
            objs.append(json.load(f).get("object"))
    df["object"] = objs
    zcol = []
    for obj, grp in df.groupby("object"):
        vals = pd.to_numeric(grp["originality_mean"], errors="coerce")
        mean = vals.mean(skipna=True)
        std  = vals.std(ddof=0, skipna=True)
        for v in vals:
            zcol.append(float((v - mean) / std) if pd.notna(v) and pd.notna(std) and std > 0 else None)
    df["originality_mean_z"] = zcol
    return df.to_dict(orient="records")

# -------- main --------
def _derive_roots() -> Tuple[str, str, str, str]:
    """Resolve run dir and in/out roots."""
    run_dir = os.path.abspath(CONFIG["RUN_DIR_4MANIP"])
    run_name = os.path.basename(run_dir.rstrip("/"))
    in_root = CONFIG["IN_FLUENCY_ROOT"] or os.path.join(CONFIG["BASE"], "outputs", "5_eval_manip", "fluency", run_name)
    out_root = CONFIG["OUT_ORIG_ROOT"] or os.path.join(CONFIG["BASE"], "outputs", "5_eval_manip", "originality", run_name)
    return run_dir, run_name, os.path.abspath(in_root), os.path.abspath(out_root)

def main():
    """Entry point."""
    run_dir, run_name, in_root, out_root = _derive_roots()
    if not os.path.isdir(in_root):
        raise SystemExit(f"Fluency root not found: {in_root}")
    os.makedirs(out_root, exist_ok=True)

    flu_paths = list_fluency_files(in_root)
    if not flu_paths:
        raise SystemExit(f"No fluency jsons under: {in_root}")

    print("[INFO] building object pools ...")
    pools = build_object_pools(flu_paths)
    print(f"[INFO] pools built for {len(pools)} objects")

    summary_rows = []
    for fp in flu_paths:
        try:
            rep = make_report_for_fluency_file(fp, pools, out_root)
            summary_rows.append(rep)
        except Exception as e:
            print(f"[SKIP] {os.path.basename(fp)} -> {e}")

    if summary_rows:
        summary_rows = add_within_object_zscores(summary_rows)
        sum_csv = os.path.join(out_root, CONFIG["SUMMARY_CSV"])
        sum_json = os.path.join(out_root, CONFIG["SUMMARY_JSON"])
        pd.DataFrame(summary_rows).to_csv(sum_csv, index=False)
        with open(sum_json, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        print(f"\nSummary: {sum_csv}\nOut root: {out_root}")
    else:
        print("[INFO] No originality reports created.")

if __name__ == "__main__":
    main()
