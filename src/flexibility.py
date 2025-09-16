import os, re, json, glob, unicodedata
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

# -------- CONFIG --------
CONFIG = {
    "BASE": "/home/beinling/qwen-reasoning",
    "RUN_DIR_4MANIP": "/home/beinling/qwen-reasoning/outputs/4_manip/20250916_040418",

    # Auto-derive from RUN_DIR_4MANIP if None
    "IN_FLUENCY_ROOT": None,   # e.g. .../outputs/5_eval_manip/fluency/<RUN_NAME>
    "OUT_FLEX_ROOT": None,     # e.g. .../outputs/5_eval_manip/flexibility/<RUN_NAME>

    # Accept both simple and old fluency formats
    "FLUENCY_GLOBS": ["*/*_fluency_simple.json", "*/*_fluency.json"],

    # Raw text fields (fallback parsing)
    "ANSWER_FIELDS_BASE": ["final_answer", "raw_output", "output_text", "text"],
    "ANSWER_FIELDS_MANIP": ["output_text_after_swap", "final_answer", "raw_output", "output_text", "text"],

    # Encoder
    "FLEX_MODEL_NAME": os.getenv("FLEX_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),

    # Optional clustering
    "ROBUSTNESS_CLUSTER": False,
    "CLUSTER_LINKAGE": "average",
    "CLUSTER_SIM_THR": 0.55,
    "CLUSTER_CAP_K": None,  # int or None
    "MIN_CLUSTER_SIZE": 2,
    "ASSIGN_THR": 0.50,

    # Summary files
    "SUMMARY_CSV": "_summary_flexibility.csv",
    "SUMMARY_JSON": "_summary_flexibility.json",
}
# ------------------------

_model = None
def _get_model():
    """Load model once."""
    global _model
    if _model is None:
        _model = SentenceTransformer(CONFIG["FLEX_MODEL_NAME"])
    return _model

def _normalize_text(s: str) -> str:
    """Light unicode/space normalization."""
    s = s.replace("\\n", "\n")
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("“", "\"").replace("”", "\"")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    return s.strip()

def _lines_from_lists(text: str) -> List[str]:
    """Extract numbered or bullet list lines; fallback to non-empty lines."""
    items = re.findall(r"(?m)^\s*\d+[.)]\s+(.*)", text)
    if not items:
        items = re.findall(r"(?m)^\s*[-–—•]\s+(.*)", text)
    if not items:
        items = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return [itm.strip() for itm in items if itm.strip()]

def _clean_idea_text(raw: str) -> str:
    """Strip simple markup and titles."""
    t = raw.strip()
    t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
    t = re.sub(r"\*(.*?)\*", r"\1", t)
    parts = re.split(r"\s+[–—-]\s+", t, maxsplit=1)
    if len(parts) == 2:
        t = parts[1].strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _read_text_from_raw(data: Dict[str, Any], file_kind: str) -> str:
    """Pick text from raw baseline/manipulated json."""
    fields = CONFIG["ANSWER_FIELDS_MANIP"] if file_kind == "manipulated" else CONFIG["ANSWER_FIELDS_BASE"]
    for k in fields:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""

def _fallback_ideas_from_source(source_file: str, file_kind: str) -> List[str]:
    """Parse ideas from the original baseline/manipulated json."""
    if not source_file or not os.path.isfile(source_file):
        return []
    with open(source_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    txt = _normalize_text(_read_text_from_raw(data, file_kind))
    items = _lines_from_lists(txt)
    ideas = [_clean_idea_text(x) for x in items if x.strip()]
    seen = set(); uniq = []
    for s in ideas:
        base = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
        base = re.sub(r"\s+", " ", base).strip()
        if base in seen:
            continue
        seen.add(base)
        uniq.append(s)
    return uniq

def _load_ideas_from_fluency(path: str) -> Tuple[str, str, List[str], Dict[str, Any]]:
    """Read ideas and metadata from a fluency report json."""
    with open(path, "r", encoding="utf-8") as f:
        flu = json.load(f)
    task_id = flu.get("task_id") or os.path.splitext(os.path.basename(path))[0].replace("_fluency", "")
    file_kind = flu.get("file_kind") or ("manipulated" if "manipulated" in os.path.basename(path) else "baseline")
    ideas = flu.get("items") or flu.get("unique_ideas") or []
    ideas = [str(x).strip() for x in ideas if isinstance(x, (str, bytes)) and str(x).strip()]
    return task_id, file_kind, ideas, flu

def compute_adjacent_distances(ideas: List[str]) -> Dict[str, Any]:
    """Compute 1 - cosine on adjacent pairs; return mean."""
    n = len(ideas)
    if n <= 1:
        return {"flex_adj_mean": None, "adj_sims": [], "adj_dists": []}
    emb = _get_model().encode(ideas, normalize_embeddings=True, convert_to_numpy=True)
    sims = np.sum(emb[:-1] * emb[1:], axis=1)
    sims = np.clip(sims, -1.0, 1.0).astype(float)
    dists = (1.0 - sims).astype(float)
    return {"flex_adj_mean": float(np.mean(dists)),
            "adj_sims": sims.tolist(),
            "adj_dists": dists.tolist()}

def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    return X @ X.T

def _medoid_index(idxs: List[int], S: np.ndarray) -> int:
    sub = S[np.ix_(idxs, idxs)]
    return idxs[int(np.argmax(sub.sum(axis=1)))]

def compute_clusters(
    ideas: List[str],
    linkage: str,
    sim_thr: float,
    cap_k: Optional[int],
    min_cluster_size: int,
    assign_thr: float
) -> Dict[str, Any]:
    """Optional HAC clustering with small-cluster reassign."""
    n = len(ideas)
    if n == 0:
        return {"k": 0, "assignments": [], "clusters": [], "exemplars": []}
    if n == 1:
        return {"k": 1, "assignments": [0], "clusters": [[0]], "exemplars": [ideas[0]]}

    emb = _get_model().encode(ideas, normalize_embeddings=True, convert_to_numpy=True)
    if cap_k:
        hac = AgglomerativeClustering(n_clusters=min(cap_k, n), linkage=linkage, metric="cosine")
    else:
        hac = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0 - float(sim_thr),
                                      linkage=linkage, metric="cosine")
    labels = hac.fit_predict(emb)

    S = _cosine_sim_matrix(emb)
    clusters: Dict[int, List[int]] = {}
    for i, c in enumerate(labels.astype(int)):
        clusters.setdefault(c, []).append(i)

    if min_cluster_size > 1 and len(clusters) > 1:
        medoids = {cid: _medoid_index(idxs, S) for cid, idxs in clusters.items()}
        medoid_ids = list(medoids.keys())
        medoid_vec = np.array([medoids[cid] for cid in medoid_ids], dtype=int)

        small = [cid for cid, idxs in clusters.items() if len(idxs) < min_cluster_size]
        for cid in small:
            for idx in clusters[cid]:
                sims = S[idx, medoid_vec]
                if cid in medoid_ids:
                    sims[medoid_ids.index(cid)] = -1.0
                best_pos = int(np.argmax(sims))
                if sims[best_pos] >= assign_thr:
                    best_cid = medoid_ids[best_pos]
                    clusters.setdefault(best_cid, []).append(idx)
            clusters[cid] = []
        clusters = {cid: sorted(idxs) for cid, idxs in clusters.items() if len(idxs) > 0}

    comps = [idxs for _, idxs in sorted(clusters.items(), key=lambda kv: kv[0])]
    exemplars = [ideas[_medoid_index(idxs, S)] for idxs in comps]

    reassign = {}
    for new_id, idxs in enumerate(comps):
        for i in idxs:
            reassign[i] = new_id
    assignments = [reassign[i] for i in range(n)]

    return {"k": len(comps), "assignments": assignments, "clusters": comps, "exemplars": exemplars}

def _list_fluency_jsons(in_root: str) -> List[str]:
    """Collect fluency json files under in_root."""
    out = []
    for pat in CONFIG["FLUENCY_GLOBS"]:
        out.extend(glob.glob(os.path.join(in_root, pat)))
    return sorted(set(p for p in out if os.path.isfile(p)))

def _derive_roots() -> Tuple[str, str, str, str]:
    """Resolve run_dir, run_name, in_root, out_root."""
    run_dir = os.path.abspath(CONFIG["RUN_DIR_4MANIP"])
    run_name = os.path.basename(run_dir.rstrip("/"))
    in_root = CONFIG["IN_FLUENCY_ROOT"] or os.path.join(CONFIG["BASE"], "outputs", "5_eval_manip", "fluency", run_name)
    out_root = CONFIG["OUT_FLEX_ROOT"] or os.path.join(CONFIG["BASE"], "outputs", "5_eval_manip", "flexibility", run_name)
    return run_dir, run_name, os.path.abspath(in_root), os.path.abspath(out_root)

def main():
    run_dir, run_name, in_root, out_root = _derive_roots()
    if not os.path.isdir(in_root):
        raise SystemExit(f"Fluency root not found: {in_root}")
    os.makedirs(out_root, exist_ok=True)

    flu_files = _list_fluency_jsons(in_root)
    if not flu_files:
        raise SystemExit(f"No fluency json files under: {in_root}")

    summary_rows = []
    for fp in flu_files:
        with open(fp, "r", encoding="utf-8") as f:
            flu = json.load(f)

        task_id = flu.get("task_id") or os.path.splitext(os.path.basename(fp))[0].replace("_fluency", "")
        file_kind = flu.get("file_kind") or ("manipulated" if "manipulated" in os.path.basename(fp) else "baseline")
        aut_folder = os.path.basename(os.path.dirname(fp))
        ideas = flu.get("items") or flu.get("unique_ideas") or []
        ideas = [str(x).strip() for x in ideas if isinstance(x, (str, bytes)) and str(x).strip()]
        if not ideas:
            source_file = flu.get("source_file")
            ideas = _fallback_ideas_from_source(source_file, file_kind)

        adj = compute_adjacent_distances(ideas)

        cluster_info = None
        if CONFIG["ROBUSTNESS_CLUSTER"]:
            cluster_info = compute_clusters(
                ideas,
                linkage=CONFIG["CLUSTER_LINKAGE"],
                sim_thr=CONFIG["CLUSTER_SIM_THR"],
                cap_k=CONFIG["CLUSTER_CAP_K"],
                min_cluster_size=CONFIG["MIN_CLUSTER_SIZE"],
                assign_thr=CONFIG["ASSIGN_THR"],
            )

        out_dir_aut = os.path.join(out_root, aut_folder)
        os.makedirs(out_dir_aut, exist_ok=True)

        base = f"{task_id}_{file_kind}_flexibility"
        out_json = os.path.join(out_dir_aut, f"{base}.json")
        out_csv  = os.path.join(out_dir_aut, f"{base}.csv")

        report = {
            "task_id": task_id,
            "file_kind": file_kind,
            "aut_folder": aut_folder,
            "encoder_model": CONFIG["FLEX_MODEL_NAME"],
            "n_ideas": len(ideas),
            "flex_adj_mean": adj["flex_adj_mean"],
            "adjacent_sims": adj["adj_sims"],
            "adjacent_dists": adj["adj_dists"],
            "source_fluency": os.path.abspath(fp),
        }
        if cluster_info:
            report.update({
                "cluster_k": cluster_info["k"],
                "cluster_exemplars": cluster_info["exemplars"],
                "clusters": cluster_info["clusters"],
            })

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        df = pd.DataFrame({
            "idx": list(range(len(ideas))),
            "idea": ideas,
            "adj_sim_to_prev": [np.nan] + report["adjacent_sims"],
            "adj_dist_to_prev": [np.nan] + report["adjacent_dists"],
        })
        if cluster_info:
            df["cluster_id"] = cluster_info["assignments"]
        df.to_csv(out_csv, index=False)

        summary_rows.append({
            "aut_folder": aut_folder,
            "task_id": task_id,
            "file_kind": file_kind,
            "n_ideas": len(ideas),
            "flex_adj_mean": report["flex_adj_mean"],
            "report_json": out_json,
            "report_csv": out_csv,
        })
        print(f"[OK] {file_kind:<11} | {task_id:<10} | ideas:{len(ideas):>3} | flex:{str(report['flex_adj_mean'])}")

    sum_csv  = os.path.join(out_root, CONFIG["SUMMARY_CSV"])
    sum_json = os.path.join(out_root, CONFIG["SUMMARY_JSON"])
    pd.DataFrame(summary_rows).to_csv(sum_csv, index=False)
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)
    print(f"\nSummary: {sum_csv}\nOut root: {out_root}")

if __name__ == "__main__":
    main()
