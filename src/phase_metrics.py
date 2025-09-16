import argparse
import json
import os
import re
import sys
import csv
import shutil
from glob import glob
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PHASES = ["Clarify", "Ideate", "Evaluate", "Finalize"]
FAMILIES = ["aut", "gsm", "crt"]
FILENAME_RE = re.compile(r"^(aut|gsm|crt)_[0-9]+\.gold\.json$")

# Canonicalization for phase keys coming from marker_usage (e.g., 'CLARIFY')
PHASE_CANON = {
    "clarify": "Clarify",
    "ideate": "Ideate",
    "evaluate": "Evaluate",
    "finalize": "Finalize",
}

def canon_phase(key: str) -> str:
    """Maps arbitrary phase keys (e.g., 'CLARIFY') to canonical names."""
    if not isinstance(key, str):
        return ""
    return PHASE_CANON.get(key.strip().lower(), "")

def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(BASE, p)

def clean_dir(p: str) -> None:
    """Removes previous output files, but never touches input files."""
    ensure_dir(p)
    for name in os.listdir(p):
        fp = os.path.join(p, name)
        if os.path.isdir(fp):
            shutil.rmtree(fp, ignore_errors=True)
        else:
            try:
                os.remove(fp)
            except FileNotFoundError:
                pass

def primary(label_list: List[str]) -> str:
    """Returns the first label as the primary phase (empty if none)."""
    if not label_list:
        return ""
    return label_list[0]

def norm_marker(s: str) -> str:
    return s.strip().lower()

def norm_for_match(s: str) -> str:
    """Normalizes text for simple substring matching (space- and char-robust)."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", s.lower())).strip()

def marker_in_text(text: str, marker: str) -> bool:
    """Heuristic: considers marker present if its normalized form occurs as a token substring."""
    nt = norm_for_match(text)
    nm = norm_for_match(marker)
    if not nt or not nm:
        return False
    return f" {nm} " in f" {nt} "

def init_confusion_with_none(phases: List[str]) -> Dict[str, Dict[str, int]]:
    """Creates Gold×Pred confusion skeleton incl. a 'NONE' column/row for missing primary labels."""
    cols = phases + ["NONE"]
    return {g: {p: 0 for p in cols} for g in cols}

def init_transitions_confusion(phases: List[str]) -> Dict[str, Dict[str, int]]:
    """Creates GoldTransition×PredTransition confusion skeleton."""
    keys = [f"{a}->{b}" for a in phases for b in phases]
    keys.append("UNK")
    return {g: {p: 0 for p in keys} for g in keys}

def add_confusion(conf: Dict[str, Dict[str, int]], gold: str, pred: str) -> None:
    if gold in conf and pred in conf[gold]:
        conf[gold][pred] += 1

def add_transition_confusion(conf: Dict[str, Dict[str, int]], gold_t: str, pred_t: str) -> None:
    if gold_t not in conf:
        gold_t = "UNK"
    if pred_t not in conf[gold_t]:
        pred_t = "UNK"
    conf[gold_t][pred_t] += 1

def collect_intended_markers(data: dict) -> Dict[str, str]:
    """
    Builds {normalized_marker -> intended_phase} from per-file 'marker_usage'.
    'trigger_markers' are ignored by design; only the declared catalog is used.
    Accepts dict values as {"marker": count, ...} or list ["marker", ...].
    """
    out = {}
    mu = data.get("marker_usage", {}) or {}
    if not isinstance(mu, dict):
        return out
    for ph_key, mpairs in mu.items():
        ph = canon_phase(ph_key)
        if not ph:
            continue
        if isinstance(mpairs, dict):
            for m in mpairs.keys():
                if isinstance(m, str) and m.strip():
                    out[norm_marker(m)] = ph
        elif isinstance(mpairs, list):
            for m in mpairs:
                if isinstance(m, str) and m.strip():
                    out[norm_marker(m)] = ph
    return out

def compute_prf_from_confusion(conf: Dict[str, Dict[str, int]], phases: List[str]) -> Dict:
    """
    Computes per-phase Precision/Recall/F1 from a Gold×Pred confusion matrix.
    Also returns micro and macro aggregates.
    """
    per_phase = {}
    total_tp = total_fp = total_fn = 0

    for ph in phases:
        tp = conf.get(ph, {}).get(ph, 0)
        fp = sum(conf.get(g, {}).get(ph, 0) for g in phases if g != ph)
        fn = sum(conf.get(ph, {}).get(p, 0) for p in phases if p != ph)

        prec = tp / (tp + fp) if (tp + fp) > 0 else None
        rec = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None

        per_phase[ph] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1
        }

        total_tp += tp
        total_fp += fp
        total_fn += fn

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else None
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else None
    micro_f1   = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec is not None and micro_rec is not None and (micro_prec + micro_rec) > 0) else None

    def mean_ignore_none(vals: List[float]) -> float:
        vals = [v for v in vals if v is not None]
        return (sum(vals) / len(vals)) if vals else None

    macro_prec = mean_ignore_none([per_phase[ph]["precision"] for ph in phases])
    macro_rec  = mean_ignore_none([per_phase[ph]["recall"] for ph in phases])
    macro_f1   = mean_ignore_none([per_phase[ph]["f1"] for ph in phases])

    return {
        "per_phase": per_phase,
        "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
        "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1}
    }

def compute_metrics_for_files(file_paths: List[str]) -> Dict:
    """
    Computes all requested metrics from a list of *.gold.json files.
    Input files are read-only; no file is modified.
    """
    counts_pred = Counter()
    counts_gold = Counter()
    files_processed = []
    total_sent = 0

    correct_any_overlap = 0
    correct_primary = 0

    per_phase_gold_tot = Counter()
    per_phase_gold_correct = Counter()

    confusion_primary = init_confusion_with_none(PHASES)

    transitions_total = 0
    transitions_correct_primary = 0
    transitions_confusion = init_transitions_confusion(PHASES)
    transition_failure_counter = Counter()
    missed_transitions = 0
    wrong_transitions = 0

    marker_counter = Counter()
    marker_stats = {}

    for path in file_paths:
        try:
            if not os.path.isfile(path):
                continue
            fname = os.path.basename(path)
            if not FILENAME_RE.match(fname):
                continue

            data = read_json(path)
            labels = data.get("labels", []) or []
            if not isinstance(labels, list):
                continue

            intended = collect_intended_markers(data)  # {marker -> intended phase}
            files_processed.append(fname)

            gold_primary_seq: List[str] = []
            pred_primary_seq: List[str] = []

            for entry in labels:
                seg = [x for x in (entry.get("seg_labels") or []) if x in PHASES]
                gold = [x for x in (entry.get("gold_labels") or []) if x in PHASES]
                text = entry.get("text", "") or ""

                seg_prim = primary(seg)
                gold_prim = primary(gold)
                total_sent += 1

                # Overview counts (multi-label)
                for p in seg:
                    counts_pred[p] += 1
                for g in gold:
                    counts_gold[g] += 1

                # Accuracies
                if seg and gold and (set(seg) & set(gold)):
                    correct_any_overlap += 1
                if seg_prim == gold_prim and seg_prim in PHASES:
                    correct_primary += 1

                # Per-phase primary accuracy
                if gold_prim in PHASES:
                    per_phase_gold_tot[gold_prim] += 1
                    if seg_prim == gold_prim:
                        per_phase_gold_correct[gold_prim] += 1

                # Primary confusion (incl. NONE)
                g_key = gold_prim if gold_prim in PHASES else "NONE"
                p_key = seg_prim if seg_prim in PHASES else "NONE"
                add_confusion(confusion_primary, g_key, p_key)

                # Marker stats (catalog-driven, text-only; trigger_markers ignored)
                if intended:
                    nt = norm_for_match(text)
                    if nt:
                        for mk, ph in intended.items():
                            if not mk:
                                continue
                            if not marker_in_text(nt, mk):
                                continue

                            marker_counter.update([mk])
                            st = marker_stats.setdefault(mk, {
                                "intended_phase": ph,
                                "total": 0,
                                "gold_correct": 0,   # Gold == intended
                                "pred_correct": 0,   # Pred == intended
                                "both_correct": 0,   # Gold == Pred == intended
                                "agree_count": 0,    # Gold == Pred (regardless of intended)
                                "gold_phase_counts": Counter(),
                                "pred_phase_counts": Counter(),
                            })
                            st["total"] += 1
                            if gold_prim in PHASES:
                                st["gold_phase_counts"][gold_prim] += 1
                            if seg_prim in PHASES:
                                st["pred_phase_counts"][seg_prim] += 1

                            if gold_prim in PHASES and seg_prim in PHASES and gold_prim == seg_prim:
                                st["agree_count"] += 1
                            if gold_prim == ph:
                                st["gold_correct"] += 1
                            if seg_prim == ph:
                                st["pred_correct"] += 1
                            if gold_prim == ph and seg_prim == ph:
                                st["both_correct"] += 1

                gold_primary_seq.append(gold_prim if gold_prim in PHASES else "")
                pred_primary_seq.append(seg_prim if seg_prim in PHASES else "")

            # Transition metrics (primary sequence only)
            for i in range(1, len(gold_primary_seq)):
                g_prev, g_curr = gold_primary_seq[i-1], gold_primary_seq[i]
                p_prev, p_curr = pred_primary_seq[i-1], pred_primary_seq[i]
                gold_t = f"{g_prev}->{g_curr}" if g_prev and g_curr else "UNK"
                pred_t = f"{p_prev}->{p_curr}" if p_prev and p_curr else "UNK"
                add_transition_confusion(transitions_confusion, gold_t, pred_t)
                transitions_total += 1

                gold_changes = (g_prev in PHASES and g_curr in PHASES and g_prev != g_curr)
                pred_changes = (p_prev in PHASES and p_curr in PHASES and p_prev != p_curr)

                if gold_t == pred_t and gold_t != "UNK":
                    transitions_correct_primary += 1
                else:
                    if gold_changes and (not pred_changes):
                        missed_transitions += 1  # missed boundary
                    elif gold_changes and pred_changes and gold_t != pred_t:
                        wrong_transitions += 1   # wrong boundary

        except Exception as e:
            print(f"[WARN] Error while processing {path}: {e}", file=sys.stderr)
            continue

    # Core accuracies
    acc_any = (correct_any_overlap / total_sent) if total_sent else 0.0
    acc_primary = (correct_primary / total_sent) if total_sent else 0.0

    per_phase_acc = {}
    for ph in PHASES:
        tot = per_phase_gold_tot[ph]
        per_phase_acc[ph] = (per_phase_gold_correct[ph] / tot) if tot else None

    # Pred primary coverage (share of sentences with a valid predicted primary)
    pred_none = sum(confusion_primary[g].get("NONE", 0) for g in confusion_primary.keys())
    pred_primary_coverage = 1.0 - (pred_none / total_sent if total_sent else 0.0)

    # Transitions
    trans_acc = (transitions_correct_primary / transitions_total) if transitions_total else 0.0
    top_failed_transitions = [
        {"gold": g, "pred": p, "count": c}
        for (g, p), c in transition_failure_counter.most_common(15)
    ]

    # PRF from confusion (exclude NONE for PRF computation)
    conf_for_prf = {g: {p: confusion_primary[g].get(p, 0) for p in PHASES} for g in PHASES}
    prf = compute_prf_from_confusion(conf_for_prf, PHASES)

    # Marker summaries
    per_marker = []
    micro_gold_align = 0
    micro_pred_align = 0
    micro_both_align = 0
    micro_agree = 0
    total_occ = 0
    min_support = 5
    macro_gold_rates = []
    macro_pred_rates = []
    macro_both_rates = []
    macro_agree_rates = []

    for mk, st in sorted(marker_stats.items(), key=lambda kv: (-kv[1]["total"], kv[0])):
        total = st["total"]
        total_occ += total

        gold_rate = (st["gold_correct"] / total) if total else 0.0
        pred_rate = (st["pred_correct"] / total) if total else 0.0
        both_rate = (st["both_correct"] / total) if total else 0.0
        agree_rate = (st["agree_count"] / total) if total else 0.0

        micro_gold_align += st["gold_correct"]
        micro_pred_align += st["pred_correct"]
        micro_both_align += st["both_correct"]
        micro_agree += st["agree_count"]

        if total >= min_support:
            macro_gold_rates.append(gold_rate)
            macro_pred_rates.append(pred_rate)
            macro_both_rates.append(both_rate)
            macro_agree_rates.append(agree_rate)

        gp = st["gold_phase_counts"]
        pp = st["pred_phase_counts"]
        gold_top_phase, gold_top_cnt = ("", 0)
        if gp:
            gold_top_phase, gold_top_cnt = max(gp.items(), key=lambda kv: kv[1])
        pred_top_phase, pred_top_cnt = ("", 0)
        if pp:
            pred_top_phase, pred_top_cnt = max(pp.items(), key=lambda kv: kv[1])

        per_marker.append({
            "marker": mk,
            "intended_phase": st["intended_phase"],
            "count": total,
            "gold_start_accuracy": gold_rate,
            "pred_start_accuracy": pred_rate,
            "both_start_accuracy": both_rate,
            "agreement_rate": agree_rate,
            "gold_phase_counts": {k: int(v) for k, v in gp.items()},
            "pred_phase_counts": {k: int(v) for k, v in pp.items()},
            "gold_top_phase": gold_top_phase,
            "gold_top_frac": (gold_top_cnt / total) if total else 0.0,
            "pred_top_phase": pred_top_phase,
            "pred_top_frac": (pred_top_cnt / total) if total else 0.0
        })

    micro_gold_alignment = (micro_gold_align / total_occ) if total_occ else 0.0
    micro_pred_alignment = (micro_pred_align / total_occ) if total_occ else 0.0
    micro_both_alignment = (micro_both_align / total_occ) if total_occ else 0.0
    micro_agreement_rate = (micro_agree / total_occ) if total_occ else 0.0

    macro_gold_alignment = (sum(macro_gold_rates) / len(macro_gold_rates)) if macro_gold_rates else None
    macro_pred_alignment = (sum(macro_pred_rates) / len(macro_pred_rates)) if macro_pred_rates else None
    macro_both_alignment = (sum(macro_both_rates) / len(macro_both_rates)) if macro_both_rates else None
    macro_agreement_rate = (sum(macro_agree_rates) / len(macro_agree_rates)) if macro_agree_rates else None

    return {
        "files_processed": sorted(files_processed),
        "sentences_total": total_sent,
        "labels_overview": {  # overview table: predicted vs gold counts per phase
            "predicted": {k: int(v) for k, v in counts_pred.items()},
            "gold": {k: int(v) for k, v in counts_gold.items()},
        },
        "accuracy": {
            "any_overlap": acc_any,
            "primary": acc_primary,
            "pred_primary_coverage": pred_primary_coverage,
            "per_phase_primary": per_phase_acc
        },
        "prf": prf,  # precision/recall/f1: per_phase, micro, macro
        "confusion_primary": {k: v for k, v in confusion_primary.items()},
        "transitions": {
            "total": transitions_total,
            "correct_primary": transitions_correct_primary,
            "accuracy_primary": trans_acc,
            "missed": missed_transitions,
            "wrong": wrong_transitions,
            "confusion": transitions_confusion,
            "top_failed_transitions": top_failed_transitions
        },
        "markers": {
            "top_20": Counter({m: c for m, c in Counter(marker_counter).most_common(20)}),
            "phase_alignment": {
                "min_support": min_support,
                "micro": {
                    "gold_start_accuracy": micro_gold_alignment,
                    "pred_start_accuracy": micro_pred_alignment,
                    "both_start_accuracy": micro_both_alignment,
                    "agreement_rate": micro_agreement_rate
                },
                "macro": {
                    "gold_start_accuracy": macro_gold_alignment,
                    "pred_start_accuracy": macro_pred_alignment,
                    "both_start_accuracy": macro_both_alignment,
                    "agreement_rate": macro_agreement_rate
                },
                "per_marker": per_marker
            }
        }
    }

def write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_confusion_csv(path: str, conf: Dict[str, Dict[str, int]], phases: List[str]) -> None:
    """
    Writes a confusion matrix to CSV. Columns are derived from the conf keys
    and include 'NONE' if present.
    """
    # Determine columns dynamically to include 'NONE' if present.
    rows = list(conf.keys())
    cols_set = set()
    for g in rows:
        cols_set.update(conf[g].keys())
    cols = sorted([c for c in cols_set if c != "NONE"] + (["NONE"] if "NONE" in cols_set else []))

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold\\pred"] + cols)
        for g in sorted(rows, key=lambda x: (x != "NONE", x)):
            row = [g] + [conf.get(g, {}).get(p, 0) for p in cols]
            writer.writerow(row)

def write_transitions_confusion_csv(path: str, conf: Dict[str, Dict[str, int]]) -> None:
    rows = sorted(conf.keys())
    cols_set = set()
    for g in rows:
        cols_set.update(conf[g].keys())
    cols = sorted(cols_set)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gold_transition\\pred_transition"] + cols)
        for g in rows:
            row = [g] + [conf[g].get(p, 0) for p in cols]
            writer.writerow(row)

def list_family_files(input_dir: str, family: str) -> List[str]:
    return sorted(
        p for p in glob(os.path.join(input_dir, f"{family}_*.gold.json"))
        if FILENAME_RE.match(os.path.basename(p))
    )

def list_all_files(input_dir: str) -> List[str]:
    all_paths = []
    for fam in FAMILIES:
        all_paths.extend(list_family_files(input_dir, fam))
    return sorted(set(all_paths))

def write_readme(out_dir: str) -> None:
    content = """Generated by phase_metrics.py

Metrics overview:
- accuracy.any_overlap: True if seg_labels ∩ gold_labels ≠ ∅.
- accuracy.primary: Primary-phase match (first label).
- accuracy.pred_primary_coverage: Share of sentences with a valid predicted primary phase.
- accuracy.per_phase_primary: Accuracy by gold primary phase.
- prf: Precision/Recall/F1 from the primary-phase confusion (per phase, micro, macro).
- confusion_primary: Gold×Pred confusion over primary phases (includes 'NONE' for missing primary).
- transitions: Primary-phase transition metrics (A->B) incl. confusion, top failures,
  and boundary errors (missed = gold changed, pred did not; wrong = both changed but mismatched).
- markers.top_20: Most frequent markers found in sentence text (catalog-driven search).
- markers.phase_alignment:
  - gold_start_accuracy / pred_start_accuracy / both_start_accuracy:
    share that the sentence containing a marker has Gold / Pred / Both equal to the marker’s intended phase.
  - agreement_rate: share that Gold == Pred on sentences where the marker occurs.
  - per_marker details include per-phase distributions and top phases.
- labels_overview: counts per phase for predicted vs. gold.
"""
    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(content)

def main():
    parser = argparse.ArgumentParser(description="Compute phase metrics per family (aut, gsm, crt) and overall.")
    parser.add_argument("--input-dir", default="outputs/2_eval_phase", help="Folder with *.gold.json files (read-only).")
    parser.add_argument("--out-dir", default="outputs/3_phase_metrics", help="Output folder for metrics.")
    args = parser.parse_args()

    input_dir = resolve_path(args.input_dir)
    out_dir = resolve_path(args.out_dir)

    if not os.path.isdir(input_dir):
        print(f"[ERR] Input folder does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)

    ensure_dir(out_dir)
    clean_dir(out_dir)

    # Per-family reports
    for fam in FAMILIES:
        fam_files = list_family_files(input_dir, fam)
        metrics = compute_metrics_for_files(fam_files)
        write_json(os.path.join(out_dir, f"phase_metrics_{fam}.json"), metrics)
        write_confusion_csv(os.path.join(out_dir, f"confusion_primary_{fam}.csv"),
                            metrics["confusion_primary"], PHASES)
        write_transitions_confusion_csv(os.path.join(out_dir, f"transitions_confusion_{fam}.csv"),
                                        metrics["transitions"]["confusion"])

    # Aggregate report
    metrics_all = compute_metrics_for_files(list_all_files(input_dir))
    write_json(os.path.join(out_dir, "phase_metrics_all.json"), metrics_all)
    write_confusion_csv(os.path.join(out_dir, "confusion_primary_all.csv"),
                        metrics_all["confusion_primary"], PHASES)
    write_transitions_confusion_csv(os.path.join(out_dir, "transitions_confusion_all.csv"),
                                    metrics_all["transitions"]["confusion"])

    write_readme(out_dir)
    print("[OK] Metrics created in:", out_dir)
    for fam in FAMILIES:
        print(f"  - {fam}: {os.path.join(out_dir, f'phase_metrics_{fam}.json')}")
    print("  - all:", os.path.join(out_dir, "phase_metrics_all.json"))

if __name__ == "__main__":
    main()
