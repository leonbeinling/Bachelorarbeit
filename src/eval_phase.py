import os, json, re

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IN_DIR  = os.path.join(BASE, "outputs", "structured_signals")
OUT_DIR = os.path.join(BASE, "outputs", "gold_labels")
os.makedirs(OUT_DIR, exist_ok=True)

# Satzgrenzen wie in deiner Segmentierung (., !, ? + evtl. schließende Quotes/Brackets + Whitespace, oder \n)
SENT_BOUNDARY_RE = re.compile(r'([.!?]["\')\]]*\s+|\n+)', re.MULTILINE)

PHASE_TCASE = {
    "CLARIFY": "Clarify",
    "IDEATE": "Ideate",
    "EVALUATE": "Evaluate",
    "FINALIZE": "Finalize"
}

def split_sentences(text: str):
    """Gibt Liste von (start, end, sent_text) zurück."""
    sents = []
    start = 0
    for m in SENT_BOUNDARY_RE.finditer(text):
        end = m.end()
        sent = text[start:end].strip()
        if sent:
            sents.append((start, end, sent))
        start = end
    # Tail
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            sents.append((start, len(text), tail))
    return sents

def build_seg_ranges(segments_char):
    """
    Erzeugt Ranges im rekonstruierten Fulltext:
    - start, end: Zeichen-Range
    - phase: Titelcase der Phase
    - marker: start_marker des Segments (falls vorhanden, lowercase wie aus struct file)
    - seg_idx: Index des Segments (stabil für Debug)
    """
    ranges = []
    offset = 0
    for i, seg in enumerate(segments_char):
        t = seg.get("text", "")
        phase_uc = (seg.get("phase") or "").upper()
        phase_tc = PHASE_TCASE.get(phase_uc, phase_uc.title() if phase_uc else "")
        ranges.append({
            "start": offset,
            "end": offset + len(t),
            "phase": phase_tc,
            "marker": (seg.get("start_marker") or None),
            "seg_idx": i
        })
        offset += len(t)
    return ranges

def phases_for_span(ranges, s_start, s_end):
    """Aggregiert Phasen, die den Satz [s_start,s_end) überlappen – in Auftretensreihenfolge, ohne Duplikate."""
    out = []
    seen = set()
    for r in ranges:
        if r["start"] < s_end and r["end"] > s_start:
            ph = r["phase"]
            if ph and ph not in seen:
                out.append(ph)
                seen.add(ph)
    return out

def markers_for_span(ranges, s_start, s_end):
    """
    Aggregiert ausgelöste Marker für den Satz, basierend auf den Segmenten,
    die den Satz überlappen. Nutzt das 'start_marker' des jeweiligen Segments.
    Reihenfolge = erstes Auftreten, ohne Duplikate.
    """
    out = []
    seen = set()
    for r in ranges:
        if r["start"] < s_end and r["end"] > s_start:
            mk = r.get("marker")
            if mk and mk not in seen:
                out.append(mk)
                seen.add(mk)
    return out

def process_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segs = (data.get("segmented_phases_signals") or {}).get("segments_char") or []
    segmeta = data.get("segmentation_meta") or {}

    if not segs:
        # Nichts zu tun -> leere Struktur schreiben
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(OUT_DIR, f"{base}.gold.json")
        out = {
            "task_id": data.get("task_id"),
            "task_type": data.get("task_type"),
            "prompt": data.get("prompt"),
            "units": "sentence",
            "note": "Gold labels left blank; seg_labels derived from structured_signals; no segments found.",
            "marker_usage": segmeta.get("marker_usage") or {},
            "marker_counts": segmeta.get("marker_counts") or {},
            "catalog_hash": segmeta.get("catalog_hash"),
            "labels": []
        }
        with open(out_path, "w", encoding="utf-8") as o:
            json.dump(out, o, ensure_ascii=False, indent=2)
        print(f"[gold] {base}: no segments -> {out_path}")
        return

    # Gesamttest rekonstruieren (Konkatenation der Segmenttexte in Reihenfolge)
    full_text = "".join(seg.get("text", "") for seg in segs)
    # Segment-Ranges im Fulltext (inkl. Marker)
    seg_ranges = build_seg_ranges(segs)
    # Sätze bestimmen
    sentences = split_sentences(full_text)

    labels = []
    for i, (s_start, s_end, sent_text) in enumerate(sentences, start=1):
        seg_phases = phases_for_span(seg_ranges, s_start, s_end)
        trig_markers = markers_for_span(seg_ranges, s_start, s_end)
        labels.append({
            "sent_id": i,
            "text": sent_text,
            "seg_labels": seg_phases,            # Phasen aus Input (ggf. mehrere)
            "trigger_markers": trig_markers,     # Marker, deren Segmente diesen Satz überlappen
            "gold_labels": []                    # Gold standard bleibt LEER (oder später gefüllt)
        })

    base = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}.gold.json")
    out = {
        "task_id": data.get("task_id"),
        "task_type": data.get("task_type"),
        "prompt": data.get("prompt"),
        "units": "sentence",
        "note": "Gold labels left blank; seg_labels and trigger_markers aggregated per sentence from structured_signals.",
        # nützliche globale Marker-Infos aus structured_signals
        "marker_usage": segmeta.get("marker_usage") or {},
        "marker_counts": segmeta.get("marker_counts") or {},
        "catalog_hash": segmeta.get("catalog_hash"),
        "labels": labels
    }
    with open(out_path, "w", encoding="utf-8") as o:
        json.dump(out, o, ensure_ascii=False, indent=2)

    print(f"[gold] {base}: {len(labels)} sentences -> {out_path}")

def main():
    files = [f for f in os.listdir(IN_DIR) if f.lower().endswith(".json")]
    if not files:
        print(f"[gold] no json files in {IN_DIR}")
        return
    for fname in sorted(files):
        process_file(os.path.join(IN_DIR, fname))

if __name__ == "__main__":
    main()
