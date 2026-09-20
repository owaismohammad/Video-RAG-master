"""
Dataset + output-sheet helpers for the Video-RAG benchmark.

Deliberately free of torch / llava imports so it can be exercised on a laptop
before renting a GPU (see benchmark/test_dataset.py).
"""

import os
import csv

import config


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------
def find_video_file(video_name):
    """<VIDEO_DIR>/<video_name>.<ext> for the first ext that exists."""
    for ext in config.VIDEO_EXTENSIONS:
        p = os.path.join(config.VIDEO_DIR, video_name + ext)
        if os.path.exists(p):
            return p
    return None


def _pick_column(fieldnames, candidates):
    lowered = {str(f).strip().lower(): f for f in fieldnames if f is not None}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def load_questions(video_name):
    """Read <QUESTION_DIR>/<video_name>.csv.

    Returns (rows, csv_path). rows is a list of
    {"question_id", "question", "ground_truth"}; None if the file is missing.
    """
    csv_path = os.path.join(config.QUESTION_DIR, video_name + ".csv")
    if not os.path.exists(csv_path):
        return None, csv_path

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        q_col = _pick_column(fieldnames, config.QUESTION_COLUMN_CANDIDATES)
        id_col = _pick_column(fieldnames, config.QUESTION_ID_COLUMN_CANDIDATES)
        gt_col = _pick_column(fieldnames, config.GROUND_TRUTH_COLUMN_CANDIDATES)
        if q_col is None and fieldnames:
            q_col = fieldnames[0]

        rows = []
        for row in reader:
            text = (row.get(q_col) or "").strip()
            if not text:
                continue
            # Auto-numbered ids count kept rows, not raw CSV rows, so that a
            # blank line in the CSV does not desync question_id from
            # question_index in the response sheet.
            auto_id = str(len(rows) + 1)
            rows.append({
                "question_id": (row.get(id_col) or "").strip() if id_col else auto_id,
                "question": text,
                "ground_truth": (row.get(gt_col) or "").strip() if gt_col else "",
            })
    return rows, csv_path


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
RESPONSE_FIELDS = [
    "video_id",
    "question_index",
    "question_id",
    "question",
    "ground_truth",
    "response",
    "t_retrieval_request_sec",
    "t_clip_frame_gate_sec",
    "num_det_frames_selected",
    "t_ape_detection_sec",
    "num_det_docs",
    "t_ocr_retrieval_sec",
    "num_ocr_docs_retrieved",
    "t_asr_retrieval_sec",
    "num_asr_docs_retrieved",
    "t_answer_generation_sec",
    "t_question_total_sec",
    "json_request",
    "error",
]

SUMMARY_FIELDS = [
    "video_id",
    "video_path",
    "video_duration_sec",
    "num_questions",
    "num_keyframes_extracted",
    "num_ocr_docs_total",
    "num_asr_docs_total",
    "indexing_total_sec",
    "indexing_keyframe_sampling_sec",
    "indexing_lvlm_preprocess_sec",
    "indexing_clip_encode_sec",
    "indexing_ocr_sec",
    "indexing_asr_sec",
    "asr_loaded_from_cache",
    "querying_total_sec",
    "avg_sec_per_question",
    "avg_answer_generation_sec",
    "wall_clock_total_sec",
]


def _write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_responses_csv(path, rows):
    _write_csv(path, RESPONSE_FIELDS, rows)


def read_responses_csv(path):
    """Read a responses.csv back (used when resuming a partially-done run)."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_summary_csv(path, rows):
    _write_csv(path, SUMMARY_FIELDS, rows)
