"""
Offline sanity check for the benchmark's dataset + output plumbing.

Run this on your laptop BEFORE renting a GPU:

    python benchmark/test_dataset.py

It builds a throwaway dataset (fake videos + question CSVs in several column
layouts), points config at it, and asserts that video discovery, question
parsing and the response/summary sheets all behave. No torch, no GPU.
"""

import os
import sys
import csv
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

_TMP = tempfile.mkdtemp(prefix="vidrag_bench_test_")
config.VIDEO_DIR = os.path.join(_TMP, "videos")
config.QUESTION_DIR = os.path.join(_TMP, "questions")
config.OUTPUT_DIR = os.path.join(_TMP, "out")
os.makedirs(config.VIDEO_DIR)
os.makedirs(config.QUESTION_DIR)
os.makedirs(config.OUTPUT_DIR)

import dataset


def _write(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


failures = []


def check(label, cond, detail=""):
    if cond:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        failures.append(label)


print("== video discovery ==")
open(os.path.join(config.VIDEO_DIR, "Video_ID_1.mp4"), "w").close()
open(os.path.join(config.VIDEO_DIR, "Video_ID_2.mkv"), "w").close()
check("finds .mp4", dataset.find_video_file("Video_ID_1").endswith("Video_ID_1.mp4"))
check("finds .mkv", dataset.find_video_file("Video_ID_2").endswith("Video_ID_2.mkv"))
check("missing -> None", dataset.find_video_file("Video_ID_99") is None)

print("== question CSV parsing ==")

# layout A: question_id / question / answer
_write(os.path.join(config.QUESTION_DIR, "Video_ID_1.csv"), [
    ["question_id", "question", "answer"],
    ["q1", "What is the person holding?", "a mug"],
    ["q2", "Where does the scene take place?", "a kitchen"],
])
rows, _ = dataset.load_questions("Video_ID_1")
check("layout A row count", len(rows) == 2, rows)
check("layout A id", rows[0]["question_id"] == "q1", rows[0])
check("layout A question", rows[0]["question"] == "What is the person holding?", rows[0])
check("layout A ground truth", rows[0]["ground_truth"] == "a mug", rows[0])

# layout B: single bare 'Questions' column, mixed case, BOM, blank line
with open(os.path.join(config.QUESTION_DIR, "Video_ID_2.csv"), "w",
          encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerows([
        ["Questions"],
        ["  Describe the opening shot.  "],
        [""],
        ["How many people speak?"],
    ])
rows, _ = dataset.load_questions("Video_ID_2")
check("layout B skips blank rows", len(rows) == 2, rows)
check("layout B strips whitespace", rows[0]["question"] == "Describe the opening shot.", rows[0])
check("layout B auto-numbers ids", [r["question_id"] for r in rows] == ["1", "2"], rows)
check("layout B empty ground truth", rows[0]["ground_truth"] == "", rows[0])

# layout C: unrecognised header -> first column is the question
_write(os.path.join(config.QUESTION_DIR, "Video_ID_3.csv"), [
    ["prompt_text", "notes"],
    ["What colour is the car?", "n/a"],
])
rows, _ = dataset.load_questions("Video_ID_3")
check("layout C falls back to first column",
      len(rows) == 1 and rows[0]["question"] == "What colour is the car?", rows)

# layout D: commas and quotes inside a question survive the round trip
_write(os.path.join(config.QUESTION_DIR, "Video_ID_4.csv"), [
    ["question"],
    ['He said "go", then what happened?'],
])
rows, _ = dataset.load_questions("Video_ID_4")
check("layout D quoting", rows[0]["question"] == 'He said "go", then what happened?', rows)

# missing file
rows, path = dataset.load_questions("Video_ID_77")
check("missing CSV -> None", rows is None, path)

print("== response sheet ==")
resp_path = os.path.join(config.OUTPUT_DIR, "responses.csv")
dataset.write_responses_csv(resp_path, [
    {
        "video_id": "Video_ID_1", "question_index": 1, "question_id": "q1",
        "question": 'Multi\nline, "quoted"', "ground_truth": "a mug",
        "response": "A mug, held in the right hand.",
        "t_retrieval_request_sec": 1.2345, "t_answer_generation_sec": 3.5,
        "t_question_total_sec": 5.9, "num_det_frames_selected": 7,
    },
])
with open(resp_path, encoding="utf-8", newline="") as f:
    read_back = list(csv.DictReader(f))
check("responses header complete",
      list(read_back[0].keys()) == dataset.RESPONSE_FIELDS, list(read_back[0].keys()))
check("newlines/quotes round-trip",
      read_back[0]["question"] == 'Multi\nline, "quoted"', repr(read_back[0]["question"]))
check("missing metrics become empty",
      read_back[0]["t_ocr_retrieval_sec"] == "", read_back[0])
check("response text preserved",
      read_back[0]["response"] == "A mug, held in the right hand.", read_back[0])

print("== summary sheet ==")
summ_path = os.path.join(config.OUTPUT_DIR, "benchmark_summary.csv")
dataset.write_summary_csv(summ_path, [
    {"video_id": "Video_ID_1", "num_keyframes_extracted": 32, "indexing_total_sec": 42.0},
    {"video_id": "Video_ID_2", "num_keyframes_extracted": 32, "indexing_total_sec": 39.5},
])
with open(summ_path, encoding="utf-8", newline="") as f:
    srows = list(csv.DictReader(f))
check("summary header complete",
      list(srows[0].keys()) == dataset.SUMMARY_FIELDS, list(srows[0].keys()))
check("summary rows", len(srows) == 2 and srows[1]["video_id"] == "Video_ID_2", srows)
check("summary keyframe count", srows[0]["num_keyframes_extracted"] == "32", srows[0])

print()
if failures:
    print(f"{len(failures)} FAILED: {failures}")
    sys.exit(1)
print("all checks passed")
print(f"(scratch dir: {_TMP})")
