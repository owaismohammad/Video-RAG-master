# Video-RAG benchmark harness

Runs the Video-RAG pipeline over your own dataset of videos + open-ended
questions and writes a response sheet per video, with indexing and querying
timings and key-frame counts.

The pipeline itself (`vidrag_pipeline/`) is **not modified**. This harness
imports `vidrag_pipeline/tools/` directly and re-uses the pipeline's own
functions and prompts verbatim. The one exception is documented under
[What differs from `vidrag_pipeline.py`](#what-differs-from-vidrag_pipelinepy).

---

## 1. Dataset layout

```
/workspace/dataset/
├── videos/
│   ├── Video_ID_1.mp4
│   ├── Video_ID_2.mp4
│   └── ... Video_ID_40.mp4
└── questions/
    ├── Video_ID_1.csv
    ├── Video_ID_2.csv
    └── ... Video_ID_40.csv
```

`.mp4 .mkv .avi .mov .webm` are all accepted; the first one that exists wins.

Each question CSV needs a header row. The question column is auto-detected
from `question / questions / question_text / query / q`; an id column from
`question_id / qid / id / index / no / s_no`; an optional reference answer
from `answer / ground_truth / gt / correct_answer / reference / gt_answer`.
Matching is case-insensitive. If no header matches, the **first column** is
used as the question and rows are numbered 1..N. Blank question rows are
skipped.

All of these work:

```csv
question_id,question,answer
q1,What is the person holding?,a mug
```

```csv
Questions
Describe the opening shot.
How many people speak?
```

Verify your real CSVs parse correctly before you rent anything — no GPU needed:

```bash
uv run python benchmark/test_dataset.py
```

---

## 2. Configure

Everything is hard-coded in [`benchmark/config.py`](config.py). There is no
argument parsing anywhere in this harness.

```python
START_VIDEO_ID = 1     # inclusive
END_VIDEO_ID   = 5     # inclusive -> runs Video_ID_1 .. Video_ID_5

VIDEO_DIR    = "/workspace/dataset/videos"
QUESTION_DIR = "/workspace/dataset/questions"
OUTPUT_DIR   = "/workspace/Benchmarking_VideoRAG-Master"

MODEL_PATH   = "/workspace/models/LLaVA-Video-7B-Qwen2"

max_frames_num = 32    # key frames sampled per video
USE_OCR = True
USE_ASR = True
USE_DET = True         # needs the APE service on port 9999
```

`SKIP_COMPLETED = True` means a video with a finished `responses.csv` is
skipped on a re-run, so a box that dies at video 23 resumes there instead of
redoing 1–22.

---

## 3. Run

```bash
bash run_ape_service.sh      # shell 1 — only if USE_DET = True
bash run_benchmark.sh        # shell 2
```

Set `USE_DET = False` in `config.py` to skip APE entirely; OCR + ASR + the
RAG retrieval still run.

---

## 4. Output

```
Benchmarking_VideoRAG-Master/
├── Video_ID_1/
│   ├── responses.csv     <- one row per question: answer + every timing
│   ├── indexing.json     <- machine-readable indexing stats
│   ├── summary.txt       <- the same, human-readable
│   └── prompts/q001.txt  <- exact RAG-augmented prompt sent to the LVLM
├── Video_ID_2/
│   └── ...
├── benchmark_summary.csv <- one row per video
└── all_responses.csv     <- every question of every video, concatenated
```

`responses.csv` is rewritten after **every question**, so an interrupted run
loses at most one answer.

### `responses.csv` columns

| Column | Meaning |
|---|---|
| `video_id`, `question_index`, `question_id` | identity; `question_index` is 1-based position in the CSV |
| `question`, `ground_truth` | carried through from your CSV |
| `response` | **the model's answer** |
| `t_retrieval_request_sec` | step 0 — LVLM generating the JSON retrieval request (text-only, no video) |
| `t_clip_frame_gate_sec` | CLIP scoring the key frames against the requested objects |
| `num_det_frames_selected` | **how many key frames passed the CLIP gate** and were sent to APE |
| `t_ape_detection_sec` | APE object detection + scene-graph text generation |
| `num_det_docs` | non-empty detection descriptions returned |
| `t_ocr_retrieval_sec` / `num_ocr_docs_retrieved` | Contriever+FAISS retrieval over the OCR docs |
| `t_asr_retrieval_sec` / `num_asr_docs_retrieved` | same over the ASR transcript chunks |
| `t_answer_generation_sec` | **LVLM generating the final answer** |
| `t_question_total_sec` | end-to-end for this question (sum of the above plus overhead) |
| `json_request` | the raw step-0 output, for debugging |
| `error` | traceback if this question failed; the run continues |

### `indexing.json` / `summary.txt`

Per-video work done **once**, before any question is asked:

| Field | Meaning |
|---|---|
| `num_keyframes_extracted` | **key frames the pipeline actually works on** (uniform sampling, `force_sample=True`, so this equals `max_frames_num` = 32) |
| `keyframe_timestamps` | the timestamp of each sampled frame |
| `video_duration_sec`, `keyframe_resolution` | source video properties |
| `t_keyframe_sampling_sec` | decord decode + uniform sampling |
| `t_lvlm_frame_preprocess_sec` | LLaVA image processor over those frames |
| `t_clip_frame_encode_sec` | CLIP preprocessing of the frames (DET only) |
| `t_ocr_sec` / `num_ocr_docs_total` | EasyOCR over the key frames |
| `t_asr_sec` / `num_asr_docs_total` | Whisper over the full audio track, in 30s chunks |
| `asr_loaded_from_cache` | true if a previous run's transcript was reused (then `t_asr_sec` is a file read, not real ASR cost) |
| `t_total_indexing_sec` | **total indexing time** — sum of the five stages above |

---

## 5. How to read "indexing" here

Video-RAG has no persistent vector index. The per-video preparation — frame
sampling, OCR, ASR transcription, CLIP frame features — is done once, held in
memory, and then reused by all 10 questions. That is what this harness reports
as indexing time.

The FAISS index over the OCR/ASR documents is built **per query** inside
`tools/rag_retriever_dynamic.py`, so its cost sits in
`t_ocr_retrieval_sec` / `t_asr_retrieval_sec`, not in the indexing total.

So per video, the honest cost model is:

```
wall_clock_total_sec = t_total_indexing_sec          (once)
                     + sum(t_question_total_sec)     (per question)
```

Both are in `benchmark_summary.csv`.

One caveat on the ASR numbers: the transcript cache in `RESTORE_DIR/audio/` is
keyed on the video filename and shared across runs. The first run pays the real
Whisper cost; a re-run reads the `.txt` and reports a near-zero `t_asr_sec`
with `asr_loaded_from_cache = true`. Delete that folder if you want clean
cold-start indexing timings.

---

## What differs from `vidrag_pipeline.py`

The step-0 retrieval prompt is read directly out of `vidrag_pipeline.py` at
startup, so it is byte-identical by construction. The thresholds
(`rag_threshold` 0.3, `clip_threshold` 0.3, `beta` 3.0), the CLIP gate, the
retrieval logic and the prompt assembly follow the original line for line.
`process_video`, `chunk_audio`, `transcribe_chunk` and `det_preprocess` are
identical to the original's (checked by comparing their ASTs); the remaining
copied functions differ only as listed below.

**One line is different.** The original ends the prompt with a multiple-choice
instruction:

```python
qs += "Select the best answer to the following multiple-choice question ... Respond with only the letter (A, B, C, or D) ..."
```

Your questions are open-ended, so that instruction would force a meaningless
single letter. It is replaced with:

```python
qs += "Answer the following question based on the video and the information (if given). Question: " + question_text
```

Related: `ANSWER_MAX_NEW_TOKENS` in `config.py` defaults to 1024.
`vidrag_pipeline.py` uses 4096 and the MCQ eval scripts use 16 — neither suits
open-ended answers. Generation stops at EOS regardless; this is only a cap.

Other deviations, none of which change what the model is asked or what it
returns (`quiet=True` on the ffmpeg call, `except Exception` in place of a bare
`except`, APE host/port read from `config.py`, and `max_new_tokens` exposed as a
parameter defaulting to the original 4096):

- The EasyOCR reader is constructed once for the whole run instead of once per
  video. The original rebuilds it inside `get_ocr_docs`, which would otherwise
  charge model-loading time to every video's OCR measurement.
- `save_frames` writes to `restore/<video_id>/frame_N.png` instead of
  `restore/frame_N.png`, so videos don't overwrite each other's frames. This
  matches what the upstream eval scripts in `evals/` already do.
