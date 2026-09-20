"""
Hard-coded configuration for the Video-RAG benchmark run.

Edit the values in this file directly -- there is no argument parsing anywhere
in this harness, by design.
"""

import os

# ---------------------------------------------------------------------------
# 1. WHICH VIDEOS TO RUN  (inclusive on both ends)
#    e.g. START_VIDEO_ID = 1, END_VIDEO_ID = 5  ->  Video_ID_1 .. Video_ID_5
# ---------------------------------------------------------------------------
START_VIDEO_ID = 1
END_VIDEO_ID   = 20

# Video files are expected as  <VIDEO_DIR>/<VIDEO_NAME_TEMPLATE>.<ext>
# Question files as            <QUESTION_DIR>/<VIDEO_NAME_TEMPLATE>.csv
VIDEO_NAME_TEMPLATE = "Video_ID_{i}"
VIDEO_EXTENSIONS    = [".mp4", ".mkv", ".avi", ".mov", ".webm"]

# ---------------------------------------------------------------------------
# 2. PATHS
# ---------------------------------------------------------------------------
VIDEO_DIR    = "/workspace/dataset/videos"
QUESTION_DIR = "/workspace/dataset/questions"
OUTPUT_DIR   = "/workspace/Benchmarking_VideoRAG-Master"

# Local path (or HF repo id) of the LVLM weights.
MODEL_PATH  = "/workspace/models/LLaVA-Video-7B-Qwen2"
MODEL_BASE  = None
MODEL_NAME  = "llava_qwen"
CONV_TEMPLATE = "qwen_1_5"

# llava's load_pretrained_model defaults to "flash_attention_2", which requires
# the flash-attn package. setup.sh installs a prebuilt wheel for it. If that
# install failed, set this to "sdpa" -- both are exact (non-approximate)
# attention, so answers are equivalent up to floating-point ordering; sdpa is
# just somewhat slower and uses a little more memory.
ATTN_IMPLEMENTATION = "flash_attention_2"

# CLIP + Whisper (HF ids; set to local dirs if you pre-downloaded them)
CLIP_PATH    = "openai/clip-vit-large-patch14-336"
WHISPER_PATH = "openai/whisper-large"

# Scratch dir the pipeline writes frames / extracted audio / ASR caches into.
RESTORE_DIR = "/workspace/restore"

# ---------------------------------------------------------------------------
# 3. VIDEO-RAG HYPER-PARAMETERS  (verbatim from vidrag_pipeline.py)
# ---------------------------------------------------------------------------
# Key frames uniformly sampled per video.
# 64 is what the paper uses for LLaVA-Video in every table (Video-MME Table 1,
# MLVU Table 2, LongVideoBench Table 3) and what evals/*.py use.
# vidrag_pipeline.py's demo default is 32, which matches no paper result.
max_frames_num  = 64
rag_threshold   = 0.3
clip_threshold  = 0.3
beta            = 3.0

USE_OCR = True
USE_ASR = True
USE_DET = True            # requires the APE service running on port 9999

APE_HOST = "0.0.0.0"
APE_PORT = 9999

# Max tokens for the open-ended answer. vidrag_pipeline.py uses 4096; the
# open-ended answers here are short, so this is a safety cap only.
ANSWER_MAX_NEW_TOKENS = 1024

# ---------------------------------------------------------------------------
# 4. QUESTION CSV PARSING
#    The harness auto-detects the question column from these candidates
#    (case-insensitive). The first match wins. Same for the optional id and
#    ground-truth columns. If nothing matches, the FIRST column is used as the
#    question and rows are numbered 1..N.
# ---------------------------------------------------------------------------
QUESTION_COLUMN_CANDIDATES     = ["question", "questions", "question_text", "query", "q"]
QUESTION_ID_COLUMN_CANDIDATES  = ["question_id", "qid", "id", "index", "no", "s_no"]
GROUND_TRUTH_COLUMN_CANDIDATES = ["answer", "ground_truth", "gt", "correct_answer", "reference", "gt_answer"]

# ---------------------------------------------------------------------------
# 5. RUN BEHAVIOUR
# ---------------------------------------------------------------------------
# Skip a video whose output folder already holds a finished responses.csv.
# Useful when a rented GPU box dies mid-run.
SKIP_COMPLETED = True

# Also dump the full RAG-augmented prompt sent to the LVLM for each question.
SAVE_PROMPTS = True

os.makedirs(RESTORE_DIR, exist_ok=True)
os.makedirs(os.path.join(RESTORE_DIR, "audio"), exist_ok=True)
