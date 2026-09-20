#!/usr/bin/env bash
# ===========================================================================
# One-shot setup for the Video-RAG benchmark on a rented GPU box
# (vast.ai / runpod / any Ubuntu + CUDA 12.x image).
#
#   git clone <this repo> && cd Video-RAG-master && bash setup.sh 2>&1 | tee setup.log
#
# Everything installs with uv. Two virtualenvs, because APE pins detectron2 and
# an older torchvision that cannot co-resolve with LLaVA-NeXT -- this is the
# same reason the upstream readme uses two separate conda envs.
#
#   .venv       -> Video-RAG pipeline + LLaVA-Video-7B-Qwen2  (the benchmark)
#   .venv-ape   -> APE detection service on port 9999         (USE_DET = True)
#
# Nothing here prompts. Edit the knobs, run, walk away.
# ===========================================================================
set -euo pipefail

# --- knobs -----------------------------------------------------------------
INSTALL_APE=1                      # 0 -> skip APE (then set USE_DET = False in config.py)
DOWNLOAD_WEIGHTS=1                 # 0 -> weights already cached
MODEL_DIR="/workspace/models"      # must match config.MODEL_PATH's parent
HF_HOME_DIR="/workspace/hf_cache"  # CLIP + Whisper land here
APE_CKPT_DIR="/checkpoints"        # hardcoded in ape_tools/ape_api.py -- do not change
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export UV_HTTP_TIMEOUT=900
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "=============================================================="
echo " Video-RAG benchmark setup   ($(date))"
echo " repo: $REPO_ROOT"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "!! no nvidia-smi"
echo "CPU cores: $(nproc --all)"

# Disk: the run needs ~25GB of weights plus your videos, and model loading is
# the one genuinely I/O-heavy step. Find out now, not three hours in.
echo "--- disk ---"
df -h /workspace 2>/dev/null || df -h .
echo -n "write speed: "
dd if=/dev/zero of=/workspace/.disktest bs=1M count=512 oflag=direct 2>&1 \
    | tail -1 || echo "(direct I/O unsupported, skipped)"
rm -f /workspace/.disktest
echo "  (<200 MB/s = slow host; model load will take minutes. >1 GB/s = NVMe, fine.)"
echo "------------"

# --- 1. system packages ----------------------------------------------------
echo ""
echo "### [1/7] system packages"
if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    SUDO=""; if [ "$(id -u)" -ne 0 ]; then SUDO="sudo"; fi
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq \
        git git-lfs curl ffmpeg libgl1 libglib2.0-0 build-essential ninja-build
    git lfs install || true
else
    echo "  !! apt-get not found; ensure ffmpeg + libGL are present"
fi

# --- 2. uv -----------------------------------------------------------------
echo ""
echo "### [2/7] uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# Vast/RunPod images ship with a venv (e.g. /venv/main, py3.12) already active.
# `uv pip install` follows VIRTUAL_ENV, so without this the wheels below would
# land in that env instead of the project's .venv (py3.10).
unset VIRTUAL_ENV
uv --version

# --- 3. main environment ---------------------------------------------------
echo ""
echo "### [3/7] uv sync  (torch 2.1.2+cu121, LLaVA-NeXT, Video-RAG deps)"
uv sync
uv run python -c "import spacy; spacy.load('en_core_web_sm')" \
    || uv run python -m spacy download en_core_web_sm

# --- 4. flash-attn (prebuilt wheel) ----------------------------------------
# llava's load_pretrained_model defaults to attn_implementation="flash_attention_2".
# Building flash-attn from source takes 30-60 min of paid GPU time; this exact
# prebuilt wheel (verified to exist) installs in seconds. cu122 wheels are
# compatible with cu121 torch builds.
echo ""
echo "### [4/7] flash-attn"
FA_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
if uv pip install --python "$REPO_ROOT/.venv/bin/python" "$FA_WHL"; then
    uv run python -c "import flash_attn; print('flash-attn', flash_attn.__version__)" \
        && echo "  OK -- leave ATTN_IMPLEMENTATION = 'flash_attention_2' in config.py"
else
    echo "  !! flash-attn install FAILED."
    echo "  !! ACTION REQUIRED: set ATTN_IMPLEMENTATION = 'sdpa' in benchmark/config.py"
fi

# --- 5. model weights ------------------------------------------------------
echo ""
echo "### [5/7] model weights"
export HF_HOME="$HF_HOME_DIR"
mkdir -p "$HF_HOME" "$MODEL_DIR"
if [ "$DOWNLOAD_WEIGHTS" = "1" ]; then
    uv run python - <<PY
import os
from huggingface_hub import snapshot_download
# The LVLM is loaded from a local path (config.MODEL_PATH).
p = snapshot_download("lmms-lab/LLaVA-Video-7B-Qwen2",
                      local_dir=os.path.join("$MODEL_DIR", "LLaVA-Video-7B-Qwen2"))
print("LVLM  ->", p)
# CLIP + Whisper are referenced by hub id and resolve out of HF_HOME.
print("CLIP  ->", snapshot_download("openai/clip-vit-large-patch14-336"))
print("ASR   ->", snapshot_download("openai/whisper-large"))
# Contriever, used by tools/rag_retriever_dynamic.py for OCR/ASR retrieval.
print("RAG   ->", snapshot_download("facebook/contriever"))
PY
else
    echo "  skipped (DOWNLOAD_WEIGHTS=0)"
fi

# --- 6. APE ----------------------------------------------------------------
echo ""
echo "### [6/7] APE detection service"
if [ "$INSTALL_APE" = "1" ]; then
    [ -d "APE" ] || git clone https://github.com/shenyunhang/APE.git

    [ -x .venv-ape/bin/python ] || uv venv .venv-ape --python 3.10
    export VIRTUAL_ENV="$REPO_ROOT/.venv-ape"
    uv pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121

    # detectron2 / detrex / CLIP / APE compile against the torch installed
    # above, so they must build WITHOUT uv's isolated build env (which has no
    # torch -> "No module named 'torch'"). That needs setuptools + wheel +
    # ninja + cython present in the venv. setuptools is pinned <70 because
    # torch 2.1.2 imports pkg_resources, which newer setuptools removed.
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export FORCE_CUDA=1 MAX_JOBS="$(nproc --all)"
    # Compile for this box's GPU only (default builds every arch: much slower).
    export TORCH_CUDA_ARCH_LIST="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
    uv pip install "setuptools==69.5.1" wheel ninja cython "numpy<2"

    # APE/requirements.txt pins torch==1.12.1, which would downgrade the
    # torch 2.1.2+cu121 installed above (and break the CUDA 12 build). Drop
    # torch/torchvision from it; everything else (incl. the pinned
    # detectron2 + detrex + CLIP commits) is kept.
    grep -v -E '^(torch|torchvision)(==|$)' APE/requirements.txt > .ape-requirements.txt
    uv pip install --no-build-isolation -r .ape-requirements.txt || \
        echo "  !! APE requirements failed (detectron2/detrex build?) -- APE will not start"
    rm -f .ape-requirements.txt

    uv pip install --no-build-isolation -e ./APE
    uv pip install decord xformers==0.0.23.post1 || true
    unset VIRTUAL_ENV

    # ape_service.py + ape_api.py belong in APE/demo (upstream readme step 4);
    # ape_api.py imports predictor_lazy, which lives there.
    cp ape_tools/ape_api.py ape_tools/ape_service.py APE/demo/

    # ape_api.py hardcodes train.init_checkpoint=/checkpoints/model_final.pth.
    # That absolute path sits on the container's root overlay, which on most
    # rented boxes is small -- the big allocated disk is mounted at /workspace.
    # Symlink it so the checkpoint lands on the large disk while ape_api.py
    # stays unmodified.
    if [ ! -e "$APE_CKPT_DIR" ]; then
        mkdir -p /workspace/checkpoints
        ln -s /workspace/checkpoints "$APE_CKPT_DIR"
        echo "  $APE_CKPT_DIR -> /workspace/checkpoints"
    fi
    mkdir -p "$APE_CKPT_DIR"

    # The config it loads is APE-L_D. Locate that exact checkpoint in the HF
    # repo rather than guessing its path (the repo is 22.9GB of all variants;
    # we pull only the one file).
    if [ ! -f "$APE_CKPT_DIR/model_final.pth" ]; then
        uv run python - <<PY
import shutil
from huggingface_hub import HfApi, hf_hub_download
KEY = "ape_deta_vitl_eva02_clip_vlf_lsj1024_cp_16x4_1080k"
files = HfApi().list_repo_files("shenyunhang/APE")
cands = [f for f in files if f.endswith(".pth") and KEY in f]
if not cands:
    cands = [f for f in files if f.endswith("model_final.pth")]
    print("!! exact APE-D checkpoint not found; candidates:", files[:40])
if not cands:
    raise SystemExit("!! no APE checkpoint found in shenyunhang/APE")
print("downloading", cands[0])
src = hf_hub_download("shenyunhang/APE", cands[0])
shutil.copy(src, "$APE_CKPT_DIR/model_final.pth")
print("APE ckpt -> $APE_CKPT_DIR/model_final.pth")
PY
    else
        echo "  APE checkpoint already present"
    fi
else
    echo "  skipped (INSTALL_APE=0) -- set USE_DET = False in benchmark/config.py"
fi

# --- 7. scratch dirs -------------------------------------------------------
echo ""
echo "### [7/7] scratch dirs"
mkdir -p /workspace/restore/audio \
         /workspace/dataset/videos \
         /workspace/dataset/questions \
         /workspace/Benchmarking_VideoRAG-Master

cat <<'EOF'

==============================================================
 Setup finished. Next:

   1. Upload videos    -> /workspace/dataset/videos     (Video_ID_1.mp4 ...)
      Upload questions -> /workspace/dataset/questions  (Video_ID_1.csv ...)

   2. Check the dataset wiring (no GPU, ~1 second):
        uv run python benchmark/test_dataset.py

   3. Smoke test ONE video first -- set END_VIDEO_ID = 1 in
      benchmark/config.py, then run step 4/5 and inspect the output.

   4. If USE_DET = True, shell 1:   bash run_ape_service.sh
   5. Shell 2:                      bash run_benchmark.sh

 Results -> /workspace/Benchmarking_VideoRAG-Master
==============================================================
EOF
