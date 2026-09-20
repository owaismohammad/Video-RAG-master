#!/usr/bin/env bash
# Run the Video-RAG benchmark. All settings live in benchmark/config.py.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export TOKENIZERS_PARALLELISM=false

# Python puts the script's own dir (benchmark/) on sys.path, so `import config`
# and `import dataset` resolve regardless of cwd.
exec uv run python benchmark/run_benchmark.py
