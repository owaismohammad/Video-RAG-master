#!/usr/bin/env bash
# Start the APE object-detection service the pipeline talks to on port 9999.
# Leave this running in its own shell while the benchmark runs.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/APE"
exec "$REPO_ROOT/.venv-ape/bin/python" demo/ape_service.py
