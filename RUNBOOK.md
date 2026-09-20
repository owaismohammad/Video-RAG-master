# Video-RAG benchmark — rented GPU runbook

Ordered steps for running 40 videos × 10 open-ended questions through Video-RAG
on a vast.ai box. Metric definitions and output format are in
[benchmark/README.md](benchmark/README.md).

**Config recap:** LLaVA-Video-7B-Qwen2, 64 frames, OCR + ASR + DET all on.

---

## Step 0 — push this repo (do it before renting)

The benchmark files must be in your GitHub repo so the VM can clone them.
The clock starts when you rent, so do this at home:

```bash
git add -A && git commit -m "Add benchmark harness" && git push
```

This includes `uv.lock`, which pins every package (torch 2.1.2+cu121, the
exact LLaVA-NeXT and transformers commits, and so on) so the VM's `uv sync`
installs exactly what was resolved here instead of re-resolving fresh. Don't
delete it. If you edit `pyproject.toml`, run `uv lock` and commit it again.

---

## Step 1 — rent the box

| | |
|---|---|
| **GPU** | 1× **48GB** — L40S, RTX A6000, or A40 (~42GB peak: 32GB benchmark + 10GB APE) |
| **Alternative** | 2× 24GB (cheaper) — pin APE to GPU 1, see Step 5 |
| **Comfortable** | 1× A100 80GB / H100 — the paper's hardware |
| **Disk** | **≥200 GB** (~25GB weights + your videos + extracted audio) |
| **CPU** | **≥16 cores.** Not optional — see the Contriever note at the bottom |
| **Image** | CUDA **12.1–12.4**, Ubuntu 22.04, Python 3.10 |

Avoid: 24GB cards with APE enabled (OOM at 64 frames), and Blackwell cards
(RTX 5090 / B200) — torch 2.1.2 has no sm_100 kernels.

---

## Step 2 — clone and upload data

```bash
cd /workspace && git clone https://github.com/owaismohammad/Video-RAG-master.git && cd Video-RAG-master
```

Upload so it looks like this:

```
/workspace/dataset/videos/Video_ID_1.mp4  ... Video_ID_40.mp4
/workspace/dataset/questions/Video_ID_1.csv ... Video_ID_40.csv
```

Upload in the background while setup runs — they're independent.

---

## Step 3 — setup (one command, ~20–40 min)

```bash
bash setup.sh 2>&1 | tee setup.log
```

Installs system packages, uv, the main venv, a prebuilt flash-attn wheel, all
model weights, and a second venv with APE + its checkpoint.

**Then grep the log for the one thing that can silently degrade the run:**

```bash
grep -i "flash-attn\|ACTION REQUIRED\|FAILED" setup.log
```

If flash-attn failed, set `ATTN_IMPLEMENTATION = "sdpa"` in
`benchmark/config.py`. Both are exact attention — sdpa is just slower.

---

## Step 4 — verify before spending GPU time

```bash
uv run python benchmark/test_dataset.py
```

Checks video discovery and that your real CSVs parse. Takes a second, no GPU.
If your question column isn't auto-detected, add its name to
`QUESTION_COLUMN_CANDIDATES` in `benchmark/config.py`.

---

## Step 5 — start APE (shell 1, leave running)

```bash
bash run_ape_service.sh
```

Wait for `Server is listening...` before Step 6.

Two 24GB GPUs instead of one 48GB? Pin APE to the second card:

```bash
CUDA_VISIBLE_DEVICES=1 bash run_ape_service.sh
```

Skipping APE entirely: set `USE_DET = False` in `benchmark/config.py` and skip
this step. OCR, ASR and RAG retrieval still run.

---

## Step 6 — smoke test ONE video (shell 2)

Do not launch all 40 first. In `benchmark/config.py`:

```python
START_VIDEO_ID = 1
END_VIDEO_ID   = 1
```

```bash
bash run_benchmark.sh
```

Then check the answers are real sentences, not empty or a single letter:

```bash
column -s, -t < /workspace/Benchmarking_VideoRAG-Master/Video_ID_1/responses.csv | cut -c1-200
cat /workspace/Benchmarking_VideoRAG-Master/Video_ID_1/summary.txt
nvidia-smi --query-gpu=memory.used --format=csv   # headroom check
```

In `responses.csv`, confirm `error` is empty and `num_det_frames_selected` is
non-zero (proves APE is actually being reached).

---

## Step 7 — full run

Set `END_VIDEO_ID = 40`, then run under `nohup` so an SSH drop doesn't kill it:

```bash
nohup bash run_benchmark.sh > /workspace/benchmark.log 2>&1 &
tail -f /workspace/benchmark.log
```

`SKIP_COMPLETED = True` means a re-run resumes at the first unfinished video.
`responses.csv` is rewritten after every question, so a crash costs one answer.

Monitor:

```bash
ls /workspace/Benchmarking_VideoRAG-Master/*/responses.csv | wc -l   # videos done
grep -c . /workspace/Benchmarking_VideoRAG-Master/all_responses.csv  # answers so far
```

---

## Step 8 — pull results down

```bash
cd /workspace && tar czf results.tar.gz Benchmarking_VideoRAG-Master
```

Then `scp` it off **before destroying the instance**:

```bash
scp -P <port> root@<host>:/workspace/results.tar.gz .
```

---

## Runtime and cost

Rough estimate for 40 videos averaging ~45 min each, 10 questions apiece, on a
48GB card. Not measured — treat as an order of magnitude.

| Stage | Per video | 40 videos |
|---|---|---|
| Whisper ASR (dominant) | 2–6 min | 2–4 h |
| Frame sampling + OCR + CLIP | ~1 min | ~40 min |
| 10 questions (LVLM ×2, APE, retrieval) | 3–6 min | 2–4 h |
| **Total** | | **≈ 5–9 h** |

Add ~40 min setup. At typical 48GB pricing that's a few dollars of compute —
but note the whole thing runs sequentially, so wall-clock is the cost driver.

**The one thing that can blow this up:**
[rag_retriever_dynamic.py](vidrag_pipeline/tools/rag_retriever_dynamic.py)
loads Contriever with no `.to(device)` — it runs **on CPU** — and re-encodes
every document on every query. A 90-minute video has ~180 ASR chunks, re-encoded
twice per question × 10 questions ≈ 3,600 CPU forward passes. On a weak vCPU
this can exceed the GPU time. This is upstream code, left unmodified on purpose.
Renting ≥16 cores is the mitigation; ask if you want the GPU + caching fix
(numerically identical — same model, same vectors, same retrieved docs).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: flash_attn` | `ATTN_IMPLEMENTATION = "sdpa"` in config.py |
| CUDA OOM during answer generation | Lower `max_frames_num` to 32, or move APE to another GPU |
| `ConnectionRefusedError` port 9999 | APE service isn't up — check shell 1, or set `USE_DET = False` |
| APE: `model_final.pth` not found | `ls -la /checkpoints/` — re-run the APE block of setup.sh |
| Model load fails on transformers version | Don't swap versions blind: `pyproject.toml` already pins the exact commit LLaVA-NeXT uses. Paste the traceback and check `uv pip list \| grep -i "transformers\|tokenizers"` first |
| All answers empty, `error` populated | Read the `error` column; it holds the full traceback |
| Questions parsed wrong | Add your column name to `QUESTION_COLUMN_CANDIDATES` |
| ASR times suspiciously near zero | Cached transcripts in `/workspace/restore/audio/` — delete for cold-start timings |
