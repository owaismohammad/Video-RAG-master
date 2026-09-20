"""
Video-RAG benchmark runner.

Runs the unmodified Video-RAG pipeline (vidrag_pipeline.py) over a custom
dataset of videos, each with its own CSV of open-ended questions, and records:

  * per-video "indexing" cost      -> key-frame sampling, CLIP frame features,
                                      OCR over key frames, Whisper ASR
  * how many key frames were used  -> sampled frames, and how many of them the
                                      CLIP gate actually forwarded to APE
  * per-question latency           -> retrieval-request step, CLIP gating,
                                      APE detection, OCR/ASR RAG retrieval,
                                      answer generation
  * the answer text itself

No argument parsing: edit benchmark/config.py.

Expected layout:
    <VIDEO_DIR>/Video_ID_1.mp4
    <QUESTION_DIR>/Video_ID_1.csv

Output layout:
    <OUTPUT_DIR>/Video_ID_1/responses.csv
    <OUTPUT_DIR>/Video_ID_1/indexing.json
    <OUTPUT_DIR>/Video_ID_1/summary.txt
    <OUTPUT_DIR>/Video_ID_1/prompts/q001.txt
    <OUTPUT_DIR>/benchmark_summary.csv
    <OUTPUT_DIR>/all_responses.csv
"""

import os
import sys

import ast
import json
import time
import socket
import pickle
import traceback

import config
from dataset import (
    find_video_file,
    load_questions,
    RESPONSE_FIELDS,
    read_responses_csv,
    write_responses_csv,
    write_summary_csv,
)

# Make the pipeline's own `tools/` package importable without copying files
# around. Nothing inside vidrag_pipeline/ is modified.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PIPELINE_DIR = os.path.join(_REPO_ROOT, "vidrag_pipeline")
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)

import copy

import numpy as np
import torch
import ffmpeg
import torchaudio
import easyocr
from PIL import Image
from tqdm import tqdm
from decord import VideoReader, cpu
from transformers import (
    CLIPProcessor,
    CLIPModel,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates

from tools.rag_retriever_dynamic import retrieve_documents_with_dynamic
from tools.filter_keywords import filter_keywords
from tools.scene_graph import generate_scene_graph_description


# ===========================================================================
# timing helper
# ===========================================================================
class Timer:
    """with Timer() as t: ...   ->   t.elapsed (seconds, float)"""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._t0
        return False


def _r(x, n=4):
    return round(float(x), n)


# ===========================================================================
# model loading  (same models / dtypes as vidrag_pipeline.py)
# ===========================================================================
print("Loading CLIP ...", flush=True)
clip_model = CLIPModel.from_pretrained(
    config.CLIP_PATH, torch_dtype=torch.float16, device_map="auto"
)
clip_processor = CLIPProcessor.from_pretrained(config.CLIP_PATH)

print("Loading Whisper ...", flush=True)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    config.WHISPER_PATH, torch_dtype=torch.float16, device_map="auto"
)
whisper_processor = WhisperProcessor.from_pretrained(config.WHISPER_PATH)

print("Loading LLaVA-Video ...", flush=True)
device = "cuda"
overwrite_config = {}
tokenizer, model, image_processor, max_length = load_pretrained_model(
    config.MODEL_PATH,
    config.MODEL_BASE,
    config.MODEL_NAME,
    torch_dtype="bfloat16",
    device_map="auto",
    attn_implementation=config.ATTN_IMPLEMENTATION,
    overwrite_config=overwrite_config,
)
model.eval()
conv_template = config.CONV_TEMPLATE

max_frames_num = config.max_frames_num
rag_threshold = config.rag_threshold
clip_threshold = config.clip_threshold
beta = config.beta
USE_OCR, USE_ASR, USE_DET = config.USE_OCR, config.USE_ASR, config.USE_DET

# One easyocr reader for the whole run (re-creating it per video wastes GPU
# time and would otherwise be charged to every video's indexing cost).
print("Loading EasyOCR ...", flush=True)
ocr_reader = easyocr.Reader(["en"])


# ===========================================================================
# pipeline functions -- copied from vidrag_pipeline.py. process_video,
# chunk_audio, transcribe_chunk and det_preprocess are identical (AST-checked).
# The others differ only as follows:
#   extract_audio   ffmpeg .run(quiet=True)  (silences ffmpeg's stderr spam)
#   get_asr_docs    `except Exception` instead of a bare `except`
#   get_ocr_docs    reuses one shared EasyOCR reader instead of building one per call
#   save_frames     writes to <RESTORE_DIR>/<video_id>/ instead of restore/
#   get_det_docs    takes that video_id; APE host/port come from config; `except Exception`
#   llava_inference max_new_tokens is a parameter (default 4096, as original)
# ===========================================================================
def process_video(video_path, max_frames_num, fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps() / fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i / fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i / vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time


def extract_audio(video_path, audio_path):
    if not os.path.exists(audio_path):
        ffmpeg.input(video_path).output(
            audio_path, acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True)


def chunk_audio(audio_path, chunk_length_s=30):
    speech, sr = torchaudio.load(audio_path)
    speech = speech.mean(dim=0)
    speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(speech)
    num_samples_per_chunk = chunk_length_s * 16000
    chunks = []
    for i in range(0, len(speech), num_samples_per_chunk):
        chunks.append(speech[i:i + num_samples_per_chunk])
    return chunks


def transcribe_chunk(chunk):
    inputs = whisper_processor(chunk, return_tensors="pt")
    inputs["input_features"] = inputs["input_features"].to(whisper_model.device, torch.float16)
    with torch.no_grad():
        predicted_ids = whisper_model.generate(
            inputs["input_features"],
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


def get_asr_docs(video_path, audio_path):
    full_transcription = []
    try:
        extract_audio(video_path, audio_path)
    except Exception:
        return full_transcription
    audio_chunks = chunk_audio(audio_path, chunk_length_s=30)

    for chunk in audio_chunks:
        transcription = transcribe_chunk(chunk)
        full_transcription.append(transcription)

    return full_transcription


def get_ocr_docs(frames):
    reader = ocr_reader
    text_set = []
    ocr_docs = []
    for img in frames:
        ocr_results = reader.readtext(img)
        det_info = ""
        for result in ocr_results:
            text = result[1]
            confidence = result[2]
            if confidence > 0.5 and text not in text_set:
                det_info += f"{text}; "
                text_set.append(text)
        if len(det_info) > 0:
            ocr_docs.append(det_info)

    return ocr_docs


def save_frames(frames, file_name):
    out_dir = os.path.join(config.RESTORE_DIR, file_name)
    os.makedirs(out_dir, exist_ok=True)
    file_paths = []
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        file_path = os.path.join(out_dir, f"frame_{i}.png")
        img.save(file_path)
        file_paths.append(file_path)
    return file_paths


def get_det_docs(frames, prompt, file_name):
    prompt = ",".join(prompt)
    frames_path = save_frames(frames, file_name)
    res = []
    if len(frames) > 0:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((config.APE_HOST, config.APE_PORT))
        data = (frames_path, prompt)
        client_socket.send(pickle.dumps(data))
        result_data = client_socket.recv(4096)
        try:
            res = pickle.loads(result_data)
        except Exception:
            res = []
    return res


def det_preprocess(det_docs, location, relation, number):
    scene_descriptions = []

    for det_doc_per_frame in det_docs:
        objects = []
        scene_description = ""
        if len(det_doc_per_frame) > 0:
            for obj_id, objs in enumerate(det_doc_per_frame.split(";")):
                obj_name = objs.split(":")[0].strip()
                obj_bbox = objs.split(":")[1].strip()
                obj_bbox = ast.literal_eval(obj_bbox)
                objects.append({"id": obj_id, "label": obj_name, "bbox": obj_bbox})

            scene_description = generate_scene_graph_description(
                objects, location, relation, number
            )
        scene_descriptions.append(scene_description)

    return scene_descriptions


def llava_inference(qs, video, max_new_tokens=4096):
    if video is not None:
        question = DEFAULT_IMAGE_TOKEN + qs
    else:
        question = qs
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    cont = model.generate(
        input_ids,
        images=video,
        modalities=["video"],
        do_sample=False,
        temperature=0,
        max_new_tokens=max_new_tokens,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
    return text_outputs


# The step-0 decoupling prompt. Read straight out of vidrag_pipeline.py (not
# retyped here) so it is byte-identical to the original by construction.
def _load_retrieve_prompt_suffix():
    src = open(os.path.join(_PIPELINE_DIR, "vidrag_pipeline.py"), encoding="utf-8").read()
    parts = [
        n.value.value
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.AugAssign)
        and isinstance(n.target, ast.Name)
        and n.target.id == "retrieve_pmt_0"
    ]
    assert len(parts) == 2, "vidrag_pipeline.py's retrieve_pmt_0 layout changed"
    return "".join(parts)


RETRIEVE_PROMPT_SUFFIX = _load_retrieve_prompt_suffix()


# ===========================================================================
# per-video: indexing  (everything that is done ONCE, before any question)
# ===========================================================================
def index_video(video_path, video_name):
    """Run the once-per-video part of Video-RAG and time each stage."""
    stats = {"video_path": video_path, "video_name": video_name}

    # --- key-frame sampling -------------------------------------------------
    with Timer() as t:
        frames, frame_time, video_time = process_video(
            video_path, max_frames_num, 1, force_sample=True
        )
    raw_video = [f for f in frames]
    stats["t_keyframe_sampling_sec"] = _r(t.elapsed)
    stats["video_duration_sec"] = _r(video_time, 2)
    stats["max_frames_num_configured"] = max_frames_num
    stats["num_keyframes_extracted"] = int(len(frames))
    stats["keyframe_timestamps"] = frame_time
    stats["keyframe_resolution"] = f"{frames.shape[2]}x{frames.shape[1]}"

    # --- LVLM visual preprocessing -----------------------------------------
    with Timer() as t:
        video = image_processor.preprocess(
            frames, return_tensors="pt"
        )["pixel_values"].cuda().bfloat16()
        video = [video]
    stats["t_lvlm_frame_preprocess_sec"] = _r(t.elapsed)

    # --- CLIP frame features (used by the DET gate) -------------------------
    video_tensor = None
    stats["t_clip_frame_encode_sec"] = 0.0
    if USE_DET:
        with Timer() as t:
            video_tensor = []
            for frame in raw_video:
                processed = clip_processor(
                    images=frame, return_tensors="pt"
                )["pixel_values"].to(clip_model.device, dtype=torch.float16)
                video_tensor.append(processed.squeeze(0))
            video_tensor = torch.stack(video_tensor, dim=0)
        stats["t_clip_frame_encode_sec"] = _r(t.elapsed)
        stats["num_clip_frames_encoded"] = int(video_tensor.shape[0])

    # --- OCR over the key frames -------------------------------------------
    ocr_docs_total = []
    stats["t_ocr_sec"] = 0.0
    if USE_OCR:
        with Timer() as t:
            ocr_docs_total = get_ocr_docs(frames)
        stats["t_ocr_sec"] = _r(t.elapsed)
        stats["num_ocr_docs_total"] = len(ocr_docs_total)
        stats["num_frames_with_ocr_text"] = len(ocr_docs_total)

    # --- ASR (Whisper) over the full audio track ----------------------------
    asr_docs_total = []
    stats["t_asr_sec"] = 0.0
    stats["asr_loaded_from_cache"] = False
    if USE_ASR:
        stem = os.path.basename(video_path).split(".")[0]
        cache_txt = os.path.join(config.RESTORE_DIR, "audio", stem + ".txt")
        with Timer() as t:
            if os.path.exists(cache_txt):
                with open(cache_txt, "r", encoding="utf-8") as f:
                    asr_docs_total = f.readlines()
                stats["asr_loaded_from_cache"] = True
            else:
                audio_path = os.path.join(config.RESTORE_DIR, "audio", stem + ".wav")
                asr_docs_total = get_asr_docs(video_path, audio_path)
                with open(cache_txt, "w", encoding="utf-8") as f:
                    for doc in asr_docs_total:
                        f.write(doc + "\n")
        stats["t_asr_sec"] = _r(t.elapsed)
        stats["num_asr_docs_total"] = len(asr_docs_total)

    stats["t_total_indexing_sec"] = _r(
        stats["t_keyframe_sampling_sec"]
        + stats["t_lvlm_frame_preprocess_sec"]
        + stats["t_clip_frame_encode_sec"]
        + stats["t_ocr_sec"]
        + stats["t_asr_sec"]
    )

    ctx = {
        "frames": frames,
        "video": video,
        "video_tensor": video_tensor,
        "ocr_docs_total": ocr_docs_total,
        "asr_docs_total": asr_docs_total,
    }
    return ctx, stats


# ===========================================================================
# per-question: the Video-RAG query path
# ===========================================================================
def answer_question(question_text, ctx, video_name):
    """One question through Video-RAG. Returns (answer, prompt, metrics)."""
    m = {k: "" for k in RESPONSE_FIELDS}
    frames = ctx["frames"]
    video = ctx["video"]
    video_tensor = ctx["video_tensor"]
    ocr_docs_total = ctx["ocr_docs_total"]
    asr_docs_total = ctx["asr_docs_total"]

    q_t0 = time.perf_counter()

    det_docs, ocr_docs, asr_docs, det_top_idx = [], [], [], []

    # ---- step 0: retrieval request (CoT decoupling) ------------------------
    retrieve_pmt_0 = "Question: " + question_text + RETRIEVE_PROMPT_SUFFIX
    with Timer() as t:
        json_request = llava_inference(retrieve_pmt_0, None)
    m["t_retrieval_request_sec"] = _r(t.elapsed)
    m["json_request"] = json_request.replace("\n", " ")[:2000]

    # ---- step 1: gather docs ----------------------------------------------
    query = [question_text]
    torch.cuda.empty_cache()

    # APE / detection
    m["t_clip_frame_gate_sec"] = 0.0
    m["t_ape_detection_sec"] = 0.0
    m["num_det_frames_selected"] = 0
    m["num_det_docs"] = 0
    if USE_DET:
        with Timer() as t:
            try:
                request_det = json.loads(json_request)["DET"]
                request_det = filter_keywords(request_det)
                clip_text = ["A picture of " + txt for txt in request_det]
                if len(clip_text) == 0:
                    clip_text = ["A picture of object"]
            except Exception:
                request_det = None
                clip_text = ["A picture of object"]

            clip_inputs = clip_processor(
                text=clip_text, return_tensors="pt", padding=True, truncation=True
            ).to(clip_model.device)
            with torch.no_grad():
                clip_img_feats = clip_model.get_image_features(video_tensor)
                text_features = clip_model.get_text_features(**clip_inputs)
                similarities = (clip_img_feats @ text_features.T).squeeze(0).mean(1).cpu()
                similarities = np.array(similarities, dtype=np.float64)
                alpha = beta * (len(similarities) / 16)
                similarities = similarities * alpha / np.sum(similarities)

            del clip_inputs, clip_img_feats, text_features
            torch.cuda.empty_cache()

            det_top_idx = [
                idx for idx in range(max_frames_num) if similarities[idx] > clip_threshold
            ]
        m["t_clip_frame_gate_sec"] = _r(t.elapsed)
        m["num_det_frames_selected"] = len(det_top_idx)

        if request_det is not None and len(request_det) > 0:
            with Timer() as t:
                det_docs = get_det_docs(frames[det_top_idx], request_det, video_name)

                L, R, N = False, False, False
                try:
                    det_retrieve_info = json.loads(json_request)["TYPE"]
                except Exception:
                    det_retrieve_info = None
                if det_retrieve_info is not None:
                    if "location" in det_retrieve_info:
                        L = True
                    if "relation" in det_retrieve_info:
                        R = True
                    if "number" in det_retrieve_info:
                        N = True
                det_docs = det_preprocess(det_docs, location=L, relation=R, number=N)
            m["t_ape_detection_sec"] = _r(t.elapsed)
            m["num_det_docs"] = len([d for d in det_docs if len(d) > 0])

    # OCR retrieval
    m["t_ocr_retrieval_sec"] = 0.0
    m["num_ocr_docs_retrieved"] = 0
    if USE_OCR:
        with Timer() as t:
            try:
                request_det = json.loads(json_request)["DET"]
                request_det = filter_keywords(request_det)
            except Exception:
                request_det = None
            ocr_docs = []
            if len(ocr_docs_total) > 0:
                ocr_query = query.copy()
                if request_det is not None and len(request_det) > 0:
                    ocr_query.extend(request_det)
                ocr_docs, _ = retrieve_documents_with_dynamic(
                    ocr_docs_total, ocr_query, threshold=rag_threshold
                )
        m["t_ocr_retrieval_sec"] = _r(t.elapsed)
        m["num_ocr_docs_retrieved"] = len(ocr_docs)

    # ASR retrieval
    m["t_asr_retrieval_sec"] = 0.0
    m["num_asr_docs_retrieved"] = 0
    if USE_ASR:
        with Timer() as t:
            asr_docs = []
            try:
                request_asr = json.loads(json_request)["ASR"]
            except Exception:
                request_asr = None
            if len(asr_docs_total) > 0:
                asr_query = query.copy()
                if request_asr is not None:
                    asr_query.append(request_asr)
                asr_docs, _ = retrieve_documents_with_dynamic(
                    asr_docs_total, asr_query, threshold=rag_threshold
                )
        m["t_asr_retrieval_sec"] = _r(t.elapsed)
        m["num_asr_docs_retrieved"] = len(asr_docs)

    # ---- step 2: build the augmented prompt --------------------------------
    qs = ""
    if USE_DET and len(det_docs) > 0:
        for i, info in enumerate(det_docs):
            if len(info) > 0:
                qs += f"Frame {str(det_top_idx[i]+1)}: " + info + "\n"
        if len(qs) > 0:
            qs = (
                f"\nVideo have {str(max_frames_num)} frames in total, "
                "the detected objects' information in specific frames: " + qs
            )
    if USE_ASR and len(asr_docs) > 0:
        qs += (
            "\nVideo Automatic Speech Recognition information "
            "(given in chronological order of the video): " + " ".join(asr_docs)
        )
    if USE_OCR and len(ocr_docs) > 0:
        qs += (
            "\nVideo OCR information (given in chronological order of the video): "
            + "; ".join(ocr_docs)
        )
    # NOTE: the only instruction that differs from vidrag_pipeline.py. The
    # original line is multiple-choice specific ("Respond with only the letter
    # (A, B, C, or D)"); this dataset is open-ended.
    qs += (
        "Answer the following question based on the video and the information "
        "(if given). Question: " + question_text
    )

    # ---- step 3: answer ----------------------------------------------------
    with Timer() as t:
        res = llava_inference(qs, video, max_new_tokens=config.ANSWER_MAX_NEW_TOKENS)
    m["t_answer_generation_sec"] = _r(t.elapsed)
    m["t_question_total_sec"] = _r(time.perf_counter() - q_t0)

    return res, qs, m


# ===========================================================================
# main loop
# ===========================================================================
def main():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("-" * 70)
    print(f"OCR {rag_threshold}: {USE_OCR}")
    print(f"ASR {rag_threshold}: {USE_ASR}")
    print(f"DET {beta}-{clip_threshold}: {USE_DET}")
    print(f"Frames: {max_frames_num}")
    print(f"Videos: {config.START_VIDEO_ID} .. {config.END_VIDEO_ID} (inclusive)")
    print(f"Output: {config.OUTPUT_DIR}")
    print("-" * 70, flush=True)

    summary_rows = []
    all_rows = []

    for i in range(config.START_VIDEO_ID, config.END_VIDEO_ID + 1):
        video_name = config.VIDEO_NAME_TEMPLATE.format(i=i)
        out_dir = os.path.join(config.OUTPUT_DIR, video_name)
        os.makedirs(out_dir, exist_ok=True)
        responses_csv = os.path.join(out_dir, "responses.csv")
        indexing_json = os.path.join(out_dir, "indexing.json")

        if config.SKIP_COMPLETED and os.path.exists(responses_csv) and os.path.exists(indexing_json):
            print(f"[{video_name}] already done -> skipping", flush=True)
            with open(indexing_json, "r", encoding="utf-8") as f:
                prev = json.load(f)
            summary_rows.append(prev.get("_summary_row", {"video_id": video_name}))
            # Pull the earlier answers back in so the combined sheets stay
            # complete across a resumed run.
            all_rows.extend(read_responses_csv(responses_csv))
            continue

        video_path = find_video_file(video_name)
        if video_path is None:
            print(f"[{video_name}] VIDEO NOT FOUND in {config.VIDEO_DIR} -> skipping", flush=True)
            continue

        questions, q_csv = load_questions(video_name)
        if not questions:
            print(f"[{video_name}] no questions at {q_csv} -> skipping", flush=True)
            continue

        print(f"\n=== [{video_name}] {len(questions)} questions | {video_path}", flush=True)

        # ---------------- indexing ----------------
        try:
            ctx, idx_stats = index_video(video_path, video_name)
        except Exception:
            print(f"[{video_name}] INDEXING FAILED:\n{traceback.format_exc()}", flush=True)
            with open(os.path.join(out_dir, "error.txt"), "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
            continue

        print(
            f"    indexing {idx_stats['t_total_indexing_sec']:.2f}s "
            f"| keyframes {idx_stats['num_keyframes_extracted']} "
            f"| ocr_docs {idx_stats.get('num_ocr_docs_total', 0)} "
            f"| asr_docs {idx_stats.get('num_asr_docs_total', 0)}",
            flush=True,
        )

        # ---------------- questions ----------------
        rows = []
        prompts_dir = os.path.join(out_dir, "prompts")
        if config.SAVE_PROMPTS:
            os.makedirs(prompts_dir, exist_ok=True)

        for qi, q in enumerate(tqdm(questions, desc=f"  {video_name}", unit="q"), start=1):
            try:
                res, prompt, m = answer_question(q["question"], ctx, video_name)
                err = ""
            except Exception:
                res, prompt = "", ""
                m = {k: "" for k in RESPONSE_FIELDS}
                err = traceback.format_exc()
                print(f"    q{qi} FAILED:\n{err}", flush=True)

            m["video_id"] = video_name
            m["question_index"] = qi
            m["question_id"] = q["question_id"]
            m["question"] = q["question"]
            m["ground_truth"] = q["ground_truth"]
            m["response"] = res
            # keep the TAIL: the exception type/message is the last line of a traceback
            m["error"] = err.replace("\n", " ")[-1000:] if err else ""
            rows.append(m)
            all_rows.append(m)

            if config.SAVE_PROMPTS and prompt:
                with open(os.path.join(prompts_dir, f"q{qi:03d}.txt"), "w", encoding="utf-8") as f:
                    f.write(prompt)

            # flush after every question so a dead box loses at most one answer
            write_responses_csv(responses_csv, rows)

        # ---------------- per-video bookkeeping ----------------
        q_times = [
            r["t_question_total_sec"] for r in rows
            if isinstance(r.get("t_question_total_sec"), float)
        ]
        ans_times = [
            r["t_answer_generation_sec"] for r in rows
            if isinstance(r.get("t_answer_generation_sec"), float)
        ]
        total_q_time = float(sum(q_times))

        summary_row = {
            "video_id": video_name,
            "video_path": video_path,
            "video_duration_sec": idx_stats["video_duration_sec"],
            "num_questions": len(rows),
            "num_keyframes_extracted": idx_stats["num_keyframes_extracted"],
            "num_ocr_docs_total": idx_stats.get("num_ocr_docs_total", 0),
            "num_asr_docs_total": idx_stats.get("num_asr_docs_total", 0),
            "indexing_total_sec": idx_stats["t_total_indexing_sec"],
            "indexing_keyframe_sampling_sec": idx_stats["t_keyframe_sampling_sec"],
            "indexing_lvlm_preprocess_sec": idx_stats["t_lvlm_frame_preprocess_sec"],
            "indexing_clip_encode_sec": idx_stats["t_clip_frame_encode_sec"],
            "indexing_ocr_sec": idx_stats["t_ocr_sec"],
            "indexing_asr_sec": idx_stats["t_asr_sec"],
            "asr_loaded_from_cache": idx_stats["asr_loaded_from_cache"],
            "querying_total_sec": _r(total_q_time),
            "avg_sec_per_question": _r(total_q_time / len(rows)) if rows else 0.0,
            "avg_answer_generation_sec": _r(sum(ans_times) / len(ans_times)) if ans_times else 0.0,
            "wall_clock_total_sec": _r(idx_stats["t_total_indexing_sec"] + total_q_time),
        }
        summary_rows.append(summary_row)

        idx_stats["_summary_row"] = summary_row
        with open(indexing_json, "w", encoding="utf-8") as f:
            json.dump(idx_stats, f, indent=2, ensure_ascii=False)

        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"Video          : {video_name}\n")
            f.write(f"Path           : {video_path}\n")
            f.write(f"Duration       : {idx_stats['video_duration_sec']} s\n")
            f.write(f"Key frames     : {idx_stats['num_keyframes_extracted']} "
                    f"(uniformly sampled, max_frames_num={max_frames_num})\n")
            f.write(f"Frame times    : {idx_stats['keyframe_timestamps']}\n\n")
            f.write("--- INDEXING (once per video) ---\n")
            f.write(f"  key-frame sampling : {idx_stats['t_keyframe_sampling_sec']:.3f} s\n")
            f.write(f"  LVLM frame prep    : {idx_stats['t_lvlm_frame_preprocess_sec']:.3f} s\n")
            f.write(f"  CLIP frame encode  : {idx_stats['t_clip_frame_encode_sec']:.3f} s\n")
            f.write(f"  OCR (easyocr)      : {idx_stats['t_ocr_sec']:.3f} s "
                    f"-> {idx_stats.get('num_ocr_docs_total', 0)} docs\n")
            f.write(f"  ASR (whisper)      : {idx_stats['t_asr_sec']:.3f} s "
                    f"-> {idx_stats.get('num_asr_docs_total', 0)} docs"
                    f"{' [from cache]' if idx_stats['asr_loaded_from_cache'] else ''}\n")
            f.write(f"  TOTAL INDEXING     : {idx_stats['t_total_indexing_sec']:.3f} s\n\n")
            f.write("--- QUERYING ---\n")
            f.write(f"  questions          : {len(rows)}\n")
            f.write(f"  total              : {summary_row['querying_total_sec']:.3f} s\n")
            f.write(f"  avg per question   : {summary_row['avg_sec_per_question']:.3f} s\n")
            f.write(f"  avg answer gen     : {summary_row['avg_answer_generation_sec']:.3f} s\n\n")
            f.write(f"WALL CLOCK TOTAL     : {summary_row['wall_clock_total_sec']:.3f} s\n")

        # free the per-video tensors before the next video
        ctx.clear()
        torch.cuda.empty_cache()

        # rolling global outputs
        if summary_rows:
            write_summary_csv(
                os.path.join(config.OUTPUT_DIR, "benchmark_summary.csv"), summary_rows
            )
        if all_rows:
            write_responses_csv(os.path.join(config.OUTPUT_DIR, "all_responses.csv"), all_rows)

    print("\nDone. Results in", config.OUTPUT_DIR, flush=True)


if __name__ == "__main__":
    main()
