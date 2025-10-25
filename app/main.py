# app/main.py
import asyncio
import logging
import sys
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from .config import MODELS
from .extract import get_pages_text
from .llm import (
    extract_page_chunk_multi,
    chat_ollama_model_retry_json,
    build_prompt,
)
from .logic import (
    aggregate,
    chunk_pages,
    compare_models,
    parse_llm_json,
)
from .schema import FinalReport, MultiModelReport, PageExtraction

# Console logger
logger = logging.getLogger("pdf-analyzer")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(_h)

app = FastAPI(
    title="PDF Statement Analyzer (Ollama Multi-Model)",
    description="Upload a PDF (with optional password) and compare qwen3:30b, gemma3:27b, and mistral:7b-instruct on transaction extraction.",
    version="1.3.0",
)

# Simple in-memory progress store
PROGRESS: Dict[str, Dict] = {}

def _init_progress(job_id: str, total_chunks: int, models: List[str]):
    PROGRESS[job_id] = {
        "chunks_total": total_chunks,
        "chunks_done": 0,
        "models": models,
        "last_step": "started",
        "per_model_ms": {m: 0.0 for m in models},
        "history": [],
        "error": None,
        "started_at": time.time(),
    }

def _bump_progress(job_id: str, step: str = "", per_model_durations: Dict[str, float] | None = None):
    p = PROGRESS.get(job_id)
    if not p:
        return
    if step:
        p["history"].append(step)
        p["last_step"] = step
    if per_model_durations:
        for m, ms in per_model_durations.items():
            p["per_model_ms"][m] = p["per_model_ms"].get(m, 0.0) + float(ms)

def _finish_progress(job_id: str):
    p = PROGRESS.get(job_id)
    if p:
        p["finished_at"] = time.time()

def _log_console_progress(job_id: str):
    p = PROGRESS.get(job_id)
    if not p:
        return
    done = int(p.get("chunks_done", 0))
    total = int(p.get("chunks_total", 1)) or 1
    pct = int(done * 100 / total)
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + " " * (bar_len - filled)
    logger.info(f"[{pct:3d}%] |{bar}| chunks {done}/{total}")

@app.get("/progress/{job_id}", summary="Check current progress")
def get_progress(job_id: str):
    if job_id not in PROGRESS:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return PROGRESS[job_id]

@app.post("/analyze-multi", response_model=MultiModelReport, summary="Analyze a PDF with multiple models")
async def analyze_pdf_multi(
    file: UploadFile = File(..., description="PDF bank statement (15–25 pages OK)"),
    password: Optional[str] = Form(default=None, description="Password if the PDF is protected"),
    pages_per_chunk: int = Form(default=2, ge=1, le=5, description="Pages to send to the model at once"),
    models: Optional[str] = Form(default=None, description="Comma-separated model list. Leave empty to use defaults."),
):
    try:
        raw_bytes = await file.read()
        pages_text = get_pages_text(raw_bytes, password)
        if not any(pages_text):
            raise HTTPException(status_code=422, detail="No readable text. If scanned, install OCR deps (poppler, tesseract).")

        use_models = [m.strip() for m in models.split(",")] if models else MODELS
        chunks = chunk_pages(pages_text, chunk_size=pages_per_chunk)

        job_id = str(uuid.uuid4())
        _init_progress(job_id, total_chunks=len(chunks), models=use_models)
        _bump_progress(job_id, step=f"Prepared {len(chunks)} chunk(s)")
        _log_console_progress(job_id)

        # Raw outputs per model (list aligned by chunk order)
        per_model_raw: Dict[str, List[str]] = {m: [] for m in use_models}

        # Process chunk-by-chunk, all models concurrently per chunk
        for idx, chunk in enumerate(chunks, start=1):
            t0 = time.time()
            _bump_progress(job_id, step=f"Chunk {idx}/{len(chunks)}: sending to models")
            raw_by_model = await extract_page_chunk_multi(use_models, chunk)
            if not isinstance(raw_by_model, dict):
                _bump_progress(job_id, step=f"Chunk {idx}: unexpected engine result; coercing to empty dict")
                raw_by_model = {}
            t1 = time.time()

            # Attempt immediate validation; if it fails, do a one-time repair retry
            for m in use_models:
                raw = raw_by_model.get(m, "")
                valid = True
                try:
                    _ = parse_llm_json(raw)
                except Exception:
                    valid = False

                if not valid:
                    try:
                        repaired = await chat_ollama_model_retry_json(m, build_prompt(chunk), raw)
                        _ = parse_llm_json(repaired)  # validate repaired
                        raw = repaired
                        _bump_progress(job_id, step=f"Chunk {idx}: {m} repaired JSON")
                    except Exception as e:
                        _bump_progress(job_id, step=f"Chunk {idx}: {m} repair failed: {type(e).__name__}")

                per_model_raw[m].append(raw)

            # Attribute chunk time across models (heuristic)
            elapsed_ms = (t1 - t0) * 1000.0
            per_model = elapsed_ms / max(1, len(use_models))
            _bump_progress(job_id, per_model_durations={m: per_model for m in use_models})

            PROGRESS[job_id]["chunks_done"] += 1
            _bump_progress(job_id, step=f"Chunk {idx} complete")
            _log_console_progress(job_id)

        # Parse & aggregate per model (final pass)
        by_model_report: Dict[str, FinalReport] = {}
        errors: Dict[str, str] = {}
        _bump_progress(job_id, step="Parsing model outputs")

        for m in use_models:
            try:
                page_list: List[PageExtraction] = []
                for raw in per_model_raw[m]:
                    try:
                        page_list.extend(parse_llm_json(raw))
                    except Exception as e:
                        errors[m] = (errors.get(m, "") + f" {type(e).__name__}: {e}").strip()
                if not page_list:
                    raise ValueError("No parsable JSON from model.")
                by_model_report[m] = aggregate(page_list)
            except Exception as e:
                errors[m] = f"{type(e).__name__}: {e}"

        comparison = compare_models(by_model_report)
        _bump_progress(job_id, step="Comparison complete")
        _finish_progress(job_id)
        _log_console_progress(job_id)  # final 100% line

        payload = MultiModelReport(
            by_model=by_model_report,
            errors=errors,
            comparison=comparison,
            meta={
                "job_id": job_id,
                "chunks_total": PROGRESS[job_id]["chunks_total"],
                "chunks_done": PROGRESS[job_id]["chunks_done"],
                "per_model_ms": PROGRESS[job_id]["per_model_ms"],
                "started_at": PROGRESS[job_id]["started_at"],
                "finished_at": PROGRESS[job_id].get("finished_at"),
                "history": PROGRESS[job_id]["history"][-12:],  # last few steps
            },
        )

        resp = JSONResponse(payload.dict())
        resp.headers["X-Job-Id"] = job_id
        return resp

    except ValueError as e:
        logger.exception("ValueError in /analyze-multi")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unhandled error in /analyze-multi")
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
