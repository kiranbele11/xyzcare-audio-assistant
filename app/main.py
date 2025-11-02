"""
XYZCare Voice Manual Demo - FastAPI scaffold

Serves:
- GET /           : frontend index (placeholder UI)
- GET /healthz    : liveness/readiness
- Static assets   : /static/* (served from ./frontend)

Stubs (to be implemented next):
- POST /api/stt
- GET  /api/manual/resolve
- GET  /api/manual/{manual_id}/pdf
- POST /api/manual/{manual_id}/search
"""
from __future__ import annotations

import os
import logging
import tempfile
import time
import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles
from openai import OpenAI
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

# For local STT
# import torch
# import torchaudio
# from transformers import pipeline
# import librosa
# import subprocess

# --- Env / Config ---
load_dotenv()

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
MANUALS_DIR = Path(os.getenv("MANUALS_DIR", "./data/manuals")).resolve()
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", "./data/sqlite/manuals.db")).resolve()
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "./data/index/faiss.index")).resolve()
FAISS_META_PATH = Path(os.getenv("FAISS_META_PATH", "./data/index/meta.json")).resolve()
ALIAS_MAP_PATH = Path(os.getenv("ALIAS_MAP_PATH", "./data/alias_map.json")).resolve()

FRONTEND_DIR = (Path(__file__).resolve().parent.parent / "frontend").resolve()
FRONTEND_INDEX = FRONTEND_DIR / "index.html"

ENABLE_TIMING_LOGS = os.getenv("ENABLE_TIMING_LOGS", "true").lower() == "true"


# --- Local STT ---
# stt_pipeline = None

# def get_stt_pipeline():
#     global stt_pipeline
#     if stt_pipeline is None:
#         try:
#             logger.info("Initializing local STT pipeline...")
#             device = "cuda:0" if torch.cuda.is_available() else "cpu"
#             stt_pipeline = pipeline(
#                 "automatic-speech-recognition",
#                 model="openai/whisper-base.en",
#                 device=device
#             )
#             logger.info("Local STT pipeline initialized.")
#         except Exception as e:
#             logger.exception("Failed to initialize local STT pipeline: %s", e)
#     return stt_pipeline


# --- App ---
logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper(), logging.INFO))
logger = logging.getLogger("xyzcare")

app = FastAPI(
    title="XYZCare Voice Manual Demo",
    version="0.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",  # can be disabled later for a pure demo
)

# CORS - permissive for local demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local demo; narrow later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ensure_dirs() -> None:
    """Create minimal data directory structure needed for the demo."""
    (DATA_DIR).mkdir(parents=True, exist_ok=True)
    (MANUALS_DIR).mkdir(parents=True, exist_ok=True)
    (SQLITE_PATH.parent).mkdir(parents=True, exist_ok=True)
    (FAISS_INDEX_PATH.parent).mkdir(parents=True, exist_ok=True)
    (FAISS_META_PATH.parent).mkdir(parents=True, exist_ok=True)


ensure_dirs()

# Serve entire frontend directory as static files.
# This allows referencing /static/app.js, /static/styles.css, etc.
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="static")


# --- Manual resolution helpers ---

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def _load_manuals_config() -> List[Dict[str, Any]]:
    """
    Load manuals config from ALIAS_MAP_PATH. Expected schema:
    {
      "manuals": [
        { "manual_id": "wp300", "title": "XYZ Widget Pro 300", "filename": "xyz_widget_pro_300.pdf", "aliases": ["WP-300", "Widget Pro 300", "WP300"] }
      ]
    }
    """
    if not ALIAS_MAP_PATH.exists():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Alias map not found")
    try:
        with open(ALIAS_MAP_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        manuals = data.get("manuals", [])
        # Basic validation
        for m in manuals:
            if "manual_id" not in m or "filename" not in m:
                raise ValueError("Invalid manual entry; requires manual_id and filename")
        return manuals
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to load alias map: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to load alias map")

def _build_alias_lookup(manuals: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup of normalized alias -> manual entry (includes explicit aliases, title, and manual_id).
    """
    lookup: Dict[str, Dict[str, Any]] = {}
    for m in manuals:
        aliases = m.get("aliases", [])
        for a in aliases:
            lookup[_normalize(a)] = m
        # Also include title and manual_id as implicit aliases
        if "title" in m:
            lookup[_normalize(m["title"])] = m
        lookup[_normalize(m["manual_id"])] = m
    return lookup

# --- Routes ---

@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """
    Serve the frontend index. If not present yet, display a helpful placeholder.
    """
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX))
    # Placeholder HTML if frontend not created yet
    html = """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>XYZCare Demo</title>
        <style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:2rem} .muted{color:#666}</style>
      </head>
      <body>
        <h1>XYZCare Voice Manual Demo</h1>
        <p class="muted">Frontend not found. Create <code>frontend/index.html</code> or visit <code>/static/</code> for assets.</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)


@app.get("/healthz", response_class=JSONResponse)
async def healthz() -> JSONResponse:
    """
    Liveness/readiness probe. Checks presence of key paths.
    """
    checks: Dict[str, Any] = {
        "frontend_index_exists": FRONTEND_INDEX.exists(),
        "data_dir_exists": DATA_DIR.exists(),
        "manuals_dir_exists": MANUALS_DIR.exists(),
        "sqlite_parent_exists": SQLITE_PATH.parent.exists(),
        "faiss_index_parent_exists": FAISS_INDEX_PATH.parent.exists(),
        "faiss_meta_parent_exists": FAISS_META_PATH.parent.exists(),
    }
    status_overall = "ok" if all(checks.values()) else "degraded"
    payload = {
        "status": status_overall,
        "version": app.version,
        "checks": checks,
    }
    return JSONResponse(payload, status_code=200 if status_overall == "ok" else 503)


# --- API stubs (to be implemented) ---

@app.post("/api/stt")
async def stt(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """
    Speech-to-text endpoint using a local model.
    - Accepts multipart file under "file" (audio/webm or audio/wav)
    - Transcribes using a local Whisper model
    """
    # Local STT is disabled due to large slug size on Heroku.
    # To enable, uncomment the related code and add torch, torchaudio, librosa to requirements.txt.
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Local STT is not available")


@app.get("/api/manual/resolve")
async def manual_resolve(q: Optional[str] = None) -> JSONResponse:
    """
    Resolve the best matching manual given a natural language query containing product name and/or model.
    Returns either a resolved manual or candidate list when ambiguous.
    """
    if not q or not q.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query 'q' is required")
    manuals = _load_manuals_config()
    alias_lookup = _build_alias_lookup(manuals)
    choices = list(alias_lookup.keys())
    target = _normalize(q)

    # Threshold is 0..1 in env; convert to 0..100
    try:
        threshold = float(os.getenv("ALIAS_MATCH_THRESHOLD", "0.85"))
    except Exception:
        threshold = 0.85
    min_score = int(max(0.0, min(1.0, threshold)) * 100)

    results = rf_process.extract(target, choices, scorer=rf_fuzz.WRatio, limit=5)
    # results: List[Tuple[str, int, int]] -> (alias, score, index)
    if not results:
        return JSONResponse({"candidates": [], "message": "no_match"}, status_code=200)

    # Map to unique manuals preserving order
    seen = set()
    candidates = []
    for alias, score, _ in results:
        m = alias_lookup.get(alias)
        if not m:
            continue
        mid = m["manual_id"]
        if mid in seen:
            continue
        seen.add(mid)
        candidates.append({
            "manual_id": mid,
            "title": m.get("title"),
            "filename": m.get("filename"),
            "score": round(score / 100.0, 3),
        })
        if len(candidates) >= 3:
            break

    top = candidates[0] if candidates else None
    if top and int(top["score"] * 100) >= min_score:
        return JSONResponse(top, status_code=200)
    else:
        return JSONResponse({"candidates": candidates, "message": "ambiguous"}, status_code=200)


@app.get("/api/manual/{manual_id}/pdf")
async def manual_pdf(manual_id: str):
    """
    Stream the resolved manual PDF for viewing. Only serves files under MANUALS_DIR.
    """
    manuals = _load_manuals_config()
    match = next((m for m in manuals if m.get("manual_id") == manual_id), None)
    filename = None
    if match:
        filename = match.get("filename")
    if not filename:
        # Fallback to {manual_id}.pdf
        filename = f"{manual_id}.pdf"
    pdf_path = (MANUALS_DIR / filename).resolve()

    # Security: ensure path is within MANUALS_DIR
    try:
        pdf_path.relative_to(MANUALS_DIR)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid manual path")

    if not pdf_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manual PDF not found")

    logger.info(f"Serving PDF: {pdf_path} for manual_id: {manual_id}")

    # Explicitly set Content-Disposition to inline to force display in browser
    response = FileResponse(str(pdf_path), media_type="application/pdf")
    response.headers["Content-Disposition"] = "inline"
    return response


@app.post("/api/manual/{manual_id}/search")
async def manual_search(manual_id: str, request: Request) -> JSONResponse:
    """
    Hybrid retrieval:
      1) Shortlist pages with SQLite FTS5 BM25 within the specified manual
      2) Re-rank shortlisted pages using OpenAI embeddings cosine similarity to the question
    Returns: { page: int, score: float, snippet: str }
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    question = (payload or {}).get("question", "")
    if not isinstance(question, str) or not question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'question'")

    # Tunables
    try:
        fts_top_k = int(os.getenv("FTS_TOP_K", "8"))
    except Exception:
        fts_top_k = 8

    # Step 1: FTS shortlist
    candidates: List[Tuple[int, str]] = []  # (page_number, text)
    conn = None
    # Testing nvim
    try:
        conn = sqlite3.connect(str(SQLITE_PATH))
        cur = conn.cursor()
        
        # Escape special characters for FTS5 query and remove stop words
        # Remove common question words and punctuation
        stop_words = {'what', 'how', 'when', 'where', 'why', 'who', 'is', 'are', 'the', 'a', 'an', 'to', 'do', 'does'}
        cleaned_question = question.replace('?', ' ').replace('.', ' ').replace('-', ' ').replace(',', ' ')
        words = [w.strip() for w in cleaned_question.lower().split() if w.strip() and w.lower() not in stop_words]
        
        # Use the cleaned words for FTS search
        escaped_question = ' '.join(words) if words else question.replace('"', '""').replace('.', ' ').replace('-', ' ').replace('?', ' ')
        
        try:
            cur.execute(
                """
                SELECT page_number, text
                FROM pages_fts
                WHERE manual_id = ? AND pages_fts MATCH ?
                ORDER BY bm25(pages_fts) ASC
                LIMIT ?
                """,
                (manual_id, escaped_question, fts_top_k),
            )
        except sqlite3.OperationalError:
            cur.execute(
                """
                SELECT page_number, text
                FROM pages_fts
                WHERE manual_id = ? AND pages_fts MATCH ?
                LIMIT ?
                """,
                (manual_id, escaped_question, fts_top_k),
            )
        rows = cur.fetchall() or []
        for row in rows:
            candidates.append((int(row[0]), row[1] or ""))
    except Exception as e:
        logger.exception("FTS shortlist failed: %s", e)
        # Fallback LIKE
        try:
            if conn is None:
                conn = sqlite3.connect(str(SQLITE_PATH))
            cur = conn.cursor()
            # Escape LIKE wildcards and use proper escaping
            escaped_question = question.replace('%', '\\%').replace('_', '\\_')
            cur.execute(
                """
                SELECT page_number, text
                FROM pages
                WHERE manual_id = ? AND text LIKE ?
                LIMIT ?
                """,
                (manual_id, f"%{escaped_question}%", fts_top_k),
            )
            rows = cur.fetchall() or []
            for row in rows:
                candidates.append((int(row[0]), row[1] or ""))
        except Exception as e2:
            logger.exception("Fallback LIKE failed: %s", e2)
            candidates = []
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass

    if not candidates:
        return JSONResponse({"message": "no_results"}, status_code=200)

    # Step 2: Embedding re-rank
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OPENAI_API_KEY not configured")
    emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    client = OpenAI(api_key=api_key)
    texts = [question] + [c[1] for c in candidates]
    try:
        resp = client.embeddings.create(model=emb_model, input=texts)
        vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    except Exception as e:
        logger.exception("Embeddings failed: %s", e)
        # Fallback to FTS order
        best_page, best_text = candidates[0]
        snippet = (best_text[:240] + "…") if len(best_text) > 240 else best_text
        return JSONResponse({"page": best_page, "score": None, "snippet": snippet}, status_code=200)

    # Normalize vectors for cosine sim
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v) + 1e-12
        return v / n

    qv = _norm(vecs[0])
    page_vecs = [_norm(v) for v in vecs[1:]]
    scores = [float(np.dot(pv, qv)) for pv in page_vecs]

    best_idx = int(np.argmax(scores))
    best_page, best_text = candidates[best_idx]
    best_score = float(scores[best_idx])

    snippet = (best_text[:240] + "…") if len(best_text) > 240 else best_text

    return JSONResponse({"page": best_page, "score": round(best_score, 4), "snippet": snippet}, status_code=200)


if __name__ == "__main__":
    # Local dev convenience: python app/main.py
    try:
        import uvicorn  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.error("Uvicorn is required to run directly: %s", exc)
        raise
    uvicorn.run(
        "app.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,  # Disable auto-reload to prevent constant restarts
        log_level=LOG_LEVEL.lower()
    )
