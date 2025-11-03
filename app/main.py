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
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from starlette.staticfiles import StaticFiles
from openai import OpenAI
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

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
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/sqlite/manuals.db")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
# Legacy SQLite path for backward compatibility
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", "./data/sqlite/manuals.db")).resolve()
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "./data/index/faiss.index")).resolve()
FAISS_META_PATH = Path(os.getenv("FAISS_META_PATH", "./data/index/meta.json")).resolve()
ALIAS_MAP_PATH = Path(os.getenv("ALIAS_MAP_PATH", "./data/alias_map.json")).resolve()

# Database engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=5,              # Maximum number of persistent connections
    max_overflow=10,          # Maximum overflow connections beyond pool_size
    pool_pre_ping=True,       # Verify connections before use
    pool_recycle=3600         # Recycle connections after 1 hour
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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

def _extract_product_from_query(query: str, manuals: List[Dict[str, Any]]) -> Optional[Tuple[str, float]]:
    """
    Extract product name from a technical query using keyword matching.
    Returns (extracted_product_name, confidence) or None.

    This function looks for product-specific keywords in the query and
    returns a product name that can be used for fuzzy matching.
    """
    query_lower = _normalize(query)
    query_words = set(query_lower.split())

    # Score each manual based on keyword matches
    best_match = None
    best_score = 0.0

    logger.debug(f"Product extraction - query_lower: '{query_lower}', query_words: {query_words}")

    for manual in manuals:
        score = 0.0
        keywords = manual.get("keywords", [])

        # Count keyword matches - check both substring and word-level matches
        keyword_matches = 0
        for kw in keywords:
            kw_norm = _normalize(kw)
            # Check if keyword appears as substring OR as individual words
            if kw_norm in query_lower or kw_norm in query_words:
                keyword_matches += 1
                logger.debug(f"  Manual {manual.get('manual_id')}: matched keyword '{kw}' (normalized: '{kw_norm}')")

        if keyword_matches > 0:
            # Weight by number of matches and keyword specificity
            # Give higher score for more matches
            score = keyword_matches / max(len(keywords), 1)
            logger.debug(f"  Manual {manual.get('manual_id')}: {keyword_matches} keyword matches, initial score: {score:.3f}")

            # Bonus for direct alias mentions (even partial)
            for alias in manual.get("aliases", []):
                alias_norm = _normalize(alias)
                # Check both substring and word overlap
                alias_words = set(alias_norm.split())
                word_overlap = len(query_words & alias_words)
                if alias_norm in query_lower or word_overlap > 0:
                    score += 0.5  # Strong boost for alias match
                    logger.debug(f"  Manual {manual.get('manual_id')}: alias match bonus ('{alias}'), score now: {score:.3f}")
                    break

            # Bonus for product-specific terms (Pixel, LG, etc.)
            manual_id_norm = _normalize(manual.get("manual_id", ""))
            if any(word in query_words for word in manual_id_norm.split()):
                score += 0.3
                logger.debug(f"  Manual {manual.get('manual_id')}: manual_id word match bonus, score now: {score:.3f}")

            if score > best_score:
                best_score = score
                # Return the first alias or title as the extracted product name
                best_match = manual.get("aliases", [manual.get("title", "")])[0] if manual.get("aliases") else manual.get("title", "")
                logger.debug(f"  New best match: {best_match} with score {best_score:.3f}")

    # Only return if we have reasonable confidence (at least 1 keyword match)
    # Lower threshold to 0.15 to catch technical queries with fewer keyword matches
    if best_score > 0.15:
        # Normalize confidence to 0-1 range, capped at 0.9
        confidence = min(0.9, best_score)
        logger.info(f"Extracted product '{best_match}' from query with confidence {confidence:.2f} (score: {best_score:.3f})")
        return (best_match, confidence)

    logger.info(f"Could not extract product from query: '{query}' (best_score: {best_score:.3f}, threshold: 0.15)")
    return None

# --- Routes ---

@app.get("/")
async def root():
    logger.info("Received GET request for root path.")
    """
    Serve the frontend index. If not present yet, display a helpful placeholder.
    """
    if FRONTEND_INDEX.exists():
        logger.info(f"FRONTEND_INDEX exists: {FRONTEND_INDEX}. Serving FileResponse.")
        return FileResponse(str(FRONTEND_INDEX), media_type="text/html")
    # Placeholder HTML if frontend not created yet
    logger.info("FRONTEND_INDEX does not exist. Serving placeholder HTML.")
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
    Speech-to-text endpoint using OpenAI Whisper API.
    - Accepts multipart file under "file" (audio/webm, audio/wav, audio/mp3, audio/m4a)
    - Transcribes using OpenAI Whisper API
    - Returns: { "transcript": "...", "duration_ms": 123 }
    """
    t0 = time.time()

    # Validate API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY not configured"
        )

    # Validate file
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No audio file provided"
        )

    # Validate MIME type
    allowed_mimes = os.getenv("ALLOWED_AUDIO_MIME", "audio/webm,audio/wav,audio/mp3,audio/mpeg,audio/m4a,audio/x-m4a").split(",")
    content_type = file.content_type or ""

    logger.info(f"Received audio file: {file.filename}, content_type: {content_type}, size: {file.size if hasattr(file, 'size') else 'unknown'}")

    # Read file content
    try:
        audio_bytes = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read audio file"
        )

    # Validate file size (OpenAI limit is 25MB)
    max_size_mb = int(os.getenv("MAX_UPLOAD_MB", "25"))
    if len(audio_bytes) > max_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file too large. Max {max_size_mb}MB allowed."
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Audio file is empty"
        )

    # Determine file extension from filename or content type
    filename = file.filename or "audio.webm"
    if filename.endswith(".webm"):
        ext = "webm"
    elif filename.endswith(".wav"):
        ext = "wav"
    elif filename.endswith(".mp3"):
        ext = "mp3"
    elif filename.endswith(".m4a"):
        ext = "m4a"
    else:
        # Infer from content type
        if "webm" in content_type:
            ext = "webm"
        elif "wav" in content_type:
            ext = "wav"
        elif "mp3" in content_type or "mpeg" in content_type:
            ext = "mp3"
        elif "m4a" in content_type:
            ext = "m4a"
        else:
            ext = "webm"  # default

    # Call OpenAI Whisper API
    try:
        client = OpenAI(api_key=api_key)

        # Create a temporary file-like object for OpenAI client
        # The client expects a file-like object with a name attribute
        import io
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = f"audio.{ext}"

        t_api_start = time.time()
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        t_api_end = time.time()

        transcript = response.strip() if isinstance(response, str) else ""

        t_total = time.time() - t0
        api_duration_ms = int((t_api_end - t_api_start) * 1000)
        total_duration_ms = int(t_total * 1000)

        if ENABLE_TIMING_LOGS:
            logger.info(f"STT completed: {total_duration_ms}ms total (API: {api_duration_ms}ms)")

        logger.info(f"Transcript: {transcript}")

        return JSONResponse({
            "transcript": transcript,
            "duration_ms": total_duration_ms,
            "api_duration_ms": api_duration_ms
        }, status_code=200)

    except Exception as e:
        t_total = time.time() - t0
        logger.exception("OpenAI Whisper API error: %s", e)

        # Provide specific error messages
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "quota" in error_msg.lower():
            detail = f"OpenAI quota exceeded. Please check billing at platform.openai.com. Error: {error_msg}"
        elif "rate_limit" in error_msg.lower():
            detail = f"Rate limit exceeded. Please try again later. Error: {error_msg}"
        elif "invalid_api_key" in error_msg.lower() or "authentication" in error_msg.lower():
            detail = f"Invalid OpenAI API key. Error: {error_msg}"
        elif "timeout" in error_msg.lower():
            detail = f"Request timeout. Please try with shorter audio. Error: {error_msg}"
        else:
            detail = f"Speech-to-text failed: {error_msg}"

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )


@app.get("/api/manual/resolve")
async def manual_resolve(q: Optional[str] = None) -> JSONResponse:
    """
    Resolve the best matching manual given a natural language query containing product name and/or model.
    Returns either a resolved manual or candidate list when ambiguous.

    Enhanced with product extraction: attempts to extract product mentions from technical queries
    before falling back to direct fuzzy matching.
    """
    logger.info(f"manual_resolve called with query: '{q}'")
    if not q or not q.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query 'q' is required")
    manuals = _load_manuals_config()
    logger.info(f"Loaded {len(manuals)} manuals from alias map")

    # Step 1: Try to extract product name from query using keyword matching
    extracted = _extract_product_from_query(q, manuals)
    if extracted:
        product_name, confidence = extracted
        logger.info(f"Product extraction found: '{product_name}' (confidence: {confidence:.2f})")
        # Lower threshold to 0.15 to use extraction for technical queries
        # Even low confidence extraction is better than matching technical terms against product names
        if confidence >= 0.15:
            target = _normalize(product_name)
            logger.info(f"Using extracted product for matching: '{target}' (confidence: {confidence:.2f})")
        else:
            # Very low confidence, fall back to original query
            target = _normalize(q)
            logger.info(f"Very low confidence extraction ({confidence:.2f}), using original query: '{target}'")
    else:
        # No product extraction, use original query
        target = _normalize(q)
        logger.info(f"No product extracted, using original query: '{target}'")

    alias_lookup = _build_alias_lookup(manuals)
    logger.info(f"Built alias lookup with {len(alias_lookup)} entries")
    choices = list(alias_lookup.keys())

    # Threshold is 0..1 in env; convert to 0..100
    try:
        threshold = float(os.getenv("ALIAS_MATCH_THRESHOLD", "0.85"))
    except Exception:
        threshold = 0.85
    min_score = int(max(0.0, min(1.0, threshold)) * 100)
    logger.info(f"Using match threshold: {threshold} (min_score: {min_score})")

    results = rf_process.extract(target, choices, scorer=rf_fuzz.WRatio, limit=5)
    logger.info(f"Fuzzy matching results: {results}")
    # results: List[Tuple[str, int, int]] -> (alias, score, index)
    if not results:
        logger.info("No fuzzy matching results found")
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

    logger.info(f"Final candidates: {candidates}")
    top = candidates[0] if candidates else None
    if top and int(top["score"] * 100) >= min_score:
        logger.info(f"Resolved to top match: {top}")
        return JSONResponse(top, status_code=200)
    else:
        logger.info("Returning ambiguous candidates")
        return JSONResponse({"candidates": candidates, "message": "ambiguous"}, status_code=200)


@app.get("/api/manual/{manual_id}/pdf")
async def manual_pdf(manual_id: str):
    """
    Generate a presigned S3 URL for the manual PDF or serve from local filesystem.
    Returns a redirect to the S3 URL for viewing (Heroku) or FileResponse (local dev).
    """
    # Check if S3 is enabled
    use_s3 = os.getenv("USE_S3", "false").lower() == "true"
    
    if not use_s3:
        # Fallback to local file serving (for local development)
        logger.info(f"manual_pdf called for manual_id: {manual_id} (local mode)")
        manuals = _load_manuals_config()
        match = next((m for m in manuals if m.get("manual_id") == manual_id), None)
        
        if not match:
            logger.warning(f"Manual '{manual_id}' not found in alias map")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manual not found")
        
        filename = match.get("filename") or f"{manual_id}.pdf"
        pdf_path = (MANUALS_DIR / filename).resolve()
        logger.info(f"Resolved PDF path: {pdf_path}")
        
        # Security: ensure path is within MANUALS_DIR
        try:
            pdf_path.relative_to(MANUALS_DIR)
        except Exception:
            logger.error(f"PDF path {pdf_path} is not within MANUALS_DIR {MANUALS_DIR}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid manual path")
        
        if not pdf_path.exists():
            logger.warning(f"PDF file not found at {pdf_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Manual PDF not found. PDFs are not available in this deployment."
            )
        
        logger.info(f"Serving PDF from local filesystem: {pdf_path}")
        response = FileResponse(str(pdf_path), media_type="application/pdf")
        response.headers["Content-Disposition"] = "inline"
        return response
    
    # S3 mode: generate presigned URL
    logger.info(f"manual_pdf called for manual_id: {manual_id} (S3 mode)")
    
    import boto3
    from botocore.exceptions import ClientError
    
    # Get S3 configuration
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    bucket_name = os.getenv("S3_BUCKET_NAME")
    
    if not all([aws_access_key, aws_secret_key, bucket_name]):
        logger.error("S3 credentials not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="S3 credentials not configured. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and S3_BUCKET_NAME."
        )
    
    # Resolve filename from alias map
    manuals = _load_manuals_config()
    match = next((m for m in manuals if m.get("manual_id") == manual_id), None)
    
    if not match:
        logger.warning(f"Manual '{manual_id}' not found in alias map")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Manual '{manual_id}' not found")
    
    filename = match.get("filename")
    if not filename:
        logger.error(f"No filename configured for manual '{manual_id}'")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail=f"Filename not configured for manual '{manual_id}'"
        )
    
    # S3 key path (PDFs should be in manuals/ subfolder)
    s3_key = f"manuals/{filename}"
    logger.info(f"Generating presigned URL for s3://{bucket_name}/{s3_key}")
    
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # Generate presigned URL (valid for 1 hour)
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': bucket_name,
                'Key': s3_key,
                'ResponseContentDisposition': 'inline',
                'ResponseContentType': 'application/pdf'
            },
            ExpiresIn=3600  # 1 hour
        )
        
        logger.info(f"✓ Generated presigned URL for manual '{manual_id}': {s3_key}")
        
        # Return 307 Temporary Redirect to presigned URL
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=presigned_url, status_code=307)
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        logger.exception(f"S3 error for {s3_key}: {error_code} - {e}")
        
        if error_code == 'NoSuchKey':
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found in S3: {s3_key}"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to access PDF in S3: {error_code}"
            )
    except Exception as e:
        logger.exception(f"Unexpected error generating presigned URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate PDF URL"
        )



@app.post("/api/manual/{manual_id}/search")
async def manual_search(manual_id: str, request: Request) -> JSONResponse:
    """
    Hybrid retrieval:
      1) Shortlist pages with PostgreSQL full-text search within the specified manual
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

    session = SessionLocal()
    try:
        # Check database type
        from urllib.parse import urlparse
        db_url = urlparse(DATABASE_URL)

        # Step 1: Database-specific full-text search shortlist
        candidates: List[Tuple[int, str]] = []  # (page_number, text)

        # Clean question for search - use minimal stop words to preserve technical terms
        # Only remove articles that don't add semantic value
        minimal_stop_words = {'the', 'a', 'an'}
        cleaned_question = question.replace('?', ' ').replace('.', ' ').replace(',', ' ')
        words = [w.strip() for w in cleaned_question.lower().split() if w.strip() and w.lower() not in minimal_stop_words]
        search_query = ' '.join(words) if words else question.replace('"', '').replace('.', ' ').replace('?', ' ')
        logger.info(f"Cleaned search query: '{search_query}' (from: '{question}')")

        if db_url.scheme == 'postgresql':
            logger.info(f"Using PostgreSQL search for manual '{manual_id}' with query: '{search_query}'")
            # Use PostgreSQL full-text search with ts_rank
            result = session.execute(
                text("""
                SELECT page_number, text_content
                FROM pages
                WHERE manual_id = :manual_id AND text_vector @@ plainto_tsquery('english', :query)
                ORDER BY ts_rank(text_vector, plainto_tsquery('english', :query)) DESC
                LIMIT :limit
                """),
                {
                    "manual_id": manual_id,
                    "query": search_query,
                    "limit": fts_top_k
                }
            )
            rows = result.fetchall()
            logger.info(f"PostgreSQL FTS returned {len(rows)} candidates.")
            for row in rows:
                candidates.append((int(row[0]), row[1] or ""))

            # Fallback to trigram similarity if no FTS results
            if not candidates:
                logger.info("FTS returned no results. Falling back to trigram similarity.")
                result = session.execute(
                    text("""
                    SELECT page_number, text_content
                    FROM pages
                    WHERE manual_id = :manual_id AND text_content % :query
                    ORDER BY similarity(text_content, :query) DESC
                    LIMIT :limit
                    """),
                    {
                        "manual_id": manual_id,
                        "query": search_query,
                        "limit": fts_top_k
                    }
                )
                rows = result.fetchall()
                logger.info(f"PostgreSQL trigram search returned {len(rows)} candidates.")
                for row in rows:
                    candidates.append((int(row[0]), row[1] or ""))

            # Final fallback to ILIKE
            if not candidates:
                logger.info("Trigram search returned no results. Falling back to ILIKE.")
                result = session.execute(
                    text("""
                    SELECT page_number, text_content
                    FROM pages
                    WHERE manual_id = :manual_id AND text_content ILIKE :query
                    LIMIT :limit
                    """),
                    {
                        "manual_id": manual_id,
                        "query": f"%{search_query}%",
                        "limit": fts_top_k
                    }
                )
                rows = result.fetchall()
                logger.info(f"PostgreSQL ILIKE search returned {len(rows)} candidates.")
                for row in rows:
                    candidates.append((int(row[0]), row[1] or ""))
        else:
            logger.info(f"Using SQLite LIKE search for manual '{manual_id}' with query: '{search_query}'")
            # SQLite fallback to LIKE
            result = session.execute(
                text("""
                SELECT page_number, text_content
                FROM pages
                WHERE manual_id = :manual_id AND text_content LIKE :query
                LIMIT :limit
                """),
                {
                    "manual_id": manual_id,
                    "query": f"%{search_query}%",
                    "limit": fts_top_k
                }
            )
            rows = result.fetchall()
            logger.info(f"SQLite LIKE search returned {len(rows)} candidates.")
            for row in rows:
                candidates.append((int(row[0]), row[1] or ""))

        if not candidates:
            return JSONResponse({"message": "no_results"}, status_code=200)

        # Filter out candidates with empty or very short text before embedding
        # Track indices to map back to original candidates
        valid_indices = []
        valid_candidates = []
        for i, (page_num, page_text) in enumerate(candidates):
            if page_text and len(page_text.strip()) > 10:
                valid_indices.append(i)
                valid_candidates.append((page_num, page_text))
        
        if not valid_candidates:
            logger.warning("All candidates have empty/insufficient text content")
            return JSONResponse({"message": "no_results"}, status_code=200)

        # Step 2: Embedding re-rank (same as before)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="OPENAI_API_KEY not configured")
        emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        client = OpenAI(api_key=api_key)
        texts = [question] + [c[1] for c in valid_candidates]
        try:
            resp = client.embeddings.create(model=emb_model, input=texts)
            vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        except Exception as e:
            logger.exception("Embeddings failed: %s", e)
            # Fallback to FTS order (use first valid candidate)
            best_page, best_text = valid_candidates[0]
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
        best_page, best_text = valid_candidates[best_idx]
        best_score = float(scores[best_idx])


        logger.info(f"Re-ranked candidates. Best page: {best_page}, Score: {best_score:.4f}")

        snippet = (best_text[:240] + "…") if len(best_text) > 240 else best_text

        return JSONResponse({"page": best_page, "score": round(best_score, 4), "snippet": snippet}, status_code=200)

    finally:
        session.close()


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
