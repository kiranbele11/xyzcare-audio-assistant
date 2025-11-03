"""
Ingest local PDF manuals into SQLite + FTS5 and build a FAISS semantic index.

- Reads alias map to discover manuals and filenames
- Extracts per-page text using PyMuPDF
- Populates SQLite tables:
    manuals(id TEXT PRIMARY KEY, title TEXT, filename TEXT, num_pages INT)
    pages(id INTEGER PK, manual_id TEXT, page_number INT, text TEXT, heading TEXT)
    pages_fts(text, manual_id UNINDEXED, page_number UNINDEXED) USING fts5
- Generates OpenAI embeddings (text-embedding-3-small) for each page and saves:
    data/index/faiss.index (FAISS IndexFlatIP with L2-normalized vectors)
    data/index/meta.json   (list of {"manual_id": str, "page_number": int})
Usage:
  python -m app.ingestion.ingest_manuals --rebuild
  or
  python app/ingestion/ingest_manuals.py --rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import numpy as np
import fitz  # PyMuPDF
import faiss  # type: ignore
from dotenv import load_dotenv
from openai import OpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# --- Load environment ---
load_dotenv()

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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Database engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=5,              # Maximum number of persistent connections
    max_overflow=10,          # Maximum overflow connections beyond pool_size
    pool_pre_ping=True,       # Verify connections before use
    pool_recycle=3600         # Recycle connections after 1 hour
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Utilities ---
def ensure_dirs() -> None:
    (DATA_DIR).mkdir(parents=True, exist_ok=True)
    (MANUALS_DIR).mkdir(parents=True, exist_ok=True)
    (SQLITE_PATH.parent).mkdir(parents=True, exist_ok=True)
    (FAISS_INDEX_PATH.parent).mkdir(parents=True, exist_ok=True)
    (FAISS_META_PATH.parent).mkdir(parents=True, exist_ok=True)

def load_alias_map() -> List[Dict]:
    if not ALIAS_MAP_PATH.exists():
        raise FileNotFoundError(f"Alias map not found at {ALIAS_MAP_PATH}")
    with open(ALIAS_MAP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    manuals = data.get("manuals", [])
    if not manuals:
        raise ValueError("No manuals defined in alias map")
    return manuals

def get_db_session():
    return SessionLocal()

def init_schema(session, rebuild: bool = False) -> None:
    # Check if we're using PostgreSQL or SQLite
    from urllib.parse import urlparse
    db_url = urlparse(DATABASE_URL)

    if rebuild:
        # Drop tables in correct order due to foreign key constraints
        # Use database-agnostic approach
        try:
            session.execute(text("DROP TABLE IF EXISTS pages CASCADE;"))
        except Exception as e:
            # CASCADE not supported in SQLite, fallback to simple DROP
            error_msg = str(e).lower()
            if 'cascade' in error_msg or 'syntax' in error_msg:
                session.execute(text("DROP TABLE IF EXISTS pages;"))
            else:
                raise
        try:
            session.execute(text("DROP TABLE IF EXISTS manuals CASCADE;"))
        except Exception as e:
            # CASCADE not supported in SQLite, fallback to simple DROP
            error_msg = str(e).lower()
            if 'cascade' in error_msg or 'syntax' in error_msg:
                session.execute(text("DROP TABLE IF EXISTS manuals;"))
            else:
                raise

    if db_url.scheme == 'postgresql':
        # Use PostgreSQL schema
        schema_path = Path(__file__).parent.parent.parent / "migrations" / "postgres_schema.sql"
        with open(schema_path, "r") as f:
            schema_sql = f.read()
        session.execute(text(schema_sql))
    else:
        # Use SQLite-compatible schema
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS manuals(
                manual_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                filename TEXT NOT NULL,
                num_pages INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS pages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manual_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                text_content TEXT,
                headings TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(manual_id, page_number),
                FOREIGN KEY (manual_id) REFERENCES manuals(manual_id) ON DELETE CASCADE
            );
        """))
        session.execute(text("CREATE INDEX IF NOT EXISTS idx_pages_manual_id ON pages(manual_id);"))

    session.commit()

def extract_pdf_text(pdf_path: Path) -> List[str]:
    texts: List[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for p in doc:
            # Prefer text layer; fall back to simple extract
            t = p.get_text("text") or ""
            # Remove NUL characters that cause PostgreSQL errors
            t = t.replace('\x00', '')
            texts.append(t.strip())
    return texts

def upsert_manual_and_pages(session, manual_id: str, title: str, filename: str, pages_text: List[str]) -> None:
    from urllib.parse import urlparse
    db_url = urlparse(DATABASE_URL)

    if db_url.scheme == 'postgresql':
        # PostgreSQL-specific upsert
        session.execute(
            text("""
            INSERT INTO manuals (manual_id, title, filename, num_pages)
            VALUES (:manual_id, :title, :filename, :num_pages)
            ON CONFLICT (manual_id) DO UPDATE SET
                title = EXCLUDED.title,
                filename = EXCLUDED.filename,
                num_pages = EXCLUDED.num_pages,
                updated_at = CURRENT_TIMESTAMP
            """),
            {
                "manual_id": manual_id,
                "title": title,
                "filename": filename,
                "num_pages": len(pages_text)
            }
        )

        # Clear previous pages for this manual
        session.execute(text("DELETE FROM pages WHERE manual_id = :manual_id"), {"manual_id": manual_id})

        # Insert pages (text_vector will be auto-updated by trigger)
        for i, page_text in enumerate(pages_text, start=1):
            session.execute(
                text("""
                INSERT INTO pages (manual_id, page_number, text_content, headings)
                VALUES (:manual_id, :page_number, :text_content, :headings)
                """),
                {
                    "manual_id": manual_id,
                    "page_number": i,
                    "text_content": page_text,
                    "headings": None
                }
            )
    else:
        # SQLite-compatible operations
        # Check if manual exists
        result = session.execute(text("SELECT manual_id FROM manuals WHERE manual_id = :manual_id"), {"manual_id": manual_id})
        if result.fetchone():
            session.execute(
                text("UPDATE manuals SET title = :title, filename = :filename, num_pages = :num_pages, updated_at = CURRENT_TIMESTAMP WHERE manual_id = :manual_id"),
                {
                    "manual_id": manual_id,
                    "title": title,
                    "filename": filename,
                    "num_pages": len(pages_text)
                }
            )
        else:
            session.execute(
                text("INSERT INTO manuals (manual_id, title, filename, num_pages) VALUES (:manual_id, :title, :filename, :num_pages)"),
                {
                    "manual_id": manual_id,
                    "title": title,
                    "filename": filename,
                    "num_pages": len(pages_text)
                }
            )

        # Clear previous pages for this manual
        session.execute(text("DELETE FROM pages WHERE manual_id = :manual_id"), {"manual_id": manual_id})

        # Insert pages
        for i, page_text in enumerate(pages_text, start=1):
            session.execute(
                text("INSERT INTO pages (manual_id, page_number, text_content, headings) VALUES (:manual_id, :page_number, :text_content, :headings)"),
                {
                    "manual_id": manual_id,
                    "page_number": i,
                    "text_content": page_text,
                    "headings": None
                }
            )

    session.commit()

def chunk_iter(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def build_faiss_index(session, client: OpenAI) -> None:
    result = session.execute(text("SELECT manual_id, page_number, text_content FROM pages ORDER BY manual_id, page_number;"))
    rows = result.fetchall()
    if not rows:
        raise RuntimeError("No pages found to embed")

    texts: List[str] = [r[2] for r in rows]
    meta: List[Dict] = [{"manual_id": r[0], "page_number": int(r[1])} for r in rows]

    vectors: List[List[float]] = []
    t0 = time.perf_counter()
    # Batch embed for throughput; OpenAI supports batching inputs
    for batch in chunk_iter(texts, 64):
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for item in resp.data:
            vectors.append(item.embedding)
    t1 = time.perf_counter()

    vecs = np.array(vectors, dtype=np.float32)
    vecs = l2_normalize(vecs)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    # Persist
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    print(f"Embeddings: {len(texts)} pages embedded in {(t1 - t0):.2f}s, dim={dim}")
    print(f"Saved index to {FAISS_INDEX_PATH} and meta to {FAISS_META_PATH}")

def ingest(rebuild: bool = False, embed: bool = True) -> None:
    ensure_dirs()
    session = get_db_session()
    try:
        init_schema(session, rebuild=rebuild)

        manuals = load_alias_map()
        ingested = 0
        skipped = 0

        for m in manuals:
            manual_id = m.get("manual_id")
            title = m.get("title", manual_id)
            filename = m.get("filename") or f"{manual_id}.pdf"
            pdf_path = (MANUALS_DIR / filename).resolve()
            if not pdf_path.exists():
                print(f"[skip] {manual_id}: PDF not found at {pdf_path}")
                skipped += 1
                continue
            pages_text = extract_pdf_text(pdf_path)
            if not any(pages_text):
                print(f"[warn] {manual_id}: no text extracted; is this a scanned PDF?")
            upsert_manual_and_pages(session, manual_id, title, filename, pages_text)
            print(f"[ok]   {manual_id}: {len(pages_text)} pages ingested")
            ingested += 1

        if ingested == 0:
            print("No manuals ingested. Aborting embedding.")
            return

        if embed:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY not set; cannot build embeddings")
            client = OpenAI(api_key=api_key)
            build_faiss_index(session, client)

        print("Ingestion complete.")
    finally:
        session.close()

def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Ingest manuals into SQLite FTS and FAISS index.")
    parser.add_argument("--rebuild", action="store_true", help="Drop and recreate tables (destructive).")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding + FAISS index step.")
    args = parser.parse_args(argv)

    try:
        ingest(rebuild=args.rebuild, embed=(not args.no_embed))
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))