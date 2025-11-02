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
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import fitz  # PyMuPDF
import faiss  # type: ignore
from dotenv import load_dotenv
from openai import OpenAI

# --- Load environment ---
load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
MANUALS_DIR = Path(os.getenv("MANUALS_DIR", "./data/manuals")).resolve()
SQLITE_PATH = Path(os.getenv("SQLITE_PATH", "./data/sqlite/manuals.db")).resolve()
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "./data/index/faiss.index")).resolve()
FAISS_META_PATH = Path(os.getenv("FAISS_META_PATH", "./data/index/meta.json")).resolve()
ALIAS_MAP_PATH = Path(os.getenv("ALIAS_MAP_PATH", "./data/alias_map.json")).resolve()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

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

def connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn

def init_schema(conn: sqlite3.Connection, rebuild: bool = False) -> None:
    cur = conn.cursor()
    if rebuild:
        cur.execute("DROP TABLE IF EXISTS pages;")
        cur.execute("DROP TABLE IF EXISTS manuals;")
        cur.execute("DROP TABLE IF EXISTS pages_fts;")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS manuals(
            id TEXT PRIMARY KEY,
            title TEXT,
            filename TEXT,
            num_pages INTEGER
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            manual_id TEXT,
            page_number INTEGER,
            text TEXT,
            heading TEXT
        );
        """
    )
    # FTS virtual table (standalone content)
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts
        USING fts5(
            text,
            manual_id UNINDEXED,
            page_number UNINDEXED,
            tokenize = 'porter'
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_manual ON pages(manual_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_manual_page ON pages(manual_id, page_number);")
    conn.commit()

def extract_pdf_text(pdf_path: Path) -> List[str]:
    texts: List[str] = []
    with fitz.open(str(pdf_path)) as doc:
        for p in doc:
            # Prefer text layer; fall back to simple extract
            t = p.get_text("text") or ""
            texts.append(t.strip())
    return texts

def upsert_manual_and_pages(conn: sqlite3.Connection, manual_id: str, title: str, filename: str, pages_text: List[str]) -> None:
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO manuals(id, title, filename, num_pages) VALUES(?,?,?,?)",
        (manual_id, title, filename, len(pages_text)),
    )
    # Clear previous pages for this manual
    cur.execute("DELETE FROM pages WHERE manual_id = ?", (manual_id,))
    cur.execute("DELETE FROM pages_fts WHERE manual_id = ?", (manual_id,))
    # Insert pages and FTS
    for i, text in enumerate(pages_text, start=1):
        cur.execute(
            "INSERT INTO pages(manual_id, page_number, text, heading) VALUES(?,?,?,?)",
            (manual_id, i, text, None),
        )
        cur.execute(
            "INSERT INTO pages_fts(text, manual_id, page_number) VALUES(?,?,?)",
            (text, manual_id, i),
        )
    conn.commit()

def chunk_iter(lst: List[str], n: int) -> List[List[str]]:
    return [lst[i : i + n] for i in range(0, len(lst), n)]

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms

def build_faiss_index(conn: sqlite3.Connection, client: OpenAI) -> None:
    cur = conn.cursor()
    cur.execute("SELECT manual_id, page_number, text FROM pages ORDER BY manual_id, page_number;")
    rows = cur.fetchall()
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
    conn = connect_db()
    init_schema(conn, rebuild=rebuild)

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
        upsert_manual_and_pages(conn, manual_id, title, filename, pages_text)
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
        build_faiss_index(conn, client)

    conn.close()
    print("Ingestion complete.")

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