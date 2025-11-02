# XYZCare Voice Manual Demo — Runbook

Goal
- Voice-first demo: speak product name plus model to open the correct manual, then ask follow-up repair questions to jump to the relevant page.
- Stack: FastAPI backend, browser mic capture, OpenAI Whisper for STT, SQLite FTS5 for shortlist, embeddings re-rank, built-in browser PDF viewing.

Key files
- Backend app and endpoints:
  - [app.main.stt()](app/main.py:189) — POST /api/stt
  - [app.main.manual_resolve()](app/main.py:256) — GET /api/manual/resolve?q=...
  - [app.main.manual_pdf()](app/main.py:308) — GET /api/manual/{manual_id}/pdf
  - [app.main.manual_search()](app/main.py:335) — POST /api/manual/{manual_id}/search
- Ingestion and indexing: [app/ingestion/ingest_manuals.py](app/ingestion/ingest_manuals.py)
- Frontend UI (mic, viewer): [frontend/index.html](frontend/index.html)
- Alias map (product/model → manual): [data/alias_map.json](data/alias_map.json)
- Python deps: [requirements.txt](requirements.txt)
- Env template: [.env.example](.env.example)

Prerequisites
- macOS (demo target) with Python 3.10+ and a modern Chrome/Chromium-based browser.
- OpenAI API key (for STT and embeddings).
- 10–20 text-native PDFs.

Setup
1) Create and activate a virtual environment
- python3 -m venv .venv
- source .venv/bin/activate

2) Install dependencies
- pip install -r requirements.txt

3) Configure environment variables
- cp .env.example .env
- Edit .env and set:
  - OPENAI_API_KEY=sk-...
  - Optional: adjust FTS_TOP_K, ALIAS_MATCH_THRESHOLD, etc.

4) Add manuals and alias map
- Place PDFs in [data/manuals](data/manuals) with names referenced by [data/alias_map.json](data/alias_map.json).
- Example entries already seeded for “Widget Pro 300”, “Alpha 200”, and “Omni X10”.
- You can edit or extend the alias map:
  - For each manual: { manual_id, title, filename, aliases: [name variants, model variants] }

5) Ingest manuals and build indexes
- python -m app.ingestion.ingest_manuals --rebuild
- This will:
  - Extract per-page text via PyMuPDF
  - Populate SQLite [data/sqlite/manuals.db](data/sqlite/manuals.db)
  - Build embeddings and FAISS artifacts in [data/index](data/index)

Run the app
- uvicorn app.main:app --reload
- Open http://localhost:8000 in Chrome
- Allow microphone permissions when prompted

Demo script
- Resolve a manual
  - Click “🎤 Speak” and say: “Widget Pro 300 manual” (or any name in your [data/alias_map.json](data/alias_map.json))
  - The app calls [app.main.stt()](app/main.py:189) then [app.main.manual_resolve()](app/main.py:256)
  - On success, the viewer loads via [app.main.manual_pdf()](app/main.py:308)
- Ask follow-up question(s)
  - Click “🎤 Speak” again and say: “How do I replace the belt?” or “Torque specs for the motor mount?”
  - Endpoint flow: [app.main.manual_search()](app/main.py:335)
    - FTS shortlist from SQLite pages_fts
    - Embedding re-rank with OpenAI text-embedding-3-small
    - Best page returned with a snippet; viewer jumps to that page

Latency targets
- STT: 1.2–1.8s typical for 4–6s clip
- FTS shortlist: 10–40ms
- Embedding re-rank: 80–150ms
- Viewer jump: 20–50ms
- End-to-end: under 3s typical on local Wi‑Fi

Troubleshooting
- Microphone denied
  - Chrome settings → Privacy → Microphone → Allow for http://localhost:8000
- STT failure with 502 or 429
  - Check OPENAI_API_KEY in [.env](.env) and network connectivity
  - Confirm STT_MODEL in [.env.example](.env.example) matches gpt-4o-mini-transcribe or update to whisper-1
  - For quota exceeded errors (429), enable mock STT by setting `USE_MOCK_STT=true` in [.env](.env)
- Manual not found (404)
  - Ensure the PDF filename matches the alias map entry and resides under [data/manuals](data/manuals)
- No text extracted
  - The PDF may be scanned; run OCR (outside current scope) or swap in a text-native PDF for the demo
- No relevant section found
  - Try rephrasing the question using manual terms; increase FTS_TOP_K in [.env](.env) from 8 to 12 for broader shortlist

Extending the demo (post-showcase)
- PDF.js for richer control and highlighting
  - Replace built-in viewer with PDF.js to programmatically highlight snippet terms
- Local-first privacy mode
  - Swap STT to faster-whisper and embeddings to a local model (e.g., bge-small), keep SQLite FTS
- More robust disambiguation
  - If manual resolution score < ALIAS_MATCH_THRESHOLD (see [app.main.manual_resolve()](app/main.py:256)), surface top-3 candidates for one-click selection

Key environment flags (from [.env.example](.env.example))
- STT_PROVIDER=openai
- STT_MODEL=gpt-4o-mini-transcribe
- EMBEDDING_MODEL=text-embedding-3-small
- FTS_TOP_K=8
- RERANK_TOP_N=3
- CONFIDENCE_THRESHOLD=0.65
- ALIAS_MATCH_THRESHOLD=0.85
- MAX_AUDIO_SECONDS=8
- MAX_UPLOAD_MB=10
- ALLOWED_AUDIO_MIME=audio/webm,audio/wav
- USE_MOCK_STT=false

Security and safety
- The app accepts audio uploads only up to MAX_UPLOAD_MB megabytes and MIME types in ALLOWED_AUDIO_MIME.
- PDF streaming is constrained to [data/manuals](data/manuals) via path checks in [app.main.manual_pdf()](app/main.py:308).
- Logs avoid storing transcripts; timings and status codes are safe to log for performance checks.

Acceptance checklist
- Voice resolves a known manual in <= 3 seconds
- Follow-up question jumps to a relevant page within the manual
- Ambiguity or no-match flows are handled gracefully (toasts, logs)
- Demo runs locally with no external data fetches aside from OpenAI APIs

Sample prompts
- Resolve
  - “Widget Pro 300 manual”
  - “Open the manual for Alpha 200”
- Ask
  - “How do I replace the drive belt?”
  - “What are the torque specs for the motor mount?”
  - “Fuse location for the control board?”

Command snippets
- Start app
  - uvicorn app.main:app --reload
- Ingest data
  - python -m app.ingestion.ingest_manuals --rebuild
- Check health probe
  - curl -s http://localhost:8000/healthz | jq

Notes
- Current viewer uses the browser’s built-in PDF renderer with page fragment navigation (#page=). PDF.js integration and highlight will be added in a subsequent step if needed for the demo.