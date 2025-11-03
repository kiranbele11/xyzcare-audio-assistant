#!/bin/bash

echo "--- Running Heroku release tasks ---"

# Run database ingestion
# The --rebuild flag ensures the database is cleared and repopulated on each release
# The --no-embed flag skips the FAISS index creation, as we are only testing Postgres search
python -m app.ingestion.ingest_manuals --rebuild --no-embed

echo "--- Release tasks complete ---"
