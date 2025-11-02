#!/bin/bash
# Start the server with proper reload configuration
# This watches only app/ and frontend/ directories, excluding .venv

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --reload-dir app \
  --reload-dir frontend \
  --reload-dir data/alias_map.json

