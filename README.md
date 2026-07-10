# VisRec — Visualization Recommendation Lab

Frontend + API wrapper for the existing `app.py` pipeline.

## Files
- `app.py` — your original pipeline (unchanged).
- `api.py` — FastAPI wrapper exposing it over HTTP.
- `index.html` — single-file frontend (upload CSV → see profile, insights, charts).
- `requirements.txt` — backend dependencies.

## Run it

```bash
pip install -r requirements.txt
uvicorn api:app --port 8000
```

The API serves the frontend itself at `http://127.0.0.1:8000` and opens it in your default browser automatically — no separate static server needed.

To point the frontend at a different API host (e.g. if you split frontend/backend across servers later), set this before the page's script runs:
```html
<script>window.VISREC_API_BASE = "http://your-host:8000";</script>
```

## Endpoints
- `POST /api/analyze` — multipart form, field `file` = your CSV. Returns profile, insights, and chart URLs.
- `GET /api/sample` — runs the pipeline on the built-in sample dataset (seaborn or synthetic fallback).
- `GET /api/result/{run_id}` — re-fetch a previous run.
- `GET /files/...` — serves the generated PNG/HTML charts.

## Notes
- Each analysis run gets its own folder under `runs/<run_id>/` (png/, html/, report.md, JSON) — nothing overwrites previous runs.
- CORS is wide open (`*`) for local development; tighten `allow_origins` in `api.py` before deploying publicly.
- First request will be slow if `sentence-transformers` needs to download its model weights.