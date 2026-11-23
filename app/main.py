import json
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.analyzer import analyze_store
from app.database import (
    add_scam_report,
    count_watchlist_hits,
    get_recent_scans,
    get_watchlist,
    init_db,
    save_scan,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
METRICS_PATH = os.path.join(BASE_DIR, "ml", "model_metrics.json")

app = FastAPI(title="FraudLens AI", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

init_db()


class ScanRequest(BaseModel):
    url: str
    insta_handle: Optional[str] = None
    mrp_price: Optional[float] = None
    offer_price: Optional[float] = None


class ReportRequest(BaseModel):
    domain: str
    insta_handle: Optional[str] = None
    reason: str


@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "FraudLens AI API is running. Add static/index.html to view the UI."}


@app.get("/api/metrics")
def api_metrics():
    if not os.path.exists(METRICS_PATH):
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found. Run ml/train_model.py first.",
        )
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/history")
def api_history():
    return {"scans": get_recent_scans(limit=15)}


@app.get("/api/watchlist")
def api_watchlist():
    return {"reports": get_watchlist(limit=20)}


@app.post("/api/report")
def api_report(payload: ReportRequest):
    try:
        saved = add_scam_report(
            domain_input=payload.domain,
            reason=payload.reason,
            insta_handle=payload.insta_handle,
        )
        return {"message": "Added to FraudLens Scam Watchlist.", "report": saved}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/scan")
def api_scan(payload: ScanRequest):
    try:
        from urllib.parse import urlparse
        import tldextract

        cleaned = payload.url.strip()
        if not cleaned.startswith(("http://", "https://")):
            cleaned = "https://" + cleaned
        parsed = urlparse(cleaned)
        hostname = (parsed.hostname or "").lower()
        ext = tldextract.extract(cleaned)
        domain = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else hostname

        hits = count_watchlist_hits(domain=domain, insta_handle=payload.insta_handle)
        result = analyze_store(
            raw_url=payload.url,
            insta_handle=payload.insta_handle,
            mrp_price=payload.mrp_price,
            offer_price=payload.offer_price,
            watchlist_hits=hits,
        )
        result["scanned_at"] = save_scan(result)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scan failed: {exc}") from exc