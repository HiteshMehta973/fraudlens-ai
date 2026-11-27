import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from analyzer import analyze_store
except ImportError:
    analyze_store = None

try:
    from app.database import (
        add_scam_report,
        count_watchlist_hits,
        get_recent_scans,
        get_watchlist,
        init_db,
        normalize_domain,
        save_scan,
    )
except ImportError:
    from database import (
        add_scam_report,
        count_watchlist_hits,
        get_recent_scans,
        get_watchlist,
        init_db,
        normalize_domain,
        save_scan,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="FraudLens AI Backend", version="2.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StoreScanRequest(BaseModel):
    raw_url: Optional[str] = None
    url: Optional[str] = None
    insta_handle: Optional[str] = None
    mrp_price: Optional[float] = None
    offer_price: Optional[float] = None
    watchlist_hits: Optional[int] = None
    ad_text: Optional[str] = ""
    claimed_discount: Optional[str] = ""


class ScamReportRequest(BaseModel):
    domain: str
    insta_handle: Optional[str] = ""
    reason: str


class StoreCheckItem(BaseModel):
    title: str
    status: str
    detail: str
    points: int = 0


class StoreScanResponse(BaseModel):
    url: str
    domain: str
    ip: str = "—"
    insta_handle: str = ""
    risk_score: int
    ml_scam_probability: Optional[float] = None
    verdict: str
    is_secure: bool = False
    advice: str
    checks: List[StoreCheckItem] = []
    disclaimer: str = "Automated risk assessment—not a 100% legal or financial guarantee."
    scanned_at: Optional[str] = None


@app.get("/")
def serve_home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/auth")
def serve_auth():
    return FileResponse(os.path.join(STATIC_DIR, "auth.html"))


@app.get("/dashboard")
def serve_dashboard():
    dashboard_path = os.path.join(STATIC_DIR, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/scanner")
def serve_scanner():
    dashboard_path = os.path.join(STATIC_DIR, "dashboard.html")
    if os.path.exists(dashboard_path):
        return FileResponse(dashboard_path)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "FraudLens AI",
        "analyzer_loaded": analyze_store is not None,
    }


@app.post("/api/analyze-url", response_model=StoreScanResponse)
def api_analyze_url(req: StoreScanRequest) -> Dict[str, Any]:
    target_url = (req.raw_url or req.url or "").strip()
    if not target_url:
        raise HTTPException(status_code=400, detail="Please enter a website URL.")

    if analyze_store is None:
        raise HTTPException(
            status_code=500,
            detail="analyzer.py could not be imported. Ensure analyzer.py is in the project root.",
        )

    domain_guess = normalize_domain(target_url)
    db_hits = count_watchlist_hits(domain_guess, req.insta_handle)
    effective_hits = max(db_hits, int(req.watchlist_hits or 0))

    try:
        result = analyze_store(
            raw_url=target_url,
            insta_handle=req.insta_handle,
            mrp_price=req.mrp_price,
            offer_price=req.offer_price,
            watchlist_hits=effective_hits,
        )
        scanned_at = save_scan(result)
        result["scanned_at"] = scanned_at
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Store analysis failed: {str(exc)}") from exc


@app.get("/api/scans")
def api_get_scans(limit: int = 15):
    return {"scans": get_recent_scans(limit=limit)}


@app.get("/api/watchlist")
def api_get_watchlist(limit: int = 20):
    return {"watchlist": get_watchlist(limit=limit)}


@app.post("/api/report-scam")
def api_report_scam(req: ScamReportRequest):
    try:
        saved = add_scam_report(
            domain_input=req.domain,
            reason=req.reason,
            insta_handle=req.insta_handle,
        )
        return {"status": "reported", "report": saved}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)