from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from urllib.parse import urlparse
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="FraudLens AI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UrlScanRequest(BaseModel):
    url: str
    ad_text: Optional[str] = ""
    claimed_discount: Optional[str] = ""


class UrlScanResponse(BaseModel):
    url: str
    domain: str
    risk_score: int
    verdict: str
    summary: str
    red_flags: List[str]
    safety_tips: List[str]


def _normalize_url(raw_url: str) -> tuple[str, str]:
    cleaned = (raw_url or "").strip()
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned
    parsed = urlparse(cleaned)
    domain = (parsed.netloc or "").lower().removeprefix("www.")
    return cleaned, domain


def _analyze_signals(normalized_url: str, domain: str, ad_text: str, claimed_discount: str) -> dict:
    flags = []
    score = 12

    if not domain or "." not in domain:
        flags.append("Invalid or incomplete domain format.")
        score += 45

    if normalized_url.startswith("http://"):
        flags.append("Uses unencrypted HTTP instead of HTTPS.")
        score += 20

    suspicious_tlds = (".xyz", ".top", ".click", ".shop", ".live", ".icu", ".buzz", ".gq", ".tk", ".ml")
    if any(domain.endswith(tld) for tld in suspicious_tlds):
        flags.append("Uses a low-trust or high-frequency promotional domain extension.")
        score += 20

    if re.search(r"\d{3,}", domain) or domain.count("-") >= 2:
        flags.append("Domain contains unusual numbers or multiple hyphens common in lookalike stores.")
        score += 18

    combined_text = f"{ad_text} {claimed_discount}".lower()
    urgency_words = ["80% off", "90% off", "flash sale", "hurry", "last chance", "limited stock", "free gift", "clearance"]
    if any(w in combined_text for w in urgency_words):
        flags.append("Aggressive urgency or steep discount claims detected.")
        score += 20

    score = max(0, min(100, score))
    if score >= 65:
        verdict = "Fake / High Risk"
        summary = "Multiple high-risk indicators detected. Avoid sharing UPI, card, or advance payment details."
    elif score >= 35:
        verdict = "Needs Caution"
        summary = "Some suspicious patterns found. Verify store reviews and return policies before ordering."
    else:
        verdict = "Website Looks Secure"
        summary = "No major URL or discount-trap red flags detected in quick check."

    return {
        "score": score,
        "verdict": verdict,
        "summary": summary,
        "flags": flags or ["No obvious high-risk URL or discount-trap signals found."],
        "tips": [
            "Check clear refund, return, and customer support contact details before paying.",
            "Avoid rushing into steep social-media ad discounts without independent reviews."
        ]
    }


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
    scanner_path = os.path.join(STATIC_DIR, "scanner.html")
    if os.path.exists(scanner_path):
        return FileResponse(scanner_path)
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health_check():
    return {"status": "online", "service": "FraudLens AI"}


@app.post("/api/analyze-url", response_model=UrlScanResponse)
def analyze_url(req: UrlScanRequest):
    normalized_url, domain = _normalize_url(req.url)
    result = _analyze_signals(normalized_url, domain, req.ad_text or "", req.claimed_discount or "")
    return UrlScanResponse(
        url=normalized_url,
        domain=domain,
        risk_score=result["score"],
        verdict=result["verdict"],
        summary=result["summary"],
        red_flags=result["flags"],
        safety_tips=result["tips"]
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)