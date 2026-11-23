import ipaddress
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import joblib
import pandas as pd
import requests
import tldextract
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "fraud_model.pkl")

SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "online", "site", "shop", "buzz", "icu", "tk", "store", "live"
}
BAIT_KEYWORDS = [
    "free", "lucky", "winner", "urgent", "cheap", "80off", "90off", "clearance", "giveaway", "sale"
]
POLICY_TERMS = [
    "refund policy", "return policy", "cancellation", "returns", "exchange policy"
]
CONTACT_TERMS = [
    "contact us", "support@", "help@", "customer care", "phone", "whatsapp"
]


def load_ml_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


ML_MODEL = load_ml_model()


def validate_and_guard_url(raw_url: str) -> Dict[str, str]:
    cleaned = raw_url.strip()
    if not cleaned:
        raise ValueError("Please enter a website URL.")

    if not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned

    parsed = urlparse(cleaned)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid URL format.")

    if hostname.lower() in {"localhost"} or hostname.lower().endswith(".local"):
        raise ValueError("Security Guard: Localhost and internal addresses cannot be scanned.")

    try:
        ip_str = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip_str)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved or ip_obj.is_link_local:
            raise ValueError("Security Guard: Private or internal IP addresses are blocked.")
    except socket.gaierror as exc:
        raise ValueError(f"Domain '{hostname}' could not be resolved.") from exc

    ext = tldextract.extract(cleaned)
    registered_domain = f"{ext.domain}.{ext.suffix}".lower() if ext.suffix else hostname.lower()
    tld = ext.suffix.lower()

    return {
        "url": cleaned,
        "hostname": hostname.lower(),
        "domain": registered_domain,
        "domain_root": ext.domain.lower(),
        "tld": tld,
        "ip": ip_str,
    }


def get_domain_age_days(domain: str) -> Optional[int]:
    try:
        resp = requests.get(f"https://rdap.org/domain/{domain}", timeout=4)
        if resp.status_code == 200:
            payload = resp.json()
            for event in payload.get("events", []):
                if event.get("eventAction") == "registration":
                    date_str = event.get("eventDate", "")
                    if date_str:
                        reg_dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                        now_dt = datetime.now(timezone.utc)
                        return max(1, (now_dt - reg_dt).days)
    except Exception:
        return None
    return None


def analyze_store(
    raw_url: str,
    insta_handle: Optional[str] = None,
    mrp_price: Optional[float] = None,
    offer_price: Optional[float] = None,
    watchlist_hits: int = 0,
) -> Dict[str, Any]:
    target = validate_and_guard_url(raw_url)
    url = target["url"]
    domain = target["domain"]
    domain_root = target["domain_root"]
    tld = target["tld"]
    ip_str = target["ip"]
    clean_ig = insta_handle.strip().lstrip("@").lower() if insta_handle else ""

    checks: List[Dict[str, Any]] = []
    score_points = 0

    f_suspicious_tld = 1 if tld in SUSPICIOUS_TLDS else 0
    f_url_length = len(url)
    f_hyphen_count = domain_root.count("-")
    f_has_bait = 1 if any(word in url.lower() for word in BAIT_KEYWORDS) else 0
    f_has_https = 1 if url.startswith("https://") else 0
    f_missing_policies = 1
    f_missing_contact = 1

    # 1. Community Watchlist Check
    if watchlist_hits > 0:
        pts = 35
        score_points += pts
        checks.append({
            "title": f"Reported in Watchlist ({watchlist_hits}x)",
            "status": "DANGER",
            "detail": "Users have previously flagged this domain or Instagram handle as suspicious/fake.",
            "points": pts,
        })

    # 2. Discount / Price Bait Check
    if mrp_price and offer_price and mrp_price > 0 and offer_price > 0:
        discount_pct = round(((mrp_price - offer_price) / mrp_price) * 100, 1)
        if discount_pct >= 75:
            pts = 30
            score_points += pts
            checks.append({
                "title": f"Extreme Discount Bait ({discount_pct}% OFF)",
                "status": "DANGER",
                "detail": f"MRP ₹{mrp_price:g} vs Offer ₹{offer_price:g} looks like a high-pressure social ad trap.",
                "points": pts,
            })
        elif discount_pct >= 50:
            pts = 10
            score_points += pts
            checks.append({
                "title": f"Heavy Discount ({discount_pct}% OFF)",
                "status": "WARNING",
                "detail": "Verify real buyer photos and return terms before paying.",
                "points": pts,
            })

    # 3. Instagram Handle Signal
    if clean_ig:
        if any(ch.isdigit() for ch in clean_ig[-3:]) or "_" in clean_ig or "store" in clean_ig or "hub" in clean_ig:
            pts = 10
            score_points += pts
            checks.append({
                "title": f"Instagram Handle Check (@{clean_ig})",
                "status": "WARNING",
                "detail": "Generic/numbered storefront handle. Check 'About This Account -> Former Usernames' on Instagram.",
                "points": pts,
            })
        else:
            checks.append({
                "title": f"Instagram Handle (@{clean_ig})",
                "status": "SAFE",
                "detail": "Handle format looks clean; still verify real comments and tagged posts.",
                "points": 0,
            })

    # 4. Domain Age (RDAP)
    age_days = get_domain_age_days(domain)
    f_domain_age = age_days if age_days is not None else 60

    if age_days is None:
        pts = 15
        score_points += pts
        checks.append({
            "title": "Domain Age Unavailable",
            "status": "WARNING",
            "detail": "Could not verify public registration date via RDAP.",
            "points": pts,
        })
    elif age_days < 90:
        pts = 35
        score_points += pts
        checks.append({
            "title": f"Newly Registered Domain ({age_days} days old)",
            "status": "DANGER",
            "detail": "Very young domains are a common red flag in short-lived shopping scams.",
            "points": pts,
        })
    elif age_days < 365:
        pts = 15
        score_points += pts
        checks.append({
            "title": f"Relatively Young Domain ({age_days} days)",
            "status": "WARNING",
            "detail": "Less than 1 year old—verify reputation carefully.",
            "points": pts,
        })
    else:
        checks.append({
            "title": f"Established Domain ({age_days // 365}+ years)",
            "status": "SAFE",
            "detail": f"Domain registered for {age_days} days.",
            "points": 0,
        })

    # 5. TLD & URL Structure
    if f_suspicious_tld:
        pts = 15
        score_points += pts
        checks.append({
            "title": f"High-Frequency Extension (.{tld})",
            "status": "WARNING",
            "detail": f"'.{tld}' domains are low-cost and frequently seen in temporary promotional sites.",
            "points": pts,
        })
    else:
        checks.append({
            "title": f"Standard Extension (.{tld})",
            "status": "SAFE",
            "detail": f"Uses '.{tld}' domain extension.",
            "points": 0,
        })

    if f_hyphen_count >= 2 or f_has_bait:
        pts = 15
        score_points += pts
        checks.append({
            "title": "Suspicious URL Pattern / Keywords",
            "status": "WARNING",
            "detail": "URL contains multiple hyphens or promotional bait keywords.",
            "points": pts,
        })

    # 6. HTTPS & Store Page Signals
    page_accessible = False
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; FraudLensAI/1.0)"}
        resp = requests.get(url, headers=headers, timeout=6, allow_redirects=True)
        page_accessible = True
        f_has_https = 1 if resp.url.startswith("https://") else 0

        if f_has_https:
            checks.append({
                "title": "HTTPS Connection Active",
                "status": "SAFE",
                "detail": "Encrypted connection enabled (note: scam sites can also use free SSL).",
                "points": 0,
            })
        else:
            pts = 25
            score_points += pts
            checks.append({
                "title": "Missing HTTPS Encryption",
                "status": "DANGER",
                "detail": "Site does not use HTTPS—do not enter personal or payment details.",
                "points": pts,
            })

        soup = BeautifulSoup(resp.text[:200000], "html.parser")
        page_text = soup.get_text(" ", strip=True).lower()

        if any(term in page_text for term in POLICY_TERMS):
            f_missing_policies = 0
            checks.append({
                "title": "Return / Refund Policy Mentioned",
                "status": "SAFE",
                "detail": "Return or refund policy text detected on the page.",
                "points": 0,
            })
        else:
            pts = 15
            score_points += pts
            checks.append({
                "title": "No Clear Return / Refund Policy",
                "status": "WARNING",
                "detail": "Missing return/refund policy is a strong warning sign for online stores.",
                "points": pts,
            })

        if any(term in page_text for term in CONTACT_TERMS):
            f_missing_contact = 0
            checks.append({
                "title": "Contact / Support Info Detected",
                "status": "SAFE",
                "detail": "Support email, phone, or contact references found.",
                "points": 0,
            })
        else:
            pts = 15
            score_points += pts
            checks.append({
                "title": "Limited / Missing Seller Contact Info",
                "status": "WARNING",
                "detail": "No clear customer care email, phone, or contact section found.",
                "points": pts,
            })
    except Exception:
        pts = 15
        score_points += pts
        checks.append({
            "title": "Store Page Check Limited",
            "status": "WARNING",
            "detail": "Website blocked automated inspection or timed out.",
            "points": pts,
        })

    # 7. ML Prediction
    ml_scam_prob = 50.0
    if ML_MODEL is not None:
        features = pd.DataFrame([{
            "is_suspicious_tld": f_suspicious_tld,
            "url_length": f_url_length,
            "hyphen_count": f_hyphen_count,
            "has_bait_keyword": f_has_bait,
            "domain_age_days": f_domain_age,
            "has_https": f_has_https,
            "missing_policies": f_missing_policies,
            "missing_contact": f_missing_contact,
        }])
        ml_scam_prob = round(float(ML_MODEL.predict_proba(features)[0][1]) * 100, 1)

    rule_score = min(100, score_points)
    risk_score = int(round((rule_score * 0.65) + (ml_scam_prob * 0.35)))
    if watchlist_hits > 0:
        risk_score = max(risk_score, 80)

    # Clear user-facing verdict
    if risk_score >= 60:
        verdict = "Fake / High Risk"
        is_secure = False
        advice = (
            "High risk detected—avoid UPI, card, or advance payment. "
            "If you already paid and suspect fraud, contact your bank immediately or call Cyber Crime Helpline 1930."
        )
    elif risk_score >= 30 or not page_accessible:
        verdict = "Needs Caution"
        is_secure = False
        advice = (
            "Some warning signs or partial checks found. Prefer Cash on Delivery (only if open-box/inspection is possible) "
            "and verify independent buyer reviews before ordering."
        )
    else:
        verdict = "Website Looks Secure"
        is_secure = True
        advice = (
            "Core technical, domain, and store trust signals look clean. "
            "Always keep order receipts and double-check size/return terms before checkout."
        )

    return {
        "url": url,
        "domain": domain,
        "ip": ip_str,
        "insta_handle": clean_ig,
        "risk_score": risk_score,
        "ml_scam_probability": ml_scam_prob,
        "verdict": verdict,
        "is_secure": is_secure,
        "advice": advice,
        "checks": checks,
        "disclaimer": "Automated risk assessment—not a 100% legal or financial guarantee.",
    }