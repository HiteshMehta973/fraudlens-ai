import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional
import tldextract

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "fraudlens.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                domain TEXT NOT NULL,
                insta_handle TEXT,
                risk_score INTEGER NOT NULL,
                ml_probability REAL NOT NULL,
                verdict TEXT NOT NULL,
                scanned_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scam_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                insta_handle TEXT,
                reason TEXT NOT NULL,
                reported_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def normalize_domain(raw_domain: str) -> str:
    cleaned = raw_domain.strip().lower()
    if not cleaned:
        return ""
    if not cleaned.startswith(("http://", "https://")):
        cleaned = "https://" + cleaned
    ext = tldextract.extract(cleaned)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return raw_domain.strip().lstrip("@").lower()


def count_watchlist_hits(domain: str, insta_handle: Optional[str] = None) -> int:
    clean_ig = insta_handle.strip().lstrip("@").lower() if insta_handle else ""
    conn = get_connection()
    try:
        cur = conn.cursor()
        if clean_ig:
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM scam_reports WHERE domain = ? OR insta_handle = ?",
                (domain.lower(), clean_ig),
            )
        else:
            cur.execute(
                "SELECT COUNT(*) AS cnt FROM scam_reports WHERE domain = ?",
                (domain.lower(),),
            )
        row = cur.fetchone()
        return int(row["cnt"]) if row else 0
    finally:
        conn.close()


def save_scan(scan_result: Dict[str, Any]) -> str:
    scanned_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO scans (url, domain, insta_handle, risk_score, ml_probability, verdict, scanned_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_result["url"],
                scan_result["domain"],
                scan_result.get("insta_handle", ""),
                int(scan_result["risk_score"]),
                float(scan_result["ml_scam_probability"]),
                scan_result["verdict"],
                scanned_at,
            ),
        )
        conn.commit()
        return scanned_at
    finally:
        conn.close()


def add_scam_report(domain_input: str, reason: str, insta_handle: Optional[str] = None) -> Dict[str, str]:
    clean_domain = normalize_domain(domain_input)
    clean_ig = insta_handle.strip().lstrip("@").lower() if insta_handle else ""
    clean_reason = reason.strip()[:300]

    if not clean_domain or len(clean_domain) < 3:
        raise ValueError("Please enter a valid domain or website link.")
    if len(clean_reason) < 3:
        raise ValueError("Please enter a brief reason for reporting.")

    reported_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO scam_reports (domain, insta_handle, reason, reported_at)
            VALUES (?, ?, ?, ?)
            """,
            (clean_domain, clean_ig, clean_reason, reported_at),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "domain": clean_domain,
        "insta_handle": clean_ig,
        "reason": clean_reason,
        "reported_at": reported_at,
    }


def get_recent_scans(limit: int = 15) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, url, domain, insta_handle, risk_score, ml_probability, verdict, scanned_at
            FROM scans
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_watchlist(limit: int = 20) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, domain, insta_handle, reason, reported_at
            FROM scam_reports
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()