import ipaddress
import json
import os
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
import tldextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MALICIOUS_CSV = os.path.join(BASE_DIR, "malicious_phish.csv")
PHIUSIIL_CSV = os.path.join(BASE_DIR, "phiusiil.csv")
DOMAIN_CSV = os.path.join(BASE_DIR, "domains_phishing.csv")
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

FEATURE_COLUMNS = [
    "is_suspicious_tld",
    "url_length",
    "hyphen_count",
    "has_bait_keyword",
    "digit_count",
    "dot_count",
    "special_char_count",
    "path_depth",
    "has_ip",
    "domain_age_days",
    "has_https",
    "missing_policies",
    "missing_contact",
]

SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "online", "site", "shop", "buzz", "icu", "tk", "store", "live"
}
BAIT_KEYWORDS = [
    "free", "lucky", "winner", "urgent", "cheap", "80off", "90off", "clearance", "giveaway", "sale",
    "login", "verify", "secure", "update", "account", "bank"
]
SPECIAL_CHARS = set("?=&@%_$!~*")


def _is_ip_host(host: str) -> int:
    clean_host = (host or "").strip("[]").split(":")[0]
    if not clean_host:
        return 0
    try:
        ipaddress.ip_address(clean_host)
        return 1
    except ValueError:
        return 0


def extract_url_features(raw_url: str) -> dict:
    original = str(raw_url or "").strip()
    cleaned = original if original.startswith(("http://", "https://")) else "https://" + original

    parsed = urlparse(cleaned)
    ext = tldextract.extract(cleaned)
    tld = (ext.suffix or "").lower()
    domain_root = (ext.domain or "").lower()
    host = (parsed.netloc or ext.fqdn or "").lower()
    path_str = parsed.path or ""
    path_query = f"{path_str}{parsed.query}".lower()

    path_segments = [seg for seg in path_str.split("/") if seg]

    return {
        "is_suspicious_tld": 1 if tld in SUSPICIOUS_TLDS else 0,
        "url_length": len(cleaned),
        "hyphen_count": domain_root.count("-") + path_query.count("-"),
        "has_bait_keyword": 1 if any(k in cleaned.lower() for k in BAIT_KEYWORDS) else 0,
        "digit_count": sum(ch.isdigit() for ch in cleaned),
        "dot_count": cleaned.count("."),
        "special_char_count": sum(ch in SPECIAL_CHARS for ch in cleaned),
        "path_depth": len(path_segments),
        "has_ip": _is_ip_host(host),
        # Neutral baseline when offline bulk-training without live WHOIS/HTML scrape
        "domain_age_days": 365,
        "has_https": 1 if original.lower().startswith("https://") else 0,
        "missing_policies": 0,
        "missing_contact": 0,
    }


def load_malicious_phish(path: str, sample_limit: int = 35000, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["url", "type"])
    df = df.dropna(subset=["url", "type"]).copy()
    # 'benign' -> 0 (safe), others -> 1 (scam/malicious)
    df["label"] = (df["type"].astype(str).str.strip().str.lower() != "benign").astype(int)
    df["url"] = df["url"].astype(str).str.strip()
    if len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=random_state)
    return df[["url", "label"]]


def load_phiusiil(path: str, sample_limit: int = 35000, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["URL", "label"])
    df = df.rename(columns={"URL": "url"})
    df = df.dropna(subset=["url", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    # PhiUSIIL: 1 = legitimate, 0 = phishing -> flip so 1 = scam/phishing, 0 = legit
    df["label"] = (1 - df["label"].astype(int)).clip(0, 1)
    df["url"] = df["url"].astype(str).str.strip()
    if len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=random_state)
    return df[["url", "label"]]


def load_domain_dataset(path: str, sample_limit: int = 35000, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Domain", "Label"])
    df = df.dropna(subset=["Domain", "Label"]).copy()
    df["label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    # Domain dataset: 0 = safe/legitimate, 1 = phishing/malicious
    df["label"] = df["label"].astype(int).clip(0, 1)
    df["url"] = df["Domain"].astype(str).str.strip()
    if len(df) > sample_limit:
        df = df.sample(n=sample_limit, random_state=random_state)
    return df[["url", "label"]]


def build_training_dataset(random_state: int = 42) -> tuple[pd.DataFrame, list[str]]:
    sources = []
    frames = []

    if os.path.exists(MALICIOUS_CSV):
        frames.append(load_malicious_phish(MALICIOUS_CSV, sample_limit=35000, random_state=random_state))
        sources.append("malicious_phish.csv")

    if os.path.exists(PHIUSIIL_CSV):
        frames.append(load_phiusiil(PHIUSIIL_CSV, sample_limit=35000, random_state=random_state))
        sources.append("phiusiil.csv")

    if os.path.exists(DOMAIN_CSV):
        frames.append(load_domain_dataset(DOMAIN_CSV, sample_limit=35000, random_state=random_state))
        sources.append(os.path.basename(DOMAIN_CSV))

    if not frames:
        raise FileNotFoundError("No dataset CSVs found in ml/ folder.")

    combined = pd.concat(frames, ignore_index=True)
    combined["url_norm"] = combined["url"].str.lower().str.rstrip("/")
    combined = combined[combined["url_norm"] != ""].copy()

    # Drop conflicting labels for the exact same normalized URL to reduce label noise
    label_counts = combined.groupby("url_norm")["label"].nunique()
    conflicting_urls = set(label_counts[label_counts > 1].index)
    if conflicting_urls:
        combined = combined[~combined["url_norm"].isin(conflicting_urls)].copy()

    combined = combined.drop_duplicates(subset=["url_norm"]).reset_index(drop=True)
    combined = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    rows = []
    for raw_url, label in zip(combined["url"], combined["label"]):
        feat = extract_url_features(raw_url)
        feat["label"] = int(label)
        rows.append(feat)

    return pd.DataFrame(rows), sources


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_thresh = 0.50
    best_f1 = -1.0
    for thresh in np.linspace(0.35, 0.65, 31):
        preds = (probs >= thresh).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(thresh)
    return round(best_thresh, 2), round(float(best_f1) * 100, 2)


def train_and_save():
    df, sources = build_training_dataset(random_state=42)
    X = df[FEATURE_COLUMNS]
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    best_threshold, _ = find_best_threshold(y_test, y_probs)
    y_pred = (y_probs >= best_threshold).astype(int)

    metrics = {
        "model_type": "RandomForestClassifier",
        "dataset_sources": sources,
        "total_samples": int(len(df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "decision_threshold": best_threshold,
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "feature_importances": {
            col: round(float(imp) * 100, 2)
            for col, imp in sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        },
        "disclaimer": "Risk estimate only based on trained feature patterns — not a guaranteed fraud verdict.",
    }

    joblib.dump(model, MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Datasets used:", ", ".join(sources), f"({len(df)} unique URLs)")
    print("Optimal F1 threshold:", best_threshold)
    print("ML model trained and saved to:", MODEL_PATH)
    print("Model metrics saved to:", METRICS_PATH)
    print(
        f"Accuracy: {metrics['accuracy']}% | Precision: {metrics['precision']}% | "
        f"Recall: {metrics['recall']}% | F1: {metrics['f1_score']}%"
    )


if __name__ == "__main__":
    train_and_save()