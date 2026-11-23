import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

FEATURE_COLUMNS = [
    "is_suspicious_tld",
    "url_length",
    "hyphen_count",
    "has_bait_keyword",
    "domain_age_days",
    "has_https",
    "missing_policies",
    "missing_contact",
]


def build_training_dataset(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a structured feature dataset representing legitimate e-commerce stores
    and high-risk social/ad scam storefronts for initial model training.
    Replace or expand this function with real labelled URL data as you scale.
    """
    rng = np.random.default_rng(random_state)
    half = n_samples // 2

    # 1. Legitimate / established store patterns (label = 0)
    legit_df = pd.DataFrame({
        "is_suspicious_tld": rng.choice([0, 1], size=half, p=[0.88, 0.12]),
        "url_length": rng.integers(18, 65, size=half),
        "hyphen_count": rng.choice([0, 1, 2], size=half, p=[0.75, 0.20, 0.05]),
        "has_bait_keyword": rng.choice([0, 1], size=half, p=[0.85, 0.15]),
        "domain_age_days": rng.integers(240, 2500, size=half),
        "has_https": rng.choice([1, 0], size=half, p=[0.96, 0.04]),
        "missing_policies": rng.choice([0, 1], size=half, p=[0.90, 0.10]),
        "missing_contact": rng.choice([0, 1], size=half, p=[0.92, 0.08]),
        "label": np.zeros(half, dtype=int),
    })

    # 2. Scam / disposable ad-store patterns (label = 1)
    scam_df = pd.DataFrame({
        "is_suspicious_tld": rng.choice([1, 0], size=half, p=[0.78, 0.22]),
        "url_length": rng.integers(32, 120, size=half),
        "hyphen_count": rng.choice([0, 1, 2, 3, 4], size=half, p=[0.20, 0.25, 0.25, 0.20, 0.10]),
        "has_bait_keyword": rng.choice([1, 0], size=half, p=[0.82, 0.18]),
        "domain_age_days": rng.integers(1, 120, size=half),
        "has_https": rng.choice([1, 0], size=half, p=[0.65, 0.35]),
        "missing_policies": rng.choice([1, 0], size=half, p=[0.85, 0.15]),
        "missing_contact": rng.choice([1, 0], size=half, p=[0.80, 0.20]),
        "label": np.ones(half, dtype=int),
    })

    df = pd.concat([legit_df, scam_df], ignore_index=True)
    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_and_save():
    df = build_training_dataset(n_samples=1200, random_state=42)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "model_type": "RandomForestClassifier",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred)) * 100, 2),
        "recall": round(float(recall_score(y_test, y_pred)) * 100, 2),
        "f1_score": round(float(f1_score(y_test, y_pred)) * 100, 2),
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

    print("ML model trained and saved to:", MODEL_PATH)
    print("Model metrics saved to:", METRICS_PATH)
    print(f"Accuracy: {metrics['accuracy']}% | Precision: {metrics['precision']}% | Recall: {metrics['recall']}% | F1: {metrics['f1_score']}%")


if __name__ == "__main__":
    train_and_save()