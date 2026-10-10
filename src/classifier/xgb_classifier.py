# src/classifier/xgb_classifier.py
# ============================================================
# Phase 4: XGBoost 4-class false positive recycler.
# Classes: PLANET, EB, VARIABLE, NOISE
#
# Features: 25 ephemeris-blind features from feature_extractor.py
# Output: class label + class probabilities
#
# Design decisions:
#   - Multi-class: softmax objective (not binary)
#   - Class weights balanced: dataset is 25 each, so equal
#   - Threshold-based REVIEW flag: if max_prob < review_threshold
#   - No early stopping on test set to avoid data leakage
#   - SHAP explainability built in
# ============================================================

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd

from config import (
    FEATURE_NAMES, CLASS_LABELS, REVIEW_THRESHOLD,
    MODEL_DIR, DATASET_PATH,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.json")
META_PATH  = os.path.join(MODEL_DIR, "model_meta.json")


# ── Training ──────────────────────────────────────────────────

def train(
    dataset_path: str = DATASET_PATH,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    test_size: float = 0.2,
    save_model: bool = True,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train XGBoost classifier on the Parquet feature dataset.

    Returns
    -------
    (model, metrics_dict)
    """
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import (
        classification_report, confusion_matrix, f1_score,
    )

    log.info("Loading dataset from %s", dataset_path)
    df = pd.read_parquet(dataset_path)
    log.info("Dataset: %d rows | %d cols", len(df), len(df.columns))

    # Validate feature completeness
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        log.error("Missing features in dataset: %s", missing)
        raise ValueError(f"Dataset missing features: {missing}")

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y_labels = df["class_label"].values

    # Encode labels
    le = LabelEncoder()
    le.fit(CLASS_LABELS)  # Fix encoding order
    y = le.transform(y_labels)

    log.info("Class distribution:\n%s", pd.Series(y_labels).value_counts().to_string())

    # ── Train/Test split ──────────────────────────────────────
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # ── XGBoost model ─────────────────────────────────────────
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="multi:softmax",
        num_class=len(CLASS_LABELS),
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=2,
    )

    log.info("Training XGBoost: %d trees | depth=%d | lr=%.3f", n_estimators, max_depth, learning_rate)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # ── Evaluate ──────────────────────────────────────────────
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_f1 = f1_score(y_train, y_pred_train, average="macro")
    test_f1  = f1_score(y_test,  y_pred_test,  average="macro")

    log.info("Train macro-F1: %.4f | Test macro-F1: %.4f", train_f1, test_f1)

    report = classification_report(
        y_test, y_pred_test,
        target_names=le.inverse_transform(range(len(CLASS_LABELS))),
        output_dict=True,
    )
    cm = confusion_matrix(y_test, y_pred_test).tolist()

    # ── Cross-validation ──────────────────────────────────────
    log.info("Running 5-fold cross-validation\u2026")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="multi:softmax", num_class=len(CLASS_LABELS),
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=random_state, n_jobs=2,
        ),
        X, y,
        cv=cv,
        scoring=["accuracy", "f1_macro"],
        return_train_score=True,
        n_jobs=2,
    )
    cv_f1_mean = float(np.mean(cv_results["test_f1_macro"]))
    cv_f1_std  = float(np.std(cv_results["test_f1_macro"]))
    log.info("CV F1: %.4f ± %.4f", cv_f1_mean, cv_f1_std)

    # ── Feature importance ────────────────────────────────────
    importances = model.feature_importances_
    feat_importance = dict(zip(FEATURE_NAMES, importances.tolist()))
    top5 = sorted(feat_importance.items(), key=lambda x: x[1], reverse=True)[:5]
    log.info("Top-5 features: %s", top5)

    metrics = {
        "train_f1_macro":  train_f1,
        "test_f1_macro":   test_f1,
        "cv_f1_mean":      cv_f1_mean,
        "cv_f1_std":       cv_f1_std,
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "class_report":    report,
        "confusion_matrix": cm,
        "feature_importance": feat_importance,
        "top5_features":   top5,
        "label_classes":   le.classes_.tolist(),
        "n_estimators":    n_estimators,
        "max_depth":       max_depth,
        "learning_rate":   learning_rate,
    }

    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        model.save_model(MODEL_PATH)
        with open(META_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        log.info("Model saved: %s | Meta: %s", MODEL_PATH, META_PATH)

    return model, metrics


# ── Inference ─────────────────────────────────────────────────

def load_model() -> Any:
    """Load the trained XGBoost model from disk."""
    import xgboost as xgb
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No trained model at {MODEL_PATH} — run train() first")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    log.info("Model loaded from %s", MODEL_PATH)
    return model


def predict_single(
    feature_dict: Dict[str, float],
    model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Predict class for a single target given its 25-feature dict.

    Returns
    -------
    dict:
        predicted_class  : str (PLANET / EB / VARIABLE / NOISE)
        max_probability  : float
        probabilities    : dict {class → probability}
        review_flag      : bool — True if max_prob < REVIEW_THRESHOLD
        feature_vector   : list[float] — input to classifier
    """
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb

    if model is None:
        model = load_model()

    le = LabelEncoder()
    le.fit(CLASS_LABELS)

    # Build feature array in FEATURE_NAMES order
    x = np.array([feature_dict.get(name, 0.0) for name in FEATURE_NAMES],
                 dtype=np.float32).reshape(1, -1)

    # XGBoost predict_proba
    proba = model.predict_proba(x)[0]   # shape: (n_classes,)
    pred_idx = int(np.argmax(proba))
    pred_class = le.inverse_transform([pred_idx])[0]
    max_prob = float(proba[pred_idx])

    prob_dict = {cls: float(p) for cls, p in zip(le.classes_, proba)}
    review = max_prob < REVIEW_THRESHOLD

    if review:
        log.info("REVIEW flagged: max_prob=%.3f < threshold=%.2f", max_prob, REVIEW_THRESHOLD)

    return {
        "predicted_class":  pred_class,
        "max_probability":  max_prob,
        "probabilities":    prob_dict,
        "review_flag":      review,
        "feature_vector":   x.flatten().tolist(),
    }


def explain_prediction(
    feature_dict: Dict[str, float],
    model: Optional[Any] = None,
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Compute SHAP values for a single prediction.

    Returns
    -------
    dict:
        shap_values    : dict {class → {feature → shap_value}}
        top_features   : list of (feature, shap_value) for predicted class
        base_value     : float — expected model output
    """
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — run: pip install shap")
        return {}

    if model is None:
        model = load_model()

    x = np.array([feature_dict.get(name, 0.0) for name in FEATURE_NAMES],
                 dtype=np.float32).reshape(1, -1)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)   # shape: (n_classes, 1, n_features)

    # Predicted class
    proba = model.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    pred_class = CLASS_LABELS[pred_idx]

    shap_for_pred = shap_values[pred_idx][0]   # shape: (n_features,)
    top = sorted(zip(FEATURE_NAMES, shap_for_pred.tolist()),
                 key=lambda x: abs(x[1]), reverse=True)[:top_n]

    return {
        "predicted_class": pred_class,
        "shap_values": {
            cls: dict(zip(FEATURE_NAMES, shap_values[i][0].tolist()))
            for i, cls in enumerate(CLASS_LABELS)
        },
        "top_features": top,
        "base_value": float(explainer.expected_value[pred_idx]) if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value),
    }
