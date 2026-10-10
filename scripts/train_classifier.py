#!/usr/bin/env python3
# scripts/train_classifier.py
# ============================================================
# Phase 4: Train and evaluate the XGBoost false positive recycler.
#
# Usage:
#   python scripts/train_classifier.py
#   python scripts/train_classifier.py --dataset data/features.parquet
#   python scripts/train_classifier.py --cv-only  # Just cross-validate
# ============================================================

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DATASET_PATH, MODEL_DIR
from src.classifier.xgb_classifier import train
from src.utils.logger import get_logger

log = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost false-positive recycler")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Path to Parquet feature dataset")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save", action="store_true", help="Don't save model to disk")
    args = parser.parse_args()

    log.info("Starting training pipeline…")
    model, metrics = train(
        dataset_path=args.dataset,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
        test_size=args.test_size,
        random_state=args.seed,
        save_model=not args.no_save,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Train macro-F1 : {metrics['train_f1_macro']:.4f}")
    print(f"  Test  macro-F1 : {metrics['test_f1_macro']:.4f}")
    print(f"  CV    macro-F1 : {metrics['cv_f1_mean']:.4f} ± {metrics['cv_f1_std']:.4f}")
    print(f"\n  Top-5 features:")
    for feat, imp in metrics["top5_features"]:
        print(f"    {feat:25s}: {imp:.4f}")

    if not args.no_save:
        print(f"\n  Model saved to: {MODEL_DIR}/xgb_model.json")

    # Per-class report
    print("\n  Per-class F1:")
    for cls in ["PLANET", "EB", "VARIABLE", "NOISE"]:
        if cls in metrics["class_report"]:
            f1 = metrics["class_report"][cls].get("f1-score", 0.0)
            prec = metrics["class_report"][cls].get("precision", 0.0)
            rec  = metrics["class_report"][cls].get("recall", 0.0)
            print(f"    {cls:10s}: F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}")


if __name__ == "__main__":
    main()
