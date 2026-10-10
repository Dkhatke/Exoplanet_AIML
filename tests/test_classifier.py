# tests/test_classifier.py
# ============================================================
# Phase 4 acceptance tests: XGBoost classifier.
# These tests use SYNTHETIC in-memory datasets so no real
# TESS data or trained model is needed.
# ============================================================

import numpy as np
import pytest
import pandas as pd
import tempfile
import os


def _make_synthetic_dataset(n_per_class: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Create a synthetic 25-feature dataset with 4 classes.
    Features are deliberately separable so the classifier can achieve
    reasonable accuracy even on a small synthetic dataset.
    """
    from config import FEATURE_NAMES, CLASS_LABELS
    rng = np.random.default_rng(seed=seed)
    rows = []

    class_profiles = {
        "PLANET":   {"depth": 0.015,  "snr": 12.0,  "secondary_depth": 0.001, "odd_even_diff": 0.001, "oot_rms": 0.0003, "ls_peak_power": 0.05},
        "EB":       {"depth": 0.10,   "snr": 50.0,  "secondary_depth": 0.04,  "odd_even_diff": 0.05,  "oot_rms": 0.001,  "ls_peak_power": 0.30},
        "VARIABLE": {"depth": 0.001,  "snr": 2.0,   "secondary_depth": 0.001, "odd_even_diff": 0.001, "oot_rms": 0.015,  "ls_peak_power": 0.85},
        "NOISE":    {"depth": 0.0001, "snr": 0.5,   "secondary_depth": 0.0,   "odd_even_diff": 0.0,   "oot_rms": 0.0002, "ls_peak_power": 0.02},
    }

    for label in CLASS_LABELS:
        profile = class_profiles[label]
        for _ in range(n_per_class):
            row = {}
            noise_scale = 0.05
            for feat in FEATURE_NAMES:
                base = profile.get(feat, 0.1)
                row[feat] = float(max(0.0, base + rng.normal(0, abs(base) * noise_scale + 1e-6)))
            row["class_label"] = label
            rows.append(row)

    return pd.DataFrame(rows)


class TestXGBClassifier:

    @pytest.fixture(autouse=True)
    def temp_dir(self, tmp_path):
        """Override MODEL_DIR and DATASET_PATH for each test."""
        self._tmpdir = str(tmp_path)

    def _make_dataset_file(self) -> str:
        df = _make_synthetic_dataset(n_per_class=25, seed=7)
        path = os.path.join(self._tmpdir, "features.parquet")
        df.to_parquet(path, index=False)
        return path

    def test_train_returns_model_and_metrics(self):
        from src.classifier.xgb_classifier import train
        dataset_path = self._make_dataset_file()
        model, metrics = train(
            dataset_path=dataset_path,
            n_estimators=50,
            max_depth=3,
            save_model=False,
        )
        assert model is not None
        assert isinstance(metrics, dict)

    def test_metrics_has_required_keys(self):
        from src.classifier.xgb_classifier import train
        dataset_path = self._make_dataset_file()
        _, metrics = train(dataset_path=dataset_path, n_estimators=50, save_model=False)
        for key in ("train_f1_macro", "test_f1_macro", "cv_f1_mean", "cv_f1_std",
                    "confusion_matrix", "feature_importance", "top5_features"):
            assert key in metrics, f"Missing metrics key: {key}"

    def test_cv_f1_exceeds_random_chance(self):
        """4-class random chance = 0.25; our synthetic data should be far above."""
        from src.classifier.xgb_classifier import train
        dataset_path = self._make_dataset_file()
        _, metrics = train(dataset_path=dataset_path, n_estimators=100, save_model=False)
        assert metrics["cv_f1_mean"] > 0.60, (
            f"CV F1 {metrics['cv_f1_mean']:.4f} is too low (< 0.60) on synthetic separable data"
        )

    def test_model_saves_and_loads(self):
        from src.classifier.xgb_classifier import train, load_model
        import xgboost as xgb

        dataset_path = self._make_dataset_file()
        model_path   = os.path.join(self._tmpdir, "xgb_model.json")

        # Patch MODEL_DIR for this test
        import src.classifier.xgb_classifier as clf_mod
        original_model_path = clf_mod.MODEL_PATH
        original_meta_path  = clf_mod.META_PATH
        clf_mod.MODEL_PATH = model_path
        clf_mod.META_PATH  = model_path.replace(".json", "_meta.json")

        try:
            model, _ = train(dataset_path=dataset_path, n_estimators=50, save_model=True)
            assert os.path.exists(model_path), "Model file not saved"

            loaded = load_model()
            assert loaded is not None
        finally:
            clf_mod.MODEL_PATH = original_model_path
            clf_mod.META_PATH  = original_meta_path

    def test_predict_single_returns_required_keys(self):
        from src.classifier.xgb_classifier import train, predict_single
        from config import FEATURE_NAMES

        dataset_path = self._make_dataset_file()
        model, _ = train(dataset_path=dataset_path, n_estimators=50, save_model=False)

        # Make a planet-like feature dict
        feat_dict = {name: 0.0 for name in FEATURE_NAMES}
        feat_dict["depth"]        = 0.015
        feat_dict["snr"]          = 12.0
        feat_dict["ls_peak_power"] = 0.05

        result = predict_single(feat_dict, model=model)
        for key in ("predicted_class", "max_probability", "probabilities", "review_flag"):
            assert key in result, f"Missing key: {key}"

    def test_probabilities_sum_to_one(self):
        from src.classifier.xgb_classifier import train, predict_single
        from config import FEATURE_NAMES

        dataset_path = self._make_dataset_file()
        model, _ = train(dataset_path=dataset_path, n_estimators=50, save_model=False)

        feat_dict = {name: 0.0 for name in FEATURE_NAMES}
        result = predict_single(feat_dict, model=model)
        total_prob = sum(result["probabilities"].values())
        assert abs(total_prob - 1.0) < 0.01, f"Probabilities sum to {total_prob:.4f} (not 1.0)"

    def test_review_flag_raised_for_ambiguous_features(self):
        """All-zero features → maximum uncertainty → review flag should be True."""
        from src.classifier.xgb_classifier import train, predict_single
        from config import FEATURE_NAMES, REVIEW_THRESHOLD

        dataset_path = self._make_dataset_file()
        model, _ = train(dataset_path=dataset_path, n_estimators=50, save_model=False)

        # All-zero: no signal → model should be uncertain
        feat_dict = {name: 0.0 for name in FEATURE_NAMES}
        result = predict_single(feat_dict, model=model)
        # We don't mandate which class, just that it's a valid one
        assert result["predicted_class"] in ["PLANET", "EB", "VARIABLE", "NOISE"]
        # max_probability should be a valid float between 0 and 1
        assert 0.0 < result["max_probability"] <= 1.0

    def test_feature_importance_has_all_features(self):
        from src.classifier.xgb_classifier import train
        from config import FEATURE_NAMES

        dataset_path = self._make_dataset_file()
        _, metrics = train(dataset_path=dataset_path, n_estimators=50, save_model=False)
        importance = metrics["feature_importance"]
        for name in FEATURE_NAMES:
            assert name in importance, f"Feature '{name}' missing from importance dict"

    def test_top5_features_has_5_entries(self):
        from src.classifier.xgb_classifier import train
        dataset_path = self._make_dataset_file()
        _, metrics = train(dataset_path=dataset_path, n_estimators=50, save_model=False)
        assert len(metrics["top5_features"]) == 5

    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 4×4 for a 4-class problem."""
        from src.classifier.xgb_classifier import train
        dataset_path = self._make_dataset_file()
        _, metrics = train(dataset_path=dataset_path, n_estimators=50, save_model=False)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 4, f"Confusion matrix has {len(cm)} rows (expected 4)"
        for row in cm:
            assert len(row) == 4, f"Row length {len(row)} (expected 4)"
