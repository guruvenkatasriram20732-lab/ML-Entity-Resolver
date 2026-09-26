"""
Lightweight Machine Learning model module for Amazon ML Challenge.
Provides dual engine: scikit-learn models (when available) and an optimized
zero-dependency pure-NumPy regularized Logistic Regression classifier.
"""

import importlib
import json
from pathlib import Path
from typing import Optional
import numpy as np  # type: ignore[import-not-found]

try:
    sklearn = importlib.import_module("sklearn")
    SklearnLogReg = importlib.import_module("sklearn.linear_model").LogisticRegression
    SklearnScaler = importlib.import_module("sklearn.preprocessing").StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    sklearn = None
    SklearnLogReg = None
    SklearnScaler = None
    SKLEARN_AVAILABLE = False


class PureNumpyClassifier:
    def __init__(self, lr: float = 0.05, epochs: int = 400, l2_reg: float = 0.01):
        self.lr, self.epochs, self.l2_reg = lr, epochs, l2_reg
        self.weights, self.bias, self.mean, self.std = None, 0.0, None, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)
        self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
        self.std[self.std == 0.0] = 1.0
        X_norm = (X - self.mean) / self.std

        n_samples, n_features = X_norm.shape
        self.weights, self.bias = np.zeros(n_features, dtype=np.float32), 0.0
        pos_weight = (n_samples - np.sum(y == 1)) / max(np.sum(y == 1), 1)
        sample_weights = np.where(y == 1, pos_weight, 1.0)

        for _ in range(self.epochs):
            preds = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X_norm, self.weights) + self.bias, -30.0, 30.0)))
            errs = (preds - y) * sample_weights
            self.weights -= self.lr * ((np.dot(X_norm.T, errs) / n_samples) + self.l2_reg * self.weights)
            self.bias -= self.lr * (float(np.sum(errs)) / n_samples)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_norm = (np.asarray(X, dtype=np.float32) - self.mean) / self.std
        p1 = 1.0 / (1.0 + np.exp(-np.clip(np.dot(X_norm, self.weights) + self.bias, -30.0, 30.0)))
        return np.column_stack([1.0 - p1, p1])


class EntityResolutionModel:
    def __init__(self, prefer_sklearn: bool = True):
        self.use_sklearn = prefer_sklearn and SKLEARN_AVAILABLE
        self.model, self.scaler, self.best_threshold = None, None, 0.60

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names=None):
        if self.use_sklearn:
            self.scaler = SklearnScaler()
            self.model = SklearnLogReg(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
            self.model.fit(self.scaler.fit_transform(X), y)
        else:
            self.model = PureNumpyClassifier()
            self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform(X) if self.use_sklearn else X)

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        meta = {"best_threshold": self.best_threshold, "use_sklearn": self.use_sklearn}
        with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
