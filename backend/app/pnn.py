from __future__ import annotations

import numpy as np


class ProbabilisticNeuralNetwork:
    """Specht PNN: Gaussian Parzen kernels over training patterns.

    Discrete class posteriors come from normalized class-conditional kernel sums.
    """

    def __init__(self, sigma: float = 0.85):
        self.sigma = float(sigma)
        self.classes_: np.ndarray | None = None
        self.patterns_: dict[int, np.ndarray] = {}
        self.n_features_: int | None = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.patterns_ = {int(c): X[y == c] for c in self.classes_}
        return self

    def _kernel_sums(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        k = len(self.classes_)
        scores = np.zeros((n, k), dtype=float)
        inv_two_sigma2 = 1.0 / (2.0 * self.sigma * self.sigma)
        for j, cls in enumerate(self.classes_):
            patterns = self.patterns_[int(cls)]
            # chunk patterns to keep memory bounded
            chunk = 512
            total = np.zeros(n, dtype=float)
            for start in range(0, len(patterns), chunk):
                batch = patterns[start : start + chunk]
                # (n, m, d)
                diff = X[:, None, :] - batch[None, :, :]
                dist2 = np.sum(diff * diff, axis=2)
                total += np.exp(-dist2 * inv_two_sigma2).sum(axis=1)
            scores[:, j] = total / max(len(patterns), 1)
        return scores

    def predict_proba(self, X):
        scores = self._kernel_sums(np.asarray(X, dtype=float))
        scores = np.maximum(scores, 1e-12)
        return scores / scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class ScaledPNN:
    """Apply an external scaler, then a ProbabilisticNeuralNetwork."""

    def __init__(self, scaler, model: ProbabilisticNeuralNetwork):
        self.scaler = scaler
        self.model = model
        self.classes_ = model.classes_

    def predict_proba(self, X):
        return self.model.predict_proba(self.scaler.transform(np.asarray(X, dtype=float)))

    def predict(self, X):
        return self.model.predict(self.scaler.transform(np.asarray(X, dtype=float)))
