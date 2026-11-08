from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def _one_hot(y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    index = {int(c): i for i, c in enumerate(classes)}
    out = np.zeros((len(y), len(classes)), dtype=float)
    for i, label in enumerate(y):
        out[i, index[int(label)]] = 1.0
    return out


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """KL(P || Q) for discrete distributions along the last axis."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


class KLProbabilisticNet:
    """Shallow probabilistic network trained with KL(target || pred) on discrete labels.

    Architecture: StandardScaler-like external scaling assumed; here we learn
    linear logits W·x + b → softmax. KL is the natural loss when comparing two
    discrete distributions over decks/switch.
    """

    def __init__(
        self,
        lr: float = 0.05,
        epochs: int = 400,
        batch_size: int = 128,
        l2: float = 1e-4,
        label_smoothing: float = 0.05,
        seed: int = 42,
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.classes_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.train_kl_: float | None = None

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.scale_

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        Xs = self._scale(X)

        targets = _one_hot(y, self.classes_)
        if self.label_smoothing > 0:
            s = self.label_smoothing
            targets = (1.0 - s) * targets + s / n_classes

        self.W_ = rng.normal(0.0, 0.05, size=(n_features, n_classes))
        self.b_ = np.zeros(n_classes)

        n = len(Xs)
        order = np.arange(n)
        last_kl = 0.0
        for _ in range(self.epochs):
            rng.shuffle(order)
            for start in range(0, n, self.batch_size):
                idx = order[start : start + self.batch_size]
                xb = Xs[idx]
                tb = targets[idx]
                logits = xb @ self.W_ + self.b_
                pred = _softmax(logits)
                # dKL/dlogits for KL(target||pred) equals (pred - target)
                grad = (pred - tb) / len(idx)
                self.W_ -= self.lr * (xb.T @ grad + self.l2 * self.W_)
                self.b_ -= self.lr * grad.sum(axis=0)
            pred_all = _softmax(Xs @ self.W_ + self.b_)
            last_kl = float(kl_divergence(targets, pred_all).mean())
        self.train_kl_ = last_kl
        return self

    def predict_proba(self, X):
        Xs = self._scale(np.asarray(X, dtype=float))
        return _softmax(Xs @ self.W_ + self.b_)

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def mean_kl(self, X, y) -> float:
        targets = _one_hot(np.asarray(y).astype(int), self.classes_)
        pred = self.predict_proba(X)
        return float(kl_divergence(targets, pred).mean())
