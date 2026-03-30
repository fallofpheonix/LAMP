from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class LogisticVisibilityModel:
    weights: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xn = (X - self.mean) / self.std
        logits = Xn @ self.weights + self.bias
        logits = np.clip(logits, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.uint8)

    def save(self, path: str) -> None:
        np.savez(
            path,
            weights=self.weights,
            bias=np.array([self.bias], dtype=np.float64),
            mean=self.mean,
            std=self.std,
        )

    @staticmethod
    def load(path: str) -> "LogisticVisibilityModel":
        data = np.load(path)
        return LogisticVisibilityModel(
            weights=data["weights"],
            bias=float(data["bias"][0]),
            mean=data["mean"],
            std=data["std"],
        )


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(np.uint8)
    y_pred = y_pred.astype(np.uint8)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_steps: int = 81,
) -> Tuple[float, Dict[str, float]]:
    best_t = 0.5
    best_m = {"f1": -1.0}

    for t in np.linspace(0.1, 0.9, n_steps):
        pred = (y_prob >= t).astype(np.uint8)
        m = binary_metrics(y_true, pred)
        if m["f1"] > best_m["f1"]:
            best_t = float(t)
            best_m = m
    return best_t, best_m


def train_logistic_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    learning_rate: float = 0.05,
    epochs: int = 300,
    l2: float = 1e-4,
) -> Tuple[LogisticVisibilityModel, Dict[str, float], Dict[str, float]]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    Xn = (X_train - mean) / std
    n_features = X_train.shape[1]
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    n = float(len(y_train))
    y = y_train.astype(np.float64)

    for _ in range(epochs):
        logits = Xn @ w + b
        logits = np.clip(logits, -50.0, 50.0)
        p = 1.0 / (1.0 + np.exp(-logits))

        # Weighted BCE to counter class imbalance.
        pos = max(np.sum(y == 1.0), 1.0)
        neg = max(np.sum(y == 0.0), 1.0)
        w_pos = neg / pos
        sample_w = np.where(y == 1.0, w_pos, 1.0)

        grad = (p - y) * sample_w
        grad_w = (Xn.T @ grad) / n + l2 * w
        grad_b = float(np.sum(grad) / n)

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    model = LogisticVisibilityModel(weights=w, bias=b, mean=mean, std=std)

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_metrics = binary_metrics(y_train, train_pred)
    val_metrics = binary_metrics(y_val, val_pred)

    return model, train_metrics, val_metrics
