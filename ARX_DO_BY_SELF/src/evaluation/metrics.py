from __future__ import annotations

import numpy as np


# Tính FIT, RMSE và Bias.
def fit_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    denom = np.linalg.norm(y_true - y_true.mean())
    fit = 100.0 * (1.0 - np.linalg.norm(err) / denom) if denom > 1e-12 else float("nan")
    return {
        "FIT": float(fit),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "Bias": float(np.mean(err)),
    }
