from __future__ import annotations

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Softmax expects a one-dimensional array.")

    if values.size == 0:
        raise ValueError("Softmax input cannot be empty.")

    if not np.isfinite(values).all():
        raise ValueError("Softmax input must contain only finite values.")

    shifted = values - np.max(values)

    exponentials = np.exp(shifted)
    total = np.sum(exponentials)

    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError(
            "Softmax normalization produced an invalid result."
        )

    result = np.asarray(exponentials / total, dtype=np.float64)
    return result

