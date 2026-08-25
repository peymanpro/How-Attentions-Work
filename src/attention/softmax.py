from __future__ import annotations

import numpy as np


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    if values.ndim != 1:
        raise ValueError("Softmax expects a one-dimensional array.")

    if values.size == 0:
        raise ValueError("Softmax input cannot be empty.")

    if not np.isfinite(values).all():
        raise ValueError(
            "Softmax input must contain only finite values."
        )

    return _softmax_finite(values)


def masked_softmax(
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    if values.ndim != 1:
        raise ValueError(
            "Masked softmax expects a one-dimensional array."
        )

    if mask.shape != values.shape:
        raise ValueError(
            "Mask shape must match the values shape."
        )

    if values.size == 0:
        raise ValueError(
            "Masked softmax input cannot be empty."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Masked softmax input must contain only finite values."
        )

    if not np.any(mask):
        raise ValueError(
            "At least one value must be unmasked."
        )

    result = np.zeros_like(values)

    valid_values = values[mask]
    valid_probabilities = _softmax_finite(
        valid_values
    )

    result[mask] = valid_probabilities

    return result


def _softmax_finite(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)

    exponentials = np.exp(shifted)
    total = np.sum(exponentials)

    if total <= 0.0 or not np.isfinite(total):
        raise FloatingPointError(
            "Softmax normalization produced an invalid result."
        )

    result = np.asarray(
        exponentials / total,
        dtype=np.float64,
    )

    return result
