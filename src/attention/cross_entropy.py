from __future__ import annotations

import numpy as np


class CrossEntropyLoss:
    def calculate(
        self,
        probabilities: np.ndarray,
        target_index: int,
    ) -> float:
        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        )

        if probabilities.ndim != 1:
            raise ValueError(
                "Probabilities must be one-dimensional."
            )

        if probabilities.size == 0:
            raise ValueError(
                "Probabilities cannot be empty."
            )

        if not np.isfinite(probabilities).all():
            raise ValueError(
                "Probabilities must be finite."
            )

        if np.any(probabilities < 0.0):
            raise ValueError(
                "Probabilities cannot be negative."
            )

        if not 0 <= target_index < probabilities.size:
            raise IndexError(
                "Target index is outside the probability vector."
            )

        total = np.sum(probabilities)

        if not np.isclose(total, 1.0):
            raise ValueError(
                "Probabilities must sum to one."
            )

        probability = max(
            probabilities[target_index],
            1e-12,
        )

        return float(-np.log(probability))
