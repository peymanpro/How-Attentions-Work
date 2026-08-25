from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix


@dataclass(frozen=True)
class AttentionSnapshot:
    weights: Matrix

    def difference(
        self,
        other: AttentionSnapshot,
    ) -> Matrix:
        if self.weights.shape != other.weights.shape:
            raise ValueError(
                "Attention snapshots must have the same shape."
            )

        return Matrix(
            other.weights.data
            - self.weights.data
        )

    def mean_absolute_change(
        self,
        other: AttentionSnapshot,
    ) -> float:
        difference = self.difference(other).data

        return float(
            np.mean(
                np.abs(difference)
            )
        )
