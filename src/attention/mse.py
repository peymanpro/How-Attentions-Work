from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix


class MeanSquaredError:
    def calculate(
        self,
        prediction: Matrix,
        target: Matrix,
    ) -> float:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes must match."
            )

        difference = (
            prediction.data - target.data
        )

        return float(
            np.mean(difference * difference)
        )

    def gradient(
        self,
        prediction: Matrix,
        target: Matrix,
    ) -> Matrix:
        if prediction.shape != target.shape:
            raise ValueError(
                "Prediction and target shapes must match."
            )

        difference = (
            prediction.data - target.data
        )

        return Matrix(
            2.0 * difference / prediction.data.size
        )
