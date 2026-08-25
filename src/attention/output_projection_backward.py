from __future__ import annotations

from dataclasses import dataclass

from src.attention.output_projection import TokenPredictions
from src.math.matrix import Matrix


@dataclass(frozen=True)
class OutputProjectionGradients:
    input: Matrix
    weights: Matrix


class OutputProjectionBackward:
    def backward(
        self,
        inputs: Matrix,
        weights: Matrix,
        predictions: TokenPredictions,
        targets: list[int],
    ) -> OutputProjectionGradients:
        if inputs.rows != len(targets):
            raise ValueError(
                "Number of targets must match number of input rows."
            )

        if weights.rows != inputs.columns:
            raise ValueError(
                "Weight rows must match input columns."
            )

        if (
            weights.columns
            != predictions.probabilities.columns
        ):
            raise ValueError(
                "Weight columns must match vocabulary size."
            )

        if (
            predictions.logits.shape
            != predictions.probabilities.shape
        ):
            raise ValueError(
                "Logits and probabilities must have the same shape."
            )

        if predictions.probabilities.rows != inputs.rows:
            raise ValueError(
                "Prediction rows must match input rows."
            )

        vocabulary_size = weights.columns

        for target in targets:
            if not 0 <= target < vocabulary_size:
                raise IndexError(
                    "Target index is outside the vocabulary."
                )

        d_logits = predictions.probabilities.data

        d_logits = d_logits.copy()

        for row_index, target in enumerate(targets):
            d_logits[row_index, target] -= 1.0

        input_gradient = Matrix(
            d_logits @ weights.data.T
        )

        weight_gradient = Matrix(
            inputs.data.T @ d_logits
        )

        return OutputProjectionGradients(
            input=input_gradient,
            weights=weight_gradient,
        )
