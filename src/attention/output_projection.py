from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.softmax import softmax
from src.math.matrix import Matrix


@dataclass(frozen=True)
class TokenPredictions:
    logits: Matrix
    probabilities: Matrix


class OutputProjection:
    def __init__(
        self,
        input_dimension: int,
        vocabulary_size: int,
        seed: int = 42,
    ) -> None:
        if input_dimension <= 0:
            raise ValueError(
                "input_dimension must be positive."
            )

        if vocabulary_size <= 0:
            raise ValueError(
                "vocabulary_size must be positive."
            )

        self._input_dimension = input_dimension
        self._vocabulary_size = vocabulary_size

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(input_dimension)

        self._weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(
                    input_dimension,
                    vocabulary_size,
                ),
            )
        )

    @property
    def input_dimension(self) -> int:
        return self._input_dimension

    @property
    def vocabulary_size(self) -> int:
        return self._vocabulary_size

    @property
    def weights(self) -> Matrix:
        return Matrix(self._weights.data)

    def forward(
        self,
        inputs: Matrix,
    ) -> TokenPredictions:
        if inputs.columns != self._input_dimension:
            raise ValueError(
                "Input dimension does not match output projection."
            )

        logits = inputs.multiply(self._weights)

        probabilities = self._softmax_rows(logits)

        return TokenPredictions(
            logits=logits,
            probabilities=probabilities,
        )

    def apply_gradient(
        self,
        gradient: Matrix,
        learning_rate: float,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        if gradient.shape != self._weights.shape:
            raise ValueError(
                "Gradient shape must match output projection weights."
            )

        self._weights = self._weights.add(
            gradient.multiply_scalar(
                -learning_rate
            )
        )

    @staticmethod
    def _softmax_rows(
        matrix: Matrix,
    ) -> Matrix:
        data = matrix.data
        result = np.empty_like(data)

        for row_index in range(data.shape[0]):
            result[row_index] = softmax(
                data[row_index]
            )

        return Matrix(result)
