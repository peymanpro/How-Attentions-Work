from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.qkv import QKV
from src.attention.softmax import softmax
from src.math.matrix import Matrix


@dataclass(frozen=True)
class AttentionResult:
    output: Matrix
    weights: Matrix
    scores: Matrix


class ScaledDotProductAttention:
    def __init__(self, key_dimension: int) -> None:
        if key_dimension <= 0:
            raise ValueError("key_dimension must be positive.")

        self._key_dimension = key_dimension

    @property
    def key_dimension(self) -> int:
        return self._key_dimension

    def forward(self, qkv: QKV) -> AttentionResult:
        scores = qkv.query.multiply(
            qkv.key.transpose()
        )

        scaled_scores = scores.multiply_scalar(
            1.0 / np.sqrt(self._key_dimension)
        )

        weights = self._softmax_rows(scaled_scores)

        output = weights.multiply(qkv.value)

        return AttentionResult(
            output=output,
            weights=weights,
            scores=scaled_scores,
        )

    @staticmethod
    def _softmax_rows(matrix: Matrix) -> Matrix:
        data = matrix.data

        result = np.empty_like(data)

        for row_index in range(data.shape[0]):
            result[row_index] = softmax(
                data[row_index]
            )

        return Matrix(result)
