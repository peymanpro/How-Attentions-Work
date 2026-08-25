from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.qkv import QKV
from src.attention.softmax import masked_softmax, softmax
from src.math.matrix import Matrix


@dataclass(frozen=True)
class AttentionResult:
    output: Matrix
    weights: Matrix
    scores: Matrix


class ScaledDotProductAttention:
    def __init__(
        self,
        key_dimension: int,
    ) -> None:
        if key_dimension <= 0:
            raise ValueError(
                "key_dimension must be positive."
            )

        self._key_dimension = key_dimension

    @property
    def key_dimension(self) -> int:
        return self._key_dimension

    def forward(
        self,
        qkv: QKV,
        causal: bool = False,
    ) -> AttentionResult:
        scores = qkv.query.multiply(
            qkv.key.transpose()
        )

        scaled_scores = scores.multiply_scalar(
            1.0 / np.sqrt(self._key_dimension)
        )

        weights = self._softmax_rows(
            scaled_scores,
            causal=causal,
        )

        output = weights.multiply(
            qkv.value
        )

        return AttentionResult(
            output=output,
            weights=weights,
            scores=scaled_scores,
        )

    @staticmethod
    def _softmax_rows(
        matrix: Matrix,
        causal: bool,
    ) -> Matrix:
        data = matrix.data
        rows, columns = data.shape

        result = np.zeros_like(data)

        for row_index in range(rows):
            if causal:
                allowed = np.arange(columns) <= row_index
            else:
                allowed = np.ones(
                    columns,
                    dtype=bool,
                )

            if causal and row_index >= columns:
                raise ValueError(
                    "Causal attention requires compatible "
                    "query and key sequence lengths."
                )

            result[row_index] = (
                masked_softmax(
                    data[row_index],
                    allowed,
                )
                if causal
                else softmax(
                    data[row_index]
                )
            )

        return Matrix(result)
