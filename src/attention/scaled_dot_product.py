from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.masking import resolve_attention_mask
from src.attention.qkv import QKV
from src.attention.softmax import masked_softmax
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
        attention_mask: np.ndarray | None = None,
    ) -> AttentionResult:
        self._validate_shapes(qkv)

        scores = qkv.query.multiply(
            qkv.key.transpose()
        )

        scaled_scores = scores.multiply_scalar(
            1.0 / self._key_dimension**0.5
        )

        mask = resolve_attention_mask(
            qkv.query.rows,
            qkv.key.rows,
            causal=causal,
            attention_mask=attention_mask,
        )

        weights = self._softmax_rows(
            scaled_scores,
            mask,
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
        mask: np.ndarray,
    ) -> Matrix:
        data = matrix.data
        allowed = mask

        result = np.zeros_like(data)

        for row_index in range(data.shape[0]):
            result[row_index] = masked_softmax(
                data[row_index],
                allowed[row_index],
            )

        return Matrix(result)

    def _validate_shapes(
        self,
        qkv: QKV,
    ) -> None:
        if qkv.query.columns != self._key_dimension:
            raise ValueError(
                "Query dimension must match key_dimension."
            )

        if qkv.key.columns != self._key_dimension:
            raise ValueError(
                "Key dimension must match key_dimension."
            )

        if qkv.key.rows != qkv.value.rows:
            raise ValueError(
                "Key and value sequence lengths must match."
            )
