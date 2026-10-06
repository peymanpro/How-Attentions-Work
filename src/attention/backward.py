from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.masking import resolve_attention_mask
from src.attention.qkv import QKV
from src.attention.scaled_dot_product import AttentionResult
from src.math.matrix import Matrix


@dataclass(frozen=True)
class AttentionGradients:
    query: Matrix
    key: Matrix
    value: Matrix


class ScaledDotProductAttentionBackward:
    def __init__(self, key_dimension: int) -> None:
        if key_dimension <= 0:
            raise ValueError(
                "key_dimension must be positive."
            )

        self._key_dimension = key_dimension

    def backward(
        self,
        qkv: QKV,
        forward_result: AttentionResult,
        output_gradient: Matrix,
        causal: bool = False,
        attention_mask: np.ndarray | None = None,
    ) -> AttentionGradients:
        self._validate_shapes(
            qkv,
            forward_result,
            output_gradient,
        )

        q = qkv.query.data
        k = qkv.key.data
        v = qkv.value.data

        attention_weights = forward_result.weights.data
        d_output = output_gradient.data

        mask = resolve_attention_mask(
            qkv.query.rows,
            qkv.key.rows,
            causal=causal,
            attention_mask=None
            if attention_mask is None
            else attention_mask,
        )

        d_value = attention_weights.T @ d_output
        d_weights = d_output @ v.T

        d_scores = np.zeros_like(d_weights)

        for row_index in range(d_weights.shape[0]):
            allowed = mask[row_index]
            valid_weights = attention_weights[row_index, allowed]
            valid_gradient = d_weights[row_index, allowed]
            dot_product = np.sum(
                valid_weights * valid_gradient
            )
            d_scores[row_index, allowed] = (
                valid_weights
                * (valid_gradient - dot_product)
            )

        scale = self._key_dimension**0.5

        d_query = d_scores @ k / scale
        d_key = d_scores.T @ q / scale

        return AttentionGradients(
            query=Matrix(d_query),
            key=Matrix(d_key),
            value=Matrix(d_value),
        )

    def _validate_shapes(
        self,
        qkv: QKV,
        forward_result: AttentionResult,
        output_gradient: Matrix,
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

        if (
            forward_result.weights.shape
            != (qkv.query.rows, qkv.key.rows)
        ):
            raise ValueError(
                "Attention weights shape is invalid."
            )

        if (
            forward_result.output.shape
            != output_gradient.shape
        ):
            raise ValueError(
                "Output gradient shape must match attention output."
            )
