from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.masking import resolve_attention_mask
from src.attention.softmax import masked_softmax
from src.math.matrix import Matrix


@dataclass(frozen=True)
class AdditiveAttentionResult:
    output: Matrix
    weights: Matrix
    scores: Matrix


class AdditiveAttention:
    """Bahdanau-style attention using an additive scoring function."""

    def __init__(
        self,
        query_dimension: int,
        key_dimension: int,
        value_dimension: int,
        attention_dimension: int,
        seed: int = 42,
    ) -> None:
        for name, value in [
            ("query_dimension", query_dimension),
            ("key_dimension", key_dimension),
            ("value_dimension", value_dimension),
            ("attention_dimension", attention_dimension),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        self._query_dimension = query_dimension
        self._key_dimension = key_dimension
        self._value_dimension = value_dimension
        self._attention_dimension = attention_dimension

        rng = np.random.default_rng(seed)
        self._query_weights = rng.normal(
            0.0,
            1.0 / np.sqrt(query_dimension),
            size=(query_dimension, attention_dimension),
        )
        self._key_weights = rng.normal(
            0.0,
            1.0 / np.sqrt(key_dimension),
            size=(key_dimension, attention_dimension),
        )
        self._score_vector = rng.normal(
            0.0,
            1.0 / np.sqrt(attention_dimension),
            size=attention_dimension,
        )

    @property
    def attention_dimension(self) -> int:
        return self._attention_dimension

    def forward(
        self,
        query_inputs: Matrix,
        key_inputs: Matrix,
        value_inputs: Matrix,
        causal: bool = False,
        attention_mask: np.ndarray | None = None,
    ) -> AdditiveAttentionResult:
        if query_inputs.columns != self._query_dimension:
            raise ValueError(
                "Query input dimension does not match additive attention."
            )

        if key_inputs.columns != self._key_dimension:
            raise ValueError(
                "Key input dimension does not match additive attention."
            )

        if value_inputs.columns != self._value_dimension:
            raise ValueError(
                "Value input dimension does not match additive attention."
            )

        if key_inputs.rows != value_inputs.rows:
            raise ValueError(
                "Key and value sequence lengths must match."
            )

        query_projection = (
            query_inputs.data @ self._query_weights
        )
        key_projection = (
            key_inputs.data @ self._key_weights
        )

        scores = np.empty(
            (query_inputs.rows, key_inputs.rows),
            dtype=np.float64,
        )

        for query_index in range(query_inputs.rows):
            combined = (
                query_projection[query_index][None, :]
                + key_projection
            )
            scores[query_index] = np.tanh(
                combined
            ) @ self._score_vector

        score_matrix = Matrix(scores)
        mask = resolve_attention_mask(
            query_inputs.rows,
            key_inputs.rows,
            causal=causal,
            attention_mask=attention_mask,
        )

        weights = np.zeros_like(scores)

        for row_index in range(scores.shape[0]):
            weights[row_index] = masked_softmax(
                scores[row_index],
                mask[row_index],
            )

        return AdditiveAttentionResult(
            output=Matrix(weights @ value_inputs.data),
            weights=Matrix(weights),
            scores=score_matrix,
        )
