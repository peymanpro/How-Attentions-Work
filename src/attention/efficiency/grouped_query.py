from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.qkv import QKV
from src.attention.scaled_dot_product import (
    AttentionResult,
    ScaledDotProductAttention,
)
from src.math.matrix import Matrix


@dataclass(frozen=True)
class GroupedQueryAttentionResult:
    output: Matrix
    concatenated: Matrix
    heads: tuple[AttentionResult, ...]


class _QueryProjector:
    def __init__(
        self,
        input_dimension: int,
        head_dimension: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        self._weights = Matrix(
            rng.normal(
                0.0,
                1.0 / np.sqrt(input_dimension),
                size=(input_dimension, head_dimension),
            )
        )

    def project(self, inputs: Matrix) -> Matrix:
        return inputs.multiply(self._weights)


class _KeyValueProjector:
    def __init__(
        self,
        input_dimension: int,
        head_dimension: int,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(input_dimension)
        self._key_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(input_dimension, head_dimension),
            )
        )
        self._value_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(input_dimension, head_dimension),
            )
        )

    def project(self, inputs: Matrix) -> tuple[Matrix, Matrix]:
        return (
            inputs.multiply(self._key_weights),
            inputs.multiply(self._value_weights),
        )


class GroupedQueryAttention:
    """Self-attention with fewer K/V heads than query heads."""

    def __init__(
        self,
        model_dimension: int,
        num_query_heads: int,
        num_key_value_heads: int,
        seed: int = 42,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError("model_dimension must be positive.")

        if num_query_heads <= 0:
            raise ValueError("num_query_heads must be positive.")

        if num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive.")

        if model_dimension % num_query_heads != 0:
            raise ValueError(
                "model_dimension must be divisible by num_query_heads."
            )

        if num_query_heads % num_key_value_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_key_value_heads."
            )

        self._model_dimension = model_dimension
        self._num_query_heads = num_query_heads
        self._num_key_value_heads = num_key_value_heads
        self._head_dimension = model_dimension // num_query_heads
        self._group_size = num_query_heads // num_key_value_heads

        self._query_projectors = tuple(
            _QueryProjector(
                model_dimension,
                self._head_dimension,
                seed + 1009 * (index + 1),
            )
            for index in range(num_query_heads)
        )

        self._key_value_projectors = tuple(
            _KeyValueProjector(
                model_dimension,
                self._head_dimension,
                seed + 2003 * (index + 1),
            )
            for index in range(num_key_value_heads)
        )

        rng = np.random.default_rng(seed + 7919)
        self._output_weights = Matrix(
            rng.normal(
                0.0,
                1.0 / np.sqrt(model_dimension),
                size=(model_dimension, model_dimension),
            )
        )

    @property
    def num_query_heads(self) -> int:
        return self._num_query_heads

    @property
    def num_key_value_heads(self) -> int:
        return self._num_key_value_heads

    def forward(
        self,
        inputs: Matrix,
        causal: bool = False,
        attention_mask: np.ndarray | None = None,
    ) -> GroupedQueryAttentionResult:
        if inputs.columns != self._model_dimension:
            raise ValueError(
                "Input dimension does not match grouped-query attention."
            )

        kv_pairs = tuple(
            projector.project(inputs)
            for projector in self._key_value_projectors
        )

        heads: list[AttentionResult] = []

        for index, query_projector in enumerate(
            self._query_projectors
        ):
            query = query_projector.project(inputs)
            group_index = index // self._group_size
            key, value = kv_pairs[group_index]

            result = ScaledDotProductAttention(
                key_dimension=self._head_dimension
            ).forward(
                QKV(
                    query=query,
                    key=key,
                    value=value,
                ),
                causal=causal,
                attention_mask=attention_mask,
            )
            heads.append(result)

        concatenated = Matrix(
            np.concatenate(
                [head.output.data for head in heads],
                axis=1,
            )
        )

        output = concatenated.multiply(
            self._output_weights
        )

        return GroupedQueryAttentionResult(
            output=output,
            concatenated=concatenated,
            heads=tuple(heads),
        )
