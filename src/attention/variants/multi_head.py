from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.qkv import QKVProjector
from src.attention.scaled_dot_product import (
    AttentionResult,
    ScaledDotProductAttention,
)
from src.attention.variants.cross_attention import (
    CrossAttentionProjector,
)
from src.math.matrix import Matrix


@dataclass(frozen=True)
class MultiHeadAttentionResult:
    output: Matrix
    concatenated: Matrix
    heads: tuple[AttentionResult, ...]


class MultiHeadAttention:
    """Multi-head attention over self- or cross-attention inputs."""

    def __init__(
        self,
        query_dimension: int,
        num_heads: int,
        key_value_dimension: int | None = None,
        seed: int = 42,
    ) -> None:
        if query_dimension <= 0:
            raise ValueError(
                "query_dimension must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if query_dimension % num_heads != 0:
            raise ValueError(
                "query_dimension must be divisible by num_heads."
            )

        if key_value_dimension is not None and key_value_dimension <= 0:
            raise ValueError(
                "key_value_dimension must be positive."
            )

        self._query_dimension = query_dimension
        self._num_heads = num_heads
        self._head_dimension = query_dimension // num_heads
        self._key_value_dimension = key_value_dimension

        if key_value_dimension is None:
            self._self_projectors = tuple(
                QKVProjector(
                    model_dimension=query_dimension,
                    attention_dimension=self._head_dimension,
                    seed=seed + 1009 * (index + 1),
                )
                for index in range(num_heads)
            )
            self._cross_projectors: tuple[CrossAttentionProjector, ...] = ()
        else:
            self._self_projectors = ()
            self._cross_projectors = tuple(
                CrossAttentionProjector(
                    query_dimension=query_dimension,
                    key_value_dimension=key_value_dimension,
                    attention_dimension=self._head_dimension,
                    seed=seed + 1009 * (index + 1),
                )
                for index in range(num_heads)
            )

        rng = np.random.default_rng(seed + 7919)
        self._output_weights = Matrix(
            rng.normal(
                0.0,
                1.0 / np.sqrt(query_dimension),
                size=(query_dimension, query_dimension),
            )
        )

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dimension(self) -> int:
        return self._head_dimension

    @property
    def output_weights(self) -> Matrix:
        return Matrix(self._output_weights.data)

    def forward(
        self,
        query_inputs: Matrix,
        key_value_inputs: Matrix | None = None,
        causal: bool = False,
        attention_mask: np.ndarray | None = None,
    ) -> MultiHeadAttentionResult:
        if query_inputs.columns != self._query_dimension:
            raise ValueError(
                "Query input dimension does not match multi-head attention."
            )

        if self._key_value_dimension is None:
            if key_value_inputs is not None:
                raise ValueError(
                    "Self-attention mode does not accept separate key/value inputs."
                )

            heads = tuple(
                self._run_self_head(
                    projector,
                    query_inputs,
                    causal,
                    attention_mask,
                )
                for projector in self._self_projectors
            )
        else:
            if key_value_inputs is None:
                raise ValueError(
                    "Cross-attention mode requires key/value inputs."
                )

            if key_value_inputs.columns != self._key_value_dimension:
                raise ValueError(
                    "Key/value input dimension does not match multi-head attention."
                )

            if causal:
                raise ValueError(
                    "Causal masking is only defined here for self-attention mode."
                )

            heads = tuple(
                self._run_cross_head(
                    projector,
                    query_inputs,
                    key_value_inputs,
                    attention_mask,
                )
                for projector in self._cross_projectors
            )

        concatenated = Matrix(
            np.concatenate(
                [head.output.data for head in heads],
                axis=1,
            )
        )

        output = concatenated.multiply(
            self._output_weights
        )

        return MultiHeadAttentionResult(
            output=output,
            concatenated=concatenated,
            heads=heads,
        )

    @staticmethod
    def _run_self_head(
        projector: QKVProjector,
        inputs: Matrix,
        causal: bool,
        attention_mask: np.ndarray | None,
    ) -> AttentionResult:
        qkv = projector.project(inputs)
        return ScaledDotProductAttention(
            key_dimension=qkv.query.columns
        ).forward(
            qkv,
            causal=causal,
            attention_mask=attention_mask,
        )

    @staticmethod
    def _run_cross_head(
        projector: CrossAttentionProjector,
        query_inputs: Matrix,
        key_value_inputs: Matrix,
        attention_mask: np.ndarray | None,
    ) -> AttentionResult:
        qkv = projector.project(
            query_inputs,
            key_value_inputs,
        )
        return ScaledDotProductAttention(
            key_dimension=qkv.query.columns
        ).forward(
            qkv,
            attention_mask=attention_mask,
        )
