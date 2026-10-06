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
class CrossAttentionResult:
    qkv: QKV
    attention: AttentionResult


class CrossAttentionProjector:
    """Project queries from one sequence and keys/values from another."""

    def __init__(
        self,
        query_dimension: int,
        key_value_dimension: int,
        attention_dimension: int,
        seed: int = 42,
    ) -> None:
        for name, value in [
            ("query_dimension", query_dimension),
            ("key_value_dimension", key_value_dimension),
            ("attention_dimension", attention_dimension),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive.")

        self._query_dimension = query_dimension
        self._key_value_dimension = key_value_dimension
        self._attention_dimension = attention_dimension

        rng = np.random.default_rng(seed)
        self._query_weights = Matrix(
            rng.normal(
                0.0,
                1.0 / np.sqrt(query_dimension),
                size=(query_dimension, attention_dimension),
            )
        )
        self._key_weights = Matrix(
            rng.normal(
                0.0,
                1.0 / np.sqrt(key_value_dimension),
                size=(key_value_dimension, attention_dimension),
            )
        )
        self._value_weights = Matrix(
            rng.normal(
                0.0,
                1.0 / np.sqrt(key_value_dimension),
                size=(key_value_dimension, attention_dimension),
            )
        )

    def project(
        self,
        query_inputs: Matrix,
        key_value_inputs: Matrix,
    ) -> QKV:
        if query_inputs.columns != self._query_dimension:
            raise ValueError(
                "Query input dimension does not match the projector."
            )

        if key_value_inputs.columns != self._key_value_dimension:
            raise ValueError(
                "Key/value input dimension does not match the projector."
            )

        return QKV(
            query=query_inputs.multiply(self._query_weights),
            key=key_value_inputs.multiply(self._key_weights),
            value=key_value_inputs.multiply(self._value_weights),
        )


class CrossAttention:
    """Attention where Q comes from one sequence and K/V from another."""

    def __init__(
        self,
        query_dimension: int,
        key_value_dimension: int,
        attention_dimension: int,
        seed: int = 42,
    ) -> None:
        self._projector = CrossAttentionProjector(
            query_dimension=query_dimension,
            key_value_dimension=key_value_dimension,
            attention_dimension=attention_dimension,
            seed=seed,
        )
        self._attention = ScaledDotProductAttention(
            key_dimension=attention_dimension,
        )

    @property
    def projector(self) -> CrossAttentionProjector:
        return self._projector

    def forward(
        self,
        query_inputs: Matrix,
        key_value_inputs: Matrix,
        attention_mask: np.ndarray | None = None,
    ) -> CrossAttentionResult:
        qkv = self._projector.project(
            query_inputs,
            key_value_inputs,
        )
        result = self._attention.forward(
            qkv,
            attention_mask=attention_mask,
        )

        return CrossAttentionResult(
            qkv=qkv,
            attention=result,
        )
