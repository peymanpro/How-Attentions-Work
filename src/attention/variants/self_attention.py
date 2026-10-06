from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.qkv import QKV, QKVProjector
from src.attention.scaled_dot_product import (
    AttentionResult,
    ScaledDotProductAttention,
)
from src.math.matrix import Matrix


@dataclass(frozen=True)
class SelfAttentionResult:
    qkv: QKV
    attention: AttentionResult


class SelfAttention:
    """Compose learned Q/K/V projections with scaled dot-product attention."""

    def __init__(
        self,
        model_dimension: int,
        attention_dimension: int | None = None,
        seed: int = 42,
    ) -> None:
        resolved_dimension = (
            model_dimension
            if attention_dimension is None
            else attention_dimension
        )

        self._projector = QKVProjector(
            model_dimension=model_dimension,
            attention_dimension=resolved_dimension,
            seed=seed,
        )
        self._attention = ScaledDotProductAttention(
            key_dimension=resolved_dimension,
        )

    @property
    def projector(self) -> QKVProjector:
        return self._projector

    @property
    def attention(self) -> ScaledDotProductAttention:
        return self._attention

    def forward(
        self,
        inputs: Matrix,
        causal: bool = False,
        attention_mask: np.ndarray | None = None,
    ) -> SelfAttentionResult:
        qkv = self._projector.project(inputs)
        result = self._attention.forward(
            qkv,
            causal=causal,
            attention_mask=attention_mask,
        )

        return SelfAttentionResult(
            qkv=qkv,
            attention=result,
        )
