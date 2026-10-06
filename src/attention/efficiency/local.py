from __future__ import annotations

from src.attention.masking import sliding_window_mask
from src.attention.qkv import QKVProjector
from src.attention.variants.self_attention import (
    SelfAttention,
    SelfAttentionResult,
)
from src.math.matrix import Matrix


class LocalSelfAttention:
    """Self-attention restricted to a sliding local context window."""

    def __init__(
        self,
        model_dimension: int,
        attention_dimension: int | None = None,
        window_size: int = 2,
        seed: int = 42,
    ) -> None:
        self._window_size = window_size
        self._self_attention = SelfAttention(
            model_dimension=model_dimension,
            attention_dimension=attention_dimension,
            seed=seed,
        )

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def projector(self) -> QKVProjector:
        return self._self_attention.projector

    def forward(
        self,
        inputs: Matrix,
        causal: bool = True,
    ) -> SelfAttentionResult:
        mask = sliding_window_mask(
            inputs.rows,
            self._window_size,
            causal=causal,
        )
        return self._self_attention.forward(
            inputs,
            causal=False,
            attention_mask=mask,
        )
