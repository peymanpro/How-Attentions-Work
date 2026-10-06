from __future__ import annotations

import numpy as np

from src.attention.efficiency.grouped_query import (
    GroupedQueryAttention,
    GroupedQueryAttentionResult,
)
from src.math.matrix import Matrix


class MultiQueryAttention:
    """Grouped-query attention with a single shared K/V head."""

    def __init__(
        self,
        model_dimension: int,
        num_query_heads: int,
        seed: int = 42,
    ) -> None:
        self._attention = GroupedQueryAttention(
            model_dimension=model_dimension,
            num_query_heads=num_query_heads,
            num_key_value_heads=1,
            seed=seed,
        )

    @property
    def num_query_heads(self) -> int:
        return self._attention.num_query_heads

    @property
    def num_key_value_heads(self) -> int:
        return self._attention.num_key_value_heads

    def forward(
        self,
        inputs: Matrix,
        causal: bool = False,
        attention_mask: np.ndarray | None = None,
    ) -> GroupedQueryAttentionResult:
        return self._attention.forward(
            inputs,
            causal=causal,
            attention_mask=attention_mask,
        )
