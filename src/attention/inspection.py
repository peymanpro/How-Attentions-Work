from __future__ import annotations

from src.attention.attention_snapshot import (
    AttentionSnapshot,
)
from src.attention.qkv import QKVProjector
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.attention.sequence import TokenEmbeddingSequence


class AttentionInspector:
    def __init__(
        self,
        sequence: TokenEmbeddingSequence,
        projector: QKVProjector,
        attention: ScaledDotProductAttention,
    ) -> None:
        self._sequence = sequence
        self._projector = projector
        self._attention = attention

    def capture(
        self,
        token_ids: list[int],
        causal: bool = True,
    ) -> AttentionSnapshot:
        inputs = self._sequence.encode(token_ids)

        qkv = self._projector.project(inputs)

        result = self._attention.forward(
            qkv,
            causal=causal,
        )

        return AttentionSnapshot(
            weights=result.weights
        )
