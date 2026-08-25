from src.attention.inspection import AttentionInspector
from src.attention.qkv import QKVProjector
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.attention.sequence import TokenEmbeddingSequence


def test_attention_inspector_should_capture_weights() -> None:
    sequence = TokenEmbeddingSequence(
        vocabulary_size=5,
        embedding_dimension=8,
        seed=42,
    )

    projector = QKVProjector(
        model_dimension=8,
        attention_dimension=8,
        seed=42,
    )

    attention = ScaledDotProductAttention(
        key_dimension=8,
    )

    inspector = AttentionInspector(
        sequence=sequence,
        projector=projector,
        attention=attention,
    )

    snapshot = inspector.capture(
        [0, 1, 2, 3]
    )

    assert snapshot.weights.shape == (4, 4)
