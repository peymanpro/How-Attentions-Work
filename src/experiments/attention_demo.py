from __future__ import annotations

from src.attention.qkv import QKVProjector
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.attention.sequence import TokenEmbeddingSequence

TOKENS = [
    "the",
    "cat",
    "drinks",
    "milk",
]


def main() -> None:
    encoder = TokenEmbeddingSequence(
        vocabulary_size=len(TOKENS),
        embedding_dimension=8,
        seed=42,
    )

    sequence = encoder.encode(
        list(range(len(TOKENS)))
    )

    projector = QKVProjector(
        model_dimension=8,
        attention_dimension=8,
        seed=42,
    )

    qkv = projector.project(sequence)

    attention = ScaledDotProductAttention(
        key_dimension=8,
    )

    result = attention.forward(
        qkv,
        causal=True,
    )

    print("HowAttentionWorks")
    print("==================")
    print()
    print("Tokens:")
    print(TOKENS)
    print()
    print("Causal Attention Weights:")
    print()

    print(
        "       "
        + "".join(
            f"{token:>10}"
            for token in TOKENS
        )
    )

    for index, token in enumerate(TOKENS):
        row = result.weights.data[index]

        values = "".join(
            f"{value:10.4f}"
            for value in row
        )

        print(
            f"{token:>6}{values}"
        )


if __name__ == "__main__":
    main()
