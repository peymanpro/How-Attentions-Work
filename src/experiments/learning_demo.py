from __future__ import annotations

import numpy as np

from src.experiments.learning_experiment import (
    AttentionLearningExperiment,
)

TOKENS = [
    "the",
    "cat",
    "drinks",
    "milk",
    "<eos>",
]


def print_attention(
    title: str,
    weights: np.ndarray,
) -> None:
    print()
    print(title)
    print("-" * len(title))
    print()

    print(
        "       "
        + "".join(
            f"{token:>10}"
            for token in TOKENS[:-1]
        )
    )

    for index, token in enumerate(TOKENS[:-1]):
        row = weights[index]

        print(
            f"{token:>7}"
            + "".join(
                f"{value:10.4f}"
                for value in row
            )
        )


def main() -> None:
    result = AttentionLearningExperiment().run(
        input_token_ids=[0, 1, 2, 3],
        target_token_ids=[1, 2, 3, 4],
        vocabulary_size=len(TOKENS),
        embedding_dimension=8,
        attention_dimension=8,
        epochs=200,
        learning_rate=0.01,
        seed=42,
    )

    print("HowAttentionWorks")
    print("=================")

    print(
        f"Initial Loss: {result.initial_loss:.6f}"
    )

    print(
        f"Final Loss:   {result.final_loss:.6f}"
    )

    print(
        f"Reduction:    "
        f"{(
            result.initial_loss
            - result.final_loss
        ):.6f}"
    )

    print(
        f"Attention Change: "
        f"{result.mean_attention_change:.6f}"
    )

    print_attention(
        "Attention Before Training",
        result.initial_attention.weights.data,
    )

    print_attention(
        "Attention After Training",
        result.final_attention.weights.data,
    )


if __name__ == "__main__":
    main()
