from __future__ import annotations

from src.attention.qkv import QKVProjector
from src.attention.retrieval_task import (
    RetrievalDataset,
    RetrievalExample,
)
from src.attention.retrieval_training import (
    QKRetrievalTrainer,
)
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.attention.sequence import (
    TokenEmbeddingSequence,
)

TOKENS = [
    "red",
    "sky",
    "blue",
    "ocean",
    "query-red",
    "query-blue",
]


def main() -> None:
    dataset = RetrievalDataset(
        [
            RetrievalExample(
                input_token_ids=(
                    0,
                    1,
                    2,
                    3,
                    4,
                ),
                query_position=4,
                target_value_position=1,
            ),
            RetrievalExample(
                input_token_ids=(
                    0,
                    1,
                    2,
                    3,
                    5,
                ),
                query_position=4,
                target_value_position=3,
            ),
        ]
    )

    sequence = TokenEmbeddingSequence(
        vocabulary_size=len(TOKENS),
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

    trainer = QKRetrievalTrainer(
        sequence=sequence,
        projector=projector,
        attention=attention,
        learning_rate=0.05,
    )

    before_weights = []

    for example in dataset.examples:
        inputs = sequence.encode(
            list(example.input_token_ids)
        )

        qkv = projector.project(inputs)

        result = attention.forward(
            qkv,
            causal=True,
        )

        before_weights.append(
            result.weights.data[
                example.query_position
            ]
        )

    history = trainer.train(
        dataset=dataset,
        epochs=500,
    )

    after_weights = []

    for example in dataset.examples:
        inputs = sequence.encode(
            list(example.input_token_ids)
        )

        qkv = projector.project(inputs)

        result = attention.forward(
            qkv,
            causal=True,
        )

        after_weights.append(
            result.weights.data[
                example.query_position
            ]
        )

    print("Q/K-Only Attention Retrieval")
    print("============================")
    print()

    print(
        f"Initial Loss: {history[0].average_loss:.8f}"
    )

    print(
        f"Final Loss:   {history[-1].average_loss:.8f}"
    )

    print(
        f"Reduction:    "
        f"{history[0].average_loss - history[-1].average_loss:.8f}"
    )

    print()

    print("Retrieval Targets")
    print("-----------------")
    print("query-red  -> sky")
    print("query-blue -> ocean")

    print()

    print(
        "Query-red initial:"
    )
    print(
        [
            round(float(value), 6)
            for value in before_weights[0]
        ]
    )

    print(
        "Query-red final:"
    )
    print(
        [
            round(float(value), 6)
            for value in after_weights[0]
        ]
    )

    print()

    print(
        "Query-blue initial:"
    )
    print(
        [
            round(float(value), 6)
            for value in before_weights[1]
        ]
    )

    print(
        "Query-blue final:"
    )
    print(
        [
            round(float(value), 6)
            for value in after_weights[1]
        ]
    )


if __name__ == "__main__":
    main()

