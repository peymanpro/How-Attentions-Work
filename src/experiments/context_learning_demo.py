from __future__ import annotations

from src.attention.context_task import (
    ContextDataset,
    ContextExample,
)
from src.experiments.context_learning_experiment import (
    ContextLearningExperiment,
)


def main() -> None:
    dataset = ContextDataset(
        [
            ContextExample(
                input_token_ids=[0, 2],
                target_token_id=3,
            ),
            ContextExample(
                input_token_ids=[1, 2],
                target_token_id=4,
            ),
        ]
    )

    result = ContextLearningExperiment().run(
        dataset=dataset,
        vocabulary_size=5,
        embedding_dimension=8,
        attention_dimension=8,
        epochs=300,
        learning_rate=0.01,
        seed=42,
    )

    print("Context-Dependent Attention Experiment")
    print("======================================")
    print()

    print(
        f"Initial Loss:       {result.initial_loss:.6f}"
    )

    print(
        f"Final Loss:         {result.final_loss:.6f}"
    )

    print(
        f"Loss Reduction:     "
        f"{result.initial_loss - result.final_loss:.6f}"
    )

    print()

    print("Parameter Changes")
    print("-----------------")
    print(
        f"Wq:                {result.q_weight_change:.6f}"
    )
    print(
        f"Wk:                {result.k_weight_change:.6f}"
    )
    print(
        f"Wv:                {result.v_weight_change:.6f}"
    )
    print(
        f"Wout:              {result.output_weight_change:.6f}"
    )

    print()

    print(
        f"Attention Change:   "
        f"{result.mean_attention_change:.6f}"
    )

    print()

    print("Initial Attention")
    print("------------------")
    print(result.initial_attention.weights.data)

    print()

    print("Final Attention")
    print("----------------")
    print(result.final_attention.weights.data)


if __name__ == "__main__":
    main()
