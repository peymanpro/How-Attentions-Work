from src.attention.context_task import (
    ContextDataset,
    ContextExample,
)
from src.experiments.context_learning_experiment import (
    ContextLearningExperiment,
)


def create_dataset() -> ContextDataset:
    return ContextDataset(
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


def test_context_experiment_should_reduce_loss() -> None:
    result = ContextLearningExperiment().run(
        dataset=create_dataset(),
        vocabulary_size=5,
        epochs=300,
        learning_rate=0.01,
        seed=42,
    )

    assert result.final_loss < result.initial_loss


def test_context_experiment_should_change_parameters() -> None:
    result = ContextLearningExperiment().run(
        dataset=create_dataset(),
        vocabulary_size=5,
        epochs=300,
        learning_rate=0.01,
        seed=42,
    )

    assert result.q_weight_change > 0.0
    assert result.k_weight_change > 0.0
    assert result.v_weight_change > 0.0
    assert result.output_weight_change > 0.0


def test_context_experiment_should_change_attention() -> None:
    result = ContextLearningExperiment().run(
        dataset=create_dataset(),
        vocabulary_size=5,
        epochs=300,
        learning_rate=0.01,
        seed=42,
    )

    assert result.mean_attention_change > 0.0
