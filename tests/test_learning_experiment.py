from src.experiments.learning_experiment import (
    AttentionLearningExperiment,
)


def test_experiment_should_reduce_loss() -> None:
    result = AttentionLearningExperiment().run(
        input_token_ids=[0, 1, 2, 3],
        target_token_ids=[1, 2, 3, 4],
        vocabulary_size=5,
        embedding_dimension=8,
        attention_dimension=8,
        epochs=200,
        learning_rate=0.01,
        seed=42,
    )

    assert result.final_loss < result.initial_loss


def test_experiment_should_change_attention_weights() -> None:
    result = AttentionLearningExperiment().run(
        input_token_ids=[0, 1, 2, 3],
        target_token_ids=[1, 2, 3, 4],
        vocabulary_size=5,
        embedding_dimension=8,
        attention_dimension=8,
        epochs=200,
        learning_rate=0.01,
        seed=42,
    )

    assert result.mean_attention_change > 0.0


def test_experiment_should_keep_causal_attention_structure() -> None:
    result = AttentionLearningExperiment().run(
        input_token_ids=[0, 1, 2, 3],
        target_token_ids=[1, 2, 3, 4],
        vocabulary_size=5,
        embedding_dimension=8,
        attention_dimension=8,
        epochs=50,
        learning_rate=0.01,
        seed=42,
    )

    weights = result.final_attention.weights.data

    assert weights[0, 1] == 0.0
    assert weights[0, 2] == 0.0
    assert weights[0, 3] == 0.0

    assert weights[1, 2] == 0.0
    assert weights[1, 3] == 0.0

    assert weights[2, 3] == 0.0
