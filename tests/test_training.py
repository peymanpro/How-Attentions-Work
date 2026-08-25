from src.attention.output_projection import OutputProjection
from src.attention.qkv import QKVProjector
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.attention.sequence import TokenEmbeddingSequence
from src.attention.training import AttentionTrainer


def create_trainer() -> AttentionTrainer:
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

    output_projection = OutputProjection(
        input_dimension=8,
        vocabulary_size=5,
        seed=42,
    )

    return AttentionTrainer(
        sequence=sequence,
        projector=projector,
        attention=attention,
        output_projection=output_projection,
        learning_rate=0.01,
    )


def test_training_should_reduce_loss() -> None:
    trainer = create_trainer()

    history = trainer.train(
        input_token_ids=[0, 1, 2, 3],
        target_token_ids=[1, 2, 3, 4],
        epochs=200,
    )

    assert len(history) == 200

    assert (
        history[-1].average_loss
        < history[0].average_loss
    )


def test_training_should_produce_one_result_per_epoch() -> None:
    trainer = create_trainer()

    history = trainer.train(
        input_token_ids=[0, 1, 2, 3],
        target_token_ids=[1, 2, 3, 4],
        epochs=10,
    )

    assert [item.epoch for item in history] == list(
        range(1, 11)
    )
