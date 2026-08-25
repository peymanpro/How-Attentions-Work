from src.attention.context_task import (
    ContextDataset,
    ContextExample,
)
from src.attention.context_training import (
    ContextAttentionTrainer,
)
from src.attention.output_projection import (
    OutputProjection,
)
from src.attention.qkv import QKVProjector
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.attention.sequence import (
    TokenEmbeddingSequence,
)


def create_trainer() -> ContextAttentionTrainer:
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

    return ContextAttentionTrainer(
        sequence=sequence,
        projector=projector,
        attention=attention,
        output_projection=output_projection,
        learning_rate=0.01,
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


def test_context_training_should_reduce_loss() -> None:
    trainer = create_trainer()

    history = trainer.train(
        dataset=create_dataset(),
        epochs=300,
    )

    assert len(history) == 300

    assert (
        history[-1].average_loss
        < history[0].average_loss
    )


def test_context_training_should_return_one_result_per_epoch() -> None:
    trainer = create_trainer()

    history = trainer.train(
        dataset=create_dataset(),
        epochs=10,
    )

    assert [item.epoch for item in history] == list(
        range(1, 11)
    )
