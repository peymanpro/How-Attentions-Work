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


def create_dataset() -> RetrievalDataset:
    return RetrievalDataset(
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


def create_trainer() -> QKRetrievalTrainer:
    sequence = TokenEmbeddingSequence(
        vocabulary_size=6,
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

    return QKRetrievalTrainer(
        sequence=sequence,
        projector=projector,
        attention=attention,
        learning_rate=0.05,
    )


def test_qk_retrieval_training_should_reduce_loss() -> None:
    trainer = create_trainer()

    history = trainer.train(
        dataset=create_dataset(),
        epochs=500,
    )

    assert len(history) == 500

    assert (
        history[-1].average_loss
        < history[0].average_loss
    )


def test_qk_retrieval_training_should_produce_one_result_per_epoch() -> None:
    trainer = create_trainer()

    history = trainer.train(
        dataset=create_dataset(),
        epochs=10,
    )

    assert [item.epoch for item in history] == list(
        range(1, 11)
    )
def test_retrieval_target_should_exist_in_value_space() -> None:
    sequence = TokenEmbeddingSequence(
        vocabulary_size=6,
        embedding_dimension=8,
        seed=42,
    )

    projector = QKVProjector(
        model_dimension=8,
        attention_dimension=8,
        seed=42,
    )

    inputs = sequence.encode(
        [0, 1, 2, 3, 4]
    )

    qkv = projector.project(inputs)

    target = qkv.value.data[1:2]

    assert target.shape == (1, 8)

    assert (
        target != sequence.get_embedding(1).data
    ).any()
