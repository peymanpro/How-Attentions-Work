import pytest

from src.attention.sequence import TokenEmbeddingSequence


def test_sequence_encoder_should_return_one_embedding_per_token() -> None:
    encoder = TokenEmbeddingSequence(
        vocabulary_size=10,
        embedding_dimension=4,
    )

    result = encoder.encode(
        [1, 3, 5, 7]
    )

    assert result.shape == (4, 4)


def test_sequence_encoder_should_preserve_token_order() -> None:
    encoder = TokenEmbeddingSequence(
        vocabulary_size=10,
        embedding_dimension=4,
        seed=42,
    )

    first = encoder.encode([1, 3])
    second = encoder.encode([3, 1])

    assert first.shape == second.shape

    assert not (
        first.data == second.data
    ).all()


def test_sequence_encoder_should_reject_empty_sequence() -> None:
    encoder = TokenEmbeddingSequence(
        vocabulary_size=10,
        embedding_dimension=4,
    )

    with pytest.raises(ValueError):
        encoder.encode([])


def test_sequence_encoder_should_reject_invalid_token_id() -> None:
    encoder = TokenEmbeddingSequence(
        vocabulary_size=10,
        embedding_dimension=4,
    )

    with pytest.raises(ValueError):
        encoder.encode([1, 10])
def test_sequence_encoder_should_return_individual_embedding() -> None:
    encoder = TokenEmbeddingSequence(
        vocabulary_size=5,
        embedding_dimension=4,
        seed=42,
    )

    sequence = encoder.encode([2])

    individual = encoder.get_embedding(2)

    assert individual.shape == (1, 4)

    assert (
        individual.data == sequence.data
    ).all()


def test_sequence_encoder_should_reject_invalid_embedding_id() -> None:
    encoder = TokenEmbeddingSequence(
        vocabulary_size=5,
        embedding_dimension=4,
    )

    with pytest.raises(ValueError):
        encoder.get_embedding(5)
