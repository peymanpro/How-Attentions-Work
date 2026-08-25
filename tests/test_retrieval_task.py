import pytest

from src.attention.retrieval_task import (
    RetrievalDataset,
    RetrievalExample,
)


def test_retrieval_dataset_should_store_examples() -> None:
    example = RetrievalExample(
        input_token_ids=(0, 1, 2, 3, 4),
        query_position=4,
        target_value_position=3,
    )

    dataset = RetrievalDataset([example])

    assert len(dataset) == 1
    assert dataset.examples[0] == example


def test_retrieval_dataset_should_expose_immutable_examples() -> None:
    dataset = RetrievalDataset(
        [
            RetrievalExample(
                input_token_ids=(0, 1),
                query_position=1,
                target_value_position=0,
            )
        ]
    )

    assert isinstance(
        dataset.examples,
        tuple,
    )


def test_retrieval_dataset_should_reject_empty_dataset() -> None:
    with pytest.raises(ValueError):
        RetrievalDataset([])


def test_retrieval_dataset_should_reject_invalid_query_position() -> None:
    with pytest.raises(IndexError):
        RetrievalDataset(
            [
                RetrievalExample(
                    input_token_ids=(0, 1),
                    query_position=2,
                    target_value_position=0,
                )
            ]
        )


def test_retrieval_dataset_should_reject_invalid_target_position() -> None:
    with pytest.raises(IndexError):
        RetrievalDataset(
            [
                RetrievalExample(
                    input_token_ids=(0, 1),
                    query_position=1,
                    target_value_position=2,
                )
            ]
        )


def test_retrieval_dataset_should_require_target_before_query() -> None:
    with pytest.raises(ValueError):
        RetrievalDataset(
            [
                RetrievalExample(
                    input_token_ids=(0, 1, 2),
                    query_position=1,
                    target_value_position=2,
                )
            ]
        )


def test_retrieval_example_should_allow_first_token_as_target() -> None:
    dataset = RetrievalDataset(
        [
            RetrievalExample(
                input_token_ids=(0, 1, 2),
                query_position=2,
                target_value_position=0,
            )
        ]
    )

    assert (
        dataset.examples[0].target_value_position
        == 0
    )
