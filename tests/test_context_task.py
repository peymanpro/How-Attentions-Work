import pytest

from src.attention.context_task import (
    ContextDataset,
    ContextExample,
)


def test_context_dataset_should_store_examples() -> None:
    dataset = ContextDataset(
        [
            ContextExample(
                input_token_ids=[0, 1],
                target_token_id=2,
            ),
            ContextExample(
                input_token_ids=[3, 1],
                target_token_id=4,
            ),
        ]
    )

    assert len(dataset) == 2
    assert dataset.examples[0].target_token_id == 2


def test_context_dataset_should_be_immutable_from_outside() -> None:
    dataset = ContextDataset(
        [
            ContextExample(
                input_token_ids=[0, 1],
                target_token_id=2,
            )
        ]
    )

    assert isinstance(dataset.examples, tuple)


def test_context_dataset_should_reject_empty_examples() -> None:
    with pytest.raises(ValueError):
        ContextDataset([])


def test_context_task_should_represent_context_dependent_targets() -> None:
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

    first = dataset.examples[0]
    second = dataset.examples[1]

    assert first.input_token_ids == [0, 2]
    assert first.target_token_id == 3

    assert second.input_token_ids == [1, 2]
    assert second.target_token_id == 4

    assert first.target_token_id != second.target_token_id
