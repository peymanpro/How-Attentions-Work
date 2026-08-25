from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalExample:
    input_token_ids: tuple[int, ...]
    query_position: int
    target_value_position: int


class RetrievalDataset:
    def __init__(
        self,
        examples: list[RetrievalExample],
    ) -> None:
        if not examples:
            raise ValueError(
                "Retrieval dataset cannot be empty."
            )

        for example in examples:
            sequence_length = len(
                example.input_token_ids
            )

            if sequence_length == 0:
                raise ValueError(
                    "Retrieval examples cannot be empty."
                )

            if not (
                0
                <= example.query_position
                < sequence_length
            ):
                raise IndexError(
                    "Query position is outside the sequence."
                )

            if not (
                0
                <= example.target_value_position
                < sequence_length
            ):
                raise IndexError(
                    "Target value position is outside the sequence."
                )

            if (
                example.target_value_position
                >= example.query_position
            ):
                raise ValueError(
                    "Target value must occur before the query "
                    "for causal retrieval."
                )

        self._examples = tuple(examples)

    @property
    def examples(self) -> tuple[RetrievalExample, ...]:
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)
