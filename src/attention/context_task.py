from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextExample:
    input_token_ids: list[int]
    target_token_id: int


class ContextDataset:
    def __init__(
        self,
        examples: list[ContextExample],
    ) -> None:
        if not examples:
            raise ValueError(
                "Dataset cannot be empty."
            )

        self._examples = tuple(examples)

    @property
    def examples(self) -> tuple[ContextExample, ...]:
        return self._examples

    def __len__(self) -> int:
        return len(self._examples)
