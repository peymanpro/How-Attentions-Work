from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix


class TokenEmbeddingSequence:
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        seed: int = 42,
    ) -> None:
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive.")

        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive.")

        rng = np.random.default_rng(seed)

        scale = 1.0 / np.sqrt(embedding_dimension)

        self._embeddings = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(vocabulary_size, embedding_dimension),
            )
        )

    @property
    def embedding_dimension(self) -> int:
        return self._embeddings.columns

    def encode(
        self,
        token_ids: list[int],
    ) -> Matrix:
        if not token_ids:
            raise ValueError("token_ids cannot be empty.")

        if any(
            token_id < 0
            or token_id >= self._embeddings.rows
            for token_id in token_ids
        ):
            raise ValueError(
                "token_ids contain an invalid vocabulary ID."
            )

        return Matrix(
            self._embeddings.data[token_ids]
        )
