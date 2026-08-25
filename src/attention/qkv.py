from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.matrix import Matrix


@dataclass(frozen=True)
class QKV:
    query: Matrix
    key: Matrix
    value: Matrix


class QKVProjector:
    def __init__(
        self,
        model_dimension: int,
        attention_dimension: int,
        seed: int = 42,
    ) -> None:
        if model_dimension <= 0:
            raise ValueError("model_dimension must be positive.")

        if attention_dimension <= 0:
            raise ValueError("attention_dimension must be positive.")

        self._model_dimension = model_dimension
        self._attention_dimension = attention_dimension

        rng = np.random.default_rng(seed)

        scale = 1.0 / np.sqrt(model_dimension)

        self._query_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(model_dimension, attention_dimension),
            )
        )

        self._key_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(model_dimension, attention_dimension),
            )
        )

        self._value_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(model_dimension, attention_dimension),
            )
        )

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

    @property
    def attention_dimension(self) -> int:
        return self._attention_dimension

    def project(self, inputs: Matrix) -> QKV:
        if inputs.columns != self._model_dimension:
            raise ValueError(
                "Input dimension does not match model dimension: "
                f"{inputs.columns} != {self._model_dimension}."
            )

        return QKV(
            query=inputs.multiply(self._query_weights),
            key=inputs.multiply(self._key_weights),
            value=inputs.multiply(self._value_weights),
        )
