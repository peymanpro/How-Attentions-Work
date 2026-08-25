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
            raise ValueError(
                "model_dimension must be positive."
            )

        if attention_dimension <= 0:
            raise ValueError(
                "attention_dimension must be positive."
            )

        self._model_dimension = model_dimension
        self._attention_dimension = attention_dimension

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(model_dimension)

        self._query_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(
                    model_dimension,
                    attention_dimension,
                ),
            )
        )

        self._key_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(
                    model_dimension,
                    attention_dimension,
                ),
            )
        )

        self._value_weights = Matrix(
            rng.normal(
                0.0,
                scale,
                size=(
                    model_dimension,
                    attention_dimension,
                ),
            )
        )

    @property
    def model_dimension(self) -> int:
        return self._model_dimension

    @property
    def attention_dimension(self) -> int:
        return self._attention_dimension

    @property
    def query_weights(self) -> Matrix:
        return Matrix(self._query_weights.data)

    @property
    def key_weights(self) -> Matrix:
        return Matrix(self._key_weights.data)

    @property
    def value_weights(self) -> Matrix:
        return Matrix(self._value_weights.data)

    def project(
        self,
        inputs: Matrix,
    ) -> QKV:
        self._validate_input(inputs)

        return QKV(
            query=inputs.multiply(
                self._query_weights
            ),
            key=inputs.multiply(
                self._key_weights
            ),
            value=inputs.multiply(
                self._value_weights
            ),
        )

    def apply_gradients(
        self,
        query_gradient: Matrix,
        key_gradient: Matrix,
        value_gradient: Matrix,
        learning_rate: float,
    ) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive and finite."
            )

        expected_shape = (
            self._model_dimension,
            self._attention_dimension,
        )

        if query_gradient.shape != expected_shape:
            raise ValueError(
                "Query gradient shape does not match query weights."
            )

        if key_gradient.shape != expected_shape:
            raise ValueError(
                "Key gradient shape does not match key weights."
            )

        if value_gradient.shape != expected_shape:
            raise ValueError(
                "Value gradient shape does not match value weights."
            )

        self._query_weights = self._query_weights.add(
            query_gradient.multiply_scalar(
                -learning_rate
            )
        )

        self._key_weights = self._key_weights.add(
            key_gradient.multiply_scalar(
                -learning_rate
            )
        )

        self._value_weights = self._value_weights.add(
            value_gradient.multiply_scalar(
                -learning_rate
            )
        )

    def _validate_input(
        self,
        inputs: Matrix,
    ) -> None:
        if inputs.columns != self._model_dimension:
            raise ValueError(
                "Input dimension does not match model dimension: "
                f"{inputs.columns} != {self._model_dimension}."
            )
