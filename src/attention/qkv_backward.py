from __future__ import annotations

from dataclasses import dataclass

from src.attention.backward import AttentionGradients
from src.math.matrix import Matrix


@dataclass(frozen=True)
class QKVWeightGradients:
    query: Matrix
    key: Matrix
    value: Matrix


class QKVBackward:
    def backward(
        self,
        inputs: Matrix,
        attention_gradients: AttentionGradients,
    ) -> QKVWeightGradients:
        if inputs.rows != attention_gradients.query.rows:
            raise ValueError(
                "Input and query gradient row counts must match."
            )

        if inputs.rows != attention_gradients.key.rows:
            raise ValueError(
                "Input and key gradient row counts must match."
            )

        if inputs.rows != attention_gradients.value.rows:
            raise ValueError(
                "Input and value gradient row counts must match."
            )

        input_transpose = inputs.transpose()

        query_gradient = input_transpose.multiply(
            attention_gradients.query
        )

        key_gradient = input_transpose.multiply(
            attention_gradients.key
        )

        value_gradient = input_transpose.multiply(
            attention_gradients.value
        )

        return QKVWeightGradients(
            query=query_gradient,
            key=key_gradient,
            value=value_gradient,
        )
