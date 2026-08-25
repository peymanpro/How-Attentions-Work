from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.backward import (
    ScaledDotProductAttentionBackward,
)
from src.attention.context_task import (
    ContextDataset,
)
from src.attention.output_projection import (
    OutputProjection,
)
from src.attention.output_projection_backward import (
    OutputProjectionBackward,
)
from src.attention.qkv import QKVProjector
from src.attention.qkv_backward import QKVBackward
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.attention.sequence import (
    TokenEmbeddingSequence,
)
from src.math.matrix import Matrix


@dataclass(frozen=True)
class ContextTrainingResult:
    epoch: int
    average_loss: float


class ContextAttentionTrainer:
    def __init__(
        self,
        sequence: TokenEmbeddingSequence,
        projector: QKVProjector,
        attention: ScaledDotProductAttention,
        output_projection: OutputProjection,
        learning_rate: float,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        self._sequence = sequence
        self._projector = projector
        self._attention = attention
        self._output_projection = output_projection
        self._learning_rate = learning_rate

    def train(
        self,
        dataset: ContextDataset,
        epochs: int,
    ) -> list[ContextTrainingResult]:
        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        history: list[ContextTrainingResult] = []

        for epoch in range(1, epochs + 1):
            total_loss = 0.0

            for example in dataset.examples:
                inputs = self._sequence.encode(
                    example.input_token_ids
                )

                qkv = self._projector.project(
                    inputs
                )

                attention_result = (
                    self._attention.forward(
                        qkv,
                        causal=True,
                    )
                )

                target_row = (
                    attention_result.output.rows - 1
                )

                target_representation = Matrix(
                    attention_result.output.data[
                        target_row : target_row + 1
                    ]
                )

                predictions = (
                    self._output_projection.forward(
                        target_representation
                    )
                )

                probabilities = (
                    predictions.probabilities.data[0]
                )

                target_probability = max(
                    probabilities[
                        example.target_token_id
                    ],
                    1e-12,
                )

                total_loss += float(
                    -np.log(target_probability)
                )

                output_gradients = (
                    OutputProjectionBackward().backward(
                        inputs=target_representation,
                        weights=(
                            self._output_projection.weights
                        ),
                        predictions=predictions,
                        targets=[
                            example.target_token_id
                        ],
                    )
                )

                full_output_gradient = (
                    np.zeros_like(
                        attention_result.output.data
                    )
                )

                full_output_gradient[
                    target_row
                ] = output_gradients.input.data[0]

                attention_gradients = (
                    ScaledDotProductAttentionBackward(
                        key_dimension=(
                            self._attention.key_dimension
                        )
                    ).backward(
                        qkv=qkv,
                        forward_result=attention_result,
                        output_gradient=Matrix(
                            full_output_gradient
                        ),
                        causal=True,
                    )
                )

                qkv_weight_gradients = (
                    QKVBackward().backward(
                        inputs=inputs,
                        attention_gradients=(
                            attention_gradients
                        ),
                    )
                )

                self._output_projection.apply_gradient(
                    output_gradients.weights,
                    self._learning_rate,
                )

                self._projector.apply_gradients(
                    query_gradient=(
                        qkv_weight_gradients.query
                    ),
                    key_gradient=(
                        qkv_weight_gradients.key
                    ),
                    value_gradient=(
                        qkv_weight_gradients.value
                    ),
                    learning_rate=self._learning_rate,
                )

            history.append(
                ContextTrainingResult(
                    epoch=epoch,
                    average_loss=(
                        total_loss
                        / len(dataset)
                    ),
                )
            )

        return history
