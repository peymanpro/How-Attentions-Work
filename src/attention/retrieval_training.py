from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.attention.backward import (
    ScaledDotProductAttentionBackward,
)
from src.attention.mse import MeanSquaredError
from src.attention.qkv import QKVProjector
from src.attention.qkv_backward import QKVBackward
from src.attention.retrieval_task import RetrievalDataset
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.attention.sequence import TokenEmbeddingSequence
from src.math.matrix import Matrix


@dataclass(frozen=True)
class RetrievalTrainingResult:
    epoch: int
    average_loss: float


class QKRetrievalTrainer:
    def __init__(
        self,
        sequence: TokenEmbeddingSequence,
        projector: QKVProjector,
        attention: ScaledDotProductAttention,
        learning_rate: float,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        self._sequence = sequence
        self._projector = projector
        self._attention = attention
        self._learning_rate = learning_rate
        self._loss = MeanSquaredError()

    def train(
        self,
        dataset: RetrievalDataset,
        epochs: int,
    ) -> list[RetrievalTrainingResult]:
        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        history: list[RetrievalTrainingResult] = []

        for epoch in range(1, epochs + 1):
            total_loss = 0.0

            for example in dataset.examples:
                inputs = self._sequence.encode(
                    list(example.input_token_ids)
                )

                qkv = self._projector.project(
                    inputs
                )

                forward_result = (
                    self._attention.forward(
                        qkv,
                        causal=True,
                    )
                )

                query_row = example.query_position

                retrieved = Matrix(
                    forward_result.output.data[
                        query_row : query_row + 1
                    ]
                )

                target = (
                    self._sequence.get_embedding(
                        example.target_value_position
                    )
                )

                loss = self._loss.calculate(
                    retrieved,
                    target,
                )

                total_loss += loss

                output_gradient = (
                    np.zeros_like(
                        forward_result.output.data
                    )
                )

                output_gradient[query_row] = (
                    self._loss.gradient(
                        retrieved,
                        target,
                    ).data[0]
                )

                attention_gradients = (
                    ScaledDotProductAttentionBackward(
                        key_dimension=(
                            self._attention.key_dimension
                        )
                    ).backward(
                        qkv=qkv,
                        forward_result=forward_result,
                        output_gradient=Matrix(
                            output_gradient
                        ),
                        causal=True,
                    )
                )

                weight_gradients = (
                    QKVBackward().backward(
                        inputs=inputs,
                        attention_gradients=(
                            attention_gradients
                        ),
                    )
                )

                self._projector.apply_query_key_gradients(
                    query_gradient=(
                        weight_gradients.query
                    ),
                    key_gradient=(
                        weight_gradients.key
                    ),
                    learning_rate=self._learning_rate,
                )

            history.append(
                RetrievalTrainingResult(
                    epoch=epoch,
                    average_loss=(
                        total_loss / len(dataset)
                    ),
                )
            )

        return history

