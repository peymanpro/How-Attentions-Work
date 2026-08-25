from __future__ import annotations

from dataclasses import dataclass

from src.attention.backward import (
    ScaledDotProductAttentionBackward,
)
from src.attention.output_projection import OutputProjection
from src.attention.output_projection_backward import (
    OutputProjectionBackward,
)
from src.attention.qkv import QKVProjector
from src.attention.qkv_backward import QKVBackward
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.attention.sequence import TokenEmbeddingSequence
from src.math.matrix import Matrix


@dataclass(frozen=True)
class TrainingResult:
    epoch: int
    average_loss: float


class AttentionTrainer:
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
        input_token_ids: list[int],
        target_token_ids: list[int],
        epochs: int,
    ) -> list[TrainingResult]:
        if not input_token_ids:
            raise ValueError(
                "input_token_ids cannot be empty."
            )

        if len(input_token_ids) != len(target_token_ids):
            raise ValueError(
                "Input and target sequences must have equal length."
            )

        if epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        inputs = self._sequence.encode(
            input_token_ids
        )

        history: list[TrainingResult] = []

        for epoch in range(1, epochs + 1):
            qkv = self._projector.project(inputs)

            attention_result = self._attention.forward(
                qkv,
                causal=True,
            )

            predictions = self._output_projection.forward(
                attention_result.output
            )

            loss = self._calculate_loss(
                predictions.probabilities,
                target_token_ids,
            )

            output_gradients = (
                OutputProjectionBackward().backward(
                    inputs=attention_result.output,
                    weights=self._output_projection.weights,
                    predictions=predictions,
                    targets=target_token_ids,
                )
            )

            attention_gradients = (
                ScaledDotProductAttentionBackward(
                    key_dimension=self._attention.key_dimension
                ).backward(
                    qkv=qkv,
                    forward_result=attention_result,
                    output_gradient=output_gradients.input,
                    causal=True,
                )
            )

            qkv_weight_gradients = (
                QKVBackward().backward(
                    inputs=inputs,
                    attention_gradients=attention_gradients,
                )
            )

            self._output_projection.apply_gradient(
                output_gradients.weights,
                self._learning_rate,
            )

            self._projector.apply_gradients(
                query_gradient=qkv_weight_gradients.query,
                key_gradient=qkv_weight_gradients.key,
                value_gradient=qkv_weight_gradients.value,
                learning_rate=self._learning_rate,
            )

            history.append(
                TrainingResult(
                    epoch=epoch,
                    average_loss=loss,
                )
            )

        return history

    @staticmethod
    def _calculate_loss(
        probabilities: Matrix,
        targets: list[int],
    ) -> float:
        import numpy as np

        data = probabilities.data

        if data.shape[0] != len(targets):
            raise ValueError(
                "Number of targets must match probability rows."
            )

        losses = []

        for row_index, target in enumerate(targets):
            probability = max(
                data[row_index, target],
                1e-12,
            )

            losses.append(
                -np.log(probability)
            )

        return float(
            np.mean(losses)
        )

