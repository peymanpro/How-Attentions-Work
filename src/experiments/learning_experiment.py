from __future__ import annotations

from dataclasses import dataclass

from src.attention.attention_snapshot import AttentionSnapshot
from src.attention.diagnostics import parameter_change
from src.attention.inspection import AttentionInspector
from src.attention.output_projection import OutputProjection
from src.attention.qkv import QKVProjector
from src.attention.scaled_dot_product import ScaledDotProductAttention
from src.attention.sequence import TokenEmbeddingSequence
from src.attention.training import AttentionTrainer


@dataclass(frozen=True)
class AttentionLearningExperimentResult:
    initial_loss: float
    final_loss: float
    initial_attention: AttentionSnapshot
    final_attention: AttentionSnapshot
    mean_attention_change: float
    q_weight_change: float
    k_weight_change: float
    v_weight_change: float
    output_weight_change: float


class AttentionLearningExperiment:
    def run(
        self,
        input_token_ids: list[int],
        target_token_ids: list[int],
        vocabulary_size: int,
        embedding_dimension: int = 8,
        attention_dimension: int = 8,
        epochs: int = 200,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> AttentionLearningExperimentResult:
        sequence = TokenEmbeddingSequence(
            vocabulary_size=vocabulary_size,
            embedding_dimension=embedding_dimension,
            seed=seed,
        )

        projector = QKVProjector(
            model_dimension=embedding_dimension,
            attention_dimension=attention_dimension,
            seed=seed,
        )

        attention = ScaledDotProductAttention(
            key_dimension=attention_dimension,
        )

        output_projection = OutputProjection(
            input_dimension=attention_dimension,
            vocabulary_size=vocabulary_size,
            seed=seed,
        )

        inspector = AttentionInspector(
            sequence=sequence,
            projector=projector,
            attention=attention,
        )

        initial_attention = inspector.capture(
            input_token_ids,
            causal=True,
        )

        initial_query_weights = projector.query_weights
        initial_key_weights = projector.key_weights
        initial_value_weights = projector.value_weights
        initial_output_weights = output_projection.weights

        trainer = AttentionTrainer(
            sequence=sequence,
            projector=projector,
            attention=attention,
            output_projection=output_projection,
            learning_rate=learning_rate,
        )

        history = trainer.train(
            input_token_ids=input_token_ids,
            target_token_ids=target_token_ids,
            epochs=epochs,
        )

        final_attention = inspector.capture(
            input_token_ids,
            causal=True,
        )

        final_query_weights = projector.query_weights
        final_key_weights = projector.key_weights
        final_value_weights = projector.value_weights
        final_output_weights = output_projection.weights

        return AttentionLearningExperimentResult(
            initial_loss=history[0].average_loss,
            final_loss=history[-1].average_loss,
            initial_attention=initial_attention,
            final_attention=final_attention,
            mean_attention_change=(
                initial_attention.mean_absolute_change(
                    final_attention
                )
            ),
            q_weight_change=parameter_change(
                initial_query_weights,
                final_query_weights,
            ),
            k_weight_change=parameter_change(
                initial_key_weights,
                final_key_weights,
            ),
            v_weight_change=parameter_change(
                initial_value_weights,
                final_value_weights,
            ),
            output_weight_change=parameter_change(
                initial_output_weights,
                final_output_weights,
            ),
        )
