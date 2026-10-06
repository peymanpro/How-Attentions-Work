import numpy as np
import pytest

from src.attention.alternatives.additive import AdditiveAttention
from src.math.matrix import Matrix


def test_additive_attention_should_support_distinct_query_and_key_lengths() -> None:
    module = AdditiveAttention(
        query_dimension=3,
        key_dimension=4,
        value_dimension=5,
        attention_dimension=6,
    )

    result = module.forward(
        query_inputs=Matrix.from_values(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        key_inputs=Matrix.from_values(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]]
        ),
        value_inputs=Matrix.from_values(
            [[1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0]]
        ),
    )

    assert result.scores.shape == (2, 3)
    assert result.weights.shape == (2, 3)
    assert result.output.shape == (2, 5)
    np.testing.assert_allclose(
        result.weights.data.sum(axis=1),
        np.ones(2),
    )


def test_additive_attention_should_respect_causal_mask() -> None:
    module = AdditiveAttention(
        query_dimension=2,
        key_dimension=2,
        value_dimension=2,
        attention_dimension=4,
    )

    result = module.forward(
        query_inputs=Matrix.from_values(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        key_inputs=Matrix.from_values(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        value_inputs=Matrix.from_values(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        causal=True,
    )

    weights = result.weights.data
    np.testing.assert_allclose(
        weights[np.triu_indices(3, k=1)],
        np.zeros(3),
    )


def test_additive_attention_should_reject_key_value_dimension_mismatch() -> None:
    module = AdditiveAttention(
        query_dimension=2,
        key_dimension=2,
        value_dimension=3,
        attention_dimension=4,
    )

    with pytest.raises(ValueError):
        module.forward(
            Matrix.from_values([[1.0, 0.0]]),
            Matrix.from_values([[1.0, 0.0]]),
            Matrix.from_values([[1.0, 0.0]]),
        )
