import numpy as np
import pytest

from src.attention.variants.cross_attention import CrossAttention
from src.math.matrix import Matrix


def test_cross_attention_should_use_distinct_query_and_key_value_lengths() -> None:
    module = CrossAttention(
        query_dimension=4,
        key_value_dimension=6,
        attention_dimension=4,
    )

    result = module.forward(
        query_inputs=Matrix.from_values(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        ),
        key_value_inputs=Matrix.from_values(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
    )

    assert result.attention.weights.shape == (2, 3)
    assert result.attention.output.shape == (2, 4)


def test_cross_attention_should_return_valid_attention_weights() -> None:
    module = CrossAttention(
        query_dimension=3,
        key_value_dimension=3,
        attention_dimension=3,
    )

    result = module.forward(
        query_inputs=Matrix.from_values([[1.0, 0.0, 0.0]]),
        key_value_inputs=Matrix.from_values(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ),
    )

    assert np.all(result.attention.weights.data >= 0.0)
    np.testing.assert_allclose(
        result.attention.weights.data.sum(axis=1),
        np.ones(1),
    )


def test_cross_attention_should_reject_wrong_query_dimension() -> None:
    module = CrossAttention(
        query_dimension=3,
        key_value_dimension=3,
        attention_dimension=3,
    )

    with pytest.raises(ValueError):
        module.forward(
            Matrix.from_values([[1.0, 2.0]]),
            Matrix.from_values([[1.0, 0.0, 0.0]]),
        )
