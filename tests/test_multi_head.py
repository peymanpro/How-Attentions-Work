import numpy as np
import pytest

from src.attention.variants.multi_head import MultiHeadAttention
from src.math.matrix import Matrix


def test_multi_head_self_attention_should_create_one_result_per_head() -> None:
    module = MultiHeadAttention(
        query_dimension=4,
        num_heads=2,
    )

    result = module.forward(
        Matrix.from_values(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]]
        ),
        causal=True,
    )

    assert len(result.heads) == 2
    assert result.concatenated.shape == (3, 4)
    assert result.output.shape == (3, 4)

    for head in result.heads:
        assert head.output.shape == (3, 2)
        assert head.weights.shape == (3, 3)


def test_multi_head_self_attention_should_preserve_causal_structure() -> None:
    module = MultiHeadAttention(
        query_dimension=4,
        num_heads=2,
    )

    result = module.forward(
        Matrix.from_values(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]]
        ),
        causal=True,
    )

    for head in result.heads:
        future = head.weights.data[
            np.triu_indices(3, k=1)
        ]
        np.testing.assert_allclose(
            future,
            np.zeros(3),
        )


def test_multi_head_attention_should_reject_non_divisible_dimensions() -> None:
    with pytest.raises(ValueError):
        MultiHeadAttention(
            query_dimension=5,
            num_heads=2,
        )


def test_multi_head_cross_attention_should_support_distinct_inputs() -> None:
    module = MultiHeadAttention(
        query_dimension=4,
        num_heads=2,
        key_value_dimension=6,
    )

    result = module.forward(
        query_inputs=Matrix.from_values(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]]
        ),
        key_value_inputs=Matrix.from_values(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
        ),
    )

    assert result.output.shape == (2, 4)
    assert len(result.heads) == 2

    for head in result.heads:
        assert head.weights.shape == (2, 3)
