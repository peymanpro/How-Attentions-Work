import numpy as np
import pytest

from src.attention.qkv import QKV
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.math.matrix import Matrix


def test_attention_should_return_expected_shapes() -> None:
    qkv = QKV(
        query=Matrix.from_values(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        key=Matrix.from_values(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        value=Matrix.from_values(
            [
                [10.0, 0.0],
                [0.0, 20.0],
            ]
        ),
    )

    attention = ScaledDotProductAttention(
        key_dimension=2
    )

    result = attention.forward(qkv)

    assert result.scores.shape == (2, 2)
    assert result.weights.shape == (2, 2)
    assert result.output.shape == (2, 2)


def test_attention_weights_should_sum_to_one_per_query() -> None:
    qkv = QKV(
        query=Matrix.from_values(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        key=Matrix.from_values(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        value=Matrix.from_values(
            [
                [10.0, 0.0],
                [0.0, 20.0],
            ]
        ),
    )

    result = ScaledDotProductAttention(
        key_dimension=2
    ).forward(qkv)

    np.testing.assert_allclose(
        np.sum(result.weights.data, axis=1),
        np.ones(2),
    )


def test_attention_weights_should_be_valid_probabilities() -> None:
    qkv = QKV(
        query=Matrix.from_values(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        key=Matrix.from_values(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        value=Matrix.from_values(
            [
                [10.0, 0.0],
                [0.0, 20.0],
            ]
        ),
    )

    result = ScaledDotProductAttention(
        key_dimension=2
    ).forward(qkv)

    assert np.all(result.weights.data >= 0.0)
    assert np.all(result.weights.data <= 1.0)


def test_identical_query_and_key_should_prefer_matching_positions() -> None:
    qkv = QKV(
        query=Matrix.from_values(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        key=Matrix.from_values(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        value=Matrix.from_values(
            [
                [10.0, 0.0],
                [0.0, 20.0],
            ]
        ),
    )

    result = ScaledDotProductAttention(
        key_dimension=2
    ).forward(qkv)

    weights = result.weights.data

    assert weights[0, 0] > weights[0, 1]
    assert weights[1, 1] > weights[1, 0]


def test_attention_should_reject_invalid_key_dimension() -> None:
    with pytest.raises(ValueError):
        ScaledDotProductAttention(0)
