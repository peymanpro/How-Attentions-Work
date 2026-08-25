import numpy as np
import pytest

from src.attention.backward import (
    ScaledDotProductAttentionBackward,
)
from src.attention.qkv import QKV
from src.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from src.math.matrix import Matrix


def create_qkv() -> QKV:
    return QKV(
        query=Matrix.from_values(
            [
                [0.3, -0.2],
                [0.1, 0.4],
            ]
        ),
        key=Matrix.from_values(
            [
                [-0.1, 0.2],
                [0.5, -0.3],
            ]
        ),
        value=Matrix.from_values(
            [
                [0.7, -0.4],
                [0.2, 0.9],
            ]
        ),
    )


def test_attention_backward_should_return_correct_shapes() -> None:
    qkv = create_qkv()

    forward = ScaledDotProductAttention(
        key_dimension=2
    )

    forward_result = forward.forward(
        qkv,
        causal=True,
    )

    output_gradient = Matrix.from_values(
        [
            [1.0, -0.5],
            [0.3, 0.8],
        ]
    )

    gradients = (
        ScaledDotProductAttentionBackward(
            key_dimension=2
        ).backward(
            qkv,
            forward_result,
            output_gradient,
            causal=True,
        )
    )

    assert gradients.query.shape == qkv.query.shape
    assert gradients.key.shape == qkv.key.shape
    assert gradients.value.shape == qkv.value.shape


def test_causal_backward_should_not_create_future_score_gradients() -> None:
    qkv = create_qkv()

    forward_result = (
        ScaledDotProductAttention(
            key_dimension=2
        ).forward(
            qkv,
            causal=True,
        )
    )

    output_gradient = Matrix.from_values(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    gradients = (
        ScaledDotProductAttentionBackward(
            key_dimension=2
        ).backward(
            qkv,
            forward_result,
            output_gradient,
            causal=True,
        )
    )

    assert np.all(
        np.isfinite(
            gradients.query.data
        )
    )

    assert np.all(
        np.isfinite(
            gradients.key.data
        )
    )

    assert np.all(
        np.isfinite(
            gradients.value.data
        )
    )


def test_attention_backward_should_match_numerical_gradient() -> None:
    qkv = create_qkv()

    attention = ScaledDotProductAttention(
        key_dimension=2
    )

    forward_result = attention.forward(
        qkv,
        causal=False,
    )

    output_gradient = Matrix.from_values(
        [
            [0.6, -0.2],
            [0.4, 0.7],
        ]
    )

    analytic = (
        ScaledDotProductAttentionBackward(
            key_dimension=2
        ).backward(
            qkv,
            forward_result,
            output_gradient,
            causal=False,
        )
    )

    def scalar_loss(
        query: np.ndarray,
    ) -> float:
        mutated = QKV(
            query=Matrix(query),
            key=qkv.key,
            value=qkv.value,
        )

        result = attention.forward(
            mutated,
            causal=False,
        )

        return float(
            np.sum(
                result.output.data
                * output_gradient.data
            )
        )

    epsilon = 1e-6

    numerical = np.zeros_like(
        qkv.query.data
    )

    for row in range(
        qkv.query.rows
    ):
        for column in range(
            qkv.query.columns
        ):
            plus = (
                qkv.query.data.copy()
            )

            minus = (
                qkv.query.data.copy()
            )

            plus[row, column] += epsilon
            minus[row, column] -= epsilon

            numerical[row, column] = (
                scalar_loss(plus)
                - scalar_loss(minus)
            ) / (
                2.0 * epsilon
            )

    np.testing.assert_allclose(
        analytic.query.data,
        numerical,
        rtol=1e-5,
        atol=1e-5,
    )


def test_attention_backward_should_reject_invalid_output_gradient() -> None:
    qkv = create_qkv()

    forward_result = (
        ScaledDotProductAttention(
            key_dimension=2
        ).forward(qkv)
    )

    with pytest.raises(ValueError):
        ScaledDotProductAttentionBackward(
            key_dimension=2
        ).backward(
            qkv,
            forward_result,
            Matrix.from_values(
                [[1.0, 2.0, 3.0]]
            ),
        )
