import numpy as np
import pytest

from src.attention.backward import ScaledDotProductAttentionBackward
from src.attention.qkv import QKV
from src.attention.scaled_dot_product import ScaledDotProductAttention
from src.math.matrix import Matrix


def create_qkv() -> QKV:
    return QKV(
        query=Matrix.from_values(
            [
                [0.3, -0.2],
                [0.1, 0.4],
                [-0.5, 0.2],
            ]
        ),
        key=Matrix.from_values(
            [
                [-0.1, 0.2],
                [0.5, -0.3],
                [0.2, 0.7],
            ]
        ),
        value=Matrix.from_values(
            [
                [0.7, -0.4],
                [0.2, 0.9],
                [-0.3, 0.1],
            ]
        ),
    )


def compare_input_gradient(
    name: str,
    qkv: QKV,
    output_gradient: Matrix,
    causal: bool,
) -> None:
    attention = ScaledDotProductAttention(key_dimension=2)
    forward_result = attention.forward(
        qkv,
        causal=causal,
    )
    analytic = ScaledDotProductAttentionBackward(
        key_dimension=2
    ).backward(
        qkv,
        forward_result,
        output_gradient,
        causal=causal,
    )

    def loss_for(candidate: QKV) -> float:
        result = attention.forward(
            candidate,
            causal=causal,
        )
        return float(
            np.sum(
                result.output.data
                * output_gradient.data
            )
        )

    epsilon = 1e-6
    candidate = qkv.query if name == "query" else (
        qkv.key if name == "key" else qkv.value
    )
    numerical = np.zeros_like(candidate.data)

    for row in range(candidate.rows):
        for column in range(candidate.columns):
            plus = candidate.data
            minus = candidate.data
            plus[row, column] += epsilon
            minus[row, column] -= epsilon

            plus_qkv = QKV(
                query=Matrix(plus if name == "query" else qkv.query.data),
                key=Matrix(plus if name == "key" else qkv.key.data),
                value=Matrix(plus if name == "value" else qkv.value.data),
            )
            minus_qkv = QKV(
                query=Matrix(minus if name == "query" else qkv.query.data),
                key=Matrix(minus if name == "key" else qkv.key.data),
                value=Matrix(minus if name == "value" else qkv.value.data),
            )

            numerical[row, column] = (
                loss_for(plus_qkv) - loss_for(minus_qkv)
            ) / (2.0 * epsilon)

    expected = {
        "query": analytic.query.data,
        "key": analytic.key.data,
        "value": analytic.value.data,
    }[name]

    np.testing.assert_allclose(
        expected,
        numerical,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("name", ["query", "key", "value"])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_backward_matches_numerical_gradient(
    name: str,
    causal: bool,
) -> None:
    qkv = create_qkv()
    output_gradient = Matrix.from_values(
        [
            [0.6, -0.2],
            [0.4, 0.7],
            [-0.3, 0.5],
        ]
    )

    compare_input_gradient(
        name,
        qkv,
        output_gradient,
        causal,
    )
