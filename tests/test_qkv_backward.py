import numpy as np

from src.attention.backward import AttentionGradients
from src.attention.qkv_backward import QKVBackward
from src.math.matrix import Matrix


def test_qkv_backward_should_return_weight_gradient_shapes() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.2],
        ]
    )

    gradients = AttentionGradients(
        query=Matrix.from_values(
            [
                [0.1, 0.2],
                [0.3, 0.4],
            ]
        ),
        key=Matrix.from_values(
            [
                [0.2, 0.1],
                [0.5, 0.3],
            ]
        ),
        value=Matrix.from_values(
            [
                [0.4, 0.5],
                [0.6, 0.7],
            ]
        ),
    )

    result = QKVBackward().backward(
        inputs,
        gradients,
    )

    assert result.query.shape == (3, 2)
    assert result.key.shape == (3, 2)
    assert result.value.shape == (3, 2)


def test_qkv_backward_should_compute_query_weight_gradient() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    query_gradient = Matrix.from_values(
        [
            [0.5, 0.1],
            [0.2, 0.3],
        ]
    )

    gradients = AttentionGradients(
        query=query_gradient,
        key=Matrix.from_values(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        value=Matrix.from_values(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )

    result = QKVBackward().backward(
        inputs,
        gradients,
    )

    expected = np.array(
        [
            [1.1, 1.0],
            [1.8, 1.4],
        ]
    )

    np.testing.assert_allclose(
        result.query.data,
        expected,
    )

