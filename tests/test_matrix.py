import numpy as np
import pytest

from src.math.matrix import Matrix


def test_matrix_should_preserve_shape() -> None:
    matrix = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    assert matrix.shape == (2, 3)
    assert matrix.rows == 2
    assert matrix.columns == 3


def test_matrix_should_transpose() -> None:
    matrix = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    result = matrix.transpose()

    np.testing.assert_array_equal(
        result.data,
        np.array(
            [
                [1.0, 4.0],
                [2.0, 5.0],
                [3.0, 6.0],
            ]
        ),
    )


def test_matrix_should_multiply() -> None:
    left = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    right = Matrix.from_values(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    result = left.multiply(right)

    np.testing.assert_array_equal(
        result.data,
        np.array(
            [
                [19.0, 22.0],
                [43.0, 50.0],
            ]
        ),
    )


def test_matrix_should_add() -> None:
    left = Matrix.from_values([[1.0, 2.0]])
    right = Matrix.from_values([[3.0, 4.0]])

    result = left.add(right)

    np.testing.assert_array_equal(
        result.data,
        np.array([[4.0, 6.0]]),
    )


def test_matrix_should_scale() -> None:
    matrix = Matrix.from_values(
        [
            [1.0, -2.0],
            [3.0, 4.0],
        ]
    )

    result = matrix.multiply_scalar(0.5)

    np.testing.assert_array_equal(
        result.data,
        np.array(
            [
                [0.5, -1.0],
                [1.5, 2.0],
            ]
        ),
    )


def test_matrix_should_reject_invalid_dimensions() -> None:
    left = Matrix.from_values([[1.0, 2.0, 3.0]])
    right = Matrix.from_values([[1.0, 2.0]])

    with pytest.raises(ValueError):
        left.multiply(right)


def test_matrix_should_reject_empty_matrix() -> None:
    with pytest.raises(ValueError):
        Matrix(np.empty((0, 0)))


def test_matrix_should_reject_non_finite_values() -> None:
    with pytest.raises(ValueError):
        Matrix.from_values([[1.0, float("nan")]])
