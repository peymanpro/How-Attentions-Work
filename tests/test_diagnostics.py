import numpy as np
import pytest

from src.attention.diagnostics import (
    frobenius_norm,
    parameter_change,
)
from src.math.matrix import Matrix


def test_frobenius_norm_should_be_calculated() -> None:
    matrix = Matrix.from_values(
        [
            [3.0, 4.0],
            [0.0, 0.0],
        ]
    )

    assert frobenius_norm(matrix) == pytest.approx(5.0)


def test_parameter_change_should_be_zero_for_identical_matrices() -> None:
    matrix = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    assert parameter_change(matrix, matrix) == pytest.approx(0.0)


def test_parameter_change_should_measure_frobenius_distance() -> None:
    before = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    after = Matrix.from_values(
        [
            [2.0, 2.0],
            [3.0, 7.0],
        ]
    )

    assert parameter_change(
        before,
        after,
    ) == pytest.approx(
        np.sqrt(10.0)
    )

