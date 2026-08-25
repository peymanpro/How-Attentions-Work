import numpy as np
import pytest

from src.attention.mse import MeanSquaredError
from src.math.matrix import Matrix


def test_mse_should_calculate_loss() -> None:
    loss = MeanSquaredError()

    prediction = Matrix.from_values(
        [
            [1.0, 3.0],
        ]
    )

    target = Matrix.from_values(
        [
            [2.0, 1.0],
        ]
    )

    assert loss.calculate(
        prediction,
        target,
    ) == pytest.approx(2.5)


def test_mse_gradient_should_be_correct() -> None:
    loss = MeanSquaredError()

    prediction = Matrix.from_values(
        [
            [1.0, 3.0],
        ]
    )

    target = Matrix.from_values(
        [
            [2.0, 1.0],
        ]
    )

    gradient = loss.gradient(
        prediction,
        target,
    )

    np.testing.assert_allclose(
        gradient.data,
        np.array(
            [
                [-1.0, 2.0],
            ]
        ),
    )
