import numpy as np
import pytest

from src.attention.output_projection import OutputProjection
from src.math.matrix import Matrix


def test_output_projection_should_return_expected_shapes() -> None:
    projection = OutputProjection(
        input_dimension=4,
        vocabulary_size=6,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    result = projection.forward(inputs)

    assert result.logits.shape == (2, 6)
    assert result.probabilities.shape == (2, 6)


def test_output_probabilities_should_sum_to_one() -> None:
    projection = OutputProjection(
        input_dimension=4,
        vocabulary_size=6,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )

    result = projection.forward(inputs)

    np.testing.assert_allclose(
        np.sum(result.probabilities.data, axis=1),
        np.ones(2),
    )


def test_output_probabilities_should_be_valid() -> None:
    projection = OutputProjection(
        input_dimension=3,
        vocabulary_size=5,
        seed=42,
    )

    inputs = Matrix.from_values(
        [
            [0.2, 0.4, -0.1],
            [0.7, -0.3, 0.5],
        ]
    )

    result = projection.forward(inputs)

    assert np.all(result.probabilities.data >= 0.0)
    assert np.all(result.probabilities.data <= 1.0)


def test_output_projection_should_be_deterministic() -> None:
    inputs = Matrix.from_values(
        [
            [0.2, 0.4],
            [0.7, -0.3],
        ]
    )

    first = OutputProjection(
        input_dimension=2,
        vocabulary_size=4,
        seed=42,
    ).forward(inputs)

    second = OutputProjection(
        input_dimension=2,
        vocabulary_size=4,
        seed=42,
    ).forward(inputs)

    np.testing.assert_allclose(
        first.logits.data,
        second.logits.data,
    )

    np.testing.assert_allclose(
        first.probabilities.data,
        second.probabilities.data,
    )


def test_output_projection_should_reject_wrong_input_dimension() -> None:
    projection = OutputProjection(
        input_dimension=4,
        vocabulary_size=6,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        projection.forward(inputs)
