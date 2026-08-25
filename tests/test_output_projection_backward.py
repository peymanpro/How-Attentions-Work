import numpy as np

from src.attention.output_projection import OutputProjection
from src.attention.output_projection_backward import (
    OutputProjectionBackward,
)
from src.math.matrix import Matrix


def test_output_projection_backward_should_return_correct_shapes() -> None:
    projection = OutputProjection(
        input_dimension=3,
        vocabulary_size=4,
        seed=42,
    )

    inputs = Matrix.from_values(
        [
            [0.2, 0.4, 0.1],
            [0.7, -0.3, 0.5],
        ]
    )

    predictions = projection.forward(inputs)

    gradients = OutputProjectionBackward().backward(
        inputs=inputs,
        weights=projection.weights,
        predictions=predictions,
        targets=[1, 2],
    )

    assert gradients.input.shape == inputs.shape
    assert gradients.weights.shape == (
        3,
        4,
    )


def test_softmax_cross_entropy_gradient_should_be_probability_minus_target() -> None:
    projection = OutputProjection(
        input_dimension=2,
        vocabulary_size=3,
        seed=42,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 0.0],
        ]
    )

    predictions = projection.forward(inputs)

    gradients = OutputProjectionBackward().backward(
        inputs=inputs,
        weights=projection.weights,
        predictions=predictions,
        targets=[2],
    )

    expected_d_logits = (
        predictions.probabilities.data.copy()
    )

    expected_d_logits[0, 2] -= 1.0

    expected_d_weights = (
        inputs.data.T @ expected_d_logits
    )

    np.testing.assert_allclose(
        gradients.weights.data,
        expected_d_weights,
    )


def test_output_projection_backward_should_reject_invalid_target() -> None:
    projection = OutputProjection(
        input_dimension=2,
        vocabulary_size=3,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 0.0],
        ]
    )

    predictions = projection.forward(inputs)

    try:
        OutputProjectionBackward().backward(
            inputs=inputs,
            weights=projection.weights,
            predictions=predictions,
            targets=[3],
        )
    except IndexError:
        return

    raise AssertionError(
        "Expected IndexError for invalid target."
    )
