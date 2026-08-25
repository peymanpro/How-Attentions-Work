import numpy as np
import pytest

from src.attention.qkv import QKVProjector
from src.math.matrix import Matrix


def test_qkv_projector_should_create_correct_shapes() -> None:
    projector = QKVProjector(
        model_dimension=4,
        attention_dimension=3,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    result = projector.project(inputs)

    assert result.query.shape == (3, 3)
    assert result.key.shape == (3, 3)
    assert result.value.shape == (3, 3)


def test_qkv_projection_should_be_deterministic_for_same_seed() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 0.5],
            [0.2, 1.0],
        ]
    )

    first = QKVProjector(
        model_dimension=2,
        attention_dimension=2,
        seed=42,
    ).project(inputs)

    second = QKVProjector(
        model_dimension=2,
        attention_dimension=2,
        seed=42,
    ).project(inputs)

    np.testing.assert_allclose(
        first.query.data,
        second.query.data,
    )

    np.testing.assert_allclose(
        first.key.data,
        second.key.data,
    )

    np.testing.assert_allclose(
        first.value.data,
        second.value.data,
    )


def test_qkv_projector_should_produce_distinct_projections() -> None:
    inputs = Matrix.from_values(
        [
            [1.0, 0.5],
            [0.2, 1.0],
        ]
    )

    result = QKVProjector(
        model_dimension=2,
        attention_dimension=2,
        seed=42,
    ).project(inputs)

    assert not np.allclose(
        result.query.data,
        result.key.data,
    )

    assert not np.allclose(
        result.query.data,
        result.value.data,
    )


def test_qkv_projector_should_reject_wrong_input_dimension() -> None:
    projector = QKVProjector(
        model_dimension=4,
        attention_dimension=3,
    )

    inputs = Matrix.from_values(
        [
            [1.0, 2.0, 3.0],
        ]
    )

    with pytest.raises(ValueError):
        projector.project(inputs)
def test_qkv_projector_should_update_weights() -> None:
    projector = QKVProjector(
        model_dimension=2,
        attention_dimension=2,
        seed=42,
    )

    before_query = projector.query_weights.data
    before_key = projector.key_weights.data
    before_value = projector.value_weights.data

    gradient = Matrix.from_values(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    projector.apply_gradients(
        query_gradient=gradient,
        key_gradient=gradient,
        value_gradient=gradient,
        learning_rate=0.1,
    )

    np.testing.assert_allclose(
        projector.query_weights.data,
        before_query - 0.1 * gradient.data,
    )

    np.testing.assert_allclose(
        projector.key_weights.data,
        before_key - 0.1 * gradient.data,
    )

    np.testing.assert_allclose(
        projector.value_weights.data,
        before_value - 0.1 * gradient.data,
    )
