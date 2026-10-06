import numpy as np
import pytest

from src.attention.variants.self_attention import SelfAttention
from src.math.matrix import Matrix


def test_self_attention_should_return_sequence_shape() -> None:
    module = SelfAttention(
        model_dimension=4,
        attention_dimension=4,
    )

    result = module.forward(
        Matrix.from_values(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        ),
        causal=True,
    )

    assert result.qkv.query.shape == (3, 4)
    assert result.attention.output.shape == (3, 4)
    assert result.attention.weights.shape == (3, 3)


def test_self_attention_should_respect_causal_mask() -> None:
    module = SelfAttention(
        model_dimension=2,
        attention_dimension=2,
    )

    result = module.forward(
        Matrix.from_values(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        ),
        causal=True,
    )

    weights = result.attention.weights.data
    np.testing.assert_allclose(
        weights[
            np.triu_indices(3, k=1)
        ],
        np.zeros(3),
    )


def test_self_attention_should_reject_invalid_input_dimension() -> None:
    module = SelfAttention(
        model_dimension=4,
        attention_dimension=4,
    )

    with pytest.raises(ValueError):
        module.forward(
            Matrix.from_values([[1.0, 2.0, 3.0]])
        )
