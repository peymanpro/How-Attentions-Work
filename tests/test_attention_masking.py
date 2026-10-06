import numpy as np
import pytest

from src.attention.masking import (
    causal_attention_mask,
    resolve_attention_mask,
)
from src.attention.qkv import QKV
from src.attention.scaled_dot_product import ScaledDotProductAttention
from src.math.matrix import Matrix


def test_causal_mask_should_have_expected_structure() -> None:
    mask = causal_attention_mask(3)

    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ]
        ),
    )


def test_resolved_mask_should_combine_causal_and_explicit_constraints() -> None:
    explicit = np.array(
        [
            [True, True, False],
            [True, True, True],
        ]
    )

    resolved = resolve_attention_mask(
        query_length=2,
        key_length=3,
        causal=True,
        attention_mask=explicit,
    )

    np.testing.assert_array_equal(
        resolved,
        np.array(
            [
                [True, False, False],
                [True, True, False],
            ]
        ),
    )


def test_attention_should_honor_an_explicit_mask() -> None:
    mask = np.array(
        [
            [True, False],
            [True, True],
        ]
    )

    result = ScaledDotProductAttention(
        key_dimension=2,
    ).forward(
        QKV(
            query=Matrix.from_values(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
            key=Matrix.from_values(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
            value=Matrix.from_values(
                [[10.0, 0.0], [0.0, 20.0]]
            ),
        ),
        attention_mask=mask,
    )

    np.testing.assert_allclose(
        result.weights.data[0],
        np.array([1.0, 0.0]),
    )


def test_attention_should_reject_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        ScaledDotProductAttention(
            key_dimension=3,
        ).forward(
            QKV(
                query=Matrix.from_values([[1.0, 0.0]]),
                key=Matrix.from_values([[1.0, 0.0]]),
                value=Matrix.from_values([[1.0, 0.0]]),
            )
        )


def test_attention_should_reject_key_value_length_mismatch() -> None:
    with pytest.raises(ValueError):
        ScaledDotProductAttention(
            key_dimension=2,
        ).forward(
            QKV(
                query=Matrix.from_values([[1.0, 0.0]]),
                key=Matrix.from_values(
                    [[1.0, 0.0], [0.0, 1.0]]
                ),
                value=Matrix.from_values([[1.0, 0.0]]),
            )
        )
