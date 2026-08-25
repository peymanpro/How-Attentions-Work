import numpy as np
import pytest

from src.attention.softmax import softmax


def test_softmax_should_return_probabilities() -> None:
    result = softmax(
        np.array([1.0, 2.0, 3.0])
    )

    assert result.shape == (3,)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_softmax_probabilities_should_sum_to_one() -> None:
    result = softmax(
        np.array([1.0, 2.0, 3.0])
    )

    assert np.isclose(
        np.sum(result),
        1.0,
    )


def test_softmax_should_preserve_relative_order() -> None:
    result = softmax(
        np.array([1.0, 2.0, 3.0])
    )

    assert result[0] < result[1] < result[2]


def test_softmax_should_be_numerically_stable() -> None:
    result = softmax(
        np.array([1000.0, 1001.0, 1002.0])
    )

    assert np.all(np.isfinite(result))
    assert np.isclose(np.sum(result), 1.0)


def test_softmax_should_be_shift_invariant() -> None:
    values = np.array([1.0, 2.0, 3.0])

    first = softmax(values)
    second = softmax(values + 1000.0)

    np.testing.assert_allclose(
        first,
        second,
    )


def test_softmax_should_reject_empty_input() -> None:
    with pytest.raises(ValueError):
        softmax(np.array([]))


def test_softmax_should_reject_non_vector_input() -> None:
    with pytest.raises(ValueError):
        softmax(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            )
        )


def test_softmax_should_reject_non_finite_input() -> None:
    with pytest.raises(ValueError):
        softmax(
            np.array(
                [
                    1.0,
                    np.nan,
                ]
            )
        )
