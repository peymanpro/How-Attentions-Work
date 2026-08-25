import numpy as np
import pytest

from src.attention.cross_entropy import CrossEntropyLoss


def test_cross_entropy_should_be_low_for_high_target_probability() -> None:
    loss = CrossEntropyLoss()

    result = loss.calculate(
        np.array([0.05, 0.9, 0.05]),
        target_index=1,
    )

    np.testing.assert_allclose(
        result,
        -np.log(0.9),
    )


def test_cross_entropy_should_be_high_for_low_target_probability() -> None:
    loss = CrossEntropyLoss()

    result = loss.calculate(
        np.array([0.45, 0.1, 0.45]),
        target_index=1,
    )

    np.testing.assert_allclose(
        result,
        -np.log(0.1),
    )


def test_cross_entropy_should_be_near_zero_for_perfect_prediction() -> None:
    loss = CrossEntropyLoss()

    result = loss.calculate(
        np.array([0.0, 1.0, 0.0]),
        target_index=1,
    )

    assert result < 1e-10


def test_cross_entropy_should_reject_invalid_target() -> None:
    loss = CrossEntropyLoss()

    with pytest.raises(IndexError):
        loss.calculate(
            np.array([0.5, 0.5]),
            target_index=2,
        )


def test_cross_entropy_should_reject_invalid_probability_sum() -> None:
    loss = CrossEntropyLoss()

    with pytest.raises(ValueError):
        loss.calculate(
            np.array([0.2, 0.2]),
            target_index=0,
        )
