import numpy as np
import pytest

from src.attention.attention_snapshot import (
    AttentionSnapshot,
)
from src.math.matrix import Matrix


def test_snapshot_should_calculate_difference() -> None:
    before = AttentionSnapshot(
        Matrix.from_values(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        )
    )

    after = AttentionSnapshot(
        Matrix.from_values(
            [
                [0.6, 0.4],
                [0.1, 0.9],
            ]
        )
    )

    difference = before.difference(after)

    np.testing.assert_allclose(
        difference.data,
        np.array(
            [
                [-0.2, 0.2],
                [-0.2, 0.2],
            ]
        ),
    )


def test_snapshot_should_calculate_mean_absolute_change() -> None:
    before = AttentionSnapshot(
        Matrix.from_values(
            [
                [0.8, 0.2],
                [0.3, 0.7],
            ]
        )
    )

    after = AttentionSnapshot(
        Matrix.from_values(
            [
                [0.6, 0.4],
                [0.1, 0.9],
            ]
        )
    )

    assert before.mean_absolute_change(after) == pytest.approx(
        0.2
    )


def test_snapshot_should_reject_different_shapes() -> None:
    before = AttentionSnapshot(
        Matrix.from_values(
            [[1.0, 0.0]]
        )
    )

    after = AttentionSnapshot(
        Matrix.from_values(
            [[1.0, 0.0], [0.0, 1.0]]
        )
    )

    with pytest.raises(ValueError):
        before.difference(after)
