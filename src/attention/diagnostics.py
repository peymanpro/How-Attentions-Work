from __future__ import annotations

import numpy as np

from src.math.matrix import Matrix


def frobenius_norm(matrix: Matrix) -> float:
    data = matrix.data

    return float(
        np.linalg.norm(
            data,
            ord="fro",
        )
    )


def parameter_change(
    before: Matrix,
    after: Matrix,
) -> float:
    if before.shape != after.shape:
        raise ValueError(
            "Matrices must have the same shape."
        )

    difference = Matrix(
        after.data - before.data
    )

    return frobenius_norm(difference)
