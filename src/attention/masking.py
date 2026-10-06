from __future__ import annotations

import numpy as np


def resolve_attention_mask(
    query_length: int,
    key_length: int,
    *,
    causal: bool = False,
    attention_mask: np.ndarray | None = None,
) -> np.ndarray:
    if query_length <= 0 or key_length <= 0:
        raise ValueError("Attention sequence lengths must be positive.")

    if attention_mask is None:
        resolved = np.ones(
            (query_length, key_length),
            dtype=bool,
        )
    else:
        resolved = np.asarray(attention_mask, dtype=bool)

        if resolved.shape != (query_length, key_length):
            raise ValueError(
                "Attention mask shape must match "
                f"({query_length}, {key_length})."
            )

        resolved = resolved.copy()

    if causal:
        query_positions = np.arange(query_length)[:, None]
        key_positions = np.arange(key_length)[None, :]
        resolved &= key_positions <= query_positions

    if not np.all(np.any(resolved, axis=1)):
        raise ValueError(
            "Every query position must have at least one allowed key."
        )

    return resolved


def sliding_window_mask(
    sequence_length: int,
    window_size: int,
    *,
    causal: bool = True,
) -> np.ndarray:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")

    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    positions = np.arange(sequence_length)
    distance = np.abs(
        positions[:, None] - positions[None, :]
    )
    mask = distance < window_size

    if causal:
        mask &= positions[None, :] <= positions[:, None]

    if not np.all(np.any(mask, axis=1)):
        raise ValueError(
            "Every query position must have at least one allowed key."
        )

    return mask
