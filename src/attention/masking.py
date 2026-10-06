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
