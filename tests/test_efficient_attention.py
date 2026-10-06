import pytest

from src.attention.efficiency.grouped_query import GroupedQueryAttention
from src.attention.efficiency.local import LocalSelfAttention
from src.attention.efficiency.multi_query import MultiQueryAttention
from src.attention.masking import sliding_window_mask
from src.math.matrix import Matrix


def create_inputs() -> Matrix:
    return Matrix.from_values(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def test_grouped_query_attention_should_share_kv_by_group() -> None:
    module = GroupedQueryAttention(
        model_dimension=4,
        num_query_heads=4,
        num_key_value_heads=2,
    )

    result = module.forward(
        create_inputs(),
        causal=True,
    )

    assert len(result.heads) == 4
    assert result.output.shape == (4, 4)
    assert all(
        head.output.shape == (4, 1)
        for head in result.heads
    )


def test_grouped_query_attention_should_reject_invalid_head_grouping() -> None:
    with pytest.raises(ValueError):
        GroupedQueryAttention(
            model_dimension=4,
            num_query_heads=3,
            num_key_value_heads=2,
        )


def test_multi_query_attention_should_use_one_kv_head() -> None:
    module = MultiQueryAttention(
        model_dimension=4,
        num_query_heads=4,
    )

    result = module.forward(
        create_inputs(),
        causal=True,
    )

    assert module.num_query_heads == 4
    assert module.num_key_value_heads == 1
    assert len(result.heads) == 4


def test_local_self_attention_should_limit_context() -> None:
    module = LocalSelfAttention(
        model_dimension=4,
        attention_dimension=4,
        window_size=2,
    )

    result = module.forward(
        create_inputs(),
        causal=True,
    )

    weights = result.attention.weights.data
    assert weights[0, 0] > 0.0
    assert weights[0, 1] == 0.0
    assert weights[1, 0] > 0.0
    assert weights[1, 1] > 0.0
    assert weights[1, 2] == 0.0
    assert weights[3, 0] == 0.0
    assert weights[3, 1] == 0.0
    assert weights[3, 2] > 0.0
    assert weights[3, 3] > 0.0


def test_sliding_window_mask_should_reject_invalid_window() -> None:
    with pytest.raises(ValueError):
        sliding_window_mask(
            sequence_length=4,
            window_size=0,
        )
