from src.attention.efficiency.grouped_query import GroupedQueryAttention
from src.attention.efficiency.local import LocalSelfAttention
from src.attention.efficiency.multi_query import MultiQueryAttention
from src.attention.variants.cross_attention import CrossAttention
from src.attention.variants.multi_head import MultiHeadAttention
from src.attention.variants.self_attention import SelfAttention
from src.math.matrix import Matrix


def main() -> None:
    tokens = Matrix.from_values(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    memory = Matrix.from_values(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ]
    )

    self_result = SelfAttention(
        model_dimension=4,
        attention_dimension=4,
        seed=42,
    ).forward(tokens, causal=True)

    cross_result = CrossAttention(
        query_dimension=4,
        key_value_dimension=6,
        attention_dimension=4,
        seed=42,
    ).forward(tokens, memory)

    multi_head_result = MultiHeadAttention(
        query_dimension=4,
        num_heads=2,
        seed=42,
    ).forward(tokens, causal=True)

    grouped_result = GroupedQueryAttention(
        model_dimension=4,
        num_query_heads=4,
        num_key_value_heads=2,
        seed=42,
    ).forward(tokens, causal=True)

    multi_query_result = MultiQueryAttention(
        model_dimension=4,
        num_query_heads=4,
        seed=42,
    ).forward(tokens, causal=True)

    local_result = LocalSelfAttention(
        model_dimension=4,
        attention_dimension=4,
        window_size=2,
        seed=42,
    ).forward(tokens, causal=True)

    print("HowAttentionWorks — Attention Family Demo")
    print("==========================================")
    print()
    print(
        "Scaled Dot-Product Attention: "
        "the shared core primitive used by every variant above."
    )
    print()
    print(f"Self-Attention output:       {self_result.attention.output.shape}")
    print(f"Cross-Attention output:      {cross_result.attention.output.shape}")
    print(
        f"Multi-Head output:           {multi_head_result.output.shape} "
        f"({len(multi_head_result.heads)} heads)"
    )
    print(
        f"Grouped-Query output:        {grouped_result.output.shape} "
        f"({grouped_result.num_key_value_heads} K/V groups)"
    )
    print(
        f"Multi-Query output:          {multi_query_result.output.shape} "
        f"({multi_query_result.num_key_value_heads} shared K/V head)"
    )
    print(f"Local Self-Attention output: {local_result.attention.output.shape}")


if __name__ == "__main__":
    main()
