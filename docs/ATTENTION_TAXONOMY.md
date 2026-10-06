# Attention Taxonomy

HowAttentionWorks treats attention as a family of related mechanisms rather than a single implementation.

## 1. Core mechanism

### Scaled Dot-Product Attention

The primitive is the standard scaled dot-product formulation.

Implemented with:

- explicit matrix operations
- numerically stable softmax
- explicit causal and arbitrary masks
- analytical backward propagation
- numerical gradient verification

Status: implemented and verified.

## 2. Attention by source relationship

### Self-Attention

Queries, keys, and values are projected from the same sequence.

Status: implemented.

### Cross-Attention

Queries come from one sequence while keys and values come from another.

Status: implemented.

## 3. Attention by composition

### Multi-Head Attention

Several attention heads operate in parallel and their outputs are concatenated and projected.

Status: implemented.

The current implementation provides forward computation and structural verification. Training and backward propagation for the complete multi-head composition remain future work.

## 4. Causality and locality

### Causal / Masked Attention

Future key positions are excluded from the normalization.

Status: implemented.

### Local / Sliding-Window Attention

Each query can attend only to a bounded neighborhood.

Status: implemented.

## 5. Efficient attention variants

### Multi-Query Attention (MQA)

Multiple query heads share a single key/value head.

Status: implemented.

### Grouped-Query Attention (GQA)

Multiple query heads share key/value projections within groups.

Status: implemented.

The current MQA/GQA implementations focus on the forward mechanism and structural properties. Training and backward propagation are intentionally deferred until the shared-parameter gradient path is developed and verified.

## 6. How the mechanisms relate

Scaled Dot-Product Attention
→ source relationship: Self / Cross
→ composition: Multi-Head
→ masking: Causal / Local
→ KV sharing: MQA / GQA

This taxonomy keeps implementations small enough to inspect while preserving the mathematical relationships between variants.

## Future candidates

The project may later cover:

- Sparse attention
- Block-sparse attention
- Relative-position-aware attention
- Memory / recurrent attention mechanisms

New variants will be added only when their mathematical and experimental scope can be documented clearly.
