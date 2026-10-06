# HowAttentionWorks

> **How does attention decide what context matters?**

HowAttentionWorks is a from-scratch implementation and exploration of the **attention family** using Python and NumPy.

The project starts from the mathematical primitive of scaled dot-product attention, then builds and compares different ways of using that primitive:

- Scaled Dot-Product Attention
- Self-Attention
- Cross-Attention
- Causal / Masked Attention
- Local / Sliding-Window Attention
- Multi-Head Attention
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)
- Additive / Bahdanau Attention

The goal is not to reproduce a production Transformer library.

The goal is to make the mathematics, data flow, masking, gradient propagation, parameter sharing, and design trade-offs explicit enough to inspect and verify.

[Attention taxonomy](docs/ATTENTION_TAXONOMY.md)

---

## Why This Project Exists

Attention is often compressed into a single equation:

$$
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

That equation is elegant, but a high-level implementation can hide the engineering details that matter when learning or debugging the mechanism.

This project opens those details up:

~~~text
Input representations
        |
        +----> Wq ----> Q
        |
        +----> Wk ----> K
        |
        +----> Wv ----> V
                         |
                         v
                       QK^T
                         |
                         v
                     Scaling
                         |
                         v
                       Mask
                         |
                         v
                      Softmax
                         |
                         v
                 Attention weights
                         |
                         v
                       AV
                         |
                         v
                  Attention output
~~~

The trainable core also follows the gradient path:

~~~text
Loss
  |
  v
dOutput
  |
  v
dAttention
  |
  v
dScores
  |
  +----> dQ
  +----> dK
  +----> dV
          |
          v
       dWq/dWk/dWv
          |
          v
    Gradient descent
~~~

---

## Attention Family

The repository is organized around relationships between mechanisms rather than a flat list of unrelated implementations.

| Family | Mechanism | Status |
|---|---|---|
| Core | Scaled Dot-Product Attention | Implemented, backward-verified |
| Source relationship | Self-Attention | Implemented |
| Source relationship | Cross-Attention | Implemented |
| Masking | Causal / Masked Attention | Implemented |
| Masking | Local / Sliding-Window Attention | Implemented |
| Composition | Multi-Head Attention | Implemented, forward-focused |
| KV sharing | Multi-Query Attention | Implemented, forward-focused |
| KV sharing | Grouped-Query Attention | Implemented, forward-focused |
| Alternative scoring | Additive / Bahdanau Attention | Implemented, forward-focused |

The key relationship is:

~~~text
Scaled Dot-Product Attention
        |
        +-- source relationship --> Self / Cross
        |
        +-- masking --------------> Causal / Local
        |
        +-- composition ----------> Multi-Head
        |
        +-- KV sharing -----------> MQA / GQA
~~~

---

## Core: Scaled Dot-Product Attention

For query, key, and value matrices:

$$
S=\frac{QK^T}{\sqrt{d_k}}
$$

$$
A=\mathrm{softmax}(S)
$$

$$
H=AV
$$

The implementation keeps these operations visible instead of delegating them to a deep-learning framework.

The core supports:

- stable Softmax
- causal masking
- arbitrary boolean attention masks
- explicit shape validation
- analytical backward propagation
- numerical gradient verification

---

## Self-Attention

Self-attention uses the same sequence as the source of queries, keys, and values:

$$
Q=XW_Q,\qquad
K=XW_K,\qquad
V=XW_V
$$

The repository provides a small Self-Attention composition around the verified scaled dot-product primitive.

Causal self-attention is supported through the same masking path rather than through a separate implementation.

---

## Cross-Attention

Cross-attention separates the query sequence from the key/value sequence:

$$
Q=XW_Q
$$

$$
K=YW_K,\qquad
V=YW_V
$$

This makes the distinction between self-attention and cross-attention explicit at the projection boundary.

The implementation supports different query and key/value sequence lengths and different input dimensions.

---

## Multi-Head Attention

Multi-head attention runs several attention heads in parallel:

~~~text
                +-- Head 1 --+
Input ----------+-- Head 2 --+-- concatenate -- output projection
                +-- Head N --+
~~~

Each head has its own Q/K/V projections and attention distribution.

The current implementation focuses on the forward mechanism and structural properties.

A complete trainable multi-head backward path is intentionally left for a later phase so that the shared concatenation and output-projection gradients can be derived and verified independently.

---

## Efficient Attention Variants

### Multi-Query Attention

MQA keeps multiple query heads but shares a single key/value head.

~~~text
Q1 --+
Q2 --+
Q3 --+-- attention with shared K/V
Q4 --+
      K
      V
~~~

### Grouped-Query Attention

GQA generalizes this idea by sharing K/V projections within groups of query heads.

~~~text
Q1 Q2 ---- K/V group 1
Q3 Q4 ---- K/V group 2
~~~

These implementations make parameter sharing explicit.

For MQA and GQA, the current scope is forward computation and structural verification. Their shared-parameter backward and training paths are not claimed to be complete yet.

---

## Additive / Bahdanau Attention

Additive attention uses a learned nonlinear compatibility score rather than the scaled dot product.

The implementation keeps the scoring function explicit and reuses the same masking abstraction. It is currently forward-focused; its backward and training path are not claimed as part of the verified learning core.

---

## Local Attention

Local attention restricts each query to a bounded neighborhood.

For a causal sliding window:

~~~text
token 0: 0
token 1: 0 1
token 2:   1 2
token 3:     2 3
~~~

The implementation expresses locality as an attention mask and reuses the same core attention operation.

This keeps masking as a reusable concern rather than creating another attention kernel.

---

## What the Experiments Teach

The original learning experiments remain intentionally small.

They demonstrate that:

~~~text
attention is differentiable
        ↓
gradients can propagate through the mechanism
        ↓
Q/K/V projections can be updated
        ↓
a complete small network can reduce a training objective
~~~

But the project explicitly avoids the stronger and unsupported conclusion:

~~~text
loss decreased
      ↓
attention must have learned a meaningful routing pattern
~~~

The experiments show why that inference is unsafe: value and output-projection parameters can absorb much of the learning signal.

That distinction is part of the project's purpose.

---

## Verification

The test suite covers:

- matrix operations and numerical invariants
- stable Softmax
- causal and explicit attention masks
- Q/K/V projection
- attention forward computation
- attention backward propagation
- finite-difference gradient checks for Q, K, and V
- output projection gradients
- training dynamics
- self-attention
- cross-attention
- multi-head structure
- local attention
- MQA / GQA structure
- additive attention structure

The project intentionally treats numerical gradient checking as a first-class correctness tool.

Run the checks with:

~~~powershell
python -m pytest
python -m ruff check .
python -m mypy src
~~~

A GitHub Actions workflow runs the same checks on pushes to main and on pull requests.

---

## Project Structure

~~~text
src/
├── attention/
│   ├── backward.py
│   ├── masking.py
│   ├── qkv.py
│   ├── scaled_dot_product.py
│   │
│   ├── alternatives/
│   │   ├── __init__.py
│   │   └── additive.py
│   │
│   ├── variants/
│   │   ├── self_attention.py
│   │   ├── cross_attention.py
│   │   └── multi_head.py
│   │
│   └── efficiency/
│       ├── local.py
│       ├── multi_query.py
│       └── grouped_query.py
│
├── experiments/
│   ├── attention_demo.py
│   ├── attention_family_demo.py
│   ├── learning_demo.py
│   ├── context_learning_demo.py
│   └── retrieval_learning_demo.py
│
└── math/
    └── matrix.py

tests/
└── ...
~~~

The original single-head learning path remains the verified foundation. Higher-level variants reuse that foundation instead of duplicating the attention algorithm.

---

## Run the Attention Family Demo

~~~powershell
python -m src.experiments.attention_family_demo
~~~

This prints the output shapes and head structure for self-attention, cross-attention, multi-head attention, local attention, MQA, and GQA.

---

## Technology

~~~text
Python 3.12
NumPy
pytest
Ruff
mypy
~~~

The core attention mechanism does not depend on PyTorch, TensorFlow, or a Transformer framework.

The project uses NumPy for numerical array operations while keeping the mathematical steps explicit in Python.

---

## Scope and Non-Goals

This repository is not intended to be a production Transformer implementation.

It currently does not attempt to implement:

- a complete Transformer block
- feed-forward Transformer layers
- residual connections
- layer normalization
- positional encoding systems
- LLM training
- distributed training
- GPU kernels

Those concerns belong to later architectural work.

The boundary of this project is the **attention family itself**.

---

## Learning Path

The repository is part of a connected AI engineering learning path:

~~~text
HowDeepLearningWorks
        ↓
HowAttentionWorks
        ↓
HowTransformersWork
        ↓
HowLLMsWork
        ↓
Production AI Systems
~~~

HowAttentionWorks is the point where the project moves from general neural-learning fundamentals into the mathematical and architectural mechanisms that underpin modern Transformer models.

---

## References

1. Vaswani, A. et al. (2017).
   Attention Is All You Need.
   https://arxiv.org/abs/1706.03762

2. Bahdanau, D. et al. (2014).
   Neural Machine Translation by Jointly Learning to Align and Translate.
   https://arxiv.org/abs/1409.0473

The repository uses these works as conceptual references; implementations here are written independently for inspection and learning.

---

## License

MIT. See LICENSE.
