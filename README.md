# HowAttentionWorks

> **How can a model learn what context matters?**

`HowAttentionWorks` is a from-scratch exploration of **scaled dot-product attention** and its learning dynamics.

The project is intentionally small, explicit, and inspectable. Instead of hiding attention behind PyTorch, TensorFlow, or a Transformer library, the core mechanism is implemented directly with **Python + NumPy** and verified with **pytest, Ruff, and mypy**.

The repository is not trying to be a production Transformer.

It is trying to answer a more fundamental engineering question:

> **What exactly happens inside attention, and how do its parameters learn through gradient-based optimization?**

---

## Why This Project Exists

Attention is often introduced with a compact equation:

```math
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

That equation is elegant, but by itself it hides most of the engineering and learning mechanics.

This project opens the equation up:

```text
Input representations
        |
        +----> Wq ----> Q
        |
        +----> Wk ----> K
        |
        +----> Wv ----> V
                         |
                         v
                    Q K^T
                         |
                         v
                    Scaling
                         |
                         v
                  Causal Masking
                         |
                         v
                      Softmax
                         |
                         v
                 Attention Weights
                         |
                         v
                   Weights x V
                         |
                         v
                    Attention Output
```

Then the learning path is traced backward:

```text
Loss
  |
  v
dOutput
  |
  v
dAttention / dScores
  |
  +----> dQ
  +----> dK
  +----> dV
          |
          v
   dWq / dWk / dWv
          |
          v
   Gradient Descent
          |
          v
   Updated Parameters
```

---

# What Is Implemented?

The repository currently contains the fundamental components required to build and train a small attention mechanism:

- Matrix operations with explicit shape validation
- Numerically stable Softmax
- Query / Key / Value projections
- Scaled dot-product attention
- Causal masking
- Token embedding lookup
- Cross-entropy loss
- Mean-squared error for controlled retrieval experiments
- Output projection to vocabulary logits
- Backpropagation through attention
- Backpropagation through the output projection
- Analytical gradient propagation into `Wq`, `Wk`, and `Wv`
- Trainable parameter updates
- End-to-end attention training
- Attention snapshots and parameter-change diagnostics
- Context-dependent learning experiments
- Controlled Q/K-only retrieval experiments
- Numerical gradient checking

The implementation deliberately avoids hiding these mechanisms behind a high-level deep-learning framework.

---

# The Core Mechanism

For an input matrix `X`, three learned projections create different views of the same representation:

```math
Q = XW_Q
```

```math
K = XW_K
```

```math
V = XW_V
```

The similarity between queries and keys is computed as:

```math
S=\frac{QK^T}{\sqrt{d_k}}
```

The scores are converted into probabilities with Softmax:

```math
A=\mathrm{softmax}(S)
```

The final attention representation is:

```math
H = AV
```

This gives the complete forward path:

```text
X
|
+-- Wq --> Q --+
|               |
+-- Wk --> K ---+--> QK^T --> /sqrt(dk) --> mask --> softmax --> A
|                                                                    |
+-- Wv --> V --------------------------------------------------------+
                                                                     |
                                                                     v
                                                                    H
```

---

# Why Scaling Exists

The dot product `QK^T` can grow with the dimensionality of the key vectors.

The attention implementation therefore uses:

```math
\frac{QK^T}{\sqrt{d_k}}
```

This keeps score magnitudes in a more useful numerical range before Softmax.

The implementation makes this scaling explicit instead of hiding it in a framework operation.

---

# Causal Attention

For autoregressive language modeling, a token must not see future tokens.

For:

```text
The cat drinks milk
```

the allowed attention pattern is:

```text
          the   cat   drinks   milk

 the      ✓     ·       ·       ·
 cat      ✓     ✓       ·       ·
drinks    ✓     ✓       ✓       ·
 milk     ✓     ✓       ✓       ✓
```

The implementation keeps masked positions out of the Softmax normalization rather than storing `-inf` inside the general-purpose `Matrix` abstraction.

That design is deliberate: the core matrix type enforces finite numerical values, while masking remains a concern of the attention operation.

---

# Backpropagation

The project does not stop at a forward implementation.

For:

```math
H = AV
```

we derive gradients for both the attention weights and the value matrix.

For:

```math
A=\mathrm{softmax}(S)
```

we propagate gradients through Softmax to obtain gradients for the score matrix.

For:

```math
S=\frac{QK^T}{\sqrt{d_k}}
```

we obtain gradients for `Q` and `K`.

For the projection layers:

$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$

we obtain:

```math
\frac{\partial L}{\partial W_Q}=X^T\frac{\partial L}{\partial Q}
```

```math
\frac{\partial L}{\partial W_K}=X^T\frac{\partial L}{\partial K}
```

```math
\frac{\partial L}{\partial W_V}=X^T\frac{\partial L}{\partial V}
```

The gradients are then applied with ordinary gradient descent.

---

# Numerical Gradient Verification

Analytical backpropagation is easy to implement incorrectly.

Therefore the repository contains a finite-difference gradient check for the attention backward pass.

Conceptually:

```text
Analytical gradient
        |
        +------------------+
                           |
                    compare with
                           |
Numerical gradient <--------+
```

The numerical approximation uses:

```math
\frac{\partial L}{\partial x}
\approx
\frac{L(x+\epsilon)-L(x-\epsilon)}{2\epsilon}
```

The analytical and numerical gradients agree within the test tolerances.

This is one of the most important correctness checks in the repository.

---

# Learning Experiments

The project deliberately includes more than one experiment because a decreasing loss does **not** automatically prove that a human-interpretable attention pattern has been learned.

## Experiment 1 — End-to-End Attention Training

A small next-token-style task was trained through the complete pipeline:

```text
Embeddings
   ↓
Q/K/V
   ↓
Causal Attention
   ↓
Output Projection
   ↓
Softmax
   ↓
Cross-Entropy
   ↓
Backpropagation
   ↓
Parameter Updates
```

Observed run:

```text
Initial Loss: 1.595562
Final Loss:   1.280878
Reduction:    0.314684
```

The model therefore successfully optimized the training objective.

However, the attention distribution itself changed very little:

```text
Mean Attention Change: 0.001022
```

The parameter changes were much larger in the value and output-projection paths:

```text
Wq    0.047282
Wk    0.023723
Wv    0.952775
Wout  1.165517
```

This is an important result.

It shows that **loss reduction alone is not evidence that the attention distribution has learned a strong, human-interpretable routing pattern**.

---

## Experiment 2 — Context-Dependent Learning

A second task used examples such as:

```text
red   likes -> sky
blue  likes -> ocean
```

The target could not be determined from the token `likes` alone; the preceding context mattered.

Observed run:

```text
Initial Loss: 1.673147
Final Loss:   0.445288
Reduction:    1.227858
```

Again, the complete network learned successfully.

But the largest parameter changes were still concentrated in the value and output paths:

```text
Wq    0.117372
Wk    0.086821
Wv    1.605810
Wout  2.052373
```

The attention distribution changed only modestly:

```text
Attention Change: 0.004338
```

This reinforced the same conclusion: the network can reduce the objective without substantially reshaping the attention distribution.

---

## Experiment 3 — Q/K-Only Retrieval

To isolate the role of query and key projections, another controlled experiment froze the value pathway and trained only `Wq` and `Wk`.

The target was defined in the same projected value space as the attention output.

Observed run:

```text
Initial Loss: 0.03800584
Final Loss:   0.03747108
Reduction:    0.00053477
```

Attention weights changed only slightly.

This is not treated as a failed implementation.

Instead, it demonstrates that **task design and objective design matter** when trying to isolate a specific learning behavior inside a neural architecture.

The repository therefore does not claim that every loss decrease corresponds to explicit attention routing.

---

# What the Experiments Actually Teach

The experiments lead to a more useful conclusion than a simple success/failure statement.

### We can clearly demonstrate:

```text
Attention is differentiable.
        ↓
Gradients can pass through it.
        ↓
Q/K/V projections can be updated.
        ↓
The complete network can reduce a training objective.
```

### But we should not conclude:

```text
Loss decreased
      ↓
Attention learned an obvious semantic routing pattern
```

That conclusion is not supported by these small experiments.

The model has several parameter pathways available to reduce loss, especially through the value and output-projection layers.

This distinction is intentionally documented because understanding **what an experiment does not prove** is part of understanding machine learning.

---

# Project Architecture

The project is intentionally organized around small responsibilities rather than a large neural-network abstraction.

```text
src/
├── attention/
│   ├── backward.py
│   ├── context_task.py
│   ├── context_training.py
│   ├── cross_entropy.py
│   ├── diagnostics.py
│   ├── inspection.py
│   ├── mse.py
│   ├── output_projection.py
│   ├── output_projection_backward.py
│   ├── qkv.py
│   ├── qkv_backward.py
│   ├── retrieval_task.py
│   ├── retrieval_training.py
│   ├── scaled_dot_product.py
│   ├── sequence.py
│   ├── softmax.py
│   └── training.py
│
├── experiments/
│   ├── attention_demo.py
│   ├── context_learning_demo.py
│   ├── learning_demo.py
│   └── retrieval_learning_demo.py
│
└── math/
    └── matrix.py
```

The important design boundary is between:

```text
Mathematical primitives
        ↓
Attention mechanism
        ↓
Learning / optimization
        ↓
Experiments / inspection
```

This keeps the implementation readable enough that the mathematics can be mapped directly to the code.

---

# Design Principles

## 1. Make the mathematics visible

Core operations are explicit.

The repository does not hide attention behind a high-level API that obscures `Q`, `K`, `V`, scores, weights, and gradients.

## 2. Test the mechanism, not just the final output

The test suite checks individual operations, shapes, numerical stability, gradient flow, and end-to-end learning.

## 3. Verify analytical gradients numerically

Backpropagation is treated as mathematics that must be verified, not code that merely needs to run.

## 4. Do not confuse optimization with understanding

A lower loss is evidence of optimization against a particular objective. It is not automatically evidence of semantic understanding.

## 5. Keep abstractions proportional to the problem

The project deliberately avoids creating interfaces and classes where a simple function or small object is enough.

## 6. Let experiments challenge the implementation

The experiments are designed not only to demonstrate success, but also to expose limitations and alternative explanations for observed behavior.

---

# Testing Strategy

The project currently contains a broad automated test suite covering:

- matrix operations
- Softmax
- Q/K/V projections
- attention scores
- causal masking
- sequence encoding
- attention backward propagation
- numerical gradient verification
- output projection
- output projection backward propagation
- trainable parameter updates
- training loops
- context-dependent tasks
- retrieval tasks
- diagnostics and parameter-change measurements
- end-to-end experiments

The test suite currently contains **94 passing tests**.

Run all tests with:

```powershell
python -m pytest
```

Run static quality checks with:

```powershell
python -m ruff check .
python -m mypy src
```

---

# Technology

```text
Python 3.12
NumPy
pytest
Ruff
mypy
```

No deep-learning framework is required for the core attention implementation.

NumPy provides numerical array operations; the attention mechanism, gradients, training flow, and experiments are implemented explicitly in Python.

---

# Running the Project

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the development dependencies listed by the project configuration.

Then run:

```powershell
python -m pytest
python -m ruff check .
python -m mypy src
```

To inspect the attention mechanism:

```powershell
python -m src.experiments.attention_demo
```

To run the main learning experiment:

```powershell
python -m src.experiments.learning_demo
```

To inspect the context-dependent experiment:

```powershell
python -m src.experiments.context_learning_demo
```

To inspect the controlled retrieval experiment:

```powershell
python -m src.experiments.retrieval_learning_demo
```

---

# What This Project Does Not Try to Be

This repository intentionally does **not** implement:

- Multi-Head Attention
- Transformer blocks
- Positional encodings
- Feed-forward Transformer layers
- Layer normalization
- Transformer decoder architecture
- Large language model training
- Hugging Face Transformers
- PyTorch training loops
- Distributed training

Those are the subject of subsequent work.

The scope of this repository ends with understanding and experimentally validating the core attention mechanism and its learning path.

---

# Why Stop Here?

A project can become less educational when additional architecture is added merely because it is available.

At this point the repository has already demonstrated the complete conceptual chain:

```text
Representation
      ↓
Q / K / V
      ↓
Scaled Similarity
      ↓
Softmax Distribution
      ↓
Weighted Context
      ↓
Loss
      ↓
Backpropagation
      ↓
Parameter Updates
      ↓
Experimental Analysis
```

The next architectural step is not another feature inside this repository.

It is the subject of a new project:

```text
HowAttentionWorks
        ↓
HowTransformersWork
```

---

# Learning Path

This repository is part of a larger progression toward AI engineering:

```text
HowDeepLearningWorks
        ↓
Neural learning fundamentals
        ↓
HowAILearnsLanguage
        ↓
Language representations and training
        ↓
HowAttentionWorks
        ↓
Attention mechanisms and gradient-based learning
        ↓
HowTransformersWork
        ↓
Transformer architecture
        ↓
HowLLMsWork
        ↓
LLM training and inference concepts
        ↓
AIMicroserviceArchitecture
        ↓
Distributed AI systems and multi-agent architecture
```

The projects are deliberately connected rather than being unrelated demonstrations of different technologies.

---

# Final Takeaway

Attention is not merely a formula.

It is a differentiable mechanism that:

```text
1. Builds queries, keys, and values.
2. Computes pairwise compatibility scores.
3. Normalizes those scores into a probability distribution.
4. Uses that distribution to mix value representations.
5. Receives gradients from a downstream objective.
6. Updates its learnable projections through optimization.
```

The experiments also reveal an important engineering lesson:

> **A successful optimization result is not automatically an explanation of what the model learned.**

To understand a neural mechanism, we need both:

```text
Implementation
     +
Mathematical derivation
     +
Automated verification
     +
Controlled experiments
     +
Careful interpretation
```

That is the purpose of `HowAttentionWorks`.

---

## License

MIT
