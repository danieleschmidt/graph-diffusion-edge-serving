# graph-diffusion-edge-serving

**Efficient inference for graph neural networks using diffusion-based message passing.**

Precomputes Personalized PageRank (PPR) matrices offline so online inference is a single matrix multiply — no graph traversal at query time.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen.svg)](#testing)

---

## The Core Idea

Standard GNN inference traverses the graph on every query:

```
query → find neighbors → aggregate → find their neighbors → ... → prediction
```

This is expensive at serving time. The key insight: **PPR diffusion captures the full neighborhood structure in a single matrix Π̂**. Precompute it once offline, and online inference becomes:

```
query → MLP(x) → Π̂ @ features → prediction   (one matmul, no graph traversal)
```

**Result: 20x lower latency, 20x higher throughput on a 1000-node graph.**

## Architecture

```
graph_diffusion/
├── model.py      # DiffusionGNN — APPNP-style GNN using PPR as message passing
├── inference.py  # EdgeOptimizedInference — precompute PPR offline, matmul online
├── engine.py     # GraphServingEngine — batching, caching, dynamic graph updates
└── benchmark.py  # ServingBenchmark — latency/throughput/memory comparison
```

### DiffusionGNN

Graph neural network that decouples feature transformation from graph propagation (APPNP-style):

1. **MLP(X)** — transform node features (no graph needed)
2. **Π̂ @ H** — propagate via PPR matrix (pure linear algebra)
3. **Softmax** — classification

The PPR matrix Π̂ = α(I - (1-α)Â)⁻¹ is the key: it captures multi-scale neighborhood information — close neighbors get high weight, distant ones get exponentially less. The teleport probability α controls the locality bias.

### EdgeOptimizedInference

```python
# OFFLINE (once, amortized):
inference = EdgeOptimizedInference(model, alpha=0.15)
inference.precompute(adj_matrix, node_features)   # O(N³) + O(N·d)

# ONLINE (hot path):
predictions = inference.infer(node_features)       # O(N·d) matmul only
```

### GraphServingEngine

Production wrapper adding:
- **Request batching** — groups queries for efficient BLAS operations
- **Feature caching** — LRU cache for MLP(xᵢ) per node (power-law graphs have hot nodes)
- **Dynamic updates** — graph change detection with cache invalidation + PPR recomputation

## Benchmark Results

On a 1000-node stochastic block model graph (9,432 edges, 5 classes, K=10):

| Method | Mean Latency | Throughput | Precompute |
|--------|-------------|------------|------------|
| Online (PPR power iteration at query time) | 1.59ms | 629 q/s | — |
| **Offline (precomputed PPR)** | **0.08ms** | **12,644 q/s** | 205ms (once) |
| **Speedup** | **20x** | **20x** | amortized over queries |

The precompute cost (205ms) is amortized over all queries. After just ~130 queries it breaks even, and savings compound indefinitely until the graph changes.

## Installation

```bash
git clone https://github.com/danieleschmidt/graph-diffusion-edge-serving
cd graph-diffusion-edge-serving
pip install torch numpy scipy
```

No other dependencies.

## Quick Start

```python
import numpy as np
from graph_diffusion import DiffusionGNN, EdgeOptimizedInference

# Your graph
adj = np.array(...)          # [N, N] adjacency matrix
node_features = np.array(... # [N, F] node feature matrix

# Build and train model
model = DiffusionGNN(
    in_features=64,
    hidden_dim=128,
    out_features=5,     # classes
    alpha=0.15,         # PPR teleport probability
    K=10,               # power iteration steps (for training)
)
# ... train model ...

# Setup serving
inference = EdgeOptimizedInference(model, alpha=0.15)
inference.precompute(adj, node_features)  # expensive, done once

# Fast inference (hot path)
log_probs = inference.infer(node_features)
predicted_classes = log_probs.argmax(axis=1)
```

## Demo

```bash
~/anaconda3/bin/python3 examples/node_classification_demo.py
```

Runs a full end-to-end example: synthetic SBM graph → train → serve → benchmark.

## Testing

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
# 28 passed in 0.82s
```

## Design Notes

**When to use this approach:**
- Your graph is relatively static (PPR recomputation cost is amortized)
- You need low-latency serving for individual node queries
- N is large enough that multi-hop graph traversal is expensive

**Tradeoffs:**
- Offline: O(N³) precomputation (matrix inversion). For very large graphs, use approximate PPR (push-based algorithms scale to millions of nodes)
- Memory: O(N²) for the full PPR matrix. Sparse approximations can reduce this significantly
- Dynamic graphs: each edge addition/deletion ideally triggers incremental PPR update (not full recomputation)

**References:**
- Klicpera et al. (2019) — [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997) (APPNP)
- Bojchevski et al. (2020) — [Scaling Graph Neural Networks with Approximate PageRank](https://arxiv.org/abs/2007.01570) (PPRGo)
