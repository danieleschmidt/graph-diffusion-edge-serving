"""
Node Classification Demo
========================

Demonstrates DiffusionGNN on a synthetic graph (1000 nodes, 5 classes).
Shows 10x+ speedup from offline PPR precomputation vs online inference.

Run:
    ~/anaconda3/bin/python3 examples/node_classification_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from graph_diffusion import DiffusionGNN, EdgeOptimizedInference, GraphServingEngine, ServingBenchmark
from graph_diffusion.engine import InferenceRequest


def make_synthetic_graph(num_nodes: int = 1000, num_classes: int = 5, avg_degree: int = 10, seed: int = 42):
    """
    Create a synthetic graph for node classification.

    Generates a stochastic block model (SBM): nodes in the same class
    are more likely to be connected than cross-class nodes.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    nodes_per_class = num_nodes // num_classes
    labels = np.repeat(np.arange(num_classes), nodes_per_class)

    # SBM adjacency: intra-class p_in >> inter-class p_out
    p_in = avg_degree / nodes_per_class * 0.8
    p_out = avg_degree / num_nodes * 0.2

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                adj[i, j] = 1.0
                adj[j, i] = 1.0

    # Node features: class-discriminative features + noise
    in_features = 64
    x = np.zeros((num_nodes, in_features), dtype=np.float32)
    for c in range(num_classes):
        mask = labels == c
        center = rng.standard_normal(in_features) * 2
        x[mask] = center + rng.standard_normal((mask.sum(), in_features)) * 0.5

    return adj, x, labels


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return (preds == labels).mean()


def main():
    print("=" * 60)
    print("Graph Diffusion Edge Serving — Node Classification Demo")
    print("=" * 60)

    # ── 1. Generate synthetic graph ──────────────────────────────
    NUM_NODES = 1000
    NUM_CLASSES = 5
    IN_FEATURES = 64
    HIDDEN_DIM = 128
    ALPHA = 0.15

    print(f"\n📊 Synthetic graph: {NUM_NODES} nodes, {NUM_CLASSES} classes")
    adj, x, labels = make_synthetic_graph(NUM_NODES, NUM_CLASSES, seed=42)
    num_edges = int(adj.sum()) // 2
    print(f"   Edges: {num_edges}")
    print(f"   Avg degree: {adj.sum(axis=1).mean():.1f}")
    print(f"   Feature dim: {IN_FEATURES}")

    # ── 2. Build model ────────────────────────────────────────────
    print("\n🧠 DiffusionGNN (APPNP-style)")
    model = DiffusionGNN(
        in_features=IN_FEATURES,
        hidden_dim=HIDDEN_DIM,
        out_features=NUM_CLASSES,
        num_layers=2,
        alpha=ALPHA,
        K=10,
    )
    print(f"   Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Alpha (teleport): {ALPHA}")
    print(f"   K (power iter steps): 10")

    # ── 3. Quick training (few epochs for demo) ───────────────────
    print("\n⚙️  Training (50 epochs for demo)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)

    # Use precomputed PPR for training too (faster)
    from graph_diffusion.model import compute_ppr_matrix
    ppr = compute_ppr_matrix(adj, alpha=ALPHA)
    ppr_t = torch.tensor(ppr, dtype=torch.float32)

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        logits = model(x_t, ppr_matrix=ppr_t)
        loss = torch.nn.functional.nll_loss(logits, y_t)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            acc = accuracy(logits.detach().numpy(), labels)
            print(f"   Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

    model.eval()
    with torch.no_grad():
        final_logits = model(x_t, ppr_matrix=ppr_t).numpy()
    final_acc = accuracy(final_logits, labels)
    print(f"   Final accuracy: {final_acc:.3f}")

    # ── 4. Serving setup ──────────────────────────────────────────
    print("\n🚀 EdgeOptimizedInference — precomputing PPR offline...")
    inference = EdgeOptimizedInference(model, alpha=ALPHA)
    inference.precompute(adj, x)

    # Verify predictions match
    offline_logits = inference.infer(x)
    offline_acc = accuracy(offline_logits, labels)
    print(f"   Offline inference accuracy: {offline_acc:.3f} (should match training)")

    # ── 5. GraphServingEngine ─────────────────────────────────────
    print("\n⚙️  GraphServingEngine — batched request serving...")
    engine = GraphServingEngine(inference, batch_size=128)
    engine.setup(adj, x)

    # Simulate 5 query batches
    rng = np.random.default_rng(99)
    requests = [
        InferenceRequest(
            node_ids=list(range(i, min(i + 50, NUM_NODES))),
            features=x[i : min(i + 50, NUM_NODES)],
        )
        for i in range(0, 250, 50)
    ]
    results = engine.serve(requests)
    served_acc = accuracy(
        np.vstack([r.predictions for r in results]),
        labels[:250],
    )
    print(f"   Served {sum(len(r.node_ids) for r in results)} nodes | Acc: {served_acc:.3f}")
    print(f"   Engine stats: {engine.stats}")

    # ── 6. Graph update demo ──────────────────────────────────────
    print("\n🔄 Dynamic graph update (adding 5% new edges)...")
    rng2 = np.random.default_rng(123)
    new_adj = adj.copy()
    extra_edges = rng2.integers(0, NUM_NODES, size=(250, 2))
    for i, j in extra_edges:
        if i != j:
            new_adj[i, j] = 1.0
            new_adj[j, i] = 1.0
    new_edges = int(new_adj.sum()) // 2 - num_edges
    print(f"   Added {new_edges} edges → triggering PPR recomputation")
    engine.update_graph(new_adj, x)
    print("   ✓ PPR updated, cache invalidated")

    # ── 7. Benchmark ──────────────────────────────────────────────
    print("\n📊 Benchmarking online vs offline inference...")
    bench = ServingBenchmark(model, num_queries=50, warmup=5)
    online_r, offline_r = bench.run(adj, x, alpha=ALPHA, verbose=True)

    speedup = online_r.mean_latency_ms / offline_r.mean_latency_ms
    if speedup >= 10:
        print(f"\n✅ {speedup:.1f}x speedup achieved (target: 10x+)")
    else:
        print(f"\n⚠️  {speedup:.1f}x speedup (graph may be too small for 10x; "
              f"speedup grows with K×|E| vs N²·d ratio)")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
