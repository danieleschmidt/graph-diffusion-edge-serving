"""Tests for GraphServingEngine."""

import numpy as np
import torch
import pytest
from graph_diffusion.model import DiffusionGNN
from graph_diffusion.inference import EdgeOptimizedInference
from graph_diffusion.engine import GraphServingEngine, FeatureCache, InferenceRequest


def make_engine(N=40, in_features=16, num_classes=4, seed=11):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = DiffusionGNN(in_features=in_features, hidden_dim=32, out_features=num_classes)
    model.eval()

    adj = (rng.random((N, N)) > 0.75).astype(np.float32)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)

    x = rng.standard_normal((N, in_features)).astype(np.float32)

    inference = EdgeOptimizedInference(model)
    engine = GraphServingEngine(inference, batch_size=16)
    engine.setup(adj, x)

    return engine, adj, x, N, num_classes


class TestFeatureCache:
    def test_basic_put_get(self):
        cache = FeatureCache(max_size=10)
        feat = np.ones(8)
        cache.put(0, feat)
        result = cache.get(0)
        np.testing.assert_array_equal(result, feat)

    def test_miss_returns_none(self):
        cache = FeatureCache(max_size=10)
        assert cache.get(99) is None

    def test_eviction(self):
        cache = FeatureCache(max_size=3)
        for i in range(4):
            cache.put(i, np.array([float(i)]))
        # Node 0 should be evicted (LRU)
        assert cache.get(0) is None
        assert cache.get(3) is not None

    def test_hit_rate(self):
        cache = FeatureCache(max_size=10)
        cache.put(1, np.ones(4))
        cache.get(1)  # hit
        cache.get(2)  # miss
        assert cache.hit_rate == 0.5

    def test_invalidate(self):
        cache = FeatureCache(max_size=10)
        cache.put(0, np.ones(4))
        cache.invalidate()
        assert cache.get(0) is None


class TestGraphServingEngine:
    def test_serve_basic(self):
        engine, adj, x, N, C = make_engine()
        requests = [InferenceRequest(node_ids=[0, 1, 2], features=x[:3])]
        results = engine.serve(requests)
        assert len(results) == 1
        assert results[0].predictions.shape == (3, C)
        assert results[0].class_labels.shape == (3,)

    def test_serve_multiple_requests(self):
        engine, adj, x, N, C = make_engine()
        requests = [
            InferenceRequest(node_ids=list(range(10)), features=x[:10]),
            InferenceRequest(node_ids=list(range(10, 20)), features=x[10:20]),
        ]
        results = engine.serve(requests)
        assert len(results) == 2
        assert all(r.predictions.shape[1] == C for r in results)

    def test_class_labels_match_argmax(self):
        engine, adj, x, N, C = make_engine()
        requests = [InferenceRequest(node_ids=list(range(N)), features=x)]
        results = engine.serve(requests)
        expected_labels = np.argmax(results[0].predictions, axis=1)
        np.testing.assert_array_equal(results[0].class_labels, expected_labels)

    def test_stats_update(self):
        engine, adj, x, N, C = make_engine()
        requests = [InferenceRequest(node_ids=list(range(10)), features=x[:10])]
        engine.serve(requests)
        stats = engine.stats
        assert stats["total_requests"] == 10
        assert stats["avg_latency_ms"] > 0

    def test_update_graph(self):
        engine, adj, x, N, C = make_engine()
        new_adj = adj.copy()
        new_adj[0, 1] = 1.0 - new_adj[0, 1]
        new_adj[1, 0] = new_adj[0, 1]
        engine.update_graph(new_adj, x)
        # Should still be able to serve
        requests = [InferenceRequest(node_ids=[0, 1], features=x[:2])]
        results = engine.serve(requests)
        assert results[0].predictions.shape == (2, C)

    def test_not_ready_before_setup(self):
        torch.manual_seed(0)
        model = DiffusionGNN(in_features=8, hidden_dim=16, out_features=3)
        inference = EdgeOptimizedInference(model)
        engine = GraphServingEngine(inference)
        x = np.random.randn(5, 8).astype(np.float32)
        requests = [InferenceRequest(node_ids=[0], features=x[:1])]
        with pytest.raises(RuntimeError):
            engine.serve(requests)
