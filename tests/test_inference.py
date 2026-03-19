"""Tests for EdgeOptimizedInference."""

import pytest
import numpy as np
import torch
from graph_diffusion.model import DiffusionGNN
from graph_diffusion.inference import EdgeOptimizedInference


def make_model_and_graph(N=30, in_features=16, num_classes=3, seed=7):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = DiffusionGNN(in_features=in_features, hidden_dim=32, out_features=num_classes)
    model.eval()

    adj = (rng.random((N, N)) > 0.7).astype(np.float32)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)

    x = rng.standard_normal((N, in_features)).astype(np.float32)
    return model, adj, x


class TestEdgeOptimizedInference:
    def test_not_ready_before_precompute(self):
        model, adj, x = make_model_and_graph()
        inference = EdgeOptimizedInference(model)
        assert not inference.is_ready
        with pytest.raises(RuntimeError):
            inference.infer(x)

    def test_ready_after_precompute(self):
        model, adj, x = make_model_and_graph()
        inference = EdgeOptimizedInference(model)
        inference.precompute(adj)
        assert inference.is_ready

    def test_infer_output_shape(self):
        model, adj, x = make_model_and_graph(N=30, in_features=16, num_classes=3)
        inference = EdgeOptimizedInference(model)
        inference.precompute(adj)
        out = inference.infer(x)
        assert out.shape == (30, 3)

    def test_infer_log_probs(self):
        model, adj, x = make_model_and_graph()
        inference = EdgeOptimizedInference(model)
        inference.precompute(adj)
        out = inference.infer(x)
        assert (out <= 0).all(), "log-softmax should be ≤ 0"

    def test_precompute_with_features(self):
        """Precomputing features should give same results as not precomputing."""
        model, adj, x = make_model_and_graph(N=20)
        model.eval()

        inf_no_feat = EdgeOptimizedInference(model)
        inf_no_feat.precompute(adj)

        inf_with_feat = EdgeOptimizedInference(model)
        inf_with_feat.precompute(adj, x)

        out1 = inf_no_feat.infer(x)
        out2 = inf_with_feat.infer(x)

        np.testing.assert_allclose(out1, out2, atol=1e-5)

    def test_update_graph_same(self):
        """Updating with same graph should not recompute."""
        model, adj, x = make_model_and_graph()
        inference = EdgeOptimizedInference(model)
        inference.precompute(adj)
        old_hash = inference._graph_hash
        inference.update_graph(adj)
        assert inference._graph_hash == old_hash

    def test_update_graph_different(self):
        """Updating with different graph should recompute."""
        model, adj, x = make_model_and_graph(N=20)
        inference = EdgeOptimizedInference(model)
        inference.precompute(adj)

        new_adj = adj.copy()
        new_adj[0, 1] = 1.0 - new_adj[0, 1]  # flip an edge
        new_adj[1, 0] = new_adj[0, 1]

        old_hash = inference._graph_hash
        inference.update_graph(new_adj)
        assert inference._graph_hash != old_hash

    def test_online_inference_shape(self):
        model, adj, x = make_model_and_graph(N=20, in_features=16, num_classes=3)
        inference = EdgeOptimizedInference(model)
        out = inference.infer_online(x, adj)
        assert out.shape == (20, 3)
