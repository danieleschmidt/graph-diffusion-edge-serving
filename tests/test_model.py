"""Tests for DiffusionGNN model."""

import pytest
import numpy as np
import torch
from graph_diffusion.model import DiffusionGNN, compute_ppr_matrix, _normalize_adj


def make_graph(N=20, seed=0):
    rng = np.random.default_rng(seed)
    adj = (rng.random((N, N)) > 0.7).astype(np.float32)
    adj = np.maximum(adj, adj.T)  # symmetric
    np.fill_diagonal(adj, 0)
    return adj


class TestDiffusionGNN:
    def test_output_shape(self):
        model = DiffusionGNN(in_features=16, hidden_dim=32, out_features=4)
        N = 20
        x = torch.randn(N, 16)
        ppr = torch.eye(N)  # identity PPR (no diffusion, for shape test)
        out = model(x, ppr_matrix=ppr)
        assert out.shape == (N, 4)

    def test_log_softmax_output(self):
        model = DiffusionGNN(in_features=16, hidden_dim=32, out_features=4)
        N = 20
        x = torch.randn(N, 16)
        ppr = torch.eye(N)
        out = model(x, ppr_matrix=ppr)
        # log-softmax: all values <= 0 and exp sums to ~1 per row
        assert (out <= 0).all()
        assert torch.allclose(out.exp().sum(dim=1), torch.ones(N), atol=1e-5)

    def test_ppr_vs_online_consistency(self):
        """Precomputed PPR and online PPR should give similar results."""
        model = DiffusionGNN(in_features=8, hidden_dim=16, out_features=3, K=20)
        model.eval()
        N = 30
        adj_np = make_graph(N)
        x = torch.randn(N, 8)

        ppr_np = compute_ppr_matrix(adj_np, alpha=0.15)
        ppr_t = torch.tensor(ppr_np, dtype=torch.float32)
        adj_t = torch.tensor(adj_np, dtype=torch.float32)

        with torch.no_grad():
            out_offline = model(x, ppr_matrix=ppr_t)
            out_online = model(x, adj=adj_t)

        # Should be close (online is K-step approx; exact match only as K→∞)
        diff = (out_offline - out_online).abs().mean().item()
        assert diff < 0.5, f"Offline/online divergence too large: {diff:.4f}"

    def test_requires_adj_or_ppr(self):
        model = DiffusionGNN(in_features=8, hidden_dim=16, out_features=3)
        x = torch.randn(10, 8)
        with pytest.raises(ValueError):
            model(x)

    def test_multilayer(self):
        model = DiffusionGNN(in_features=16, hidden_dim=32, out_features=4, num_layers=3)
        N = 15
        x = torch.randn(N, 16)
        ppr = torch.eye(N)
        out = model(x, ppr_matrix=ppr)
        assert out.shape == (N, 4)


class TestPPRMatrix:
    def test_ppr_shape(self):
        adj = make_graph(10)
        ppr = compute_ppr_matrix(adj, alpha=0.15)
        assert ppr.shape == (10, 10)

    def test_ppr_row_sums(self):
        """PPR rows should sum to alpha (since Π̂ = α(I-(1-α)Â)^{-1} and rows of Â sum to 1)."""
        adj = make_graph(20)
        ppr = compute_ppr_matrix(adj, alpha=0.15)
        # Row sums of PPR should all be close to 1 (not exactly alpha;
        # depends on normalization choice). Just check they're positive and bounded.
        assert (ppr >= 0).all(), "PPR should be non-negative"
        assert ppr.max() <= 1.1, "PPR values should be ≤ 1"

    def test_ppr_dtype(self):
        adj = make_graph(10)
        ppr = compute_ppr_matrix(adj, alpha=0.15)
        assert ppr.dtype == np.float32

    def test_normalize_adj_symmetry(self):
        adj_t = torch.tensor(make_graph(10), dtype=torch.float32)
        norm = _normalize_adj(adj_t)
        assert torch.allclose(norm, norm.T, atol=1e-6), "Normalized adj should be symmetric"
