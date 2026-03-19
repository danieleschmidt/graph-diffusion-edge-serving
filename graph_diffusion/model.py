"""
DiffusionGNN: Graph Neural Network with diffusion-based message passing.

Uses Personalized PageRank (PPR) approximation for multi-scale neighborhood
aggregation. The key insight: PPR captures multi-hop structure in a single
matrix, acting as a learned, topology-aware aggregation kernel.

Reference: APPNP (Klicpera et al., 2019) — "Predict then Propagate"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import scipy.sparse as sp
import numpy as np
from typing import Optional


class DiffusionGNN(nn.Module):
    """
    Graph neural network using PPR diffusion as message passing.

    Architecture:
        1. Transform node features with an MLP (no graph ops)
        2. Propagate via precomputed PPR matrix Π̂ (graph structure)
        3. Final linear layer for classification

    This decoupled design (APPNP-style) means the graph propagation
    step can be precomputed and cached for serving.

    Args:
        in_features:  Dimension of input node features
        hidden_dim:   Hidden layer width
        out_features: Number of output classes
        num_layers:   MLP depth before propagation
        alpha:        PPR teleport probability (restart probability)
        K:            Number of power-iteration steps for PPR approximation
        dropout:      Dropout rate during training
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        num_layers: int = 2,
        alpha: float = 0.15,
        K: int = 10,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha
        self.K = K
        self.dropout = dropout

        # Feature transformation MLP (no graph needed)
        layers: list[nn.Module] = []
        dims = [in_features] + [hidden_dim] * (num_layers - 1) + [out_features]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
        ppr_matrix: Optional[Tensor] = None,
        adj: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass.

        Args:
            x:          Node features [N, in_features]
            ppr_matrix: Precomputed PPR matrix [N, N] (fast serving path)
            adj:        Adjacency matrix [N, N] (fallback online path)

        Returns:
            Log-softmax class scores [N, out_classes]
        """
        # Feature transformation (pure MLP, no graph)
        h = F.dropout(x, p=self.dropout, training=self.training)
        h = self.mlp(h)

        # Graph propagation via PPR
        if ppr_matrix is not None:
            # Fast path: precomputed PPR — just a matrix multiply
            h = ppr_matrix @ h
        elif adj is not None:
            # Online path: iterative PPR approximation
            h = self._ppr_propagate(h, adj)
        else:
            raise ValueError("Must provide either ppr_matrix or adj")

        return F.log_softmax(h, dim=1)

    def _ppr_propagate(self, h: Tensor, adj: Tensor) -> Tensor:
        """
        Approximate PPR via power iteration:
            H^(k) = (1 - α) * Â * H^(k-1) + α * H^(0)

        where Â is the normalized adjacency (with self-loops).
        """
        # Normalize adjacency: Â = D^{-1/2} (A + I) D^{-1/2}
        adj_hat = _normalize_adj(adj)
        h0 = h
        for _ in range(self.K):
            h = (1 - self.alpha) * (adj_hat @ h) + self.alpha * h0
        return h


def _normalize_adj(adj: Tensor) -> Tensor:
    """Symmetric normalization: Â = D^{-1/2} A D^{-1/2} (with self-loops)."""
    adj = adj + torch.eye(adj.size(0), device=adj.device)
    deg = adj.sum(dim=1)
    d_inv_sqrt = torch.pow(deg, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    return d_inv_sqrt.unsqueeze(1) * adj * d_inv_sqrt.unsqueeze(0)


def compute_ppr_matrix(
    adj: np.ndarray | sp.spmatrix,
    alpha: float = 0.15,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute exact Personalized PageRank matrix.

    Π̂ = α (I - (1-α) Â)^{-1}

    For serving, this is precomputed ONCE offline. Online inference
    then only needs: H = Π̂ @ MLP(X), a single BLAS call.

    Args:
        adj:   Adjacency matrix (dense numpy or scipy sparse)
        alpha: Teleport probability (higher = more local)
        eps:   Small value for numerical stability

    Returns:
        PPR matrix of shape [N, N]
    """
    if sp.issparse(adj):
        adj = adj.toarray().astype(np.float32)
    else:
        adj = adj.astype(np.float32)

    N = adj.shape[0]
    # Add self-loops and normalize
    adj = adj + np.eye(N)
    deg = adj.sum(axis=1, keepdims=True)
    deg = np.maximum(deg, eps)
    adj_norm = adj / deg  # Row-normalized

    # Exact PPR: α (I - (1-α) A_norm)^{-1}
    M = np.eye(N) - (1 - alpha) * adj_norm
    ppr = alpha * np.linalg.inv(M)
    return ppr.astype(np.float32)
