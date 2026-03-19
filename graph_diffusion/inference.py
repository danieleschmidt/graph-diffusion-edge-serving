"""
EdgeOptimizedInference: Serving system that precomputes PPR offline.

The core insight for edge-optimized serving:
  - OFFLINE: compute Π̂ = α(I - (1-α)Â)^{-1}  [expensive, one-time]
  - ONLINE:  H = Π̂ @ MLP(x_query)              [cheap, just matmul]

This eliminates all graph traversal from the hot path. Inference for
a single node is O(N·d) instead of O(|E|·d) per layer.
"""

from __future__ import annotations

import time
import torch
import numpy as np
from torch import Tensor
from typing import Optional
import logging

from .model import DiffusionGNN, compute_ppr_matrix

logger = logging.getLogger(__name__)


class EdgeOptimizedInference:
    """
    Serving wrapper that precomputes the PPR diffusion matrix offline,
    so online inference requires no graph traversal.

    Usage:
        # Setup (done once, offline)
        engine = EdgeOptimizedInference(model, alpha=0.15)
        engine.precompute(adj_matrix)

        # Serving (hot path — pure matmul)
        logits = engine.infer(node_features)
    """

    def __init__(
        self,
        model: DiffusionGNN,
        alpha: float = 0.15,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.alpha = alpha
        self.device = device

        self._ppr_matrix: Optional[Tensor] = None
        self._precomputed_features: Optional[Tensor] = None  # MLP(X) cache
        self._graph_hash: Optional[int] = None

    @property
    def is_ready(self) -> bool:
        return self._ppr_matrix is not None

    def precompute(self, adj: np.ndarray, node_features: Optional[np.ndarray] = None) -> None:
        """
        Precompute the PPR diffusion matrix (and optionally MLP features).

        This is the expensive offline step. For a graph with N nodes:
          - PPR computation: O(N^3) via matrix inversion
          - Feature precomputation: O(N * hidden_dim)

        After this, online inference is O(N) per node query.

        Args:
            adj:           Adjacency matrix [N, N]
            node_features: Optional node feature matrix [N, F].
                          If provided, precomputes MLP(X) for all nodes.
        """
        logger.info(f"Precomputing PPR matrix for graph with {adj.shape[0]} nodes...")
        t0 = time.perf_counter()

        ppr = compute_ppr_matrix(adj, alpha=self.alpha)
        self._ppr_matrix = torch.tensor(ppr, dtype=torch.float32).to(self.device)

        elapsed = time.perf_counter() - t0
        logger.info(f"PPR precomputed in {elapsed*1000:.1f}ms")

        if node_features is not None:
            self._precompute_features(node_features)

        self._graph_hash = hash(adj.tobytes())

    def _precompute_features(self, node_features: np.ndarray) -> None:
        """Precompute MLP(X) for all nodes — eliminates MLP from hot path."""
        logger.info("Precomputing MLP features for all nodes...")
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            # Run only the MLP part (no propagation)
            h = self.model.mlp(x)
            self._precomputed_features = h
        logger.info(f"Precomputed features shape: {self._precomputed_features.shape}")

    def infer(self, node_features: np.ndarray) -> np.ndarray:
        """
        Fast online inference — no graph traversal, just matrix multiply.

        If features were precomputed, this is Π̂ @ cached_features (slice).
        Otherwise: Π̂ @ MLP(X).

        Args:
            node_features: Node feature matrix [N, F]

        Returns:
            Log-softmax class scores [N, C]
        """
        if not self.is_ready:
            raise RuntimeError("Call precompute() before infer()")

        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            if self._precomputed_features is not None:
                # Ultra-fast: PPR @ precomputed MLP features
                h = self._ppr_matrix @ self._precomputed_features
                import torch.nn.functional as F
                return F.log_softmax(h, dim=1).cpu().numpy()
            else:
                # Fast: PPR @ MLP(X) — still no graph traversal
                return self.model(x, ppr_matrix=self._ppr_matrix).cpu().numpy()

    def infer_online(self, node_features: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        Baseline online inference with graph traversal (for benchmarking).

        Runs full PPR power iteration at inference time — the slow path.
        """
        x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
        a = torch.tensor(adj, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(x, adj=a).cpu().numpy()

    def update_graph(self, new_adj: np.ndarray, node_features: Optional[np.ndarray] = None) -> None:
        """
        Handle dynamic graph updates by recomputing PPR.

        For production systems, consider incremental PPR updates
        rather than full recomputation.
        """
        new_hash = hash(new_adj.tobytes())
        if new_hash != self._graph_hash:
            logger.info("Graph changed, recomputing PPR...")
            self._precomputed_features = None
            self.precompute(new_adj, node_features)
        else:
            logger.info("Graph unchanged, skipping recomputation")
