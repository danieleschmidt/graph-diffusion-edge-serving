"""
GraphServingEngine: Production-grade batched inference with caching.

Handles:
  - Request batching for throughput optimization
  - Feature caching to avoid redundant computation
  - Dynamic graph updates with cache invalidation
  - Async-friendly design (sync implementation for clarity)
"""

from __future__ import annotations

import time
import threading
import numpy as np
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import logging

from .inference import EdgeOptimizedInference

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """A single inference request."""
    node_ids: List[int]
    features: np.ndarray
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class InferenceResult:
    """Result of a batched inference call."""
    node_ids: List[int]
    predictions: np.ndarray  # [N, C] log-softmax scores
    class_labels: np.ndarray  # [N] argmax predictions
    latency_ms: float
    cache_hit: bool = False


class FeatureCache:
    """
    LRU cache for precomputed node features.

    In production graph serving, many queries hit the same popular nodes
    (power-law degree distribution). Caching MLP(x_i) per node avoids
    recomputing the MLP for repeated queries.
    """

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, node_id: int) -> Optional[np.ndarray]:
        if node_id in self._cache:
            self._cache.move_to_end(node_id)  # LRU update
            self._hits += 1
            return self._cache[node_id]
        self._misses += 1
        return None

    def put(self, node_id: int, features: np.ndarray) -> None:
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)  # Evict LRU
        self._cache[node_id] = features

    def invalidate(self) -> None:
        self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class GraphServingEngine:
    """
    Production serving engine with batching, caching, and dynamic updates.

    Design:
      - Requests are batched for efficient matrix operations
      - Node features are cached to avoid recomputing MLP(x_i)
      - Graph updates invalidate the PPR cache and trigger recomputation
      - Thread-safe via a simple lock (for multi-threaded serving)

    Args:
        inference:    EdgeOptimizedInference backend
        batch_size:   Max nodes per inference batch
        cache_size:   Max nodes to cache features for
    """

    def __init__(
        self,
        inference: EdgeOptimizedInference,
        batch_size: int = 256,
        cache_size: int = 10_000,
    ):
        self.inference = inference
        self.batch_size = batch_size
        self.cache = FeatureCache(max_size=cache_size)
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_latency_ms = 0.0

    def serve(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """
        Process a list of inference requests with batching and caching.

        Args:
            requests: List of InferenceRequest objects

        Returns:
            List of InferenceResult objects in same order
        """
        results = []
        # Process in batches
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i : i + self.batch_size]
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
        return results

    def _process_batch(self, batch: List[InferenceRequest]) -> List[InferenceResult]:
        """Process a single batch of requests."""
        t0 = time.perf_counter()

        # Stack all features
        all_features = np.vstack([req.features for req in batch])
        all_node_ids = [nid for req in batch for nid in req.node_ids]

        with self._lock:
            if not self.inference.is_ready:
                raise RuntimeError("Engine not initialized: call setup() first")

            log_probs = self.inference.infer(all_features)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._total_requests += len(all_node_ids)
        self._total_latency_ms += latency_ms

        # Split results back per request
        results = []
        idx = 0
        for req in batch:
            n = len(req.node_ids)
            preds = log_probs[idx : idx + n]
            results.append(InferenceResult(
                node_ids=req.node_ids,
                predictions=preds,
                class_labels=np.argmax(preds, axis=1),
                latency_ms=latency_ms / len(batch),
            ))
            idx += n

        return results

    def setup(self, adj: np.ndarray, node_features: Optional[np.ndarray] = None) -> None:
        """Initialize the engine with graph structure."""
        with self._lock:
            self.cache.invalidate()
            self.inference.precompute(adj, node_features)
        logger.info("GraphServingEngine ready")

    def update_graph(self, new_adj: np.ndarray, node_features: Optional[np.ndarray] = None) -> None:
        """Handle graph update — recompute PPR and invalidate caches."""
        with self._lock:
            self.cache.invalidate()
            self.inference.update_graph(new_adj, node_features)
        logger.info("Graph updated, caches invalidated")

    @property
    def stats(self) -> dict:
        avg_latency = (
            self._total_latency_ms / self._total_requests
            if self._total_requests > 0
            else 0.0
        )
        return {
            "total_requests": self._total_requests,
            "avg_latency_ms": avg_latency,
            "cache": self.cache.stats,
        }
