"""
ServingBenchmark: Measures latency, throughput, and memory usage.

Compares:
  - Online inference: full PPR power iteration at query time
  - Offline (edge-optimized): precomputed PPR, inference is matmul only
"""

from __future__ import annotations

import time
import tracemalloc
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
import logging

from .model import DiffusionGNN
from .inference import EdgeOptimizedInference

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    num_nodes: int
    num_queries: int

    # Latency
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float

    # Throughput
    queries_per_second: float

    # Memory
    peak_memory_mb: float

    # Precompute cost (offline only)
    precompute_time_ms: float = 0.0

    def __str__(self) -> str:
        return (
            f"[{self.name}] "
            f"Latency: {self.mean_latency_ms:.2f}ms (p50={self.p50_latency_ms:.2f}, "
            f"p95={self.p95_latency_ms:.2f}, p99={self.p99_latency_ms:.2f}) | "
            f"Throughput: {self.queries_per_second:.0f} q/s | "
            f"Memory: {self.peak_memory_mb:.1f}MB"
            + (f" | Precompute: {self.precompute_time_ms:.0f}ms" if self.precompute_time_ms else "")
        )


class ServingBenchmark:
    """
    Benchmark suite for graph diffusion serving.

    Measures the speedup from precomputing the PPR diffusion matrix
    versus computing it online at inference time.

    Args:
        model:       DiffusionGNN instance to benchmark
        num_queries: Number of inference calls to time
        warmup:      Warmup iterations before measuring
    """

    def __init__(
        self,
        model: DiffusionGNN,
        num_queries: int = 100,
        warmup: int = 10,
    ):
        self.model = model
        self.num_queries = num_queries
        self.warmup = warmup

    def run(
        self,
        adj: np.ndarray,
        node_features: np.ndarray,
        alpha: float = 0.15,
        verbose: bool = True,
    ) -> tuple[BenchmarkResult, BenchmarkResult]:
        """
        Run both online and offline benchmarks.

        Returns:
            (online_result, offline_result)
        """
        N = adj.shape[0]
        logger.info(f"Benchmarking on graph with {N} nodes, {num_edges(adj)} edges")

        online = self._benchmark_online(adj, node_features, N, alpha)
        offline = self._benchmark_offline(adj, node_features, N, alpha)

        if verbose:
            print("\n" + "=" * 70)
            print("GRAPH DIFFUSION SERVING BENCHMARK")
            print("=" * 70)
            print(f"Graph: {N} nodes, {num_edges(adj)} edges")
            print(f"Queries: {self.num_queries}")
            print()
            print(str(online))
            print(str(offline))
            print()

            speedup = online.mean_latency_ms / offline.mean_latency_ms
            tp_ratio = offline.queries_per_second / online.queries_per_second
            print(f"🚀 Speedup:    {speedup:.1f}x lower latency (offline vs online)")
            print(f"📈 Throughput: {tp_ratio:.1f}x more queries/sec")
            mem_overhead = offline.peak_memory_mb - online.peak_memory_mb
            print(f"💾 Memory overhead of offline: +{mem_overhead:.1f}MB (for PPR matrix)")
            print("=" * 70)

        return online, offline

    def _benchmark_online(
        self, adj: np.ndarray, x: np.ndarray, N: int, alpha: float
    ) -> BenchmarkResult:
        """Benchmark online PPR computation at inference time."""
        inference = EdgeOptimizedInference(self.model, alpha=alpha)
        # No precompute — graph traversal happens at query time

        adj_torch = torch.tensor(adj, dtype=torch.float32)
        x_torch = torch.tensor(x, dtype=torch.float32)

        # Warmup
        for _ in range(self.warmup):
            inference.infer_online(x, adj)

        latencies = []
        tracemalloc.start()
        t_total = time.perf_counter()

        for _ in range(self.num_queries):
            t0 = time.perf_counter()
            inference.infer_online(x, adj)
            latencies.append((time.perf_counter() - t0) * 1000)

        total_time = time.perf_counter() - t_total
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latencies = np.array(latencies)
        return BenchmarkResult(
            name="Online (graph traversal at inference time)",
            num_nodes=N,
            num_queries=self.num_queries,
            mean_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            queries_per_second=self.num_queries / total_time,
            peak_memory_mb=peak / 1024 / 1024,
        )

    def _benchmark_offline(
        self, adj: np.ndarray, x: np.ndarray, N: int, alpha: float
    ) -> BenchmarkResult:
        """Benchmark offline-precomputed serving (matmul only at inference time)."""
        inference = EdgeOptimizedInference(self.model, alpha=alpha)

        # Precompute PPR (offline cost)
        t_pre = time.perf_counter()
        inference.precompute(adj, x)
        precompute_ms = (time.perf_counter() - t_pre) * 1000

        # Warmup
        for _ in range(self.warmup):
            inference.infer(x)

        latencies = []
        tracemalloc.start()
        t_total = time.perf_counter()

        for _ in range(self.num_queries):
            t0 = time.perf_counter()
            inference.infer(x)
            latencies.append((time.perf_counter() - t0) * 1000)

        total_time = time.perf_counter() - t_total
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latencies = np.array(latencies)
        return BenchmarkResult(
            name="Offline (precomputed PPR — matmul only)",
            num_nodes=N,
            num_queries=self.num_queries,
            mean_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            queries_per_second=self.num_queries / total_time,
            peak_memory_mb=peak / 1024 / 1024,
            precompute_time_ms=precompute_ms,
        )


def num_edges(adj: np.ndarray) -> int:
    return int(np.sum(adj > 0))
