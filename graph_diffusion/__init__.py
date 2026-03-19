"""
Graph Diffusion Edge Serving
============================

Efficient inference for graph neural networks using diffusion-based message passing.
Precomputes Personalized PageRank (PPR) matrices offline so online inference
is a single matrix multiply — no graph traversal at inference time.
"""

from .model import DiffusionGNN
from .inference import EdgeOptimizedInference
from .engine import GraphServingEngine
from .benchmark import ServingBenchmark

__all__ = [
    "DiffusionGNN",
    "EdgeOptimizedInference",
    "GraphServingEngine",
    "ServingBenchmark",
]
__version__ = "0.1.0"
