//! Optimization modules for quantization, caching, and performance tuning

pub mod quantization;
pub mod caching;
pub mod pruning;
pub mod performance;

pub use quantization::{AdaptiveQuantizer, QuantizationConfig};
pub use caching::{SmartCache, CacheConfig};
pub use pruning::{GraphPruner, PruningConfig};
pub use performance::{PerformanceOptimizer, PerformanceConfig, VectorizedOps};