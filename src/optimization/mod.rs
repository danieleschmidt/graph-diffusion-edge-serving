//! Optimization modules for quantization, caching, and performance tuning

pub mod quantization;
pub mod caching;
pub mod pruning;
pub mod performance;
pub mod resource_pool;

pub use quantization::{AdaptiveQuantizer, QuantizationConfig};
pub use caching::{SmartCache, CacheConfig, CacheStats, create_cache_key, create_graph_cache_key};
pub use pruning::{GraphPruner, PruningConfig};
pub use performance::{PerformanceOptimizer, PerformanceConfig, VectorizedOps};
pub use resource_pool::{ResourcePool, PoolConfig, PoolableResource, PooledConnection};