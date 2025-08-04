//! Optimization modules for quantization, caching, and performance tuning

pub mod quantization;
pub mod caching;
pub mod pruning;

pub use quantization::{AdaptiveQuantizer, QuantizationConfig};
pub use caching::{SmartCache, CacheConfig};
pub use pruning::{GraphPruner, PruningConfig};