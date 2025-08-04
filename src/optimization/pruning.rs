//! Graph pruning for performance optimization

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    pub sparsity_threshold: f32,
    pub min_degree: usize,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            sparsity_threshold: 0.1,
            min_degree: 2,
        }
    }
}

pub struct GraphPruner {
    config: PruningConfig,
}

impl GraphPruner {
    pub fn new(config: PruningConfig) -> Self {
        Self { config }
    }

    pub fn should_prune_edge(&self, weight: f32) -> bool {
        weight.abs() < self.config.sparsity_threshold
    }
}