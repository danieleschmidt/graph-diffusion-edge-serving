//! Smart caching for graph diffusion results

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl_seconds: 3600,
        }
    }
}

pub struct SmartCache {
    config: CacheConfig,
    cache: HashMap<String, Vec<f32>>,
}

impl SmartCache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Vec<f32>> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: String, value: Vec<f32>) {
        if self.cache.len() >= self.config.max_entries {
            // Simple eviction - remove first entry
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, value);
    }
}