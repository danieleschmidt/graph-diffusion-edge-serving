//! Smart caching system for graph embeddings and computation results

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub max_entries: usize,
    pub ttl_seconds: u64,
    pub enable_lru: bool,
    pub enable_compression: bool,
    pub max_memory_mb: usize,
    pub cleanup_interval_seconds: u64,
    pub enable_metrics: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            ttl_seconds: 3600, // 1 hour
            enable_lru: true,
            enable_compression: false,
            max_memory_mb: 512,
            cleanup_interval_seconds: 300, // 5 minutes
            enable_metrics: true,
        }
    }
}

#[derive(Debug, Clone)]
struct CacheEntry<T> {
    data: T,
    created_at: Instant,
    last_accessed: Instant,
    access_count: u64,
    size_bytes: usize,
}

impl<T> CacheEntry<T> {
    fn new(data: T, size_bytes: usize) -> Self {
        let now = Instant::now();
        Self {
            data,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            size_bytes,
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
        self.access_count += 1;
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub memory_usage_mb: f64,
    pub hit_rate: f64,
    pub total_requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub expired_entries: u64,
    pub average_access_count: f64,
}

pub struct SmartCache<T> 
where 
    T: Clone + Send + Sync + 'static,
{
    cache: Arc<RwLock<HashMap<String, CacheEntry<T>>>>,
    lru_order: Arc<RwLock<VecDeque<String>>>,
    config: CacheConfig,
    stats: Arc<RwLock<CacheStats>>,
    last_cleanup: Arc<RwLock<Instant>>,
}

impl<T> SmartCache<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub fn new(config: CacheConfig) -> Self {
        info!("Initializing smart cache with {} max entries, {}MB memory limit", 
              config.max_entries, config.max_memory_mb);
        
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            lru_order: Arc::new(RwLock::new(VecDeque::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats {
                total_entries: 0,
                memory_usage_mb: 0.0,
                hit_rate: 0.0,
                total_requests: 0,
                cache_hits: 0,
                cache_misses: 0,
                evictions: 0,
                expired_entries: 0,
                average_access_count: 0.0,
            })),
            last_cleanup: Arc::new(RwLock::new(Instant::now())),
        }
    }

    pub async fn get(&self, key: &str) -> Option<T> {
        // Periodic cleanup if needed
        self.cleanup_if_needed().await;

        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        
        stats.total_requests += 1;

        if let Some(entry) = cache.get_mut(key) {
            // Check if entry is expired
            if entry.is_expired(Duration::from_secs(self.config.ttl_seconds)) {
                let size_mb = entry.size_bytes as f64 / 1024.0 / 1024.0;
                cache.remove(key);
                stats.expired_entries += 1;
                stats.cache_misses += 1;
                stats.total_entries -= 1;
                stats.memory_usage_mb -= size_mb;
                
                if self.config.enable_lru {
                    let mut lru = self.lru_order.write().await;
                    lru.retain(|k| k != key);
                }
                
                debug!("Cache entry expired for key: {}", key);
                return None;
            }

            // Update access information
            entry.touch();
            stats.cache_hits += 1;
            
            if self.config.enable_lru {
                // Move to back of LRU queue
                let mut lru = self.lru_order.write().await;
                lru.retain(|k| k != key);
                lru.push_back(key.to_string());
            }
            
            debug!("Cache hit for key: {} (access count: {})", key, entry.access_count);
            Some(entry.data.clone())
        } else {
            stats.cache_misses += 1;
            debug!("Cache miss for key: {}", key);
            None
        }
    }

    pub async fn put(&self, key: String, value: T) -> bool {
        let size_bytes = self.estimate_size(&value);
        
        // Check memory constraints
        let stats = self.stats.read().await;
        let projected_memory = stats.memory_usage_mb + (size_bytes as f64 / 1024.0 / 1024.0);
        drop(stats);

        if projected_memory > self.config.max_memory_mb as f64 {
            warn!("Cache memory limit would be exceeded ({:.2}MB > {}MB), evicting entries", 
                  projected_memory, self.config.max_memory_mb);
            self.evict_lru().await;
        }

        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        // Check entry count limit
        if cache.len() >= self.config.max_entries {
            drop(cache);
            drop(stats);
            self.evict_lru().await;
            cache = self.cache.write().await;
            stats = self.stats.write().await;
        }

        let entry = CacheEntry::new(value, size_bytes);
        
        // If key already exists, update memory calculation
        if let Some(old_entry) = cache.insert(key.clone(), entry) {
            let old_size = old_entry.size_bytes as f64 / 1024.0 / 1024.0;
            stats.memory_usage_mb -= old_size;
        } else {
            stats.total_entries += 1;
        }

        stats.memory_usage_mb += size_bytes as f64 / 1024.0 / 1024.0;

        if self.config.enable_lru {
            let mut lru = self.lru_order.write().await;
            lru.retain(|k| k != &key); // Remove if exists
            lru.push_back(key.clone());
        }

        debug!("Cached entry for key: {} ({} bytes, {:.2}MB total)", 
               key, size_bytes, stats.memory_usage_mb);
        true
    }

    pub async fn remove(&self, key: &str) -> Option<T> {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;

        if let Some(entry) = cache.remove(key) {
            stats.total_entries -= 1;
            stats.memory_usage_mb -= entry.size_bytes as f64 / 1024.0 / 1024.0;

            if self.config.enable_lru {
                let mut lru = self.lru_order.write().await;
                lru.retain(|k| k != key);
            }

            debug!("Removed cache entry for key: {}", key);
            Some(entry.data)
        } else {
            None
        }
    }

    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        let mut lru = self.lru_order.write().await;

        cache.clear();
        lru.clear();
        stats.total_entries = 0;
        stats.memory_usage_mb = 0.0;
        
        info!("Cache cleared");
    }

    pub async fn get_stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let mut stats = self.stats.write().await;
        
        // Update hit rate
        if stats.total_requests > 0 {
            stats.hit_rate = stats.cache_hits as f64 / stats.total_requests as f64;
        }

        // Calculate average access count
        if !cache.is_empty() {
            let total_accesses: u64 = cache.values().map(|entry| entry.access_count).sum();
            stats.average_access_count = total_accesses as f64 / cache.len() as f64;
        }
        
        stats.clone()
    }

    pub async fn contains_key(&self, key: &str) -> bool {
        let cache = self.cache.read().await;
        
        if let Some(entry) = cache.get(key) {
            !entry.is_expired(Duration::from_secs(self.config.ttl_seconds))
        } else {
            false
        }
    }

    pub async fn size(&self) -> usize {
        let cache = self.cache.read().await;
        cache.len()
    }

    pub async fn memory_usage_mb(&self) -> f64 {
        let stats = self.stats.read().await;
        stats.memory_usage_mb
    }

    async fn evict_lru(&self) {
        if !self.config.enable_lru {
            return;
        }

        let key_to_evict = {
            let mut lru = self.lru_order.write().await;
            lru.pop_front()
        };

        if let Some(key) = key_to_evict {
            let mut cache = self.cache.write().await;
            let mut stats = self.stats.write().await;

            if let Some(entry) = cache.remove(&key) {
                stats.total_entries -= 1;
                stats.memory_usage_mb -= entry.size_bytes as f64 / 1024.0 / 1024.0;
                stats.evictions += 1;
                
                debug!("Evicted LRU cache entry: {} (accessed {} times)", 
                       key, entry.access_count);
            }
        }
    }

    async fn cleanup_if_needed(&self) {
        let last_cleanup = *self.last_cleanup.read().await;
        let cleanup_interval = Duration::from_secs(self.config.cleanup_interval_seconds);

        if last_cleanup.elapsed() > cleanup_interval {
            self.cleanup_expired().await;
        }
    }

    async fn cleanup_expired(&self) {
        let ttl = Duration::from_secs(self.config.ttl_seconds);
        let mut expired_keys = Vec::new();

        // Find expired keys
        {
            let cache = self.cache.read().await;
            for (key, entry) in cache.iter() {
                if entry.is_expired(ttl) {
                    expired_keys.push(key.clone());
                }
            }
        }

        // Remove expired keys
        if !expired_keys.is_empty() {
            let mut cache = self.cache.write().await;
            let mut stats = self.stats.write().await;
            let mut lru = self.lru_order.write().await;

            for key in &expired_keys {
                if let Some(entry) = cache.remove(key) {
                    stats.total_entries -= 1;
                    stats.memory_usage_mb -= entry.size_bytes as f64 / 1024.0 / 1024.0;
                    stats.expired_entries += 1;
                }
                lru.retain(|k| k != key);
            }

            info!("Cleaned up {} expired cache entries", expired_keys.len());
        }

        // Update last cleanup time
        *self.last_cleanup.write().await = Instant::now();
    }

    fn estimate_size(&self, _value: &T) -> usize {
        // Enhanced size estimation based on type
        let base_size = std::mem::size_of::<T>();
        
        // Add overhead for cache metadata
        let metadata_overhead = std::mem::size_of::<CacheEntry<T>>();
        
        // Simple heuristic: assume dynamic data is roughly 10x the base size
        let estimated_dynamic_size = base_size * 10;
        
        base_size + metadata_overhead + estimated_dynamic_size
    }

    pub async fn get_keys(&self) -> Vec<String> {
        let cache = self.cache.read().await;
        cache.keys().cloned().collect()
    }

    pub async fn get_hottest_keys(&self, limit: usize) -> Vec<(String, u64)> {
        let cache = self.cache.read().await;
        let mut entries: Vec<_> = cache
            .iter()
            .map(|(k, v)| (k.clone(), v.access_count))
            .collect();
        
        entries.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by access count descending
        entries.truncate(limit);
        entries
    }

    pub async fn warmup<F, Fut>(&self, key_generator: F) -> crate::Result<usize>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Vec<(String, T)>>,
    {
        let entries = key_generator().await;
        let mut warmed_count = 0;

        for (key, value) in entries {
            if self.put(key, value).await {
                warmed_count += 1;
            }
        }

        info!("Cache warmed up with {} entries", warmed_count);
        Ok(warmed_count)
    }

    pub async fn export_metrics(&self) -> Option<String> {
        if !self.config.enable_metrics {
            return None;
        }

        let stats = self.get_stats().await;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metrics = format!(
            "# HELP cache_total_entries Total number of cache entries\n\
             # TYPE cache_total_entries gauge\n\
             cache_total_entries {} {}\n\
             # HELP cache_memory_usage_mb Memory usage in megabytes\n\
             # TYPE cache_memory_usage_mb gauge\n\
             cache_memory_usage_mb {:.2} {}\n\
             # HELP cache_hit_rate Hit rate percentage\n\
             # TYPE cache_hit_rate gauge\n\
             cache_hit_rate {:.4} {}\n\
             # HELP cache_total_requests Total number of cache requests\n\
             # TYPE cache_total_requests counter\n\
             cache_total_requests {} {}\n\
             # HELP cache_hits Total number of cache hits\n\
             # TYPE cache_hits counter\n\
             cache_hits {} {}\n\
             # HELP cache_misses Total number of cache misses\n\
             # TYPE cache_misses counter\n\
             cache_misses {} {}\n\
             # HELP cache_evictions Total number of evictions\n\
             # TYPE cache_evictions counter\n\
             cache_evictions {} {}\n\
             # HELP cache_expired_entries Total number of expired entries\n\
             # TYPE cache_expired_entries counter\n\
             cache_expired_entries {} {}",
            stats.total_entries, timestamp,
            stats.memory_usage_mb, timestamp,
            stats.hit_rate, timestamp,
            stats.total_requests, timestamp,
            stats.cache_hits, timestamp,
            stats.cache_misses, timestamp,
            stats.evictions, timestamp,
            stats.expired_entries, timestamp
        );

        Some(metrics)
    }
}

impl<T> Clone for SmartCache<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn clone(&self) -> Self {
        Self {
            cache: self.cache.clone(),
            lru_order: self.lru_order.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
            last_cleanup: self.last_cleanup.clone(),
        }
    }
}

pub fn create_cache_key<H: Hash>(hashable: &H) -> String {
    use std::collections::hash_map::DefaultHasher;
    
    let mut hasher = DefaultHasher::new();
    hashable.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub fn create_graph_cache_key(
    num_nodes: usize, 
    num_edges: usize, 
    feature_checksum: u64, 
    config_hash: u64
) -> String {
    format!("graph_{}n_{}e_{:x}_{:x}", num_nodes, num_edges, feature_checksum, config_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[tokio::test]
    async fn test_basic_cache_operations() {
        let config = CacheConfig {
            max_entries: 100,
            ttl_seconds: 10,
            ..Default::default()
        };
        let cache = SmartCache::new(config);

        // Test put and get
        assert!(cache.put("key1".to_string(), "value1".to_string()).await);
        assert_eq!(cache.get("key1").await, Some("value1".to_string()));

        // Test miss
        assert_eq!(cache.get("nonexistent").await, None);

        // Test contains_key
        assert!(cache.contains_key("key1").await);
        assert!(!cache.contains_key("nonexistent").await);

        // Test remove
        assert_eq!(cache.remove("key1").await, Some("value1".to_string()));
        assert_eq!(cache.get("key1").await, None);
    }

    #[tokio::test]
    async fn test_ttl_expiration() {
        let config = CacheConfig {
            max_entries: 100,
            ttl_seconds: 1, // 1 second TTL
            ..Default::default()
        };
        let cache = SmartCache::new(config);

        cache.put("key1".to_string(), "value1".to_string()).await;
        assert_eq!(cache.get("key1").await, Some("value1".to_string()));

        // Wait for expiration
        sleep(TokioDuration::from_secs(2)).await;
        assert_eq!(cache.get("key1").await, None);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let config = CacheConfig {
            max_entries: 2, // Small cache for testing
            ttl_seconds: 60,
            enable_lru: true,
            ..Default::default()
        };
        let cache = SmartCache::new(config);

        // Fill cache
        cache.put("key1".to_string(), "value1".to_string()).await;
        cache.put("key2".to_string(), "value2".to_string()).await;

        // Access key1 to make it more recently used
        cache.get("key1").await;

        // Add third entry, should evict key2 (least recently used)
        cache.put("key3".to_string(), "value3".to_string()).await;

        assert_eq!(cache.get("key1").await, Some("value1".to_string()));
        assert_eq!(cache.get("key2").await, None); // Evicted
        assert_eq!(cache.get("key3").await, Some("value3".to_string()));
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let config = CacheConfig::default();
        let cache = SmartCache::new(config);

        // Perform some operations
        cache.put("key1".to_string(), "value1".to_string()).await;
        cache.get("key1").await; // Hit
        cache.get("nonexistent").await; // Miss

        let stats = cache.get_stats().await;
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 1);
        assert_eq!(stats.hit_rate, 0.5);
    }

    #[tokio::test]
    async fn test_cache_clear() {
        let config = CacheConfig::default();
        let cache = SmartCache::new(config);

        cache.put("key1".to_string(), "value1".to_string()).await;
        cache.put("key2".to_string(), "value2".to_string()).await;
        
        assert_eq!(cache.size().await, 2);
        
        cache.clear().await;
        assert_eq!(cache.size().await, 0);
        assert_eq!(cache.get("key1").await, None);
    }

    #[test]
    fn test_cache_key_generation() {
        let key1 = create_cache_key(&"test_string");
        let key2 = create_cache_key(&"test_string");
        let key3 = create_cache_key(&"different_string");

        assert_eq!(key1, key2); // Same input should produce same hash
        assert_ne!(key1, key3); // Different input should produce different hash
    }

    #[tokio::test]
    async fn test_hottest_keys() {
        let config = CacheConfig::default();
        let cache = SmartCache::new(config);

        cache.put("key1".to_string(), "value1".to_string()).await;
        cache.put("key2".to_string(), "value2".to_string()).await;
        
        // Access key1 multiple times
        cache.get("key1").await;
        cache.get("key1").await;
        cache.get("key1").await;
        
        // Access key2 once
        cache.get("key2").await;

        let hottest = cache.get_hottest_keys(2).await;
        assert_eq!(hottest.len(), 2);
        assert_eq!(hottest[0].0, "key1"); // Most accessed
        assert!(hottest[0].1 > hottest[1].1); // Higher access count
    }

    #[tokio::test]
    async fn test_memory_tracking() {
        let config = CacheConfig {
            max_memory_mb: 1, // Very small limit for testing
            ..Default::default()
        };
        let cache = SmartCache::new(config);

        cache.put("key1".to_string(), "value1".to_string()).await;
        let memory_usage = cache.memory_usage_mb().await;
        assert!(memory_usage > 0.0);
    }
}