//! Advanced scaling, performance optimization, and resource management

use std::sync::{Arc, atomic::{AtomicUsize, AtomicU64, Ordering}};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Semaphore, Notify, mpsc};
use tokio::time::{interval, sleep};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug, instrument};
use crate::core::{Graph};
use crate::core::dgdm::DiffusionResult;
use crate::optimization::resource_pool::ResourcePool;

// Adaptive load balancing and auto-scaling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub min_workers: usize,
    pub max_workers: usize,
    pub target_cpu_utilization: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub scale_up_cooldown_seconds: u64,
    pub scale_down_cooldown_seconds: u64,
    pub queue_size_threshold: usize,
    pub enable_auto_scaling: bool,
    pub enable_adaptive_batching: bool,
    pub max_batch_size: usize,
    pub batch_timeout_ms: u64,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: num_cpus::get() * 2,
            target_cpu_utilization: 70.0,
            scale_up_threshold: 80.0,
            scale_down_threshold: 30.0,
            scale_up_cooldown_seconds: 60,
            scale_down_cooldown_seconds: 300,
            queue_size_threshold: 100,
            enable_auto_scaling: true,
            enable_adaptive_batching: true,
            max_batch_size: 32,
            batch_timeout_ms: 100,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkerMetrics {
    pub worker_id: usize,
    pub tasks_processed: u64,
    pub average_processing_time_ms: f64,
    pub current_queue_size: usize,
    pub cpu_utilization: f64,
    pub memory_usage_mb: f64,
    pub last_activity: Instant,
}

#[derive(Debug, Clone)]
pub struct ScalingMetrics {
    pub active_workers: usize,
    pub total_queue_size: usize,
    pub average_cpu_utilization: f64,
    pub requests_per_second: f64,
    pub average_response_time_ms: f64,
    pub last_scale_action: Option<Instant>,
    pub scale_action_count: u64,
}

// Task queue for distributed processing
#[derive(Debug)]
pub struct ProcessingTask {
    pub id: String,
    pub graph: Graph,
    pub priority: TaskPriority,
    pub submitted_at: Instant,
    pub timeout: Duration,
    pub response_tx: tokio::sync::oneshot::Sender<Result<DiffusionResult, String>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

// High-performance distributed task queue
pub struct DistributedTaskQueue {
    high_priority: Arc<RwLock<VecDeque<ProcessingTask>>>,
    normal_priority: Arc<RwLock<VecDeque<ProcessingTask>>>,
    low_priority: Arc<RwLock<VecDeque<ProcessingTask>>>,
    total_queued: AtomicUsize,
    processed_count: AtomicU64,
    failed_count: AtomicU64,
    semaphore: Arc<Semaphore>,
    notify: Arc<Notify>,
}

impl Default for DistributedTaskQueue {
    fn default() -> Self {
        Self::new(1000) // Default max queue size
    }
}

impl DistributedTaskQueue {
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            high_priority: Arc::new(RwLock::new(VecDeque::new())),
            normal_priority: Arc::new(RwLock::new(VecDeque::new())),
            low_priority: Arc::new(RwLock::new(VecDeque::new())),
            total_queued: AtomicUsize::new(0),
            processed_count: AtomicU64::new(0),
            failed_count: AtomicU64::new(0),
            semaphore: Arc::new(Semaphore::new(max_queue_size)),
            notify: Arc::new(Notify::new()),
        }
    }

    pub async fn enqueue(&self, task: ProcessingTask) -> Result<(), String> {
        let _permit = self.semaphore.acquire().await
            .map_err(|_| "Queue is full".to_string())?;
        
        let queue = match task.priority {
            TaskPriority::Critical | TaskPriority::High => &self.high_priority,
            TaskPriority::Normal => &self.normal_priority,
            TaskPriority::Low => &self.low_priority,
        };

        {
            let mut queue_guard = queue.write().await;
            queue_guard.push_back(task);
        }

        self.total_queued.fetch_add(1, Ordering::Relaxed);
        self.notify.notify_one();
        Ok(())
    }

    pub async fn dequeue(&self) -> Option<ProcessingTask> {
        // Priority-based dequeue: High -> Normal -> Low
        for queue in [&self.high_priority, &self.normal_priority, &self.low_priority].iter() {
            let mut queue_guard = queue.write().await;
            if let Some(task) = queue_guard.pop_front() {
                self.total_queued.fetch_sub(1, Ordering::Relaxed);
                return Some(task);
            }
        }
        None
    }

    pub async fn wait_for_task(&self) {
        self.notify.notified().await;
    }

    pub fn queue_size(&self) -> usize {
        self.total_queued.load(Ordering::Relaxed)
    }

    pub fn processed_count(&self) -> u64 {
        self.processed_count.load(Ordering::Relaxed)
    }

    pub fn mark_processed(&self) {
        self.processed_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn mark_failed(&self) {
        self.failed_count.fetch_add(1, Ordering::Relaxed);
    }
}

// Adaptive batching for improved throughput
pub struct AdaptiveBatcher {
    config: ScalingConfig,
    pending_tasks: Arc<RwLock<Vec<ProcessingTask>>>,
    batch_notify: Arc<Notify>,
    last_batch_time: Arc<RwLock<Instant>>,
}

impl AdaptiveBatcher {
    pub fn new(config: ScalingConfig) -> Self {
        Self {
            config,
            pending_tasks: Arc::new(RwLock::new(Vec::new())),
            batch_notify: Arc::new(Notify::new()),
            last_batch_time: Arc::new(RwLock::new(Instant::now())),
        }
    }

    pub async fn add_task(&self, task: ProcessingTask) {
        let mut tasks = self.pending_tasks.write().await;
        tasks.push(task);
        
        // Check if we should trigger a batch
        let should_batch = tasks.len() >= self.config.max_batch_size ||
                          self.should_timeout_batch().await;
        
        if should_batch {
            self.batch_notify.notify_one();
        }
    }

    async fn should_timeout_batch(&self) -> bool {
        let last_batch = *self.last_batch_time.read().await;
        let timeout = Duration::from_millis(self.config.batch_timeout_ms);
        last_batch.elapsed() > timeout
    }

    pub async fn get_batch(&self) -> Vec<ProcessingTask> {
        let mut tasks = self.pending_tasks.write().await;
        let batch_size = std::cmp::min(tasks.len(), self.config.max_batch_size);
        
        if batch_size == 0 {
            return Vec::new();
        }

        let batch = tasks.drain(0..batch_size).collect();
        *self.last_batch_time.write().await = Instant::now();
        batch
    }

    pub async fn wait_for_batch(&self) {
        self.batch_notify.notified().await;
    }
}

// Worker pool with auto-scaling capabilities
pub struct ScalableWorkerPool {
    config: ScalingConfig,
    workers: Arc<RwLock<HashMap<usize, WorkerHandle>>>,
    next_worker_id: AtomicUsize,
    task_queue: Arc<DistributedTaskQueue>,
    metrics: Arc<RwLock<ScalingMetrics>>,
    shutdown_tx: Option<mpsc::Sender<()>>,
}

struct WorkerHandle {
    id: usize,
    shutdown_tx: mpsc::Sender<()>,
    metrics: Arc<RwLock<WorkerMetrics>>,
    handle: tokio::task::JoinHandle<()>,
}

impl ScalableWorkerPool {
    pub fn new(config: ScalingConfig, task_queue: Arc<DistributedTaskQueue>) -> Self {
        Self {
            config,
            workers: Arc::new(RwLock::new(HashMap::new())),
            next_worker_id: AtomicUsize::new(0),
            task_queue,
            metrics: Arc::new(RwLock::new(ScalingMetrics {
                active_workers: 0,
                total_queue_size: 0,
                average_cpu_utilization: 0.0,
                requests_per_second: 0.0,
                average_response_time_ms: 0.0,
                last_scale_action: None,
                scale_action_count: 0,
            })),
            shutdown_tx: None,
        }
    }

    pub async fn start(&mut self) -> Result<(), String> {
        // Start initial workers
        for _ in 0..self.config.min_workers {
            self.spawn_worker().await?;
        }

        // Start auto-scaling monitor if enabled
        if self.config.enable_auto_scaling {
            self.start_auto_scaler().await;
        }

        info!("Scalable worker pool started with {} workers", self.config.min_workers);
        Ok(())
    }

    async fn spawn_worker(&self) -> Result<(), String> {
        let worker_id = self.next_worker_id.fetch_add(1, Ordering::Relaxed);
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel(1);
        
        let worker_metrics = Arc::new(RwLock::new(WorkerMetrics {
            worker_id,
            tasks_processed: 0,
            average_processing_time_ms: 0.0,
            current_queue_size: 0,
            cpu_utilization: 0.0,
            memory_usage_mb: 0.0,
            last_activity: Instant::now(),
        }));

        let task_queue = self.task_queue.clone();
        let metrics = worker_metrics.clone();
        
        let handle = tokio::spawn(async move {
            info!("Worker {} started", worker_id);
            
            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        info!("Worker {} shutting down", worker_id);
                        break;
                    }
                    _ = task_queue.wait_for_task() => {
                        if let Some(task) = task_queue.dequeue().await {
                            Self::process_task(worker_id, task, &metrics).await;
                        }
                    }
                    _ = sleep(Duration::from_millis(100)) => {
                        // Periodic cleanup and metrics update
                        Self::update_worker_metrics(worker_id, &metrics).await;
                    }
                }
            }
        });

        let worker_handle = WorkerHandle {
            id: worker_id,
            shutdown_tx,
            metrics: worker_metrics,
            handle,
        };

        {
            let mut workers = self.workers.write().await;
            workers.insert(worker_id, worker_handle);
        }

        Ok(())
    }

    async fn process_task(worker_id: usize, task: ProcessingTask, metrics: &Arc<RwLock<WorkerMetrics>>) {
        let start_time = Instant::now();
        
        // Check for task timeout
        if start_time.duration_since(task.submitted_at) > task.timeout {
            warn!("Task {} timed out before processing", task.id);
            let _ = task.response_tx.send(Err("Task timed out".to_string()));
            return;
        }

        debug!("Worker {} processing task {}", worker_id, task.id);

        // Simulate graph processing (replace with actual processing)
        let result = Self::simulate_graph_processing(&task.graph).await;
        
        let processing_time = start_time.elapsed();
        
        // Update worker metrics
        {
            let mut worker_metrics = metrics.write().await;
            worker_metrics.tasks_processed += 1;
            worker_metrics.average_processing_time_ms = 
                (worker_metrics.average_processing_time_ms + processing_time.as_millis() as f64) / 2.0;
            worker_metrics.last_activity = Instant::now();
        }

        // Send response
        let _ = task.response_tx.send(result);
        
        info!("Worker {} completed task {} in {}ms", 
              worker_id, task.id, processing_time.as_millis());
    }

    async fn simulate_graph_processing(_graph: &Graph) -> Result<DiffusionResult, String> {
        // Simulate processing time based on graph size
        let processing_time = Duration::from_millis(10 + rand::random::<u64>() % 50);
        sleep(processing_time).await;
        
        // Simulate success/failure
        if rand::random::<f64>() < 0.95 {
            Ok(DiffusionResult {
                embeddings: ndarray::Array2::zeros((100, 64)), // Dummy result
                processing_time_ms: processing_time.as_millis() as f64,
                num_iterations: 5,
                convergence_score: 0.95,
            })
        } else {
            Err("Simulated processing failure".to_string())
        }
    }

    async fn update_worker_metrics(_worker_id: usize, _metrics: &Arc<RwLock<WorkerMetrics>>) {
        // Update CPU and memory metrics
        // This would integrate with system monitoring in production
    }

    async fn start_auto_scaler(&self) {
        let workers = self.workers.clone();
        let config = self.config.clone();
        let metrics = self.metrics.clone();
        let task_queue = self.task_queue.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                let should_scale = Self::should_scale(&config, &workers, &task_queue, &metrics).await;
                
                match should_scale {
                    ScaleDecision::ScaleUp(count) => {
                        info!("Auto-scaling: Adding {} workers", count);
                        // Implementation would spawn new workers
                    }
                    ScaleDecision::ScaleDown(count) => {
                        info!("Auto-scaling: Removing {} workers", count);
                        // Implementation would gracefully shutdown workers
                    }
                    ScaleDecision::NoAction => {
                        debug!("Auto-scaling: No action needed");
                    }
                }
            }
        });
    }

    async fn should_scale(
        config: &ScalingConfig,
        workers: &Arc<RwLock<HashMap<usize, WorkerHandle>>>,
        task_queue: &Arc<DistributedTaskQueue>,
        _metrics: &Arc<RwLock<ScalingMetrics>>,
    ) -> ScaleDecision {
        let current_workers = workers.read().await.len();
        let queue_size = task_queue.queue_size();
        
        // Scale up if queue is growing
        if queue_size > config.queue_size_threshold && current_workers < config.max_workers {
            let scale_count = std::cmp::min(
                (queue_size / config.queue_size_threshold).saturating_sub(current_workers),
                config.max_workers - current_workers
            );
            return ScaleDecision::ScaleUp(scale_count);
        }
        
        // Scale down if workers are idle and we have more than minimum
        if queue_size < config.queue_size_threshold / 4 && current_workers > config.min_workers {
            let scale_count = std::cmp::min(
                current_workers - config.min_workers,
                (current_workers - config.min_workers) / 2
            );
            return ScaleDecision::ScaleDown(scale_count);
        }

        ScaleDecision::NoAction
    }

    pub async fn submit_task(&self, task: ProcessingTask) -> Result<(), String> {
        self.task_queue.enqueue(task).await
    }

    pub async fn get_metrics(&self) -> ScalingMetrics {
        let metrics = self.metrics.read().await;
        let workers = self.workers.read().await;
        
        ScalingMetrics {
            active_workers: workers.len(),
            total_queue_size: self.task_queue.queue_size(),
            average_cpu_utilization: metrics.average_cpu_utilization,
            requests_per_second: metrics.requests_per_second,
            average_response_time_ms: metrics.average_response_time_ms,
            last_scale_action: metrics.last_scale_action,
            scale_action_count: metrics.scale_action_count,
        }
    }
}

#[derive(Debug)]
enum ScaleDecision {
    ScaleUp(usize),
    ScaleDown(usize),
    NoAction,
}

// Smart caching system with LRU and TTL
pub struct SmartCache<K, V> 
where
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    cache: Arc<RwLock<HashMap<K, CacheEntry<V>>>>,
    max_size: usize,
    default_ttl: Duration,
    access_order: Arc<RwLock<VecDeque<K>>>,
    size_bytes: AtomicU64,
    max_size_bytes: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry<V> {
    value: V,
    inserted_at: Instant,
    last_accessed: Instant,
    ttl: Duration,
    access_count: u64,
    size_bytes: u64,
}

impl<K, V> SmartCache<K, V>
where
    K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    pub fn new(max_size: usize, max_size_bytes: u64, default_ttl: Duration) -> Self {
        let cache = Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_size,
            default_ttl,
            access_order: Arc::new(RwLock::new(VecDeque::new())),
            size_bytes: AtomicU64::new(0),
            max_size_bytes,
        };

        // Start cleanup task
        cache.start_cleanup_task();
        cache
    }

    pub async fn get(&self, key: &K) -> Option<V> {
        let mut cache = self.cache.write().await;
        
        if let Some(entry) = cache.get_mut(key) {
            // Check TTL
            if entry.inserted_at.elapsed() > entry.ttl {
                cache.remove(key);
                self.remove_from_access_order(key).await;
                return None;
            }

            // Update access info
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Update access order
            self.update_access_order(key.clone()).await;
            
            Some(entry.value.clone())
        } else {
            None
        }
    }

    pub async fn insert(&self, key: K, value: V, size_bytes: u64) {
        let entry = CacheEntry {
            value,
            inserted_at: Instant::now(),
            last_accessed: Instant::now(),
            ttl: self.default_ttl,
            access_count: 0,
            size_bytes,
        };

        // Check if we need to evict entries
        self.ensure_capacity(size_bytes).await;
        
        {
            let mut cache = self.cache.write().await;
            cache.insert(key.clone(), entry);
        }

        self.size_bytes.fetch_add(size_bytes, Ordering::Relaxed);
        self.update_access_order(key).await;
    }

    async fn ensure_capacity(&self, new_entry_size: u64) {
        let current_size = self.size_bytes.load(Ordering::Relaxed);
        
        // Check size-based eviction
        while current_size + new_entry_size > self.max_size_bytes {
            if !self.evict_lru().await {
                break; // No more entries to evict
            }
        }

        // Check count-based eviction
        let cache_size = self.cache.read().await.len();
        while cache_size >= self.max_size {
            if !self.evict_lru().await {
                break;
            }
        }
    }

    async fn evict_lru(&self) -> bool {
        let key_to_evict = {
            let access_order = self.access_order.read().await;
            access_order.front().cloned()
        };

        if let Some(key) = key_to_evict {
            let removed_size = {
                let mut cache = self.cache.write().await;
                cache.remove(&key).map(|entry| entry.size_bytes).unwrap_or(0)
            };

            self.size_bytes.fetch_sub(removed_size, Ordering::Relaxed);
            self.remove_from_access_order(&key).await;
            true
        } else {
            false
        }
    }

    async fn update_access_order(&self, key: K) {
        let mut access_order = self.access_order.write().await;
        
        // Remove existing entry if present
        if let Some(pos) = access_order.iter().position(|k| k == &key) {
            access_order.remove(pos);
        }
        
        // Add to back (most recently used)
        access_order.push_back(key);
    }

    async fn remove_from_access_order(&self, key: &K) {
        let mut access_order = self.access_order.write().await;
        if let Some(pos) = access_order.iter().position(|k| k == key) {
            access_order.remove(pos);
        }
    }

    fn start_cleanup_task(&self) {
        let cache = self.cache.clone();
        let access_order = self.access_order.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                let mut expired_keys = Vec::new();
                
                // Find expired entries
                {
                    let cache_read = cache.read().await;
                    for (key, entry) in cache_read.iter() {
                        if entry.inserted_at.elapsed() > entry.ttl {
                            expired_keys.push(key.clone());
                        }
                    }
                }
                
                // Remove expired entries
                if !expired_keys.is_empty() {
                    let mut cache_write = cache.write().await;
                    let mut access_write = access_order.write().await;
                    
                    for key in expired_keys {
                        cache_write.remove(&key);
                        if let Some(pos) = access_write.iter().position(|k| k == &key) {
                            access_write.remove(pos);
                        }
                    }
                }
            }
        });
    }

    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let total_entries = cache.len();
        let total_size_bytes = self.size_bytes.load(Ordering::Relaxed);
        
        let (total_access_count, avg_access_count) = if total_entries > 0 {
            let total_accesses: u64 = cache.values().map(|e| e.access_count).sum();
            (total_accesses, total_accesses as f64 / total_entries as f64)
        } else {
            (0, 0.0)
        };

        CacheStats {
            total_entries,
            total_size_bytes,
            max_size: self.max_size,
            max_size_bytes: self.max_size_bytes,
            total_access_count,
            average_access_count: avg_access_count,
            hit_rate: 0.0, // Would need hit/miss counters
            fill_ratio: total_entries as f64 / self.max_size as f64,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CacheStats {
    pub total_entries: usize,
    pub total_size_bytes: u64,
    pub max_size: usize,
    pub max_size_bytes: u64,
    pub total_access_count: u64,
    pub average_access_count: f64,
    pub hit_rate: f64,
    pub fill_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_distributed_task_queue() {
        let queue = DistributedTaskQueue::new(10);
        
        // Test basic enqueue/dequeue
        assert_eq!(queue.queue_size(), 0);
        
        // Would need to create a proper ProcessingTask for full testing
        // This is a simplified test structure
    }

    #[tokio::test]
    async fn test_smart_cache() {
        let cache: SmartCache<String, String> = SmartCache::new(
            100, 
            1024 * 1024, 
            Duration::from_secs(60)
        );
        
        // Test basic operations
        cache.insert("key1".to_string(), "value1".to_string(), 10).await;
        let result = cache.get(&"key1".to_string()).await;
        assert_eq!(result, Some("value1".to_string()));
        
        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 1);
    }

    #[test]
    fn test_scaling_config() {
        let config = ScalingConfig::default();
        assert!(config.min_workers >= 1);
        assert!(config.max_workers >= config.min_workers);
        assert!(config.scale_up_threshold > config.scale_down_threshold);
    }
}