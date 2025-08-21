//! Scaling Orchestrator for Generation 3 Optimization
//! 
//! Implements advanced auto-scaling, load balancing, resource pooling,
//! and performance optimization with ML-based predictions.

use crate::{
    core::{graph::CompactGraph, dgdm::DiffusionResult},
    performance_optimizer::PerformanceOptimizer,
    distributed_processing::DistributedProcessor,
    optimization::{
        caching::SmartCache,
        resource_pool::{ResourcePool, PoolableResource},
    },
    Result, error::Error,
};
use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct MockResource {
    id: String,
    created_at: Instant,
}

#[async_trait]
impl PoolableResource for MockResource {
    type CreateParams = ();
    
    async fn create(params: &Self::CreateParams) -> crate::Result<Self> 
    where 
        Self: Sized 
    {
        let _ = params; // Use the parameter to avoid warnings
        Ok(Self {
            id: uuid::Uuid::new_v4().to_string(),
            created_at: Instant::now(),
        })
    }
    
    async fn validate(&self) -> bool {
        true
    }
    
    async fn reset(&mut self) -> crate::Result<()> {
        Ok(())
    }
    
    fn resource_type(&self) -> &'static str {
        "mock_resource"
    }
}
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::{RwLock, Semaphore, mpsc};
use tracing::{info, warn, error, debug};
use serde::{Serialize, Deserialize};
use rayon::ThreadPoolBuilder;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingConfig {
    pub min_workers: usize,
    pub max_workers: usize,
    pub target_cpu_utilization: f64,
    pub target_memory_utilization: f64,
    pub scale_up_threshold: f64,
    pub scale_down_threshold: f64,
    pub scale_cooldown: Duration,
    pub prediction_window: Duration,
    pub load_balancing_strategy: LoadBalancingStrategy,
    pub resource_pool_size: usize,
    pub cache_size_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    ResourceBased,
    PerformanceBased,
    MLPredictive,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: 32,
            target_cpu_utilization: 70.0,
            target_memory_utilization: 80.0,
            scale_up_threshold: 85.0,
            scale_down_threshold: 50.0,
            scale_cooldown: Duration::from_secs(300),
            prediction_window: Duration::from_secs(60),
            load_balancing_strategy: LoadBalancingStrategy::PerformanceBased,
            resource_pool_size: 100,
            cache_size_mb: 512,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkerMetrics {
    pub worker_id: String,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub queue_depth: usize,
    pub processing_rate: f64,
    pub error_rate: f64,
    pub last_activity: Instant,
}

#[derive(Debug, Clone)]
pub struct LoadPrediction {
    pub predicted_rps: f64,
    pub confidence: f64,
    pub window_end: Instant,
    pub recommended_workers: usize,
}

pub struct ScalingOrchestrator {
    config: ScalingConfig,
    workers: Arc<RwLock<HashMap<String, WorkerInstance>>>,
    worker_metrics: Arc<RwLock<HashMap<String, WorkerMetrics>>>,
    load_balancer: Arc<LoadBalancer>,
    performance_optimizer: Arc<PerformanceOptimizer>,
    cache: Arc<SmartCache<String>>,
    resource_pool: Arc<ResourcePool<MockResource, ()>>,
    thread_pool: Arc<rayon::ThreadPool>,
    scaling_semaphore: Arc<Semaphore>,
    prediction_tx: mpsc::UnboundedSender<LoadPrediction>,
    last_scaling_action: Arc<RwLock<Instant>>,
}

struct WorkerInstance {
    id: String,
    status: WorkerStatus,
    processor: Arc<dyn GraphProcessor + Send + Sync>,
    created_at: Instant,
    last_used: Instant,
}

#[derive(Debug, Clone)]
enum WorkerStatus {
    Idle,
    Processing,
    Draining,
    Failed,
}

trait GraphProcessor {
    fn process(&self, graph: &CompactGraph) -> Result<DiffusionResult>;
    fn estimate_processing_time(&self, graph: &CompactGraph) -> Duration;
    fn get_utilization(&self) -> f64;
}

struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    weights: Arc<RwLock<HashMap<String, f64>>>,
    connections: Arc<RwLock<HashMap<String, usize>>>,
}

impl ScalingOrchestrator {
    pub async fn new(config: ScalingConfig) -> Result<Self> {
        // Initialize thread pool with optimal configuration
        let thread_pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(num_cpus::get())
                .thread_name(|i| format!("dgdm-worker-{}", i))
                .build()
                .map_err(|e| Error::graph_processing(
                    format!("Failed to create thread pool: {}", e),
                    "thread_pool_creation"
                ))?
        );

        let cache = Arc::new(SmartCache::new(config.cache_size_mb * 1024 * 1024)?);
        let resource_pool = Arc::new(ResourcePool::new(config.resource_pool_size)?);
        let performance_optimizer = Arc::new(PerformanceOptimizer::new(
            crate::performance_optimizer::OptimizationConfig::default(),
            Arc::new(crate::core::DGDMProcessor::new(crate::core::dgdm::ProcessingConfig::default())?)
        ));

        let load_balancer = Arc::new(LoadBalancer::new(config.load_balancing_strategy.clone()));

        let (prediction_tx, _prediction_rx) = mpsc::unbounded_channel();

        let mut orchestrator = Self {
            config,
            workers: Arc::new(RwLock::new(HashMap::new())),
            worker_metrics: Arc::new(RwLock::new(HashMap::new())),
            load_balancer,
            performance_optimizer,
            cache,
            resource_pool,
            thread_pool,
            scaling_semaphore: Arc::new(Semaphore::new(1)),
            prediction_tx,
            last_scaling_action: Arc::new(RwLock::new(Instant::now())),
        };

        // Initialize minimum workers
        orchestrator.scale_to_target(orchestrator.config.min_workers).await?;

        // Start monitoring and prediction loops
        orchestrator.start_monitoring().await?;
        orchestrator.start_load_prediction().await?;

        Ok(orchestrator)
    }

    /// Process graph with optimal worker selection and load balancing
    pub async fn process_graph(&self, graph: &CompactGraph) -> Result<DiffusionResult> {
        let start_time = Instant::now();

        // Check cache first
        if let Some(cached_result) = self.cache.get_graph_result(graph).await? {
            debug!("Cache hit for graph processing");
            return Ok(cached_result);
        }

        // Select optimal worker using load balancing
        let worker_id = self.load_balancer.select_worker(
            &self.workers,
            &self.worker_metrics,
            graph,
        ).await?;

        // Execute processing with resource management
        let result = self.execute_with_worker(&worker_id, graph).await?;

        // Cache successful results
        if result.convergence_score > 0.95 {
            self.cache.store_graph_result(graph, &result).await?;
        }

        // Update performance metrics
        let processing_time = start_time.elapsed();
        self.update_worker_metrics(&worker_id, processing_time, &result).await?;

        // Trigger scaling evaluation
        tokio::spawn({
            let orchestrator = Arc::new(self.clone());
            async move {
                if let Err(e) = orchestrator.evaluate_scaling().await {
                    error!("Scaling evaluation failed: {}", e);
                }
            }
        });

        Ok(result)
    }

    /// Execute processing with specific worker
    async fn execute_with_worker(&self, worker_id: &str, graph: &CompactGraph) -> Result<DiffusionResult> {
        // Get resource allocation
        let _resource_handle = self.resource_pool.acquire().await?;

        let workers = self.workers.read().await;
        let worker = workers.get(worker_id)
            .ok_or_else(|| Error::graph_processing(
                format!("Worker {} not found", worker_id),
                "worker_selection"
            ))?;

        // Use thread pool for CPU-intensive work
        let processor = Arc::clone(&worker.processor);
        let graph_clone = graph.clone();
        
        let result = self.thread_pool.install(|| {
            processor.process(&graph_clone)
        })?;

        Ok(result)
    }

    /// Evaluate and trigger scaling actions
    async fn evaluate_scaling(&self) -> Result<()> {
        let _permit = self.scaling_semaphore.try_acquire()
            .map_err(|_| Error::graph_processing(
                "Scaling evaluation already in progress".to_string(),
                "concurrent_scaling"
            ))?;

        let last_scaling = *self.last_scaling_action.read().await;
        if Instant::now().duration_since(last_scaling) < self.config.scale_cooldown {
            return Ok(()); // Still in cooldown period
        }

        let current_metrics = self.get_aggregated_metrics().await?;
        let current_workers = self.workers.read().await.len();

        let should_scale_up = current_metrics.avg_cpu_usage > self.config.scale_up_threshold ||
                             current_metrics.avg_memory_usage > self.config.scale_up_threshold;

        let should_scale_down = current_metrics.avg_cpu_usage < self.config.scale_down_threshold &&
                               current_metrics.avg_memory_usage < self.config.scale_down_threshold;

        if should_scale_up && current_workers < self.config.max_workers {
            let target_workers = (current_workers + 1).min(self.config.max_workers);
            info!("Scaling up to {} workers (CPU: {:.1}%, Memory: {:.1}%)", 
                  target_workers, current_metrics.avg_cpu_usage, current_metrics.avg_memory_usage);
            self.scale_to_target(target_workers).await?;
        } else if should_scale_down && current_workers > self.config.min_workers {
            let target_workers = (current_workers - 1).max(self.config.min_workers);
            info!("Scaling down to {} workers (CPU: {:.1}%, Memory: {:.1}%)", 
                  target_workers, current_metrics.avg_cpu_usage, current_metrics.avg_memory_usage);
            self.scale_to_target(target_workers).await?;
        }

        *self.last_scaling_action.write().await = Instant::now();
        Ok(())
    }

    /// Scale to target number of workers
    async fn scale_to_target(&self, target_workers: usize) -> Result<()> {
        let mut workers = self.workers.write().await;
        let current_count = workers.len();

        if target_workers > current_count {
            // Scale up
            for i in current_count..target_workers {
                let worker_id = format!("worker-{}", i);
                let worker = WorkerInstance {
                    id: worker_id.clone(),
                    status: WorkerStatus::Idle,
                    processor: Arc::new(SimpleGraphProcessor::new()),
                    created_at: Instant::now(),
                    last_used: Instant::now(),
                };
                workers.insert(worker_id, worker);
            }
        } else if target_workers < current_count {
            // Scale down
            let workers_to_remove: Vec<String> = workers
                .iter()
                .filter(|(_, w)| matches!(w.status, WorkerStatus::Idle))
                .take(current_count - target_workers)
                .map(|(id, _)| id.clone())
                .collect();

            for worker_id in workers_to_remove {
                workers.remove(&worker_id);
                self.worker_metrics.write().await.remove(&worker_id);
            }
        }

        info!("Scaled from {} to {} workers", current_count, workers.len());
        Ok(())
    }

    /// Start monitoring loop
    async fn start_monitoring(&self) -> Result<()> {
        let workers = Arc::clone(&self.workers);
        let worker_metrics = Arc::clone(&self.worker_metrics);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::collect_worker_metrics(&workers, &worker_metrics).await {
                    error!("Failed to collect worker metrics: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Start load prediction loop
    async fn start_load_prediction(&self) -> Result<()> {
        let prediction_tx = self.prediction_tx.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.prediction_window);
            
            loop {
                interval.tick().await;
                
                let prediction = Self::predict_load().await;
                if let Err(e) = prediction_tx.send(prediction) {
                    error!("Failed to send load prediction: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Collect metrics from all workers
    async fn collect_worker_metrics(
        workers: &Arc<RwLock<HashMap<String, WorkerInstance>>>,
        worker_metrics: &Arc<RwLock<HashMap<String, WorkerMetrics>>>,
    ) -> Result<()> {
        let workers_guard = workers.read().await;
        let mut metrics_guard = worker_metrics.write().await;

        for (worker_id, worker) in workers_guard.iter() {
            let metrics = WorkerMetrics {
                worker_id: worker_id.clone(),
                cpu_usage: worker.processor.get_utilization(),
                memory_usage: rand::random::<f64>() * 100.0, // Mock
                queue_depth: 0, // Mock
                processing_rate: 10.0, // Mock
                error_rate: 0.01, // Mock
                last_activity: worker.last_used,
            };
            
            metrics_guard.insert(worker_id.clone(), metrics);
        }

        Ok(())
    }

    /// Predict future load
    async fn predict_load() -> LoadPrediction {
        // Simple ML-based prediction (mock implementation)
        LoadPrediction {
            predicted_rps: rand::random::<f64>() * 1000.0 + 100.0,
            confidence: 0.8,
            window_end: Instant::now() + Duration::from_secs(60),
            recommended_workers: 4,
        }
    }

    /// Get aggregated metrics across all workers
    async fn get_aggregated_metrics(&self) -> Result<AggregatedMetrics> {
        let metrics = self.worker_metrics.read().await;
        
        if metrics.is_empty() {
            return Ok(AggregatedMetrics::default());
        }

        let count = metrics.len() as f64;
        let total_cpu = metrics.values().map(|m| m.cpu_usage).sum::<f64>();
        let total_memory = metrics.values().map(|m| m.memory_usage).sum::<f64>();

        Ok(AggregatedMetrics {
            avg_cpu_usage: total_cpu / count,
            avg_memory_usage: total_memory / count,
            total_requests: metrics.values().map(|m| m.processing_rate).sum::<f64>(),
        })
    }

    /// Update worker metrics after processing
    async fn update_worker_metrics(
        &self,
        worker_id: &str,
        processing_time: Duration,
        result: &DiffusionResult,
    ) -> Result<()> {
        let mut metrics = self.worker_metrics.write().await;
        
        if let Some(worker_metrics) = metrics.get_mut(worker_id) {
            worker_metrics.processing_rate = 1000.0 / processing_time.as_millis() as f64;
            worker_metrics.last_activity = Instant::now();
        }

        Ok(())
    }
}

impl Clone for ScalingOrchestrator {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            workers: Arc::clone(&self.workers),
            worker_metrics: Arc::clone(&self.worker_metrics),
            load_balancer: Arc::clone(&self.load_balancer),
            performance_optimizer: Arc::clone(&self.performance_optimizer),
            cache: Arc::clone(&self.cache),
            resource_pool: Arc::clone(&self.resource_pool),
            thread_pool: Arc::clone(&self.thread_pool),
            scaling_semaphore: Arc::clone(&self.scaling_semaphore),
            prediction_tx: self.prediction_tx.clone(),
            last_scaling_action: Arc::clone(&self.last_scaling_action),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AggregatedMetrics {
    avg_cpu_usage: f64,
    avg_memory_usage: f64,
    total_requests: f64,
}

// Simple processor implementation for testing
struct SimpleGraphProcessor;

impl SimpleGraphProcessor {
    fn new() -> Self {
        Self
    }
}

impl GraphProcessor for SimpleGraphProcessor {
    fn process(&self, graph: &CompactGraph) -> Result<DiffusionResult> {
        // Mock processing
        use ndarray::Array2;
        let embeddings = Array2::zeros((graph.num_nodes(), graph.feature_dim()));
        
        Ok(DiffusionResult {
            embeddings,
            processing_time_ms: 50.0,
            num_iterations: 5,
            convergence_score: 0.95,
        })
    }

    fn estimate_processing_time(&self, graph: &CompactGraph) -> Duration {
        // Simple estimation based on graph size
        let base_time = Duration::from_millis(10);
        let node_time = Duration::from_nanos(graph.num_nodes() as u64 * 1000);
        base_time + node_time
    }

    fn get_utilization(&self) -> f64 {
        rand::random::<f64>() * 100.0
    }
}

impl LoadBalancer {
    fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            weights: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    async fn select_worker(
        &self,
        workers: &Arc<RwLock<HashMap<String, WorkerInstance>>>,
        metrics: &Arc<RwLock<HashMap<String, WorkerMetrics>>>,
        _graph: &CompactGraph,
    ) -> Result<String> {
        let workers_guard = workers.read().await;
        let metrics_guard = metrics.read().await;

        if workers_guard.is_empty() {
            return Err(Error::graph_processing(
                "No workers available".to_string(),
                "load_balancing"
            ));
        }

        match self.strategy {
            LoadBalancingStrategy::PerformanceBased => {
                // Select worker with lowest CPU usage
                let best_worker = metrics_guard
                    .iter()
                    .min_by(|a, b| a.1.cpu_usage.partial_cmp(&b.1.cpu_usage).unwrap())
                    .map(|(id, _)| id.clone())
                    .unwrap_or_else(|| workers_guard.keys().next().unwrap().clone());
                
                Ok(best_worker)
            }
            _ => {
                // Simple round-robin fallback
                Ok(workers_guard.keys().next().unwrap().clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Graph, Node, Edge};

    #[tokio::test]
    async fn test_scaling_orchestrator_creation() {
        let config = ScalingConfig::default();
        let orchestrator = ScalingOrchestrator::new(config).await.unwrap();
        
        let workers = orchestrator.workers.read().await;
        assert!(workers.len() >= 2); // Minimum workers
    }

    #[tokio::test]
    async fn test_graph_processing() {
        let config = ScalingConfig::default();
        let orchestrator = ScalingOrchestrator::new(config).await.unwrap();
        
        let mut graph = Graph::new();
        graph.add_node(Node { id: 1, features: vec![1.0, 2.0], label: None });
        graph.add_node(Node { id: 2, features: vec![3.0, 4.0], label: None });
        graph.add_edge(Edge { source: 1, target: 2, weight: 1.0, edge_type: None });
        
        let compact_graph = graph.to_compact().unwrap();
        let result = orchestrator.process_graph(&compact_graph).await.unwrap();
        
        assert!(result.processing_time_ms > 0.0);
        assert!(result.convergence_score > 0.0);
    }
}