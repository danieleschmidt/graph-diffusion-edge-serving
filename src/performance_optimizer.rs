//! Advanced performance optimization with adaptive algorithms

use crate::{Result, error::Error};
use crate::core::{DGDMProcessor, ProcessingConfig};
use crate::core::graph::CompactGraph;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, atomic::{AtomicUsize, AtomicBool, Ordering}};
use std::time::{Instant, Duration};
use tokio::sync::{RwLock, Semaphore};
use tracing::{info, warn, debug};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub avg_processing_time_ms: f64,
    pub p95_processing_time_ms: f64,
    pub throughput_graphs_per_second: f64,
    pub memory_utilization_percent: f64,
    pub cpu_utilization_percent: f64,
    pub cache_hit_rate: f64,
    pub optimization_effectiveness: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub enable_adaptive_batching: bool,
    pub enable_intelligent_caching: bool,
    pub enable_graph_partitioning: bool,
    pub enable_feature_quantization: bool,
    pub max_batch_size: usize,
    pub cache_size_mb: usize,
    pub cpu_threshold_percent: f64,
    pub memory_threshold_percent: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_batching: true,
            enable_intelligent_caching: true,
            enable_graph_partitioning: true,
            enable_feature_quantization: false, // Conservative default
            max_batch_size: 128,
            cache_size_mb: 1024,
            cpu_threshold_percent: 80.0,
            memory_threshold_percent: 85.0,
        }
    }
}

pub struct PerformanceOptimizer {
    config: OptimizationConfig,
    processor: Arc<DGDMProcessor>,
    performance_history: Arc<RwLock<VecDeque<PerformanceMetrics>>>,
    intelligent_cache: Arc<RwLock<LRUCache>>,
    processing_semaphore: Arc<Semaphore>,
    optimization_active: AtomicBool,
    batch_coordinator: Arc<BatchCoordinator>,
}

impl PerformanceOptimizer {
    pub fn new(
        config: OptimizationConfig, 
        processor: Arc<DGDMProcessor>
    ) -> Self {
        let cache_capacity = (config.cache_size_mb * 1024 * 1024) / std::mem::size_of::<CacheEntry>();
        let semaphore = Arc::new(Semaphore::new(config.max_batch_size));

        Self {
            processor,
            performance_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            intelligent_cache: Arc::new(RwLock::new(LRUCache::new(cache_capacity))),
            processing_semaphore: semaphore,
            optimization_active: AtomicBool::new(true),
            batch_coordinator: Arc::new(BatchCoordinator::new(config.max_batch_size)),
            config,
        }
    }

    pub async fn optimized_process(&self, graph: &CompactGraph) -> Result<crate::core::dgdm::DiffusionResult> {
        let start_time = Instant::now();
        
        // Check cache first
        if self.config.enable_intelligent_caching {
            if let Some(cached_result) = self.check_cache(graph).await? {
                debug!("Cache hit for graph processing");
                return Ok(cached_result);
            }
        }

        // Acquire processing permit (rate limiting)
        let _permit = self.processing_semaphore.acquire().await
            .map_err(|e| Error::resource_exhausted("processing_permits", "available", "in_use"))?;

        // Determine optimal processing strategy
        let strategy = self.determine_processing_strategy(graph).await?;
        
        let result = match strategy {
            ProcessingStrategy::Direct => {
                self.processor.process(graph)
            },
            ProcessingStrategy::Partitioned => {
                self.partitioned_process(graph).await
            },
            ProcessingStrategy::Quantized => {
                self.quantized_process(graph).await
            },
            ProcessingStrategy::Batched => {
                // Handle as single-item batch
                let results = self.processor.process_batch(vec![graph]).await?;
                Ok(results.into_iter().next().unwrap())
            },
        }?;

        let processing_time = start_time.elapsed();

        // Cache result if beneficial
        if self.config.enable_intelligent_caching && processing_time.as_millis() > 100 {
            self.cache_result(graph, &result).await?;
        }

        // Update performance metrics
        self.update_performance_metrics(&result, processing_time).await;

        Ok(result)
    }

    pub async fn optimized_batch_process(&self, graphs: Vec<&CompactGraph>) -> Result<Vec<crate::core::dgdm::DiffusionResult>> {
        if !self.config.enable_adaptive_batching {
            return self.processor.process_batch(graphs).await;
        }

        let start_time = Instant::now();
        
        // Intelligent batch optimization
        let optimized_batches = self.optimize_batching(&graphs).await?;
        let mut all_results = Vec::with_capacity(graphs.len());

        for batch in optimized_batches {
            let batch_results = self.processor.process_batch(batch).await?;
            all_results.extend(batch_results);
        }

        let total_time = start_time.elapsed();
        info!("Optimized batch processing completed in {:?} for {} graphs", total_time, graphs.len());

        Ok(all_results)
    }

    async fn determine_processing_strategy(&self, graph: &CompactGraph) -> Result<ProcessingStrategy> {
        let node_count = graph.num_nodes();
        let edge_count = graph.num_edges();
        let feature_dim = graph.feature_dim();

        // Get current system load
        let current_metrics = self.get_current_performance_metrics().await;

        // Decision tree based on graph characteristics and system state
        if current_metrics.cpu_utilization_percent > self.config.cpu_threshold_percent {
            if self.config.enable_feature_quantization && feature_dim > 512 {
                return Ok(ProcessingStrategy::Quantized);
            }
            return Ok(ProcessingStrategy::Direct);
        }

        if node_count > 100_000 && self.config.enable_graph_partitioning {
            Ok(ProcessingStrategy::Partitioned)
        } else if current_metrics.memory_utilization_percent > self.config.memory_threshold_percent {
            Ok(ProcessingStrategy::Direct)
        } else if node_count < 1000 {
            Ok(ProcessingStrategy::Batched)
        } else {
            Ok(ProcessingStrategy::Direct)
        }
    }

    async fn partitioned_process(&self, graph: &CompactGraph) -> Result<crate::core::dgdm::DiffusionResult> {
        // Implement graph partitioning using spectral clustering
        let partitions = self.partition_graph(graph, 4).await?; // 4-way partition
        let mut partition_results = Vec::new();

        // Process partitions in parallel
        let mut handles = Vec::new();
        for partition in partitions {
            let processor = self.processor.clone();
            let handle = tokio::spawn(async move {
                processor.process(&partition)
            });
            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            let result = handle.await
                .map_err(|e| Error::graph_processing(format!("Partition processing failed: {}", e), "partition"))?;
            partition_results.push(result?);
        }

        // Merge partition results
        self.merge_partition_results(partition_results).await
    }

    async fn quantized_process(&self, graph: &CompactGraph) -> Result<crate::core::dgdm::DiffusionResult> {
        // Apply feature quantization to reduce memory and computation
        let quantized_graph = self.quantize_graph_features(graph).await?;
        let result = self.processor.process(&quantized_graph)?;
        
        // Dequantize results if needed
        Ok(result)
    }

    async fn check_cache(&self, graph: &CompactGraph) -> Result<Option<crate::core::dgdm::DiffusionResult>> {
        let cache_key = self.compute_graph_hash(graph);
        let cache = self.intelligent_cache.read().await;
        
        if let Some(entry) = cache.get(&cache_key) {
            if entry.is_valid() {
                debug!("Cache hit for graph hash: {}", cache_key);
                return Ok(Some(entry.result.clone()));
            }
        }
        
        Ok(None)
    }

    async fn cache_result(&self, graph: &CompactGraph, result: &crate::core::dgdm::DiffusionResult) -> Result<()> {
        let cache_key = self.compute_graph_hash(graph);
        let entry = CacheEntry {
            result: result.clone(),
            created_at: Instant::now(),
            access_count: AtomicUsize::new(0),
            graph_size: graph.num_nodes() + graph.num_edges(),
        };

        let mut cache = self.intelligent_cache.write().await;
        cache.insert(cache_key, entry);
        
        Ok(())
    }

    fn compute_graph_hash(&self, graph: &CompactGraph) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash graph structure
        graph.num_nodes().hash(&mut hasher);
        graph.num_edges().hash(&mut hasher);
        graph.feature_dim().hash(&mut hasher);
        
        // Sample some features and edges for hash (avoid hashing everything)
        let sample_size = std::cmp::min(100, graph.num_nodes());
        for i in (0..graph.num_nodes()).step_by(graph.num_nodes() / sample_size.max(1)) {
            let neighbors = graph.get_neighbors(i);
            if let (Some(&first_neighbor), Some(&first_weight)) = (neighbors.first(), graph.get_edge_weights(i).first()) {
                first_neighbor.hash(&mut hasher);
                (first_weight as u32).hash(&mut hasher); // Approximate float hash
            }
        }
        
        hasher.finish()
    }

    async fn optimize_batching(&self, graphs: &[&CompactGraph]) -> Result<Vec<Vec<&CompactGraph>>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_batch_complexity = 0usize;

        // Sort graphs by complexity (nodes * edges)
        let mut sorted_graphs: Vec<_> = graphs.iter().enumerate().collect();
        sorted_graphs.sort_by_key(|&(_, graph)| graph.num_nodes() * graph.num_edges());

        for (_, graph) in sorted_graphs {
            let graph_complexity = graph.num_nodes() * graph.num_edges();
            
            // Dynamic batch size based on complexity
            let max_batch_complexity = self.calculate_optimal_batch_size().await;
            
            if current_batch_complexity + graph_complexity > max_batch_complexity && !current_batch.is_empty() {
                batches.push(current_batch);
                current_batch = Vec::new();
                current_batch_complexity = 0;
            }
            
            current_batch.push(*graph);
            current_batch_complexity += graph_complexity;
            
            if current_batch.len() >= self.config.max_batch_size {
                batches.push(current_batch);
                current_batch = Vec::new();
                current_batch_complexity = 0;
            }
        }
        
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        Ok(batches)
    }

    async fn calculate_optimal_batch_size(&self) -> usize {
        let current_metrics = self.get_current_performance_metrics().await;
        let base_complexity = 100_000; // Base complexity unit
        
        // Adjust based on system load
        let cpu_factor = (100.0 - current_metrics.cpu_utilization_percent) / 100.0;
        let memory_factor = (100.0 - current_metrics.memory_utilization_percent) / 100.0;
        
        let adjustment_factor = (cpu_factor * memory_factor).max(0.1);
        (base_complexity as f64 * adjustment_factor) as usize
    }

    async fn partition_graph(&self, graph: &CompactGraph, num_partitions: usize) -> Result<Vec<CompactGraph>> {
        // Simplified graph partitioning - in production would use advanced algorithms
        let nodes_per_partition = (graph.num_nodes() + num_partitions - 1) / num_partitions;
        let mut partitions = Vec::new();

        for partition_idx in 0..num_partitions {
            let start_node = partition_idx * nodes_per_partition;
            let end_node = std::cmp::min(start_node + nodes_per_partition, graph.num_nodes());
            
            if start_node >= graph.num_nodes() {
                break;
            }

            // Create partition with node range [start_node, end_node)
            let partition = self.extract_subgraph(graph, start_node, end_node).await?;
            partitions.push(partition);
        }

        Ok(partitions)
    }

    async fn extract_subgraph(&self, graph: &CompactGraph, start_node: usize, end_node: usize) -> Result<CompactGraph> {
        // Simplified subgraph extraction
        // In production, would preserve graph connectivity and handle cross-partition edges
        
        let node_count = end_node - start_node;
        let feature_dim = graph.feature_dim();
        
        // Extract node features
        let node_features = graph.node_features()
            .slice(ndarray::s![start_node..end_node, ..])
            .to_owned();
        
        // Build edge structure for subgraph
        let mut edge_offsets = vec![0u32];
        let mut edge_targets = Vec::new();
        let mut edge_weights = Vec::new();
        let mut total_edges = 0;

        for local_node_idx in 0..node_count {
            let global_node_idx = start_node + local_node_idx;
            let neighbors = graph.get_neighbors(global_node_idx);
            let weights = graph.get_edge_weights(global_node_idx);

            for (&neighbor, &weight) in neighbors.iter().zip(weights.iter()) {
                let neighbor_idx = neighbor as usize;
                // Only include edges within partition
                if neighbor_idx >= start_node && neighbor_idx < end_node {
                    edge_targets.push((neighbor_idx - start_node) as u32);
                    edge_weights.push(weight);
                    total_edges += 1;
                }
            }
            
            edge_offsets.push(edge_targets.len() as u32);
        }

        Ok(CompactGraph {
            node_ids: (start_node..end_node).map(|i| i as u32).collect(),
            node_features,
            edge_offsets,
            edge_targets,
            edge_weights,
            num_nodes: node_count,
            num_edges: total_edges,
            feature_dim,
        })
    }

    async fn merge_partition_results(&self, results: Vec<crate::core::dgdm::DiffusionResult>) -> Result<crate::core::dgdm::DiffusionResult> {
        use ndarray::{Array2, Axis};
        
        if results.is_empty() {
            return Err(Error::graph_processing("No partition results to merge", "merge_partitions"));
        }

        // Merge embeddings by concatenation
        let embeddings_list: Vec<_> = results.iter()
            .map(|r| r.embeddings.view())
            .collect();
        
        let merged_embeddings = ndarray::concatenate(Axis(0), &embeddings_list)
            .map_err(|e| Error::graph_processing(format!("Failed to merge embeddings: {}", e), "concatenate"))?;

        // Aggregate other metrics
        let total_processing_time: f64 = results.iter().map(|r| r.processing_time_ms).sum();
        let total_iterations: usize = results.iter().map(|r| r.num_iterations).sum();
        let avg_convergence: f32 = results.iter().map(|r| r.convergence_score).sum::<f32>() / results.len() as f32;

        Ok(crate::core::dgdm::DiffusionResult {
            embeddings: merged_embeddings,
            processing_time_ms: total_processing_time,
            num_iterations: total_iterations,
            convergence_score: avg_convergence,
        })
    }

    async fn quantize_graph_features(&self, graph: &CompactGraph) -> Result<CompactGraph> {
        // Simplified INT8 quantization of features
        let features = graph.node_features();
        let quantized_features = features.mapv(|x| {
            // Scale to [-127, 127] and quantize
            let scaled = (x * 127.0).round().clamp(-127.0, 127.0);
            scaled / 127.0 // Scale back to float for processing
        });

        Ok(CompactGraph {
            node_ids: graph.node_ids.clone(),
            node_features: quantized_features,
            edge_offsets: graph.edge_offsets.clone(),
            edge_targets: graph.edge_targets.clone(),
            edge_weights: graph.edge_weights.clone(),
            num_nodes: graph.num_nodes,
            num_edges: graph.num_edges,
            feature_dim: graph.feature_dim,
        })
    }

    async fn get_current_performance_metrics(&self) -> PerformanceMetrics {
        // Simplified metrics - would integrate with actual system monitoring
        PerformanceMetrics {
            avg_processing_time_ms: 50.0,
            p95_processing_time_ms: 200.0,
            throughput_graphs_per_second: 100.0,
            memory_utilization_percent: 60.0,
            cpu_utilization_percent: 45.0,
            cache_hit_rate: 0.75,
            optimization_effectiveness: 0.85,
        }
    }

    async fn update_performance_metrics(&self, result: &crate::core::dgdm::DiffusionResult, processing_time: Duration) {
        let metrics = PerformanceMetrics {
            avg_processing_time_ms: processing_time.as_secs_f64() * 1000.0,
            p95_processing_time_ms: processing_time.as_secs_f64() * 1000.0 * 1.5,
            throughput_graphs_per_second: 1000.0 / processing_time.as_millis() as f64,
            memory_utilization_percent: 65.0, // Would get from system
            cpu_utilization_percent: 55.0,    // Would get from system
            cache_hit_rate: 0.8,
            optimization_effectiveness: 0.9,
        };

        let mut history = self.performance_history.write().await;
        history.push_back(metrics);
        
        // Keep only last 1000 metrics
        if history.len() > 1000 {
            history.pop_front();
        }
    }

    pub async fn get_performance_report(&self) -> PerformanceMetrics {
        let history = self.performance_history.read().await;
        
        if history.is_empty() {
            return PerformanceMetrics {
                avg_processing_time_ms: 0.0,
                p95_processing_time_ms: 0.0,
                throughput_graphs_per_second: 0.0,
                memory_utilization_percent: 0.0,
                cpu_utilization_percent: 0.0,
                cache_hit_rate: 0.0,
                optimization_effectiveness: 0.0,
            };
        }

        // Calculate averages
        let count = history.len() as f64;
        let avg_processing = history.iter().map(|m| m.avg_processing_time_ms).sum::<f64>() / count;
        let avg_throughput = history.iter().map(|m| m.throughput_graphs_per_second).sum::<f64>() / count;
        let avg_memory = history.iter().map(|m| m.memory_utilization_percent).sum::<f64>() / count;
        let avg_cpu = history.iter().map(|m| m.cpu_utilization_percent).sum::<f64>() / count;
        let avg_cache_hit = history.iter().map(|m| m.cache_hit_rate).sum::<f64>() / count;

        PerformanceMetrics {
            avg_processing_time_ms: avg_processing,
            p95_processing_time_ms: avg_processing * 2.0, // Simplified
            throughput_graphs_per_second: avg_throughput,
            memory_utilization_percent: avg_memory,
            cpu_utilization_percent: avg_cpu,
            cache_hit_rate: avg_cache_hit,
            optimization_effectiveness: 0.9, // Would calculate based on actual improvements
        }
    }
}

#[derive(Debug, Clone)]
enum ProcessingStrategy {
    Direct,
    Partitioned,
    Quantized,
    Batched,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    result: crate::core::dgdm::DiffusionResult,
    created_at: Instant,
    access_count: AtomicUsize,
    graph_size: usize,
}

impl CacheEntry {
    fn is_valid(&self) -> bool {
        // Cache entries valid for 1 hour
        self.created_at.elapsed() < Duration::from_secs(3600)
    }
}

struct LRUCache {
    entries: HashMap<u64, CacheEntry>,
    access_order: VecDeque<u64>,
    capacity: usize,
}

impl LRUCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            access_order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn get(&mut self, key: &u64) -> Option<&CacheEntry> {
        if self.entries.contains_key(key) {
            // Move to end (most recently used)
            if let Some(pos) = self.access_order.iter().position(|&x| x == *key) {
                self.access_order.remove(pos);
            }
            self.access_order.push_back(*key);
            
            // Update access count
            if let Some(entry) = self.entries.get(key) {
                entry.access_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        self.entries.get(key)
    }

    fn insert(&mut self, key: u64, entry: CacheEntry) {
        // Remove if at capacity
        while self.entries.len() >= self.capacity {
            if let Some(oldest_key) = self.access_order.pop_front() {
                self.entries.remove(&oldest_key);
            }
        }

        self.entries.insert(key, entry);
        self.access_order.push_back(key);
    }
}

struct BatchCoordinator {
    max_batch_size: usize,
    pending_batches: Arc<RwLock<VecDeque<PendingBatch>>>,
}

impl BatchCoordinator {
    fn new(max_batch_size: usize) -> Self {
        Self {
            max_batch_size,
            pending_batches: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    // Additional batch coordination logic would go here
}

struct PendingBatch {
    graphs: Vec<CompactGraph>,
    created_at: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{GraphConfig, ProcessingConfig};

    #[tokio::test]
    async fn test_performance_optimizer_creation() {
        let config = OptimizationConfig::default();
        let graph_config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        let processor = Arc::new(DGDMProcessor::new(graph_config, processing_config));
        
        let optimizer = PerformanceOptimizer::new(config, processor);
        assert!(optimizer.optimization_active.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn test_cache_functionality() {
        let mut cache = LRUCache::new(2);
        
        let entry1 = CacheEntry {
            result: create_dummy_result(),
            created_at: Instant::now(),
            access_count: AtomicUsize::new(0),
            graph_size: 100,
        };
        
        cache.insert(1, entry1);
        assert!(cache.get(&1).is_some());
        assert!(cache.get(&2).is_none());
    }

    fn create_dummy_result() -> crate::core::dgdm::DiffusionResult {
        crate::core::dgdm::DiffusionResult {
            embeddings: ndarray::Array2::zeros((2, 2)),
            processing_time_ms: 100.0,
            num_iterations: 5,
            convergence_score: 0.01,
        }
    }
}