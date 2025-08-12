//! Performance optimization utilities for graph processing

use crate::core::graph::CompactGraph;
use ndarray::{Array2, s};
use serde::{Serialize, Deserialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_prefetching: bool,
    pub enable_vectorization: bool,
    pub enable_memory_pooling: bool,
    pub chunk_size: usize,
    pub thread_pool_size: usize,
    pub enable_numa_aware: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_prefetching: true,
            enable_vectorization: true,
            enable_memory_pooling: true,
            chunk_size: 1024,
            thread_pool_size: num_cpus::get(),
            enable_numa_aware: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub cache_hit_rate: f64,
    pub avg_processing_time_ms: f64,
    pub throughput_graphs_per_sec: f64,
    pub memory_efficiency: f64,
    pub cpu_utilization: f64,
    pub parallel_speedup: f64,
}

pub struct PerformanceOptimizer {
    config: PerformanceConfig,
    metrics: Arc<RwLock<PerformanceMetrics>>,
    memory_pool: Arc<RwLock<Vec<Array2<f32>>>>,
    processing_history: Arc<RwLock<Vec<ProcessingRecord>>>,
}

#[derive(Debug, Clone)]
struct ProcessingRecord {
    graph_size: usize,
    processing_time_ms: f64,
    memory_used_mb: f64,
    timestamp: std::time::Instant,
}

impl PerformanceOptimizer {
    pub fn new(config: PerformanceConfig) -> Self {
        info!("Initializing performance optimizer with {} threads", config.thread_pool_size);
        
        Self {
            config,
            metrics: Arc::new(RwLock::new(PerformanceMetrics {
                cache_hit_rate: 0.0,
                avg_processing_time_ms: 0.0,
                throughput_graphs_per_sec: 0.0,
                memory_efficiency: 0.0,
                cpu_utilization: 0.0,
                parallel_speedup: 1.0,
            })),
            memory_pool: Arc::new(RwLock::new(Vec::new())),
            processing_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn allocate_embeddings(&self, nodes: usize, features: usize) -> Array2<f32> {
        if self.config.enable_memory_pooling {
            let mut pool = self.memory_pool.write().await;
            
            // Try to reuse a buffer from the pool
            for (i, buffer) in pool.iter().enumerate() {
                if buffer.nrows() >= nodes && buffer.ncols() >= features {
                    let reused = pool.swap_remove(i);
                    debug!("Reused memory buffer: {}x{}", reused.nrows(), reused.ncols());
                    return reused.slice(s![0..nodes, 0..features]).to_owned();
                }
            }
        }
        
        // Allocate new buffer
        debug!("Allocating new memory buffer: {}x{}", nodes, features);
        Array2::zeros((nodes, features))
    }

    pub async fn return_embeddings(&self, buffer: Array2<f32>) {
        if self.config.enable_memory_pooling {
            let mut pool = self.memory_pool.write().await;
            
            // Only keep reasonable number of buffers to avoid memory bloat
            if pool.len() < 10 {
                pool.push(buffer);
                debug!("Returned buffer to memory pool (pool size: {})", pool.len());
            }
        }
    }

    pub async fn record_processing(&self, graph: &CompactGraph, processing_time_ms: f64, memory_used_mb: f64) {
        let record = ProcessingRecord {
            graph_size: graph.num_nodes(),
            processing_time_ms,
            memory_used_mb,
            timestamp: std::time::Instant::now(),
        };

        let mut history = self.processing_history.write().await;
        history.push(record);
        
        // Keep only recent history (last 1000 records)
        if history.len() > 1000 {
            history.drain(0..500); // Remove oldest 500
        }

        // Update metrics
        self.update_metrics().await;
    }

    async fn update_metrics(&self) {
        let history = self.processing_history.read().await;
        
        if history.is_empty() {
            return;
        }

        let total_time: f64 = history.iter().map(|r| r.processing_time_ms).sum();
        let avg_time = total_time / history.len() as f64;
        
        let recent_records: Vec<_> = history.iter()
            .rev()
            .take(100)
            .collect();
            
        let throughput = if !recent_records.is_empty() {
            let time_span = recent_records.first().unwrap().timestamp
                .duration_since(recent_records.last().unwrap().timestamp)
                .as_secs_f64();
            if time_span > 0.0 {
                recent_records.len() as f64 / time_span
            } else {
                0.0
            }
        } else {
            0.0
        };

        let avg_memory: f64 = recent_records.iter().map(|r| r.memory_used_mb).sum::<f64>() 
            / recent_records.len().max(1) as f64;
        
        // Simple memory efficiency heuristic
        let memory_efficiency = if avg_memory > 0.0 {
            (100.0 / avg_memory).min(1.0)
        } else {
            1.0
        };

        let mut metrics = self.metrics.write().await;
        metrics.avg_processing_time_ms = avg_time;
        metrics.throughput_graphs_per_sec = throughput;
        metrics.memory_efficiency = memory_efficiency;
        
        debug!("Updated performance metrics: {:.2}ms avg, {:.2} graphs/sec", avg_time, throughput);
    }

    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }

    pub async fn optimize_for_workload(&self, graphs: &[&CompactGraph]) -> OptimizationStrategy {
        let total_nodes: usize = graphs.iter().map(|g| g.num_nodes()).sum();
        let total_edges: usize = graphs.iter().map(|g| g.num_edges()).sum();
        let avg_nodes = total_nodes / graphs.len().max(1);
        let avg_edges = total_edges / graphs.len().max(1);
        
        let density = avg_edges as f64 / (avg_nodes * avg_nodes).max(1) as f64;
        
        info!("Analyzing workload: {} graphs, avg {:.0} nodes, {:.0} edges, density {:.6}", 
              graphs.len(), avg_nodes as f64, avg_edges as f64, density);

        if avg_nodes > 500_000 {
            OptimizationStrategy::LargeGraphs
        } else if density > 0.1 {
            OptimizationStrategy::DenseGraphs
        } else if graphs.len() > 100 {
            OptimizationStrategy::BatchProcessing
        } else {
            OptimizationStrategy::Standard
        }
    }

    pub fn get_optimal_chunk_size(&self, strategy: &OptimizationStrategy) -> usize {
        match strategy {
            OptimizationStrategy::LargeGraphs => 1,
            OptimizationStrategy::DenseGraphs => 2,
            OptimizationStrategy::BatchProcessing => 8,
            OptimizationStrategy::Standard => 4,
        }
    }
}

#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    LargeGraphs,    // Single-threaded processing for very large graphs
    DenseGraphs,    // Limited parallelism for memory-intensive dense graphs
    BatchProcessing, // High parallelism for many small graphs
    Standard,       // Balanced approach
}

// Vectorized operations for performance
pub struct VectorizedOps;

impl VectorizedOps {
    pub fn sparse_matrix_multiply(
        indices: &[usize],
        values: &[f32],
        dense_matrix: &Array2<f32>,
        output: &mut Array2<f32>
    ) {
        // Vectorized sparse matrix multiplication
        // In a real implementation, this would use SIMD instructions
        for (&idx, &val) in indices.iter().zip(values.iter()) {
            if idx < dense_matrix.nrows() {
                let row = dense_matrix.row(idx);
                for (i, &elem) in row.iter().enumerate() {
                    if i < output.ncols() {
                        output[[idx, i]] += val * elem;
                    }
                }
            }
        }
    }

    pub fn parallel_normalize_rows(matrix: &mut Array2<f32>) {
        
        
        // Simple sequential normalization for now
        for mut row in matrix.rows_mut() {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-8 {
                row /= norm;
            }
        }
    }

    pub fn compute_graph_statistics(graph: &CompactGraph) -> GraphStatistics {
        use rayon::prelude::*;
        
        let degrees: Vec<usize> = (0..graph.num_nodes())
            .into_par_iter()
            .map(|i| graph.get_neighbors(i).len())
            .collect();
            
        let max_degree = degrees.iter().max().copied().unwrap_or(0);
        let avg_degree = degrees.iter().sum::<usize>() as f64 / degrees.len().max(1) as f64;
        
        let degree_variance = degrees.iter()
            .map(|&d| (d as f64 - avg_degree).powi(2))
            .sum::<f64>() / degrees.len().max(1) as f64;
            
        GraphStatistics {
            avg_degree,
            max_degree,
            degree_variance,
            clustering_coefficient: 0.0, // Placeholder - complex to compute
            diameter_estimate: estimate_diameter(graph),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphStatistics {
    pub avg_degree: f64,
    pub max_degree: usize,
    pub degree_variance: f64,
    pub clustering_coefficient: f64,
    pub diameter_estimate: usize,
}

fn estimate_diameter(graph: &CompactGraph) -> usize {
    // Simple BFS-based diameter estimation
    // In practice, would use more sophisticated algorithms
    let sample_size = (graph.num_nodes() / 100).max(1).min(10);
    let mut max_distance = 0;
    
    for start_node in (0..graph.num_nodes()).step_by(graph.num_nodes() / sample_size) {
        let distance = bfs_max_distance(graph, start_node);
        max_distance = max_distance.max(distance);
    }
    
    max_distance
}

fn bfs_max_distance(graph: &CompactGraph, start: usize) -> usize {
    use std::collections::{VecDeque, HashSet};
    
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut max_dist = 0;
    
    queue.push_back((start, 0));
    visited.insert(start);
    
    while let Some((node, dist)) = queue.pop_front() {
        max_dist = max_dist.max(dist);
        
        for &neighbor in graph.get_neighbors(node) {
            let neighbor_idx = neighbor as usize;
            if neighbor_idx < graph.num_nodes() && !visited.contains(&neighbor_idx) {
                visited.insert(neighbor_idx);
                queue.push_back((neighbor_idx, dist + 1));
            }
        }
    }
    
    max_dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{Graph, Node, Edge};

    #[tokio::test]
    async fn test_performance_optimizer() {
        let config = PerformanceConfig::default();
        let optimizer = PerformanceOptimizer::new(config);
        
        // Test memory allocation
        let embeddings = optimizer.allocate_embeddings(100, 50).await;
        assert_eq!(embeddings.shape(), &[100, 50]);
        
        // Test return to pool
        optimizer.return_embeddings(embeddings).await;
        
        // Test metrics
        let metrics = optimizer.get_metrics().await;
        assert_eq!(metrics.avg_processing_time_ms, 0.0);
    }

    #[test]
    fn test_vectorized_ops() {
        let mut matrix = Array2::from_shape_vec((3, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0
        ]).unwrap();
        
        VectorizedOps::parallel_normalize_rows(&mut matrix);
        
        // Check that rows are normalized
        for row in matrix.rows() {
            let norm = row.dot(&row).sqrt();
            assert!((norm - 1.0).abs() < 1e-6, "Row norm should be 1.0, got {}", norm);
        }
    }
}