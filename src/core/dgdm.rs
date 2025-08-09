//! Dynamic Graph Diffusion Model (DGDM) core implementation

use crate::core::graph::{CompactGraph, GraphConfig};
use ndarray::Array2;
use tracing;
use std::sync::atomic::{AtomicUsize, Ordering};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub temperature: f32,
    pub dropout_rate: f32,
    pub use_attention: bool,
    pub normalize_features: bool,
    pub mixed_precision: bool,
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            dropout_rate: 0.1,
            use_attention: false,
            normalize_features: true,
            mixed_precision: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionResult {
    pub embeddings: Array2<f32>,
    pub processing_time_ms: f64,
    pub num_iterations: usize,
    pub convergence_score: f32,
}

pub struct DGDMProcessor {
    config: GraphConfig,
    processing_config: ProcessingConfig,
}

impl DGDMProcessor {
    pub fn new(config: GraphConfig, processing_config: ProcessingConfig) -> Self {
        Self {
            config,
            processing_config,
        }
    }

    pub fn process(&self, graph: &CompactGraph) -> crate::Result<DiffusionResult> {
        let start_time = std::time::Instant::now();

        // Enhanced input validation
        if graph.num_nodes() == 0 {
            return Err(crate::error::Error::validation(
                "Cannot process empty graph",
                "graph.num_nodes() == 0",
                "num_nodes > 0"
            ));
        }

        if graph.num_nodes() > self.config.max_nodes {
            return Err(crate::error::Error::validation(
                format!(
                    "Graph has {} nodes, exceeds maximum of {}",
                    graph.num_nodes(),
                    self.config.max_nodes
                ),
                format!("{}", graph.num_nodes()),
                format!("<= {}", self.config.max_nodes)
            ));
        }
        
        if graph.num_edges() > self.config.max_edges {
            return Err(crate::error::Error::validation(
                format!(
                    "Graph has {} edges, exceeds maximum of {}",
                    graph.num_edges(),
                    self.config.max_edges
                ),
                format!("{}", graph.num_edges()),
                format!("<= {}", self.config.max_edges)
            ));
        }
        
        // Feature dimension validation
        if graph.feature_dim() == 0 {
            return Err(crate::error::Error::validation(
                "Graph features cannot be empty",
                "feature_dim == 0",
                "feature_dim > 0"
            ));
        }
        
        if graph.feature_dim() > 10_000 {
            return Err(crate::error::Error::validation(
                format!(
                    "Feature dimension {} exceeds maximum of 10,000",
                    graph.feature_dim()
                ),
                format!("{}", graph.feature_dim()),
                "<= 10000"
            ));
        }

        let mut embeddings = graph.node_features().clone();

        if self.processing_config.normalize_features {
            self.normalize_features(&mut embeddings)?;
        }

        let mut convergence_score = 0.0;
        let diffusion_steps = self.config.diffusion_steps;

        for step in 0..diffusion_steps {
            let prev_embeddings = embeddings.clone();
            
            // Robust diffusion step with error recovery
            if let Err(e) = self.diffusion_step(graph, &mut embeddings) {
                tracing::warn!("Diffusion step {} failed, attempting recovery: {}", step, e);
                // Simple recovery: restore previous embeddings and continue
                embeddings = prev_embeddings.clone();
                continue;
            }
            
            // NaN/Inf detection and handling
            if embeddings.iter().any(|&x| !x.is_finite()) {
                tracing::warn!("Non-finite values detected at step {}, resetting to previous state", step);
                embeddings = prev_embeddings.clone();
                continue;
            }
            
            if self.processing_config.use_attention {
                if let Err(e) = self.apply_attention(&mut embeddings) {
                    tracing::warn!("Attention application failed at step {}: {}", step, e);
                    // Continue without attention
                }
            }

            convergence_score = self.compute_convergence(&prev_embeddings, &embeddings);
            
            if convergence_score < 1e-6 && step > 2 {
                tracing::debug!("Converged early at step {} with score {}", step, convergence_score);
                break;
            }
            
            // Detect divergence
            if convergence_score > 100.0 {
                return Err(crate::error::Error::graph_processing(
                    format!("Processing diverged at step {} (convergence score: {})", step, convergence_score),
                    format!("step={}, convergence={}", step, convergence_score)
                ));
            }
        }

        let processing_time = start_time.elapsed();

        Ok(DiffusionResult {
            embeddings,
            processing_time_ms: processing_time.as_secs_f64() * 1000.0,
            num_iterations: diffusion_steps,
            convergence_score,
        })
    }

    fn normalize_features(&self, embeddings: &mut Array2<f32>) -> crate::Result<()> {
        for mut row in embeddings.rows_mut() {
            let norm = row.dot(&row).sqrt();
            if norm > 1e-8 {
                row /= norm;
            }
        }
        Ok(())
    }

    fn diffusion_step(&self, graph: &CompactGraph, embeddings: &mut Array2<f32>) -> crate::Result<()> {
        let num_nodes = graph.num_nodes();
        let feature_dim = graph.feature_dim();
        let mut new_embeddings = Array2::zeros((num_nodes, feature_dim));

        for node_idx in 0..num_nodes {
            let neighbors = graph.get_neighbors(node_idx);
            let weights = graph.get_edge_weights(node_idx);
            
            if neighbors.is_empty() {
                new_embeddings.row_mut(node_idx).assign(&embeddings.row(node_idx));
                continue;
            }

            let mut aggregated = Array2::zeros((1, feature_dim));
            let mut total_weight = 0.0;

            for (&neighbor_idx, &weight) in neighbors.iter().zip(weights.iter()) {
                if (neighbor_idx as usize) < num_nodes {
                    let neighbor_features = embeddings.row(neighbor_idx as usize);
                    aggregated.row_mut(0).scaled_add(weight, &neighbor_features);
                    total_weight += weight;
                }
            }

            if total_weight > 0.0 {
                aggregated /= total_weight;
            }

            let alpha = 0.5; // Mixing parameter
            let current_features = embeddings.row(node_idx);
            let mixed_features = &current_features * (1.0 - alpha) + 
                               &aggregated.row(0) * alpha;
            
            new_embeddings.row_mut(node_idx).assign(&mixed_features);
        }

        *embeddings = new_embeddings;
        Ok(())
    }

    fn apply_attention(&self, embeddings: &mut Array2<f32>) -> crate::Result<()> {
        let temperature = self.processing_config.temperature;
        
        for mut row in embeddings.rows_mut() {
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            
            for val in row.iter_mut() {
                *val = ((*val - max_val) / temperature).exp();
            }
            
            let sum: f32 = row.sum();
            if sum > 1e-8 {
                row /= sum;
            }
        }
        
        Ok(())
    }

    fn compute_convergence(&self, prev: &Array2<f32>, current: &Array2<f32>) -> f32 {
        let diff = current - prev;
        let norm_diff = diff.mapv(|x| x * x).sum().sqrt();
        let norm_current = current.mapv(|x| x * x).sum().sqrt();
        
        if norm_current > 1e-8 {
            norm_diff / norm_current
        } else {
            0.0
        }
    }

    pub async fn process_batch(&self, graphs: Vec<&CompactGraph>) -> crate::Result<Vec<DiffusionResult>> {
        use rayon::prelude::*;

        let graph_count = graphs.len();
        tracing::info!("Processing batch of {} graphs with parallel execution", graph_count);
        
        // Dynamic chunk size based on graph complexity
        let avg_nodes = graphs.iter().map(|g| g.num_nodes()).sum::<usize>() / graph_count.max(1);
        let chunk_size = if avg_nodes > 100_000 { 1 } else if avg_nodes > 10_000 { 2 } else { 4 };
        
        let processed_count = AtomicUsize::new(0);
        
        let results: Result<Vec<_>, _> = graphs
            .into_par_iter()
            .with_max_len(chunk_size)
            .map(|graph| {
                let result = self.process(graph);
                let count = processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                if count % 10 == 0 {
                    tracing::debug!("Processed {}/{} graphs in batch", count, graph_count);
                }
                result
            })
            .collect();

        match &results {
            Ok(res) => tracing::info!("Batch processing completed: {} results", res.len()),
            Err(e) => tracing::error!("Batch processing failed: {}", e),
        }
        
        results
    }

    pub fn estimate_memory_usage(&self, graph: &CompactGraph) -> usize {
        let node_features_size = graph.num_nodes() * graph.feature_dim() * std::mem::size_of::<f32>();
        let edges_size = graph.num_edges() * std::mem::size_of::<u32>() * 2; // targets + weights
        let temporary_buffers = node_features_size * 2; // For diffusion computation
        
        node_features_size + edges_size + temporary_buffers
    }

    pub fn can_process(&self, graph: &CompactGraph) -> bool {
        graph.num_nodes() <= self.config.max_nodes &&
        graph.num_edges() <= self.config.max_edges &&
        graph.feature_dim() > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{Graph, Node, Edge};

    fn create_test_graph() -> CompactGraph {
        let mut graph = Graph::new();
        
        graph.add_node(Node {
            id: 1,
            features: vec![1.0, 0.0],
            label: None,
        });
        
        graph.add_node(Node {
            id: 2,
            features: vec![0.0, 1.0],
            label: None,
        });
        
        graph.add_node(Node {
            id: 3,
            features: vec![0.5, 0.5],
            label: None,
        });
        
        graph.add_edge(Edge {
            source: 1,
            target: 2,
            weight: 1.0,
            edge_type: None,
        });
        
        graph.add_edge(Edge {
            source: 2,
            target: 3,
            weight: 1.0,
            edge_type: None,
        });
        
        graph.add_edge(Edge {
            source: 3,
            target: 1,
            weight: 1.0,
            edge_type: None,
        });

        graph.to_compact().unwrap()
    }

    #[test]
    fn test_dgdm_processing() {
        let graph = create_test_graph();
        let config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        
        let processor = DGDMProcessor::new(config, processing_config);
        let result = processor.process(&graph).unwrap();

        assert_eq!(result.embeddings.nrows(), 3);
        assert_eq!(result.embeddings.ncols(), 2);
        assert!(result.processing_time_ms > 0.0);
        assert!(result.num_iterations > 0);
    }

    #[test]
    fn test_memory_estimation() {
        let graph = create_test_graph();
        let config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        
        let processor = DGDMProcessor::new(config, processing_config);
        let memory_usage = processor.estimate_memory_usage(&graph);
        
        assert!(memory_usage > 0);
    }

    #[test]
    fn test_can_process() {
        let graph = create_test_graph();
        let mut config = GraphConfig::default();
        config.max_nodes = 2; // Less than graph size
        
        let processing_config = ProcessingConfig::default();
        let processor = DGDMProcessor::new(config, processing_config);
        
        assert!(!processor.can_process(&graph));
    }

    #[tokio::test]
    async fn test_batch_processing() {
        let graph1 = create_test_graph();
        let graph2 = create_test_graph();
        
        let config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        let processor = DGDMProcessor::new(config, processing_config);
        
        let results = processor.process_batch(vec![&graph1, &graph2]).await.unwrap();
        
        assert_eq!(results.len(), 2);
        for result in results {
            assert_eq!(result.embeddings.nrows(), 3);
            assert_eq!(result.embeddings.ncols(), 2);
        }
    }
}