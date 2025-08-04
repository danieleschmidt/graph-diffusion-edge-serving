//! High-performance diffusion engine for graph neural networks

use crate::core::graph::CompactGraph;
use ndarray::{Array2, Array1};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionConfig {
    pub diffusion_type: DiffusionType,
    pub num_steps: usize,
    pub learning_rate: f32,
    pub momentum: f32,
    pub regularization: f32,
    pub early_stopping: bool,
    pub tolerance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffusionType {
    Heat,
    PageRank,
    Random,
    Personalized { reset_probability: f32 },
}

impl Default for DiffusionConfig {
    fn default() -> Self {
        Self {
            diffusion_type: DiffusionType::Heat,
            num_steps: 10,
            learning_rate: 0.01,
            momentum: 0.9,
            regularization: 1e-4,
            early_stopping: true,
            tolerance: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DiffusionMetrics {
    pub final_loss: f32,
    pub convergence_steps: usize,
    pub energy: f32,
    pub entropy: f32,
}

pub struct DiffusionEngine {
    config: DiffusionConfig,
    cache: HashMap<String, Array2<f32>>,
}

impl DiffusionEngine {
    pub fn new(config: DiffusionConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
        }
    }

    pub fn diffuse(
        &mut self,
        graph: &CompactGraph,
        initial_features: &Array2<f32>,
    ) -> crate::Result<(Array2<f32>, DiffusionMetrics)> {
        if graph.num_nodes() == 0 {
            return Err(crate::error::Error::GraphProcessing(
                "Cannot diffuse on empty graph".to_string(),
            ));
        }

        let cache_key = self.compute_cache_key(graph);
        
        if let Some(cached_result) = self.cache.get(&cache_key) {
            let metrics = DiffusionMetrics {
                final_loss: 0.0,
                convergence_steps: 0,
                energy: self.compute_energy(cached_result, graph),
                entropy: self.compute_entropy(cached_result),
            };
            return Ok((cached_result.clone(), metrics));
        }

        let result = match self.config.diffusion_type {
            DiffusionType::Heat => self.heat_diffusion(graph, initial_features)?,
            DiffusionType::PageRank => self.pagerank_diffusion(graph, initial_features)?,
            DiffusionType::Random => self.random_walk_diffusion(graph, initial_features)?,
            DiffusionType::Personalized { reset_probability } => {
                self.personalized_pagerank(graph, initial_features, reset_probability)?
            }
        };

        self.cache.insert(cache_key, result.0.clone());
        Ok(result)
    }

    fn heat_diffusion(
        &self,
        graph: &CompactGraph,
        initial_features: &Array2<f32>,
    ) -> crate::Result<(Array2<f32>, DiffusionMetrics)> {
        let mut features = initial_features.clone();
        // let mut prev_features = features.clone();
        let mut velocity: Array2<f32> = Array2::zeros((graph.num_nodes(), graph.feature_dim()));
        
        let mut final_loss = 0.0;
        let mut convergence_steps = self.config.num_steps;

        for step in 0..self.config.num_steps {
            let prev_features = features.clone();
            
            let gradient = self.compute_heat_gradient(graph, &features)?;
            
            velocity = velocity * self.config.momentum - gradient * self.config.learning_rate;
            features = features + &velocity;
            
            if self.config.regularization > 0.0 {
                features = features * (1.0 - self.config.regularization);
            }

            let loss = self.compute_diffusion_loss(&prev_features, &features);
            final_loss = loss;

            if self.config.early_stopping && loss < self.config.tolerance {
                convergence_steps = step + 1;
                break;
            }
        }

        let metrics = DiffusionMetrics {
            final_loss,
            convergence_steps,
            energy: self.compute_energy(&features, graph),
            entropy: self.compute_entropy(&features),
        };

        Ok((features, metrics))
    }

    fn pagerank_diffusion(
        &self,
        graph: &CompactGraph,
        initial_features: &Array2<f32>,
    ) -> crate::Result<(Array2<f32>, DiffusionMetrics)> {
        let damping = 0.85;
        let mut features = initial_features.clone();
        let teleport = (1.0 - damping) / graph.num_nodes() as f32;
        
        let mut final_loss = 0.0;
        let mut convergence_steps = self.config.num_steps;

        for step in 0..self.config.num_steps {
            let prev_features = features.clone();
            let mut new_features = Array2::zeros((graph.num_nodes(), graph.feature_dim()));

            for node_idx in 0..graph.num_nodes() {
                let neighbors = graph.get_neighbors(node_idx);
                let weights = graph.get_edge_weights(node_idx);
                
                let mut aggregation: Array1<f32> = Array1::zeros(graph.feature_dim());
                let mut total_weight = 0.0;

                for (&neighbor_idx, &weight) in neighbors.iter().zip(weights.iter()) {
                    if (neighbor_idx as usize) < graph.num_nodes() {
                        let neighbor_features = features.row(neighbor_idx as usize);
                        aggregation = aggregation + neighbor_features.to_owned() * weight;
                        total_weight += weight;
                    }
                }

                if total_weight > 0.0 {
                    aggregation = aggregation / total_weight;
                }

                let teleport_contribution = initial_features.row(node_idx).to_owned() * teleport;
                let diffusion_contribution = aggregation * damping;
                
                new_features.row_mut(node_idx).assign(&(teleport_contribution + diffusion_contribution));
            }

            let loss = self.compute_diffusion_loss(&prev_features, &new_features);
            final_loss = loss;

            if self.config.early_stopping && loss < self.config.tolerance {
                convergence_steps = step + 1;
                features = new_features;
                break;
            }

            features = new_features;
        }

        let metrics = DiffusionMetrics {
            final_loss,
            convergence_steps,
            energy: self.compute_energy(&features, graph),
            entropy: self.compute_entropy(&features),
        };

        Ok((features, metrics))
    }

    fn random_walk_diffusion(
        &self,
        graph: &CompactGraph,
        initial_features: &Array2<f32>,
    ) -> crate::Result<(Array2<f32>, DiffusionMetrics)> {
        let mut features = initial_features.clone();
        let alpha = 0.5; // Restart probability
        
        let mut final_loss = 0.0;
        let mut convergence_steps = self.config.num_steps;

        for step in 0..self.config.num_steps {
            let prev_features = features.clone();
            let mut new_features = Array2::zeros((graph.num_nodes(), graph.feature_dim()));

            for node_idx in 0..graph.num_nodes() {
                let neighbors = graph.get_neighbors(node_idx);
                
                if neighbors.is_empty() {
                    new_features.row_mut(node_idx).assign(&initial_features.row(node_idx));
                    continue;
                }

                let random_neighbor_idx = neighbors[step % neighbors.len()] as usize;
                if random_neighbor_idx < graph.num_nodes() {
                    let neighbor_features = features.row(random_neighbor_idx);
                    let current_features = initial_features.row(node_idx);
                    
                    let mixed = neighbor_features.to_owned() * (1.0 - alpha) + current_features.to_owned() * alpha;
                    new_features.row_mut(node_idx).assign(&mixed);
                } else {
                    new_features.row_mut(node_idx).assign(&initial_features.row(node_idx));
                }
            }

            let loss = self.compute_diffusion_loss(&prev_features, &new_features);
            final_loss = loss;

            if self.config.early_stopping && loss < self.config.tolerance {
                convergence_steps = step + 1;
                features = new_features;
                break;
            }

            features = new_features;
        }

        let metrics = DiffusionMetrics {
            final_loss,
            convergence_steps,
            energy: self.compute_energy(&features, graph),
            entropy: self.compute_entropy(&features),
        };

        Ok((features, metrics))
    }

    fn personalized_pagerank(
        &self,
        graph: &CompactGraph,
        initial_features: &Array2<f32>,
        reset_probability: f32,
    ) -> crate::Result<(Array2<f32>, DiffusionMetrics)> {
        let mut features = initial_features.clone();
        let mut final_loss = 0.0;
        let mut convergence_steps = self.config.num_steps;

        for step in 0..self.config.num_steps {
            let prev_features = features.clone();
            let mut new_features = Array2::zeros((graph.num_nodes(), graph.feature_dim()));

            for node_idx in 0..graph.num_nodes() {
                let neighbors = graph.get_neighbors(node_idx);
                let weights = graph.get_edge_weights(node_idx);
                
                let mut neighbor_contribution: Array1<f32> = Array1::zeros(graph.feature_dim());
                let mut total_weight = 0.0;

                for (&neighbor_idx, &weight) in neighbors.iter().zip(weights.iter()) {
                    if (neighbor_idx as usize) < graph.num_nodes() {
                        let neighbor_features = features.row(neighbor_idx as usize);
                        neighbor_contribution = neighbor_contribution + neighbor_features.to_owned() * weight;
                        total_weight += weight;
                    }
                }

                if total_weight > 0.0 {
                    neighbor_contribution = neighbor_contribution / total_weight;
                }

                let reset_contribution = initial_features.row(node_idx).to_owned() * reset_probability;
                let propagation_contribution = neighbor_contribution * (1.0 - reset_probability);
                
                new_features.row_mut(node_idx).assign(&(reset_contribution + propagation_contribution));
            }

            let loss = self.compute_diffusion_loss(&prev_features, &new_features);
            final_loss = loss;

            if self.config.early_stopping && loss < self.config.tolerance {
                convergence_steps = step + 1;
                features = new_features;
                break;
            }

            features = new_features;
        }

        let metrics = DiffusionMetrics {
            final_loss,
            convergence_steps,
            energy: self.compute_energy(&features, graph),
            entropy: self.compute_entropy(&features),
        };

        Ok((features, metrics))
    }

    fn compute_heat_gradient(
        &self,
        graph: &CompactGraph,
        features: &Array2<f32>,
    ) -> crate::Result<Array2<f32>> {
        let mut gradient = Array2::zeros((graph.num_nodes(), graph.feature_dim()));

        for node_idx in 0..graph.num_nodes() {
            let neighbors = graph.get_neighbors(node_idx);
            let weights = graph.get_edge_weights(node_idx);
            let current_features = features.row(node_idx);

            let mut node_gradient = Array1::zeros(graph.feature_dim());
            
            for (&neighbor_idx, &weight) in neighbors.iter().zip(weights.iter()) {
                if (neighbor_idx as usize) < graph.num_nodes() {
                    let neighbor_features = features.row(neighbor_idx as usize);
                    let diff = neighbor_features.to_owned() - current_features.to_owned();
                    node_gradient = node_gradient + &diff * weight;
                }
            }

            gradient.row_mut(node_idx).assign(&node_gradient);
        }

        Ok(gradient)
    }

    fn compute_diffusion_loss(&self, prev: &Array2<f32>, current: &Array2<f32>) -> f32 {
        let diff = current - prev;
        ((&diff) * &diff).sum().sqrt()
    }

    fn compute_energy(&self, features: &Array2<f32>, graph: &CompactGraph) -> f32 {
        let mut energy = 0.0;

        for node_idx in 0..graph.num_nodes() {
            let neighbors = graph.get_neighbors(node_idx);
            let weights = graph.get_edge_weights(node_idx);
            let current_features = features.row(node_idx);

            for (&neighbor_idx, &weight) in neighbors.iter().zip(weights.iter()) {
                if (neighbor_idx as usize) < graph.num_nodes() {
                    let neighbor_features = features.row(neighbor_idx as usize);
                    let diff = current_features.to_owned() - neighbor_features.to_owned();
                    energy += weight * ((&diff) * &diff).sum();
                }
            }
        }

        energy / 2.0
    }

    fn compute_entropy(&self, features: &Array2<f32>) -> f32 {
        let mut entropy = 0.0;
        let total_elements = (features.nrows() * features.ncols()) as f32;

        for &val in features.iter() {
            let prob = val.abs() / total_elements;
            if prob > 1e-10 {
                entropy -= prob * prob.ln();
            }
        }

        entropy
    }

    fn compute_cache_key(&self, graph: &CompactGraph) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        graph.num_nodes().hash(&mut hasher);
        graph.num_edges().hash(&mut hasher);
        graph.feature_dim().hash(&mut hasher);
        
        format!("{}_{:?}_{}", 
            hasher.finish(), 
            self.config.diffusion_type, 
            self.config.num_steps
        )
    }

    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{Graph, Node, Edge};

    fn create_test_graph() -> CompactGraph {
        let mut graph = Graph::new();
        
        for i in 0..4 {
            graph.add_node(Node {
                id: i,
                features: vec![i as f32, (i * 2) as f32],
                label: None,
            });
        }
        
        graph.add_edge(Edge { source: 0, target: 1, weight: 1.0, edge_type: None });
        graph.add_edge(Edge { source: 1, target: 2, weight: 1.0, edge_type: None });
        graph.add_edge(Edge { source: 2, target: 3, weight: 1.0, edge_type: None });
        graph.add_edge(Edge { source: 3, target: 0, weight: 1.0, edge_type: None });

        graph.to_compact().unwrap()
    }

    #[test]
    fn test_heat_diffusion() {
        let graph = create_test_graph();
        let config = DiffusionConfig {
            diffusion_type: DiffusionType::Heat,
            num_steps: 5,
            ..Default::default()
        };

        let mut engine = DiffusionEngine::new(config);
        let initial_features = graph.node_features().clone();
        
        let (result, metrics) = engine.diffuse(&graph, &initial_features).unwrap();
        
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        assert!(metrics.convergence_steps <= 5);
        assert!(metrics.energy >= 0.0);
    }

    #[test]
    fn test_pagerank_diffusion() {
        let graph = create_test_graph();
        let config = DiffusionConfig {
            diffusion_type: DiffusionType::PageRank,
            num_steps: 10,
            ..Default::default()
        };

        let mut engine = DiffusionEngine::new(config);
        let initial_features = graph.node_features().clone();
        
        let (result, metrics) = engine.diffuse(&graph, &initial_features).unwrap();
        
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        assert!(metrics.final_loss >= 0.0);
    }

    #[test]
    fn test_personalized_pagerank() {
        let graph = create_test_graph();
        let config = DiffusionConfig {
            diffusion_type: DiffusionType::Personalized { reset_probability: 0.15 },
            num_steps: 8,
            ..Default::default()
        };

        let mut engine = DiffusionEngine::new(config);
        let initial_features = graph.node_features().clone();
        
        let (result, metrics) = engine.diffuse(&graph, &initial_features).unwrap();
        
        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
        assert!(metrics.entropy >= 0.0);
    }

    #[test]
    fn test_cache_functionality() {
        let graph = create_test_graph();
        let config = DiffusionConfig::default();
        let mut engine = DiffusionEngine::new(config);
        
        let initial_features = graph.node_features().clone();
        
        // First call should compute
        let (result1, _) = engine.diffuse(&graph, &initial_features).unwrap();
        assert_eq!(engine.cache_size(), 1);
        
        // Second call should use cache
        let (result2, _) = engine.diffuse(&graph, &initial_features).unwrap();
        assert_eq!(engine.cache_size(), 1);
        
        // Results should be identical
        assert_eq!(result1, result2);
        
        // Clear cache
        engine.clear_cache();
        assert_eq!(engine.cache_size(), 0);
    }
}