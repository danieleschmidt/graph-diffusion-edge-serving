//! Graph pruning algorithms for reducing computational complexity

use crate::core::graph::{Graph, Edge};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    pub method: PruningMethod,
    pub target_sparsity: f32,
    pub importance_threshold: f32,
    pub preserve_connectivity: bool,
    pub max_iterations: usize,
    pub adaptive_threshold: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PruningMethod {
    MagnitudeBased,
    GradientBased,
    StructuralImportance,
    RandomPruning,
    HybridPruning,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            method: PruningMethod::StructuralImportance,
            target_sparsity: 0.3, // Remove 30% of edges
            importance_threshold: 0.1,
            preserve_connectivity: true,
            max_iterations: 10,
            adaptive_threshold: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PruningStats {
    pub original_nodes: usize,
    pub original_edges: usize,
    pub pruned_nodes: usize,
    pub pruned_edges: usize,
    pub sparsity_achieved: f32,
    pub connectivity_preserved: bool,
    pub computation_reduction: f32,
}

pub struct GraphPruner {
    config: PruningConfig,
    node_importance: HashMap<u32, f32>,
    edge_importance: HashMap<(u32, u32), f32>,
    connected_components: Vec<HashSet<u32>>,
}

impl GraphPruner {
    pub fn new(config: PruningConfig) -> Self {
        info!("Initializing graph pruner with method: {:?}", config.method);
        
        Self {
            config,
            node_importance: HashMap::new(),
            edge_importance: HashMap::new(),
            connected_components: Vec::new(),
        }
    }

    pub fn prune_graph(&mut self, graph: &Graph) -> crate::Result<Graph> {
        info!("Starting graph pruning with target sparsity: {:.2}", self.config.target_sparsity);

        // Compute importance scores
        self.compute_importance_scores(graph)?;
        
        // Analyze connectivity if preservation is required
        if self.config.preserve_connectivity {
            self.analyze_connectivity(graph);
        }

        // Perform pruning based on method
        let pruned_graph = match self.config.method {
            PruningMethod::MagnitudeBased => self.magnitude_based_pruning(graph)?,
            PruningMethod::GradientBased => self.gradient_based_pruning(graph)?,
            PruningMethod::StructuralImportance => self.structural_importance_pruning(graph)?,
            PruningMethod::RandomPruning => self.random_pruning(graph)?,
            PruningMethod::HybridPruning => self.hybrid_pruning(graph)?,
        };

        // Validate result
        if self.config.preserve_connectivity {
            self.validate_connectivity(&pruned_graph)?;
        }

        info!("Graph pruning completed: {} -> {} nodes, {} -> {} edges", 
              graph.num_nodes(), pruned_graph.num_nodes(),
              graph.num_edges(), pruned_graph.num_edges());

        Ok(pruned_graph)
    }

    pub fn get_pruning_stats(&self, original: &Graph, pruned: &Graph) -> PruningStats {
        let sparsity = 1.0 - (pruned.num_edges() as f32 / original.num_edges() as f32);
        let computation_reduction = self.estimate_computation_reduction(original, pruned);
        
        PruningStats {
            original_nodes: original.num_nodes(),
            original_edges: original.num_edges(),
            pruned_nodes: pruned.num_nodes(),
            pruned_edges: pruned.num_edges(),
            sparsity_achieved: sparsity,
            connectivity_preserved: self.check_connectivity_preserved(original, pruned),
            computation_reduction,
        }
    }

    pub fn should_prune_edge(&self, weight: f32) -> bool {
        weight.abs() < self.config.importance_threshold
    }

    fn compute_importance_scores(&mut self, graph: &Graph) -> crate::Result<()> {
        debug!("Computing importance scores for {} nodes and {} edges", 
               graph.num_nodes(), graph.num_edges());

        // Compute node importance based on centrality measures
        self.compute_node_centrality(graph);
        
        // Compute edge importance based on weights and structure
        self.compute_edge_importance(graph);

        Ok(())
    }

    fn compute_node_centrality(&mut self, graph: &Graph) {
        let mut degree_centrality = HashMap::new();
        let mut betweenness_centrality = HashMap::new();

        // Build adjacency list for faster computation
        let mut adjacency: HashMap<u32, Vec<u32>> = HashMap::new();
        for edge in &graph.edges {
            adjacency.entry(edge.source).or_insert_with(Vec::new).push(edge.target);
            adjacency.entry(edge.target).or_insert_with(Vec::new).push(edge.source);
        }

        // Compute degree centrality
        for node in &graph.nodes {
            let degree = adjacency.get(&node.id).map_or(0, |neighbors| neighbors.len());
            degree_centrality.insert(node.id, degree as f32);
        }

        // Simplified betweenness centrality (approximation for large graphs)
        for node in &graph.nodes {
            let betweenness = self.approximate_betweenness_centrality(node.id, &adjacency);
            betweenness_centrality.insert(node.id, betweenness);
        }

        // Combine centrality measures
        for node in &graph.nodes {
            let degree = degree_centrality.get(&node.id).unwrap_or(&0.0);
            let betweenness = betweenness_centrality.get(&node.id).unwrap_or(&0.0);
            
            // Weighted combination of centrality measures
            let importance = 0.6 * degree + 0.4 * betweenness;
            self.node_importance.insert(node.id, importance);
        }
    }

    fn approximate_betweenness_centrality(&self, node_id: u32, adjacency: &HashMap<u32, Vec<u32>>) -> f32 {
        // Simplified approximation: count paths through this node
        let empty_vec = vec![];
        let neighbors = adjacency.get(&node_id).unwrap_or(&empty_vec);
        let neighbor_count = neighbors.len() as f32;
        
        if neighbor_count < 2.0 {
            return 0.0;
        }

        // Approximation based on local clustering
        let mut triangles = 0;
        for &neighbor1 in neighbors {
            for &neighbor2 in neighbors {
                if neighbor1 != neighbor2 {
                    if let Some(n1_neighbors) = adjacency.get(&neighbor1) {
                        if n1_neighbors.contains(&neighbor2) {
                            triangles += 1;
                        }
                    }
                }
            }
        }

        let max_triangles = neighbor_count * (neighbor_count - 1.0);
        if max_triangles > 0.0 {
            1.0 - (triangles as f32 / max_triangles)
        } else {
            0.0
        }
    }

    fn compute_edge_importance(&mut self, graph: &Graph) {
        for edge in &graph.edges {
            let source_importance = self.node_importance.get(&edge.source).unwrap_or(&1.0);
            let target_importance = self.node_importance.get(&edge.target).unwrap_or(&1.0);
            
            // Edge importance based on weight and connected node importance
            let importance = edge.weight.abs() * (source_importance + target_importance) / 2.0;
            self.edge_importance.insert((edge.source, edge.target), importance);
        }
    }

    fn analyze_connectivity(&mut self, graph: &Graph) {
        debug!("Analyzing graph connectivity");
        
        let mut visited = HashSet::new();
        self.connected_components.clear();

        for node in &graph.nodes {
            if !visited.contains(&node.id) {
                let component = self.dfs_component(node.id, graph, &mut visited);
                self.connected_components.push(component);
            }
        }

        info!("Found {} connected components", self.connected_components.len());
    }

    fn dfs_component(&self, start_node: u32, graph: &Graph, visited: &mut HashSet<u32>) -> HashSet<u32> {
        let mut component = HashSet::new();
        let mut stack = vec![start_node];

        // Build adjacency list for this traversal
        let mut adjacency: HashMap<u32, Vec<u32>> = HashMap::new();
        for edge in &graph.edges {
            adjacency.entry(edge.source).or_insert_with(Vec::new).push(edge.target);
            adjacency.entry(edge.target).or_insert_with(Vec::new).push(edge.source);
        }

        while let Some(current) = stack.pop() {
            if visited.insert(current) {
                component.insert(current);
                
                if let Some(neighbors) = adjacency.get(&current) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }

        component
    }

    fn magnitude_based_pruning(&self, graph: &Graph) -> crate::Result<Graph> {
        debug!("Performing magnitude-based pruning");
        
        let mut edges_with_importance: Vec<_> = graph.edges.iter()
            .map(|edge| {
                let default_importance = edge.weight.abs();
                let importance = self.edge_importance.get(&(edge.source, edge.target))
                    .unwrap_or(&default_importance);
                (edge.clone(), *importance)
            })
            .collect();

        // Sort by importance (descending)
        edges_with_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top edges based on target sparsity
        let edges_to_keep = ((1.0 - self.config.target_sparsity) * graph.num_edges() as f32) as usize;
        let pruned_edges: Vec<_> = edges_with_importance.into_iter()
            .take(edges_to_keep)
            .map(|(edge, _)| edge)
            .collect();

        // Create pruned graph
        let mut pruned_graph = Graph::new();
        pruned_graph.nodes = graph.nodes.clone();
        pruned_graph.edges = pruned_edges;
        pruned_graph.metadata = graph.metadata.clone();

        Ok(pruned_graph)
    }

    fn structural_importance_pruning(&self, graph: &Graph) -> crate::Result<Graph> {
        debug!("Performing structural importance pruning");
        
        let mut important_edges = Vec::new();
        let threshold = if self.config.adaptive_threshold {
            self.compute_adaptive_threshold(graph)
        } else {
            self.config.importance_threshold
        };

        for edge in &graph.edges {
            let importance = self.edge_importance.get(&(edge.source, edge.target))
                .unwrap_or(&0.0);
            
            if *importance >= threshold {
                important_edges.push(edge.clone());
            }
        }

        // Ensure minimum connectivity if required
        if self.config.preserve_connectivity {
            important_edges = self.ensure_connectivity(graph, important_edges)?;
        }

        let mut pruned_graph = Graph::new();
        pruned_graph.nodes = graph.nodes.clone();
        pruned_graph.edges = important_edges;
        pruned_graph.metadata = graph.metadata.clone();

        Ok(pruned_graph)
    }

    fn gradient_based_pruning(&self, _graph: &Graph) -> crate::Result<Graph> {
        // Placeholder for gradient-based pruning
        // In a real implementation, this would use gradient information from training
        warn!("Gradient-based pruning not implemented, falling back to structural importance");
        self.structural_importance_pruning(_graph)
    }

    fn random_pruning(&self, graph: &Graph) -> crate::Result<Graph> {
        debug!("Performing random pruning");
        
        let edges_to_keep = ((1.0 - self.config.target_sparsity) * graph.num_edges() as f32) as usize;
        let mut pruned_edges = graph.edges.clone();
        
        // Shuffle and truncate
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        pruned_edges.shuffle(&mut rng);
        pruned_edges.truncate(edges_to_keep);

        let mut pruned_graph = Graph::new();
        pruned_graph.nodes = graph.nodes.clone();
        pruned_graph.edges = pruned_edges;
        pruned_graph.metadata = graph.metadata.clone();

        Ok(pruned_graph)
    }

    fn hybrid_pruning(&self, graph: &Graph) -> crate::Result<Graph> {
        debug!("Performing hybrid pruning");
        
        // Combine multiple pruning strategies
        let structural_weight = 0.6;
        let magnitude_weight = 0.4;

        let mut edges_with_score: Vec<_> = graph.edges.iter()
            .map(|edge| {
                let structural_importance = self.edge_importance.get(&(edge.source, edge.target))
                    .unwrap_or(&0.0);
                let magnitude_importance = edge.weight.abs();
                
                let combined_score = structural_weight * structural_importance + 
                                   magnitude_weight * magnitude_importance;
                
                (edge.clone(), combined_score)
            })
            .collect();

        // Sort by combined score (descending)
        edges_with_score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top edges
        let edges_to_keep = ((1.0 - self.config.target_sparsity) * graph.num_edges() as f32) as usize;
        let pruned_edges: Vec<_> = edges_with_score.into_iter()
            .take(edges_to_keep)
            .map(|(edge, _)| edge)
            .collect();

        let mut pruned_graph = Graph::new();
        pruned_graph.nodes = graph.nodes.clone();
        pruned_graph.edges = pruned_edges;
        pruned_graph.metadata = graph.metadata.clone();

        Ok(pruned_graph)
    }

    fn compute_adaptive_threshold(&self, _graph: &Graph) -> f32 {
        let importance_values: Vec<f32> = self.edge_importance.values().cloned().collect();
        
        if importance_values.is_empty() {
            return self.config.importance_threshold;
        }

        // Use percentile-based threshold
        let mut sorted_importance = importance_values;
        sorted_importance.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let percentile_index = (self.config.target_sparsity * sorted_importance.len() as f32) as usize;
        let threshold = sorted_importance.get(percentile_index)
            .unwrap_or(&self.config.importance_threshold);

        debug!("Adaptive threshold computed: {:.4}", threshold);
        *threshold
    }

    fn ensure_connectivity(&self, original_graph: &Graph, mut pruned_edges: Vec<Edge>) -> crate::Result<Vec<Edge>> {
        // Check if pruned graph maintains connectivity
        let mut temp_graph = Graph::new();
        temp_graph.nodes = original_graph.nodes.clone();
        temp_graph.edges = pruned_edges.clone();

        let original_components = self.count_connected_components(original_graph);
        let pruned_components = self.count_connected_components(&temp_graph);

        if pruned_components > original_components {
            warn!("Connectivity lost during pruning ({} -> {} components), adding bridge edges", 
                  original_components, pruned_components);
            
            // Add minimum spanning tree edges to maintain connectivity
            pruned_edges = self.add_bridge_edges(original_graph, pruned_edges)?;
        }

        Ok(pruned_edges)
    }

    fn count_connected_components(&self, graph: &Graph) -> usize {
        let mut visited = HashSet::new();
        let mut components = 0;

        for node in &graph.nodes {
            if !visited.contains(&node.id) {
                self.dfs_component(node.id, graph, &mut visited);
                components += 1;
            }
        }

        components
    }

    fn add_bridge_edges(&self, original_graph: &Graph, mut pruned_edges: Vec<Edge>) -> crate::Result<Vec<Edge>> {
        // Find edges that are critical for connectivity
        let pruned_edge_set: HashSet<(u32, u32)> = pruned_edges.iter()
            .map(|e| (e.source, e.target))
            .collect();

        // Add edges with highest importance that restore connectivity
        let mut candidate_edges: Vec<_> = original_graph.edges.iter()
            .filter(|e| !pruned_edge_set.contains(&(e.source, e.target)))
            .map(|e| {
                let default_importance = e.weight.abs();
                let importance = self.edge_importance.get(&(e.source, e.target))
                    .unwrap_or(&default_importance);
                (e.clone(), *importance)
            })
            .collect();

        candidate_edges.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Add edges until connectivity is restored
        for (edge, _) in candidate_edges {
            pruned_edges.push(edge);
            
            let mut temp_graph = Graph::new();
            temp_graph.nodes = original_graph.nodes.clone();
            temp_graph.edges = pruned_edges.clone();
            
            if self.count_connected_components(&temp_graph) <= self.count_connected_components(original_graph) {
                break;
            }
        }

        Ok(pruned_edges)
    }

    fn validate_connectivity(&self, graph: &Graph) -> crate::Result<()> {
        let components = self.count_connected_components(graph);
        debug!("Pruned graph has {} connected components", components);
        Ok(())
    }

    fn check_connectivity_preserved(&self, original: &Graph, pruned: &Graph) -> bool {
        self.count_connected_components(original) >= self.count_connected_components(pruned)
    }

    fn estimate_computation_reduction(&self, original: &Graph, pruned: &Graph) -> f32 {
        // Estimate computation reduction based on edge reduction
        // This is a simplified model - real computation depends on algorithm complexity
        let edge_reduction = 1.0 - (pruned.num_edges() as f32 / original.num_edges() as f32);
        let node_reduction = 1.0 - (pruned.num_nodes() as f32 / original.num_nodes() as f32);
        
        // Weighted combination (edges typically dominate computation)
        0.8 * edge_reduction + 0.2 * node_reduction
    }
}

pub fn prune_feature_matrix(features: &Array2<f32>, importance_scores: &[f32], sparsity: f32) -> Array2<f32> {
    let features_to_keep = ((1.0 - sparsity) * features.ncols() as f32) as usize;
    
    // Get indices of most important features
    let mut indexed_scores: Vec<(usize, f32)> = importance_scores.iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();
    
    indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let important_indices: Vec<usize> = indexed_scores.into_iter()
        .take(features_to_keep)
        .map(|(idx, _)| idx)
        .collect();
    
    // Create pruned feature matrix
    let mut pruned_features = Array2::zeros((features.nrows(), features_to_keep));
    
    for (new_col, &old_col) in important_indices.iter().enumerate() {
        if old_col < features.ncols() {
            pruned_features.column_mut(new_col).assign(&features.column(old_col));
        }
    }
    
    pruned_features
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::Node;

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new();
        
        // Add nodes
        for i in 0..5 {
            graph.add_node(Node {
                id: i,
                features: vec![i as f32, (i * 2) as f32],
                label: None,
            });
        }
        
        // Add edges forming a connected graph
        let edges = vec![
            (0, 1, 1.0), (1, 2, 0.5), (2, 3, 0.8), 
            (3, 4, 0.3), (0, 4, 0.9), (1, 3, 0.2)
        ];
        
        for (src, tgt, weight) in edges {
            graph.add_edge(Edge {
                source: src,
                target: tgt,
                weight,
                edge_type: None,
            });
        }
        
        graph
    }

    #[test]
    fn test_graph_pruner_creation() {
        let config = PruningConfig::default();
        let pruner = GraphPruner::new(config);
        assert_eq!(pruner.node_importance.len(), 0);
        assert_eq!(pruner.edge_importance.len(), 0);
    }

    #[test]
    fn test_magnitude_based_pruning() {
        let graph = create_test_graph();
        let config = PruningConfig {
            method: PruningMethod::MagnitudeBased,
            target_sparsity: 0.5,
            ..Default::default()
        };
        
        let mut pruner = GraphPruner::new(config);
        let pruned = pruner.prune_graph(&graph).unwrap();
        
        assert_eq!(pruned.num_nodes(), graph.num_nodes());
        assert!(pruned.num_edges() < graph.num_edges());
    }

    #[test]
    fn test_structural_importance_pruning() {
        let graph = create_test_graph();
        let config = PruningConfig {
            method: PruningMethod::StructuralImportance,
            target_sparsity: 0.3,
            preserve_connectivity: true,
            ..Default::default()
        };
        
        let mut pruner = GraphPruner::new(config);
        let pruned = pruner.prune_graph(&graph).unwrap();
        
        assert_eq!(pruned.num_nodes(), graph.num_nodes());
        assert!(pruned.num_edges() <= graph.num_edges());
    }

    #[test]
    fn test_connectivity_preservation() {
        let graph = create_test_graph();
        let config = PruningConfig {
            preserve_connectivity: true,
            target_sparsity: 0.8, // Aggressive pruning
            ..Default::default()
        };
        
        let mut pruner = GraphPruner::new(config);
        let pruned = pruner.prune_graph(&graph).unwrap();
        
        // Graph should remain connected
        let original_components = pruner.count_connected_components(&graph);
        let pruned_components = pruner.count_connected_components(&pruned);
        assert!(pruned_components <= original_components);
    }

    #[test]
    fn test_pruning_stats() {
        let graph = create_test_graph();
        let config = PruningConfig::default();
        let mut pruner = GraphPruner::new(config);
        let pruned = pruner.prune_graph(&graph).unwrap();
        
        let stats = pruner.get_pruning_stats(&graph, &pruned);
        
        assert_eq!(stats.original_nodes, graph.num_nodes());
        assert_eq!(stats.original_edges, graph.num_edges());
        assert_eq!(stats.pruned_nodes, pruned.num_nodes());
        assert_eq!(stats.pruned_edges, pruned.num_edges());
        assert!(stats.sparsity_achieved >= 0.0);
        assert!(stats.computation_reduction >= 0.0);
    }

    #[test]
    fn test_feature_matrix_pruning() {
        use ndarray::Array;
        
        let features = Array::from_shape_vec((3, 4), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let importance = vec![0.8, 0.3, 0.9, 0.1]; // Feature 2 most important, then 0, then 1, then 3
        
        let pruned = prune_feature_matrix(&features, &importance, 0.5); // Keep 50%
        
        assert_eq!(pruned.shape(), &[3, 2]); // Should keep 2 features
    }

    #[test] 
    fn test_adaptive_threshold() {
        let graph = create_test_graph();
        let config = PruningConfig {
            adaptive_threshold: true,
            ..Default::default()
        };
        
        let mut pruner = GraphPruner::new(config);
        pruner.compute_importance_scores(&graph).unwrap();
        
        let threshold = pruner.compute_adaptive_threshold(&graph);
        assert!(threshold >= 0.0);
    }
}