//! Graph data structures and operations optimized for TPU processing

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: u32,
    pub features: Vec<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: u32,
    pub target: u32,
    pub weight: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct CompactGraph {
    pub node_ids: Vec<u32>,
    pub node_features: Array2<f32>,
    pub edge_offsets: Vec<u32>,
    pub edge_targets: Vec<u32>,
    pub edge_weights: Vec<f32>,
    pub num_nodes: usize,
    pub num_edges: usize,
    pub feature_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub feature_dim: usize,
    pub diffusion_steps: usize,
    pub batch_size: usize,
    pub quantization: String,
    pub enable_caching: bool,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_edges: 1_000_000,
            feature_dim: 128,
            diffusion_steps: 5,
            batch_size: 32,
            quantization: "fp32".to_string(),
            enable_caching: true,
        }
    }
}

impl GraphConfig {
    pub fn builder() -> GraphConfigBuilder {
        GraphConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct GraphConfigBuilder {
    config: GraphConfig,
}

impl GraphConfigBuilder {
    pub fn max_nodes(mut self, max_nodes: usize) -> Self {
        self.config.max_nodes = max_nodes;
        self
    }

    pub fn max_edges(mut self, max_edges: usize) -> Self {
        self.config.max_edges = max_edges;
        self
    }

    pub fn diffusion_steps(mut self, steps: usize) -> Self {
        self.config.diffusion_steps = steps;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn quantization(mut self, quantization: &str) -> Self {
        self.config.quantization = quantization.to_string();
        self
    }

    pub fn build(self) -> GraphConfig {
        self.config
    }
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: Node) {
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge);
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    pub fn validate(&self) -> crate::Result<()> {
        if self.nodes.is_empty() {
            return Err(crate::error::Error::validation(
                "Graph must contain at least one node",
                format!("nodes: {}, edges: {}", self.nodes.len(), self.edges.len()),
                "min_nodes >= 1"
            ));
        }

        let node_ids: std::collections::HashSet<u32> = 
            self.nodes.iter().map(|n| n.id).collect();

        for edge in &self.edges {
            if !node_ids.contains(&edge.source) {
                return Err(crate::error::Error::validation(
                    format!("Edge references non-existent source node: {}", edge.source),
                    format!("edge: {} -> {}, available_nodes: {:?}", edge.source, edge.target, node_ids.len()),
                    "source_node must exist in graph"
                ));
            }
            if !node_ids.contains(&edge.target) {
                return Err(crate::error::Error::validation(
                    format!("Edge references non-existent target node: {}", edge.target),
                    format!("edge: {} -> {}, available_nodes: {:?}", edge.source, edge.target, node_ids.len()),
                    "target_node must exist in graph"
                ));
            }
        }

        Ok(())
    }

    pub fn to_compact(&self) -> crate::Result<CompactGraph> {
        self.validate()?;

        if self.nodes.is_empty() {
            return Err(crate::error::Error::graph_processing(
                "Cannot convert empty graph",
                "graph is empty after validation"
            ));
        }

        let feature_dim = self.nodes[0].features.len();
        let num_nodes = self.nodes.len();
        let num_edges = self.edges.len();

        let mut node_ids = Vec::with_capacity(num_nodes);
        let mut features_flat = Vec::with_capacity(num_nodes * feature_dim);

        for node in &self.nodes {
            if node.features.len() != feature_dim {
                return Err(crate::error::Error::GraphProcessing(
                    "All nodes must have the same feature dimension".to_string(),
                ));
            }
            node_ids.push(node.id);
            features_flat.extend_from_slice(&node.features);
        }

        let node_features = Array2::from_shape_vec(
            (num_nodes, feature_dim),
            features_flat,
        ).map_err(|e| crate::error::Error::GraphProcessing(
            format!("Failed to create feature matrix: {}", e)
        ))?;

        let node_id_to_idx: HashMap<u32, usize> = 
            node_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        let mut adjacency: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_nodes];
        for edge in &self.edges {
            let src_idx = node_id_to_idx[&edge.source];
            let tgt_idx = node_id_to_idx[&edge.target];
            adjacency[src_idx].push((tgt_idx, edge.weight));
        }

        let mut edge_offsets = Vec::with_capacity(num_nodes + 1);
        let mut edge_targets = Vec::with_capacity(num_edges);
        let mut edge_weights = Vec::with_capacity(num_edges);

        edge_offsets.push(0);
        for adj_list in adjacency {
            for (target, weight) in adj_list {
                edge_targets.push(target as u32);
                edge_weights.push(weight);
            }
            edge_offsets.push(edge_targets.len() as u32);
        }

        Ok(CompactGraph {
            node_ids,
            node_features,
            edge_offsets,
            edge_targets,
            edge_weights,
            num_nodes,
            num_edges,
            feature_dim,
        })
    }
}

impl CompactGraph {
    pub fn get_neighbors(&self, node_idx: usize) -> &[u32] {
        if node_idx >= self.num_nodes {
            return &[];
        }
        
        let start = self.edge_offsets[node_idx] as usize;
        let end = self.edge_offsets[node_idx + 1] as usize;
        &self.edge_targets[start..end]
    }

    pub fn get_edge_weights(&self, node_idx: usize) -> &[f32] {
        if node_idx >= self.num_nodes {
            return &[];
        }
        
        let start = self.edge_offsets[node_idx] as usize;
        let end = self.edge_offsets[node_idx + 1] as usize;
        &self.edge_weights[start..end]
    }

    pub fn node_features(&self) -> &Array2<f32> {
        &self.node_features
    }

    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new();
        
        graph.add_node(Node {
            id: 1,
            features: vec![0.1, 0.2, 0.3],
            label: None,
        });
        
        graph.add_node(Node {
            id: 2,
            features: vec![0.4, 0.5, 0.6],
            label: None,
        });
        
        graph.add_edge(Edge {
            source: 1,
            target: 2,
            weight: 0.8,
            edge_type: None,
        });

        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_compact_graph_conversion() {
        let mut graph = Graph::new();
        
        graph.add_node(Node {
            id: 1,
            features: vec![1.0, 2.0],
            label: None,
        });
        
        graph.add_node(Node {
            id: 2,
            features: vec![3.0, 4.0],
            label: None,
        });
        
        graph.add_edge(Edge {
            source: 1,
            target: 2,
            weight: 0.5,
            edge_type: None,
        });

        let compact = graph.to_compact().unwrap();
        
        assert_eq!(compact.num_nodes(), 2);
        assert_eq!(compact.num_edges(), 1);
        assert_eq!(compact.feature_dim(), 2);
        
        let neighbors = compact.get_neighbors(0);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], 1);
    }

    #[test]
    fn test_graph_config_builder() {
        let config = GraphConfig::builder()
            .max_nodes(50000)
            .diffusion_steps(10)
            .quantization("int8")
            .build();

        assert_eq!(config.max_nodes, 50000);
        assert_eq!(config.diffusion_steps, 10);
        assert_eq!(config.quantization, "int8");
    }
}