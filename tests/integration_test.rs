//! Integration tests for DGDM server

use graph_diffusion_edge::{
    core::{Graph, Node, Edge, GraphConfig, ProcessingConfig, DGDMProcessor},
    serving::{DGDMServer, start_server_with_config},
    validation::{GraphValidator, SecurityValidator},
};
use serde_json::json;
use std::sync::Arc;
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn test_basic_server_functionality() {
    let config = GraphConfig::default();
    let processing_config = ProcessingConfig::default();
    
    // Test processor creation
    let processor = DGDMProcessor::new(config, processing_config);
    assert!(processor.can_process(&create_test_graph()));
}

#[tokio::test]
async fn test_graph_validation() {
    let validator = GraphValidator::new();
    
    // Valid graph
    let valid_graph = create_test_graph_struct();
    assert!(validator.validate_graph(&valid_graph).is_ok());
    
    // Invalid graph - empty
    let empty_graph = Graph::new();
    assert!(validator.validate_graph(&empty_graph).is_err());
}

#[tokio::test]
async fn test_security_validation() {
    let validator = SecurityValidator::new();
    
    // Safe strings
    assert!(validator.validate_string("safe_input", "test_field").is_ok());
    assert!(validator.validate_string("Hello World 123", "test_field").is_ok());
    
    // Malicious patterns
    assert!(validator.validate_string("DROP TABLE users", "test_field").is_err());
    assert!(validator.validate_string("<script>alert('xss')</script>", "test_field").is_err());
    assert!(validator.validate_string("../../../etc/passwd", "test_field").is_err());
}

#[tokio::test]
async fn test_graph_processing_robustness() {
    let config = GraphConfig::default();
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(config, processing_config);
    
    let graph = create_test_graph();
    
    // Test timeout behavior
    let result = timeout(
        Duration::from_secs(5),
        async { processor.process(&graph) }
    ).await;
    
    assert!(result.is_ok(), "Processing should complete within timeout");
    let processing_result = result.unwrap();
    assert!(processing_result.is_ok(), "Processing should succeed");
}

#[tokio::test]
async fn test_batch_processing_limits() {
    let config = GraphConfig::default();
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(config, processing_config);
    
    let graph = create_test_graph();
    let graphs: Vec<&_> = vec![&graph; 3]; // Small batch
    
    let result = processor.process_batch(graphs).await;
    assert!(result.is_ok(), "Batch processing should succeed");
    
    let results = result.unwrap();
    assert_eq!(results.len(), 3, "Should process all graphs in batch");
}

#[tokio::test]
async fn test_error_handling() {
    let config = GraphConfig {
        max_nodes: 2, // Very restrictive
        ..Default::default()
    };
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(config, processing_config);
    
    let large_graph = create_large_test_graph();
    assert!(!processor.can_process(&large_graph), "Should reject large graphs");
}

fn create_test_graph_struct() -> Graph {
    let mut graph = Graph::new();
    
    graph.add_node(Node {
        id: 1,
        features: vec![1.0, 0.0],
        label: Some("node1".to_string()),
    });
    
    graph.add_node(Node {
        id: 2,
        features: vec![0.0, 1.0],
        label: Some("node2".to_string()),
    });
    
    graph.add_edge(Edge {
        source: 1,
        target: 2,
        weight: 1.0,
        edge_type: Some("connection".to_string()),
    });
    
    graph
}

fn create_test_graph() -> graph_diffusion_edge::core::graph::CompactGraph {
    create_test_graph_struct().to_compact().unwrap()
}

fn create_large_test_graph() -> graph_diffusion_edge::core::graph::CompactGraph {
    let mut graph = Graph::new();
    
    // Create a graph with 5 nodes (should exceed limit of 2)
    for i in 1..=5 {
        graph.add_node(Node {
            id: i,
            features: vec![i as f32, i as f32],
            label: None,
        });
    }
    
    for i in 1..5 {
        graph.add_edge(Edge {
            source: i,
            target: i + 1,
            weight: 1.0,
            edge_type: None,
        });
    }
    
    graph.to_compact().unwrap()
}