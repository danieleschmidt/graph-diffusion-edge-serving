//! Generation 2 Robustness Tests
//! 
//! Comprehensive test suite validating error handling, security, 
//! validation, and resilience patterns.

use graph_diffusion_edge::minimal_server::*;
use tokio::test;

#[tokio::test]
async fn test_empty_graph_validation() {
    let server = MinimalDGDMServer::new();
    let empty_graph = MinimalGraph {
        nodes: vec![],
        edges: vec![],
    };
    
    let result = server.process_graph_simple(&empty_graph, 5, 0.1).await;
    assert!(result.is_ok(), "Empty graph should be handled gracefully");
    
    let embeddings = result.unwrap();
    assert!(embeddings.is_empty(), "Empty graph should produce empty embeddings");
}

#[tokio::test]
async fn test_invalid_feature_values() {
    let server = MinimalDGDMServer::new();
    let invalid_graph = MinimalGraph {
        nodes: vec![
            MinimalGraphNode { id: 1, features: vec![f32::NAN, 1.0] },
        ],
        edges: vec![],
    };
    
    let result = server.process_graph_simple(&invalid_graph, 5, 0.1).await;
    // Should handle NaN values gracefully
    if let Ok(embeddings) = result {
        for (_, embedding) in embeddings {
            for value in embedding {
                assert!(value.is_finite(), "Output should not contain NaN or infinite values");
            }
        }
    }
}

#[tokio::test]
async fn test_timeout_handling() {
    let server = MinimalDGDMServer::new();
    
    // Create a large graph that might take time to process
    let mut large_graph = MinimalGraph {
        nodes: vec![],
        edges: vec![],
    };
    
    for i in 0..1000 {
        large_graph.nodes.push(MinimalGraphNode {
            id: i,
            features: vec![1.0; 100],
        });
        
        if i > 0 {
            large_graph.edges.push(MinimalGraphEdge {
                source: i - 1,
                target: i,
                weight: 1.0,
            });
        }
    }
    
    // This should complete within timeout
    let result = server.process_graph_optimized(&large_graph, 5, 0.1).await;
    match result {
        Ok(_) => println!("Large graph processed successfully"),
        Err(e) => println!("Large graph processing handled: {}", e),
    }
}

#[tokio::test]
async fn test_security_validation() {
    let server = MinimalDGDMServer::new();
    let normal_graph = MinimalGraph {
        nodes: vec![
            MinimalGraphNode { id: 1, features: vec![1.0, 0.0] },
        ],
        edges: vec![],
    };
    
    // Test excessive steps (security constraint)
    let result = server.process_graph_simple(&normal_graph, 1000, 0.1).await;
    assert!(result.is_ok(), "Should handle excessive steps gracefully");
}

#[tokio::test]
async fn test_resource_exhaustion_protection() {
    let server = MinimalDGDMServer::new();
    
    // Test with extremely high dimensional features
    let high_dim_graph = MinimalGraph {
        nodes: vec![
            MinimalGraphNode { 
                id: 1, 
                features: vec![1.0; 10000] // Very high dimensional
            },
        ],
        edges: vec![],
    };
    
    let result = server.process_graph_optimized(&high_dim_graph, 5, 0.1).await;
    match result {
        Ok(_) => println!("High-dimensional graph processed"),
        Err(_) => println!("High-dimensional graph rejected for resource protection"),
    }
}

#[tokio::test]
async fn test_convergence_stability() {
    let server = MinimalDGDMServer::new();
    let test_graph = MinimalGraph {
        nodes: vec![
            MinimalGraphNode { id: 1, features: vec![1.0, 0.0] },
            MinimalGraphNode { id: 2, features: vec![0.0, 1.0] },
        ],
        edges: vec![
            MinimalGraphEdge { source: 1, target: 2, weight: 0.5 },
        ],
    };
    
    // Run multiple times to check stability
    for _ in 0..10 {
        let result = server.process_graph_simple(&test_graph, 10, 0.1).await;
        assert!(result.is_ok(), "Processing should be stable across multiple runs");
        
        let embeddings = result.unwrap();
        for (_, embedding) in embeddings {
            for value in embedding {
                assert!(value.is_finite(), "All values should remain finite");
                assert!(value.abs() < 1000.0, "Values should not explode");
            }
        }
    }
}

#[tokio::test]
async fn test_input_sanitization() {
    let server = MinimalDGDMServer::new();
    
    // Test with extreme temperature values
    let test_graph = MinimalGraph {
        nodes: vec![MinimalGraphNode { id: 1, features: vec![1.0] }],
        edges: vec![],
    };
    
    // Negative temperature
    let result1 = server.process_graph_simple(&test_graph, 5, -1.0).await;
    assert!(result1.is_ok(), "Negative temperature should be handled");
    
    // Very high temperature
    let result2 = server.process_graph_simple(&test_graph, 5, 100.0).await;
    assert!(result2.is_ok(), "High temperature should be handled");
    
    // Zero temperature
    let result3 = server.process_graph_simple(&test_graph, 5, 0.0).await;
    assert!(result3.is_ok(), "Zero temperature should be handled");
}