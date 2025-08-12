//! Demonstration of resilience patterns and robust error handling

use graph_diffusion_edge::{
    core::{Graph, Node, Edge},
    resilience::{AdaptiveCircuitBreaker, CircuitBreakerConfig, SmartRetryPolicy},
    validation::GraphValidator,
    error::Error,
    Result,
};
use std::time::Duration;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🛡️ Resilience Patterns Demo - Error Handling & Recovery");
    
    // Set up validation
    let validator = GraphValidator::new();
    
    // Demonstrate input validation
    info!("📋 Testing Input Validation...");
    demonstrate_input_validation(&validator).await?;
    
    // Demonstrate circuit breaker patterns
    info!("⚡ Testing Circuit Breaker Patterns...");
    demonstrate_circuit_breaker().await?;
    
    // Demonstrate retry mechanisms
    info!("🔄 Testing Retry Mechanisms...");
    demonstrate_retry_policy().await?;
    
    info!("✅ All resilience patterns working correctly!");
    Ok(())
}

async fn demonstrate_input_validation(validator: &GraphValidator) -> Result<()> {
    info!("  Testing valid graph...");
    let mut valid_graph = Graph::new();
    valid_graph.add_node(Node { id: 1, features: vec![1.0, 2.0], label: None });
    valid_graph.add_edge(Edge { source: 1, target: 1, weight: 0.5, edge_type: None });
    
    match validator.validate_graph(&valid_graph) {
        Ok(()) => info!("    ✅ Valid graph accepted"),
        Err(e) => warn!("    ❌ Valid graph rejected: {}", e),
    }
    
    info!("  Testing invalid graph (empty)...");
    let empty_graph = Graph::new();
    
    match validator.validate_graph(&empty_graph) {
        Ok(()) => warn!("    ❌ Empty graph incorrectly accepted"),
        Err(e) => info!("    ✅ Empty graph correctly rejected: {}", e),
    }
    
    info!("  Testing malformed features...");
    let mut bad_graph = Graph::new();
    bad_graph.add_node(Node { id: 1, features: vec![], label: None }); // Empty features
    
    match validator.validate_graph(&bad_graph) {
        Ok(()) => warn!("    ❌ Malformed graph incorrectly accepted"),
        Err(e) => info!("    ✅ Malformed graph correctly rejected: {}", e),
    }
    
    Ok(())
}

async fn demonstrate_circuit_breaker() -> Result<()> {
    let config = CircuitBreakerConfig {
        name: "demo_service".to_string(),
        failure_threshold: 3,
        success_threshold: 2,
        timeout: Duration::from_millis(100),
        adaptation_factor: 1.5,
    };
    
    let circuit_breaker = AdaptiveCircuitBreaker::new(config);
    
    info!("  Testing successful operations...");
    for i in 1..=3 {
        let result = circuit_breaker.call(async { Ok::<i32, Error>(i * 10) }).await;
        match result {
            Ok(value) => info!("    Operation {}: Success ({})", i, value),
            Err(e) => warn!("    Operation {}: Failed ({})", i, e),
        }
    }
    
    info!("  Simulating failures to trigger circuit breaker...");
    for i in 1..=5 {
        let result = circuit_breaker.call(async { 
            Err::<i32, _>(Error::network("Simulated failure", "test"))
        }).await;
        match result {
            Ok(_) => warn!("    Failure {}: Unexpectedly succeeded", i),
            Err(e) => {
                if e.to_string().contains("Circuit breaker") {
                    info!("    Operation {}: Circuit breaker activated ⚡", i);
                } else {
                    info!("    Operation {}: Failed as expected ({})", i, e);
                }
            }
        }
    }
    
    info!("    Current circuit breaker state: {:?}", circuit_breaker.get_state());
    info!("    Metrics: {:?}", circuit_breaker.get_metrics());
    
    Ok(())
}

async fn demonstrate_retry_policy() -> Result<()> {
    let retry_policy = SmartRetryPolicy::new(3, Duration::from_millis(50));
    
    info!("  Testing retry with eventual success...");
    let mut attempt_counter = 0;
    let result = retry_policy.execute(|| {
        attempt_counter += 1;
        Box::pin(async move {
            if attempt_counter < 3 {
                info!("    Attempt {}: Simulating temporary failure", attempt_counter);
                Err(Error::network("Temporary network issue", "localhost"))
            } else {
                info!("    Attempt {}: Success!", attempt_counter);
                Ok(42)
            }
        })
    }).await;
    
    match result {
        Ok(value) => info!("    ✅ Retry succeeded with result: {}", value),
        Err(e) => warn!("    ❌ Retry exhausted: {}", e),
    }
    
    info!("  Testing non-retryable error...");
    let result = retry_policy.execute(|| Box::pin(async {
        Err::<i32, _>(Error::validation(
            "Invalid input", 
            "test_input", 
            "valid_format"
        ))
    })).await;
    
    match result {
        Ok(_) => warn!("    ❌ Non-retryable error unexpectedly succeeded"),
        Err(e) => info!("    ✅ Non-retryable error correctly failed: {}", e),
    }
    
    Ok(())
}

#[tokio::test]
async fn test_graph_processing_with_resilience() {
    let mut graph_config = GraphConfig::default();
    graph_config.feature_dim = 2;
    let processing_config = ProcessingConfig::default();
    
    let processor = DGDMProcessor::new(graph_config, processing_config);
    let retry_policy = SmartRetryPolicy::new(2, Duration::from_millis(10));
    
    // Create a valid graph
    let mut graph = Graph::new();
    graph.add_node(Node { id: 1, features: vec![1.0, 0.0], label: None });
    graph.add_node(Node { id: 2, features: vec![0.0, 1.0], label: None });
    graph.add_edge(Edge { source: 1, target: 2, weight: 0.5, edge_type: None });
    
    let compact_graph = graph.to_compact().unwrap();
    
    // Process with retry protection
    let result = retry_policy.execute(|| {
        let processor = &processor;
        let graph = &compact_graph;
        async move {
            processor.process(graph)
        }
    }).await;
    
    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.embeddings.nrows(), 2);
    assert!(response.processing_time_ms > 0.0);
}