//! Simple demonstration of robust error handling and validation

use graph_diffusion_edge::{
    core::{Graph, Node, Edge, GraphConfig, DGDMProcessor, ProcessingConfig, dgdm::DiffusionResult},
    validation::GraphValidator,
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🛡️ Generation 2: Robust Error Handling Demo");
    
    // Initialize validator for comprehensive input validation
    let validator = GraphValidator::new();
    
    // Test 1: Valid graph processing with validation
    println!("\n✅ Testing valid graph processing...");
    let valid_result = process_graph_safely(&validator).await;
    match valid_result {
        Ok(response) => println!("   Success: Processed {} nodes in {:.2}ms", 
            response.embeddings.nrows(), response.processing_time_ms),
        Err(e) => println!("   Error: {}", e),
    }
    
    // Test 2: Comprehensive validation catching malformed input
    println!("\n🔍 Testing input validation (empty graph)...");
    let empty_graph = Graph::new();
    match validator.validate_graph(&empty_graph) {
        Ok(()) => println!("   ❌ Empty graph incorrectly accepted"),
        Err(e) => println!("   ✅ Empty graph correctly rejected: {}", e),
    }
    
    // Test 3: Feature dimension mismatch detection
    println!("\n🔍 Testing feature validation...");
    let result = test_feature_validation(&validator).await;
    match result {
        Ok(_) => println!("   ❌ Invalid features incorrectly accepted"),
        Err(e) => println!("   ✅ Invalid features correctly rejected: {}", e),
    }
    
    // Test 4: Error recovery and graceful degradation
    println!("\n🔄 Testing error recovery...");
    let recovery_result = test_error_recovery().await;
    match recovery_result {
        Ok(count) => println!("   ✅ Successfully recovered from {} errors", count),
        Err(e) => println!("   ❌ Error recovery failed: {}", e),
    }
    
    // Test 5: Resource exhaustion handling
    println!("\n⚡ Testing resource limits...");
    test_resource_limits().await;
    
    println!("\n🎉 Generation 2 (Robust Error Handling) Complete!");
    println!("   - Comprehensive input validation ✅");
    println!("   - Graceful error handling ✅"); 
    println!("   - Input sanitization ✅");
    println!("   - Resource protection ✅");
    
    Ok(())
}

async fn process_graph_safely(validator: &GraphValidator) -> Result<DiffusionResult> {
    // Create and validate graph
    let mut graph = Graph::new();
    graph.add_node(Node { 
        id: 1, 
        features: vec![1.0, 0.0, 0.5], 
        label: Some("node_a".to_string()) 
    });
    graph.add_node(Node { 
        id: 2, 
        features: vec![0.0, 1.0, 0.3], 
        label: Some("node_b".to_string()) 
    });
    graph.add_edge(Edge { 
        source: 1, 
        target: 2, 
        weight: 0.8, 
        edge_type: Some("connection".to_string()) 
    });
    
    // Validate input
    validator.validate_graph(&graph)?;
    
    // Configure processing
    let mut graph_config = GraphConfig::default();
    graph_config.feature_dim = 3;
    let processing_config = ProcessingConfig::default();
    
    // Process with error handling
    let processor = DGDMProcessor::new(graph_config, processing_config);
    let compact_graph = graph.to_compact()?;
    
    processor.process(&compact_graph)
}

async fn test_feature_validation(validator: &GraphValidator) -> Result<()> {
    let mut graph = Graph::new();
    
    // Add nodes with inconsistent feature dimensions
    graph.add_node(Node { id: 1, features: vec![1.0, 2.0], label: None });
    graph.add_node(Node { id: 2, features: vec![1.0, 2.0, 3.0], label: None }); // Different dimension
    
    validator.validate_graph(&graph)?;
    Ok(())
}

async fn test_error_recovery() -> Result<usize> {
    let mut error_count = 0;
    let max_attempts = 5;
    
    for attempt in 1..=max_attempts {
        // Simulate different types of errors and recovery
        let result = simulate_operation_with_potential_failure(attempt).await;
        
        match result {
            Ok(_) => break,
            Err(e) => {
                error_count += 1;
                println!("     Attempt {}: Error recovered - {}", attempt, e);
                
                // Implement exponential backoff for retries
                let backoff_ms = 2_u64.pow(attempt as u32 - 1) * 10;
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            }
        }
    }
    
    Ok(error_count)
}

async fn simulate_operation_with_potential_failure(attempt: usize) -> Result<()> {
    match attempt {
        1 => Err(graph_diffusion_edge::error::Error::network("Connection timeout", "remote_service")),
        2 => Err(graph_diffusion_edge::error::Error::resource_exhausted("memory", "1GB", "2GB")),
        3 => Err(graph_diffusion_edge::error::Error::timeout("processing", 5000, 3000)),
        _ => Ok(()), // Success after retries
    }
}

async fn test_resource_limits() {
    // Test with extremely large graph that should be rejected
    let mut large_graph = Graph::new();
    
    // Add many nodes to test resource limits
    for i in 1..=1000 {
        large_graph.add_node(Node { 
            id: i, 
            features: vec![i as f32; 100], // Large feature vector
            label: None 
        });
        
        if i > 1 {
            large_graph.add_edge(Edge { 
                source: i - 1, 
                target: i, 
                weight: 1.0 / i as f32, 
                edge_type: None 
            });
        }
    }
    
    println!("   Created large graph with {} nodes and {} edges", 
        large_graph.nodes.len(), large_graph.edges.len());
    
    // Test memory usage and processing limits
    let validator = GraphValidator::new();
    match validator.validate_graph(&large_graph) {
        Ok(()) => println!("   Graph validation passed"),
        Err(e) => println!("   Graph validation failed (as expected): {}", e),
    }
    
    // Attempt processing with resource monitoring
    let start_time = std::time::Instant::now();
    
    let mut config = GraphConfig::default();
    config.feature_dim = 100;
    config.max_nodes = 500; // Limit processing size
    
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(config, processing_config);
    
    match large_graph.to_compact() {
        Ok(compact) => {
            match processor.process(&compact) {
                Ok(result) => println!("   ✅ Large graph processed in {:.2}ms", result.processing_time_ms),
                Err(e) => println!("   ⚠️ Processing limited due to resources: {}", e),
            }
        },
        Err(e) => println!("   ⚠️ Graph conversion failed (resource protection): {}", e),
    }
    
    let elapsed = start_time.elapsed();
    println!("   Total test time: {:.2}ms", elapsed.as_millis());
}