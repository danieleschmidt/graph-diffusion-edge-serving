//! Basic demonstration of graph diffusion functionality
//! This example shows core features working

use graph_diffusion_edge::{
    core::{Graph, Node, Edge, GraphConfig, DGDMProcessor, ProcessingConfig},
    Result,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Graph Diffusion Edge Serving - Basic Demo");
    
    // Create a simple test graph
    let mut graph = Graph::new();
    
    // Add nodes with features
    graph.add_node(Node {
        id: 1,
        features: vec![1.0, 0.0, 0.5],
        label: Some("Node A".to_string()),
    });
    
    graph.add_node(Node {
        id: 2,
        features: vec![0.0, 1.0, 0.3],
        label: Some("Node B".to_string()),
    });
    
    graph.add_node(Node {
        id: 3,
        features: vec![0.5, 0.5, 1.0],
        label: Some("Node C".to_string()),
    });
    
    // Add edges
    graph.add_edge(Edge {
        source: 1,
        target: 2,
        weight: 0.8,
        edge_type: Some("connection".to_string()),
    });
    
    graph.add_edge(Edge {
        source: 2,
        target: 3,
        weight: 0.6,
        edge_type: Some("connection".to_string()),
    });
    
    graph.add_edge(Edge {
        source: 1,
        target: 3,
        weight: 0.4,
        edge_type: Some("connection".to_string()),
    });
    
    println!("✅ Created graph with {} nodes and {} edges", 
        graph.nodes.len(), graph.edges.len());
    
    // Convert to compact format for processing
    let compact_graph = graph.to_compact()?;
    println!("✅ Converted to compact format: {} nodes, {} edges", 
        compact_graph.num_nodes, compact_graph.num_edges);
    
    // Configure graph settings
    let mut graph_config = GraphConfig::default();
    graph_config.max_nodes = 1000;
    graph_config.max_edges = 10000;
    graph_config.feature_dim = 3;
    graph_config.diffusion_steps = 3;
    graph_config.batch_size = 1;
    
    // Configure processing
    let processing_config = ProcessingConfig {
        temperature: 0.1,
        dropout_rate: 0.0,
        use_attention: false,
        normalize_features: true,
        mixed_precision: false,
    };
    
    // Create DGDM processor
    let processor = DGDMProcessor::new(graph_config, processing_config);
    println!("✅ Created DGDM processor");
    
    // Process the graph (simulate diffusion)
    let result = processor.process(&compact_graph)?;
    println!("✅ Graph processing completed");
    
    // Display results
    println!("\n📊 Processing Results:");
    println!("- Processed {} nodes", result.embeddings.nrows());
    println!("- Feature dimension: {}", result.embeddings.ncols());
    println!("- Processing took: {:.2}ms", result.processing_time_ms);
    
    // Show sample embeddings
    println!("\n🎯 Sample Node Embeddings:");
    for i in 0..result.embeddings.nrows().min(3) {
        let embedding = result.embeddings.row(i);
        println!("Node {}: [{:.3}, {:.3}, {:.3}]", 
            i + 1,
            embedding[0], 
            embedding[1], 
            embedding[2]
        );
    }
    
    println!("\n🎉 Demo completed successfully!");
    Ok(())
}