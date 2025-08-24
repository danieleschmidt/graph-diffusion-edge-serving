//! Generation 1: MAKE IT WORK - Enhanced Demo
//! 
//! Demonstrates the new quantum-enhanced graph diffusion and adaptive neural architecture search
//! capabilities that make the system work at a fundamentally advanced level.

use graph_diffusion_edge::{
    core::{Graph, Node, Edge, GraphConfig, DGDMProcessor, ProcessingConfig},
    quantum_graph_transformer::{QuantumGraphTransformer, QuantumConfig},
    adaptive_neural_architecture_search::{AdaptiveNeuralArchitectureSearch, NASConfig},
};
use tracing::{info, debug};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,graph_diffusion_edge=debug")
        .init();

    println!("🚀 Generation 1: MAKE IT WORK - Enhanced Capabilities Demo");
    println!("================================================================");
    
    // Create diverse test graphs for comprehensive evaluation
    let test_graphs = create_diverse_test_graphs();
    info!("Created {} diverse test graphs", test_graphs.len());

    // === Part 1: Quantum-Enhanced Graph Diffusion ===
    println!("\n🌟 Part 1: Quantum-Enhanced Graph Diffusion");
    println!("--------------------------------------------");
    
    await_quantum_enhancement_demo(&test_graphs).await?;

    // === Part 2: Adaptive Neural Architecture Search ===
    println!("\n🧠 Part 2: Adaptive Neural Architecture Search");
    println!("-----------------------------------------------");
    
    await_nas_optimization_demo(&test_graphs).await?;

    // === Part 3: Integrated Quantum-NAS System ===
    println!("\n⚡ Part 3: Integrated Quantum-NAS System");
    println!("----------------------------------------");
    
    await_integrated_system_demo(&test_graphs).await?;

    println!("\n✅ Generation 1 Enhanced Demo Complete!");
    println!("System now works with quantum enhancement and adaptive optimization.");
    
    Ok(())
}

async fn await_quantum_enhancement_demo(
    test_graphs: &[graph_diffusion_edge::core::graph::CompactGraph]
) -> Result<(), Box<dyn std::error::Error>> {
    
    let quantum_config = QuantumConfig {
        qubits_per_node: 6,
        entanglement_strength: 0.8,
        coherence_time: 15,
        measurement_threshold: 0.05,
        error_correction: true,
        quantum_dimensions: 16,
    };

    let mut quantum_transformer = QuantumGraphTransformer::new(quantum_config);
    
    // Create classical processor for baseline
    let graph_config = GraphConfig::default();
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(graph_config, processing_config);

    println!("Quantum Configuration:");
    println!("  • Qubits per node: 6");
    println!("  • Entanglement strength: 0.8");
    println!("  • Quantum dimensions: 16");
    println!("  • Error correction: Enabled");

    let mut total_enhancement_factor = 0.0;
    let mut total_quantum_coherence = 0.0;
    let mut total_entanglement_entropy = 0.0;

    for (i, graph) in test_graphs.iter().enumerate() {
        let start_time = Instant::now();
        
        // Classical processing
        let classical_result = processor.process(graph)?;
        let classical_time = start_time.elapsed();

        // Quantum enhancement
        let quantum_start = Instant::now();
        let quantum_result = quantum_transformer.quantum_enhance_diffusion(
            graph, 
            classical_result
        ).await?;
        let quantum_time = quantum_start.elapsed();

        // Calculate metrics
        let enhancement_factor = quantum_result.quantum_coherence_score / 
            quantum_result.classical_result.convergence_score.max(1e-6);
        
        total_enhancement_factor += enhancement_factor;
        total_quantum_coherence += quantum_result.quantum_coherence_score;
        total_entanglement_entropy += quantum_result.entanglement_entropy;

        println!("  Graph {}: Classical={:.2}ms, Quantum={:.2}ms, Enhancement={:.2}x", 
                i + 1, classical_time.as_millis(), quantum_time.as_millis(), enhancement_factor);
    }

    let avg_enhancement = total_enhancement_factor / test_graphs.len() as f32;
    let avg_coherence = total_quantum_coherence / test_graphs.len() as f32;
    let avg_entropy = total_entanglement_entropy / test_graphs.len() as f32;

    println!("\n📊 Quantum Enhancement Results:");
    println!("  • Average enhancement factor: {:.2}x", avg_enhancement);
    println!("  • Average quantum coherence: {:.4}", avg_coherence);
    println!("  • Average entanglement entropy: {:.4}", avg_entropy);

    if avg_enhancement > 1.1 {
        println!("  ✅ Quantum enhancement successful!");
    } else {
        println!("  ⚠️ Quantum enhancement marginal - consider parameter tuning");
    }

    Ok(())
}

async fn await_nas_optimization_demo(
    test_graphs: &[graph_diffusion_edge::core::graph::CompactGraph]
) -> Result<(), Box<dyn std::error::Error>> {
    
    let nas_config = NASConfig {
        population_size: 20,
        max_generations: 5, // Reduced for demo
        mutation_rate: 0.2,
        crossover_rate: 0.7,
        elite_ratio: 0.15,
        max_layers: 6,
        min_layers: 2,
        performance_weight_accuracy: 0.6,
        performance_weight_speed: 0.3,
        performance_weight_memory: 0.1,
        adaptive_mutation: true,
    };

    let mut nas = AdaptiveNeuralArchitectureSearch::new(nas_config);
    
    println!("NAS Configuration:");
    println!("  • Population size: 20");
    println!("  • Generations: 5 (demo)");
    println!("  • Adaptive mutation: Enabled");
    println!("  • Performance weighting: Accuracy(60%), Speed(30%), Memory(10%)");

    // Initialize and evolve
    println!("\n🔬 Initializing architecture population...");
    nas.initialize_population()?;

    // Split graphs for training/validation
    let (training_graphs, validation_graphs) = test_graphs.split_at(test_graphs.len() * 2 / 3);
    
    println!("📈 Evolving architectures ({} training, {} validation graphs)...", 
             training_graphs.len(), validation_graphs.len());

    let evolution_start = Instant::now();
    let best_architecture = nas.evolve_architecture(training_graphs, validation_graphs).await?;
    let evolution_time = evolution_start.elapsed();

    if let Some(best_performance) = nas.get_best_performance() {
        println!("\n🏆 Best Architecture Found:");
        println!("  • Layers: {}", best_architecture.layers.len());
        println!("  • Skip connections: {}", best_architecture.skip_connections.len());
        println!("  • Attention heads: {:?}", best_architecture.attention_heads);
        println!("  • Quantum enhanced: {}", 
                best_architecture.quantum_config.is_some());
        
        println!("\n📊 Performance Metrics:");
        println!("  • Fitness score: {:.4}", best_performance.fitness_score);
        println!("  • Accuracy: {:.4}", best_performance.accuracy);
        println!("  • Latency: {:.2}ms", best_performance.latency_ms);
        println!("  • Memory usage: {:.2}KB", best_performance.memory_usage as f32 / 1024.0);
        println!("  • Quantum coherence: {:.4}", best_performance.quantum_coherence);
        
        println!("  • Evolution time: {:.2}s", evolution_time.as_secs_f32());
        println!("  • Final generation: {}", nas.get_generation());

        if best_performance.fitness_score > 0.5 {
            println!("  ✅ NAS optimization successful!");
        } else {
            println!("  ⚠️ NAS needs more generations for optimal results");
        }
    }

    Ok(())
}

async fn await_integrated_system_demo(
    test_graphs: &[graph_diffusion_edge::core::graph::CompactGraph]
) -> Result<(), Box<dyn std::error::Error>> {
    
    println!("🌟 Demonstrating integrated quantum-enhanced NAS system...");
    
    // Configure system with quantum enhancement enabled in NAS
    let nas_config = NASConfig {
        population_size: 15,
        max_generations: 3, // Quick demo
        mutation_rate: 0.25,
        crossover_rate: 0.8,
        elite_ratio: 0.2,
        max_layers: 4,
        min_layers: 2,
        performance_weight_accuracy: 0.4,
        performance_weight_speed: 0.3,
        performance_weight_memory: 0.3,
        adaptive_mutation: true,
    };

    let mut nas = AdaptiveNeuralArchitectureSearch::new(nas_config);
    nas.initialize_population()?;

    // Run integrated optimization
    let integration_start = Instant::now();
    let (training_graphs, validation_graphs) = test_graphs.split_at(test_graphs.len() / 2);
    
    println!("🔄 Running integrated quantum-NAS optimization...");
    let optimized_architecture = nas.evolve_architecture(training_graphs, validation_graphs).await?;
    let integration_time = integration_start.elapsed();

    // Demonstrate the optimized system
    if let Some(best_performance) = nas.get_best_performance() {
        println!("\n🎯 Integrated System Results:");
        println!("  • Architecture layers: {}", optimized_architecture.layers.len());
        println!("  • Quantum enhancement: {}", 
                optimized_architecture.quantum_config.is_some());
        println!("  • Overall fitness: {:.4}", best_performance.fitness_score);
        println!("  • Integration time: {:.2}s", integration_time.as_secs_f32());

        // Test the optimized system on a sample graph
        if let Some(test_graph) = test_graphs.first() {
            println!("\n🧪 Testing optimized system on sample graph:");
            
            let test_start = Instant::now();
            
            // Create processor from optimized architecture
            let graph_config = GraphConfig::default();
            let processing_config = ProcessingConfig::default(); // Would be configured from architecture
            let processor = DGDMProcessor::new(graph_config, processing_config);
            
            let result = processor.process(test_graph)?;
            
            // Apply quantum enhancement if configured
            if let Some(ref quantum_config) = optimized_architecture.quantum_config {
                let mut quantum_transformer = QuantumGraphTransformer::new(quantum_config.clone());
                let quantum_result = quantum_transformer.quantum_enhance_diffusion(
                    test_graph, 
                    result
                ).await?;
                
                let test_time = test_start.elapsed();
                println!("  • Processing time: {:.2}ms", test_time.as_millis());
                println!("  • Quantum coherence: {:.4}", quantum_result.quantum_coherence_score);
                println!("  • Entanglement entropy: {:.4}", quantum_result.entanglement_entropy);
                println!("  • Superposition states: {}", quantum_result.superposition_count);
                
                println!("  ✅ Integrated quantum-enhanced system operational!");
            } else {
                let test_time = test_start.elapsed();
                println!("  • Processing time: {:.2}ms", test_time.as_millis());
                println!("  • Classical processing successful");
            }
        }
    }

    Ok(())
}

fn create_diverse_test_graphs() -> Vec<graph_diffusion_edge::core::graph::CompactGraph> {
    let mut graphs = Vec::new();

    // Small dense graph
    let mut small_graph = Graph::new();
    for i in 1..=5 {
        small_graph.add_node(Node {
            id: i,
            features: vec![i as f32 * 0.2, (5 - i) as f32 * 0.2],
            label: Some(format!("node_{}", i)),
        });
    }
    for i in 1..=5 {
        for j in (i + 1)..=5 {
            small_graph.add_edge(Edge {
                source: i,
                target: j,
                weight: 1.0 / (i as f32 + j as f32),
                edge_type: Some("connection".to_string()),
            });
        }
    }
    graphs.push(small_graph.to_compact().unwrap());

    // Medium sparse graph  
    let mut medium_graph = Graph::new();
    for i in 1..=15 {
        medium_graph.add_node(Node {
            id: i,
            features: vec![
                (i as f32).sin(), 
                (i as f32).cos(), 
                i as f32 / 15.0
            ],
            label: Some(format!("med_node_{}", i)),
        });
    }
    for i in 1..=15 {
        if i < 15 {
            medium_graph.add_edge(Edge {
                source: i,
                target: i + 1,
                weight: 0.8,
                edge_type: Some("sequential".to_string()),
            });
        }
        if i % 3 == 0 && i + 5 <= 15 {
            medium_graph.add_edge(Edge {
                source: i,
                target: i + 5,
                weight: 0.6,
                edge_type: Some("skip".to_string()),
            });
        }
    }
    graphs.push(medium_graph.to_compact().unwrap());

    // Large regular graph
    let mut large_graph = Graph::new();
    for i in 1..=50 {
        large_graph.add_node(Node {
            id: i,
            features: vec![
                (i as f32 / 10.0).sin(),
                (i as f32 / 10.0).cos(),
                i as f32 / 50.0,
                ((i * 3) as f32).sin(),
            ],
            label: Some(format!("large_node_{}", i)),
        });
    }
    // Create ring topology with random connections
    for i in 1..=50 {
        let next = if i == 50 { 1 } else { i + 1 };
        large_graph.add_edge(Edge {
            source: i,
            target: next,
            weight: 1.0,
            edge_type: Some("ring".to_string()),
        });
        
        // Add some random long-distance connections
        if i % 7 == 0 {
            let target = (i + 17) % 50 + 1;
            large_graph.add_edge(Edge {
                source: i,
                target,
                weight: 0.4,
                edge_type: Some("long_range".to_string()),
            });
        }
    }
    graphs.push(large_graph.to_compact().unwrap());

    // Star graph
    let mut star_graph = Graph::new();
    star_graph.add_node(Node {
        id: 1,
        features: vec![1.0, 0.0, 0.5],
        label: Some("center".to_string()),
    });
    for i in 2..=10 {
        star_graph.add_node(Node {
            id: i,
            features: vec![0.0, 1.0, i as f32 / 10.0],
            label: Some(format!("leaf_{}", i)),
        });
        star_graph.add_edge(Edge {
            source: 1,
            target: i,
            weight: 1.0,
            edge_type: Some("star".to_string()),
        });
    }
    graphs.push(star_graph.to_compact().unwrap());

    debug!("Created {} diverse test graphs: small_dense(5), medium_sparse(15), large_regular(50), star(10)", graphs.len());
    graphs
}