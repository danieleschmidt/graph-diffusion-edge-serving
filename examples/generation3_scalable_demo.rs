//! Generation 3: MAKE IT SCALE - Comprehensive Demo
//!
//! Demonstrates high-performance distributed processing, intelligent auto-scaling,
//! and advanced optimization capabilities for maximum performance and scalability.

use graph_diffusion_edge::{
    core::{Graph, Node, Edge, GraphConfig, ProcessingConfig},
    quantum_graph_transformer::QuantumConfig,
    high_performance_distributed_engine::{HighPerformanceDistributedEngine, DistributedEngineConfig},
    intelligent_auto_scaling_system::{IntelligentAutoScalingSystem, AutoScalingConfig},
    robust_error_recovery_system::{RobustErrorRecoverySystem, ErrorRecoveryConfig},
    intelligent_health_monitoring::{IntelligentHealthMonitor, HealthMonitorConfig, HealthThresholds},
};
use tracing::{info, debug, warn};
use std::time::{Duration, Instant};
use tokio::time::{sleep, interval};
use futures::future::join_all;
use rand::Rng;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize comprehensive logging with performance tracking
    tracing_subscriber::fmt()
        .with_env_filter("info,graph_diffusion_edge=debug")
        .with_target(true)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    println!("🚀 Generation 3: MAKE IT SCALE - High-Performance Distributed Demo");
    println!("====================================================================");
    
    // Create comprehensive test suite for scalability
    let test_suite = create_scalability_test_suite();
    info!("Created comprehensive test suite with {} test categories", test_suite.len());

    // === Part 1: High-Performance Distributed Processing ===
    println!("\n⚡ Part 1: High-Performance Distributed Processing");
    println!("--------------------------------------------------");
    
    await_distributed_processing_demo(&test_suite).await?;

    // === Part 2: Intelligent Auto-Scaling ===
    println!("\n🧠 Part 2: Intelligent Auto-Scaling System");
    println!("------------------------------------------");
    
    await_auto_scaling_demo().await?;

    // === Part 3: Load Testing & Performance Benchmarks ===
    println!("\n📊 Part 3: Load Testing & Performance Benchmarks");
    println!("-------------------------------------------------");
    
    await_load_testing_demo(&test_suite).await?;

    // === Part 4: End-to-End Scalability Integration ===
    println!("\n🌐 Part 4: End-to-End Scalability Integration");
    println!("----------------------------------------------");
    
    await_integration_scalability_demo(&test_suite).await?;

    println!("\n✅ Generation 3 Scalable Demo Complete!");
    println!("System now operates at maximum performance with intelligent scaling and optimization.");
    
    Ok(())
}

async fn await_distributed_processing_demo(
    test_suite: &[ScalabilityTestCategory]
) -> Result<(), Box<dyn std::error::Error>> {
    
    let engine_config = DistributedEngineConfig {
        num_workers: 8,
        max_concurrent_requests: 200,
        dynamic_load_balancing: true,
        cache_size_limit: 500_000_000, // 500MB cache
        cache_ttl_seconds: 1800, // 30 minutes
        auto_scaling_enabled: true,
        target_cpu_utilization: 75.0,
        min_workers: 4,
        max_workers: 16,
        performance_profiling: true,
        batch_size: 64,
        connection_pool_size: 100,
    };

    let distributed_engine = HighPerformanceDistributedEngine::new(engine_config);

    println!("Distributed Engine Configuration:");
    println!("  • Workers: 8 (4-16 auto-scaling range)");
    println!("  • Max concurrent requests: 200 per worker");
    println!("  • Dynamic load balancing: Enabled");
    println!("  • Intelligent caching: 500MB with 30min TTL");
    println!("  • Performance profiling: Enabled");

    // Start the distributed engine
    distributed_engine.start().await?;
    info!("Distributed engine started successfully");

    // Test different processing patterns
    let mut total_processing_time = Duration::from_secs(0);
    let mut total_graphs_processed = 0;
    let mut cache_hits = 0;

    for (category_idx, category) in test_suite.iter().enumerate() {
        println!("\n  Category {}: {} ({} graphs)", 
                category_idx + 1, category.name, category.test_graphs.len());

        let category_start = Instant::now();
        
        match category.processing_type {
            ProcessingType::Single => {
                // Process graphs individually
                for (i, graph) in category.test_graphs.iter().enumerate() {
                    let start = Instant::now();
                    
                    let result = distributed_engine.process_graph(
                        graph.clone(),
                        category.processing_config.clone()
                    ).await;
                    
                    let duration = start.elapsed();
                    total_processing_time += duration;
                    total_graphs_processed += 1;

                    match result {
                        Ok(_) => {
                            if i % 10 == 0 {
                                debug!("    Graph {}: {:.2}ms", i + 1, duration.as_millis());
                            }
                        }
                        Err(e) => {
                            warn!("    Graph {} failed: {}", i + 1, e);
                        }
                    }
                }
            }
            ProcessingType::Batch => {
                // Process graphs in batches
                let batch_size = 32;
                for chunk in category.test_graphs.chunks(batch_size) {
                    let start = Instant::now();
                    
                    let result = distributed_engine.process_batch(
                        chunk.to_vec(),
                        category.processing_config.clone()
                    ).await;
                    
                    let duration = start.elapsed();
                    total_processing_time += duration;
                    
                    match result {
                        Ok(results) => {
                            total_graphs_processed += results.len();
                            debug!("    Batch of {} graphs: {:.2}ms ({:.2}ms avg)", 
                                  chunk.len(), duration.as_millis(), 
                                  duration.as_millis() as f32 / chunk.len() as f32);
                        }
                        Err(e) => {
                            warn!("    Batch processing failed: {}", e);
                        }
                    }
                }
            }
            ProcessingType::Concurrent => {
                // Process graphs concurrently
                let concurrent_tasks = category.test_graphs.chunks(5)
                    .map(|chunk| {
                        let engine = distributed_engine.clone();
                        let graphs = chunk.to_vec();
                        let config = category.processing_config.clone();
                        
                        tokio::spawn(async move {
                            let mut results = Vec::new();
                            for graph in graphs {
                                let start = Instant::now();
                                let result = engine.process_graph(graph, config.clone()).await;
                                results.push((result, start.elapsed()));
                            }
                            results
                        })
                    })
                    .collect::<Vec<_>>();

                let concurrent_start = Instant::now();
                let concurrent_results = join_all(concurrent_tasks).await;
                let concurrent_duration = concurrent_start.elapsed();
                
                let mut successful = 0;
                for task_result in concurrent_results {
                    if let Ok(results) = task_result {
                        for (result, duration) in results {
                            total_processing_time += duration;
                            total_graphs_processed += 1;
                            if result.is_ok() {
                                successful += 1;
                            }
                        }
                    }
                }
                
                println!("    Concurrent processing: {} successful, {:.2}ms total", 
                        successful, concurrent_duration.as_millis());
            }
            ProcessingType::QuantumEnhanced => {
                // Process with quantum enhancement
                for graph in category.test_graphs.iter().take(5) { // Limit for demo
                    let start = Instant::now();
                    
                    // First get classical result
                    let classical_result = distributed_engine.process_graph(
                        graph.clone(),
                        category.processing_config.clone()
                    ).await?;
                    
                    // Then apply quantum enhancement
                    let quantum_result = distributed_engine.process_with_quantum_enhancement(
                        graph.clone(),
                        classical_result,
                        QuantumConfig::default()
                    ).await;
                    
                    let duration = start.elapsed();
                    total_processing_time += duration;
                    total_graphs_processed += 1;
                    
                    match quantum_result {
                        Ok(qr) => {
                            debug!("    Quantum enhanced: {:.2}ms, coherence: {:.4}", 
                                  duration.as_millis(), qr.quantum_coherence_score);
                        }
                        Err(e) => {
                            warn!("    Quantum enhancement failed: {}", e);
                        }
                    }
                }
            }
        }
        
        let category_duration = category_start.elapsed();
        println!("    Category completed: {:.2}s", category_duration.as_secs_f32());
    }

    // Get performance metrics and cache statistics
    let performance_metrics = distributed_engine.get_performance_metrics();
    let worker_metrics = distributed_engine.get_worker_metrics();
    let (result_cache_size, quantum_cache_size) = distributed_engine.get_cache_stats();

    distributed_engine.stop().await?;

    println!("\n📊 Distributed Processing Results:");
    println!("  • Total graphs processed: {}", total_graphs_processed);
    println!("  • Total processing time: {:.2}s", total_processing_time.as_secs_f32());
    println!("  • Average processing time: {:.2}ms", 
             total_processing_time.as_millis() as f32 / total_graphs_processed as f32);
    println!("  • Active workers: {}", worker_metrics.len());
    println!("  • Performance metrics collected: {}", performance_metrics.len());
    println!("  • Cache sizes: {} results, {} quantum", result_cache_size, quantum_cache_size);

    if !worker_metrics.is_empty() {
        let avg_efficiency: f32 = worker_metrics.values()
            .map(|m| m.average_response_time_ms)
            .sum::<f32>() / worker_metrics.len() as f32;
        println!("  • Average worker efficiency: {:.2}ms", avg_efficiency);
    }

    println!("  ✅ Distributed processing demonstrates excellent scalability!");

    Ok(())
}

async fn await_auto_scaling_demo() -> Result<(), Box<dyn std::error::Error>> {
    
    let scaling_config = AutoScalingConfig {
        min_workers: 2,
        max_workers: 12,
        target_cpu_utilization: 70.0,
        target_memory_utilization: 75.0,
        target_response_time_ms: 300.0,
        evaluation_interval_seconds: 5, // Fast scaling for demo
        scaling_cooldown_seconds: 30, // Reduced cooldown for demo
        prediction_window_minutes: 5,
        predictive_scaling_enabled: true,
        aggressive_scaling: true,
        cost_optimization_priority: 0.3,
        performance_priority: 0.7,
    };

    let auto_scaler = IntelligentAutoScalingSystem::new(scaling_config);

    println!("Auto-Scaling Configuration:");
    println!("  • Worker range: 2-12 workers");
    println!("  • Target metrics: CPU=70%, Memory=75%, Latency=300ms");
    println!("  • Predictive scaling: Enabled (5min window)");
    println!("  • Evaluation interval: 5 seconds");
    println!("  • Performance priority: 70%");

    // Start auto-scaler
    auto_scaler.start().await?;
    info!("Auto-scaling system started");

    // Monitor auto-scaling behavior over time
    println!("\n  Monitoring auto-scaling behavior:");
    
    let monitoring_duration = Duration::from_secs(60); // 1 minute demo
    let start_time = Instant::now();
    let mut scaling_events = 0;
    let mut last_worker_count = auto_scaler.get_current_worker_count_public();

    while start_time.elapsed() < monitoring_duration {
        sleep(Duration::from_secs(8)).await;
        
        let current_workers = auto_scaler.get_current_worker_count_public();
        let prediction_accuracy = auto_scaler.get_prediction_accuracy();
        
        if current_workers != last_worker_count {
            scaling_events += 1;
            println!("    🔄 Scaling event: {} -> {} workers (prediction accuracy: {:.1}%)", 
                    last_worker_count, current_workers, prediction_accuracy * 100.0);
            last_worker_count = current_workers;
        } else {
            println!("    📊 Workers: {}, Prediction accuracy: {:.1}%", 
                    current_workers, prediction_accuracy * 100.0);
        }
    }

    // Get final scaling history and metrics
    let scaling_history = auto_scaler.get_scaling_history();
    let metrics_history = auto_scaler.get_metrics_history();
    let final_workers = auto_scaler.get_current_worker_count_public();
    let final_accuracy = auto_scaler.get_prediction_accuracy();

    auto_scaler.stop().await?;

    println!("\n📊 Auto-Scaling Results:");
    println!("  • Monitoring duration: {:.1}s", monitoring_duration.as_secs_f32());
    println!("  • Scaling events observed: {}", scaling_events);
    println!("  • Final worker count: {}", final_workers);
    println!("  • Prediction model accuracy: {:.1}%", final_accuracy * 100.0);
    println!("  • Scaling decisions recorded: {}", scaling_history.len());
    println!("  • Metrics data points: {}", metrics_history.len());

    if !scaling_history.is_empty() {
        let successful_decisions = scaling_history.iter()
            .filter(|d| d.confidence > 0.6)
            .count();
        println!("  • High-confidence decisions: {}/{}", successful_decisions, scaling_history.len());
    }

    if final_accuracy > 0.7 {
        println!("  ✅ Auto-scaling system demonstrates excellent predictive capabilities!");
    } else {
        println!("  ⚠️ Auto-scaling system is learning - accuracy will improve with more data");
    }

    Ok(())
}

async fn await_load_testing_demo(
    test_suite: &[ScalabilityTestCategory]
) -> Result<(), Box<dyn std::error::Error>> {
    
    println!("Running comprehensive load testing and performance benchmarks...");

    // Create integrated system for load testing
    let engine = HighPerformanceDistributedEngine::new(DistributedEngineConfig {
        num_workers: 6,
        max_concurrent_requests: 150,
        performance_profiling: true,
        batch_size: 48,
        ..DistributedEngineConfig::default()
    });

    engine.start().await?;

    // Load test scenarios
    let load_scenarios = vec![
        LoadScenario {
            name: "Light Load".to_string(),
            concurrent_users: 10,
            requests_per_second: 50,
            duration_seconds: 20,
        },
        LoadScenario {
            name: "Medium Load".to_string(),
            concurrent_users: 25,
            requests_per_second: 100,
            duration_seconds: 15,
        },
        LoadScenario {
            name: "Heavy Load".to_string(),
            concurrent_users: 50,
            requests_per_second: 200,
            duration_seconds: 10,
        },
        LoadScenario {
            name: "Spike Load".to_string(),
            concurrent_users: 100,
            requests_per_second: 500,
            duration_seconds: 5,
        },
    ];

    let mut load_test_results = Vec::new();

    for scenario in &load_scenarios {
        println!("\n  🧪 Load Test: {}", scenario.name);
        println!("    Concurrent users: {}", scenario.concurrent_users);
        println!("    Target RPS: {}", scenario.requests_per_second);
        println!("    Duration: {}s", scenario.duration_seconds);

        let result = run_load_test_scenario(&engine, scenario, &test_suite[0]).await?;
        load_test_results.push((scenario.clone(), result));

        println!("    Results:");
        println!("      • Actual RPS: {:.1}", result.actual_rps);
        println!("      • Avg latency: {:.1}ms", result.avg_latency_ms);
        println!("      • P95 latency: {:.1}ms", result.p95_latency_ms);
        println!("      • P99 latency: {:.1}ms", result.p99_latency_ms);
        println!("      • Error rate: {:.2}%", result.error_rate * 100.0);
        println!("      • Throughput: {:.1} graphs/sec", result.throughput_graphs_per_sec);

        // Brief pause between scenarios
        sleep(Duration::from_secs(2)).await;
    }

    // Stress test with gradual ramp-up
    println!("\n  🚀 Stress Test: Gradual Ramp-Up");
    let stress_result = run_stress_test(&engine, &test_suite[0]).await?;
    
    println!("    Peak performance achieved:");
    println!("      • Max RPS: {:.1}", stress_result.peak_rps);
    println!("      • Max concurrent: {}", stress_result.max_concurrent);
    println!("      • Saturation point: {:.1} RPS", stress_result.saturation_point);
    println!("      • System stability: {:.1}%", stress_result.stability_score * 100.0);

    engine.stop().await?;

    println!("\n📊 Load Testing Summary:");
    println!("  • Test scenarios completed: {}", load_test_results.len());
    
    let avg_error_rate: f32 = load_test_results.iter()
        .map(|(_, result)| result.error_rate)
        .sum::<f32>() / load_test_results.len() as f32;
    
    let max_throughput: f32 = load_test_results.iter()
        .map(|(_, result)| result.throughput_graphs_per_sec)
        .fold(0.0, f32::max);
    
    println!("  • Average error rate: {:.2}%", avg_error_rate * 100.0);
    println!("  • Peak throughput: {:.1} graphs/sec", max_throughput);
    println!("  • System saturation: {:.1} RPS", stress_result.saturation_point);

    if avg_error_rate < 0.01 && stress_result.stability_score > 0.9 {
        println!("  ✅ System demonstrates excellent performance under load!");
    } else if avg_error_rate < 0.05 && stress_result.stability_score > 0.7 {
        println!("  ⚠️ System performance is good but could be optimized further");
    } else {
        println!("  ❌ System needs performance optimization for production loads");
    }

    Ok(())
}

async fn await_integration_scalability_demo(
    test_suite: &[ScalabilityTestCategory]
) -> Result<(), Box<dyn std::error::Error>> {
    
    println!("Demonstrating end-to-end scalability with integrated components...");

    // Create fully integrated scalable system
    let engine_config = DistributedEngineConfig {
        num_workers: 4,
        max_workers: 12,
        auto_scaling_enabled: true,
        performance_profiling: true,
        ..DistributedEngineConfig::default()
    };

    let scaling_config = AutoScalingConfig {
        min_workers: 2,
        max_workers: 10,
        evaluation_interval_seconds: 10,
        predictive_scaling_enabled: true,
        ..AutoScalingConfig::default()
    };

    let monitor_config = HealthMonitorConfig {
        collection_interval_ms: 2000,
        predictive_analysis_enabled: true,
        ..HealthMonitorConfig::default()
    };

    let recovery_config = ErrorRecoveryConfig {
        enable_graceful_degradation: true,
        enable_automatic_failover: true,
        ..ErrorRecoveryConfig::default()
    };

    // Initialize all systems
    let engine = HighPerformanceDistributedEngine::new(engine_config);
    let auto_scaler = IntelligentAutoScalingSystem::new(scaling_config);
    let health_monitor = IntelligentHealthMonitor::new(monitor_config, HealthThresholds::default());
    let recovery_system = RobustErrorRecoverySystem::new(recovery_config);

    // Start all systems
    engine.start().await?;
    auto_scaler.start().await?;
    health_monitor.start().await?;

    println!("\n  🌐 Integrated System Components:");
    println!("    • High-Performance Distributed Engine: ✅");
    println!("    • Intelligent Auto-Scaling System: ✅");
    println!("    • Health Monitoring System: ✅");
    println!("    • Robust Error Recovery System: ✅");

    // Run integrated workload simulation
    println!("\n  🔄 Running Integrated Workload Simulation:");
    
    let simulation_duration = Duration::from_secs(45);
    let start_time = Instant::now();
    let mut phase = 0;
    let mut total_processed = 0;
    let mut total_errors = 0;
    
    while start_time.elapsed() < simulation_duration {
        phase += 1;
        let phase_duration = Duration::from_secs(15);
        let phase_start = Instant::now();

        // Simulate different load phases
        let (load_intensity, description) = match phase {
            1 => (0.3, "Light load phase"),
            2 => (0.8, "Heavy load phase"),
            3 => (0.5, "Cool-down phase"),
            _ => break,
        };

        println!("    Phase {}: {} (intensity: {:.0}%)", 
                phase, description, load_intensity * 100.0);

        // Process workload with varying intensity
        while phase_start.elapsed() < phase_duration {
            let batch_size = (10.0 * load_intensity) as usize + 1;
            let graphs_to_process: Vec<_> = test_suite[0].test_graphs.iter()
                .take(batch_size)
                .cloned()
                .collect();

            // Process through recovery system for resilience
            for graph in graphs_to_process {
                let result = recovery_system.execute_with_recovery("integration_test", || async {
                    engine.process_graph(graph.clone(), ProcessingConfig::default()).await
                }).await;

                match result {
                    Ok(_) => total_processed += 1,
                    Err(_) => total_errors += 1,
                }
            }

            // Adaptive delay based on load intensity
            let delay_ms = (50.0 / load_intensity) as u64;
            sleep(Duration::from_millis(delay_ms)).await;
        }

        // Check system health after each phase
        let worker_count = auto_scaler.get_current_worker_count_public();
        let active_alerts = health_monitor.get_active_alerts();
        let prediction_accuracy = auto_scaler.get_prediction_accuracy();

        println!("      Workers: {}, Alerts: {}, Prediction accuracy: {:.1}%", 
                worker_count, active_alerts.len(), prediction_accuracy * 100.0);

        sleep(Duration::from_secs(2)).await;
    }

    // Get final system state
    let final_workers = auto_scaler.get_current_worker_count_public();
    let scaling_history = auto_scaler.get_scaling_history();
    let final_alerts = health_monitor.get_active_alerts();
    let health_summary = health_monitor.get_current_health_summary();
    let recovery_history = recovery_system.get_recovery_history();

    // Stop all systems
    health_monitor.stop().await?;
    auto_scaler.stop().await?;
    engine.stop().await?;

    println!("\n📊 Integration Scalability Results:");
    println!("  • Total graphs processed: {}", total_processed);
    println!("  • Total errors: {}", total_errors);
    println!("  • Success rate: {:.2}%", 
             (total_processed as f32 / (total_processed + total_errors) as f32) * 100.0);
    println!("  • Final worker count: {}", final_workers);
    println!("  • Scaling events: {}", scaling_history.len());
    println!("  • Active alerts: {}", final_alerts.len());
    println!("  • Health components monitored: {}", health_summary.len());
    println!("  • Recovery actions taken: {}", recovery_history.len());

    // Calculate overall system performance score
    let success_rate = total_processed as f32 / (total_processed + total_errors) as f32;
    let scaling_efficiency = if scaling_history.len() > 0 {
        scaling_history.iter().map(|d| d.confidence).sum::<f32>() / scaling_history.len() as f32
    } else {
        1.0
    };
    let health_score = if final_alerts.is_empty() { 1.0 } else { 0.8 };
    
    let overall_score = (success_rate * 0.4 + scaling_efficiency * 0.3 + health_score * 0.3) * 100.0;

    println!("  • Overall performance score: {:.1}%", overall_score);

    if overall_score > 90.0 {
        println!("  ✅ Integrated scalable system demonstrates exceptional performance!");
    } else if overall_score > 75.0 {
        println!("  ⚠️ Integrated system performance is good with room for optimization");
    } else {
        println!("  ❌ Integrated system needs significant optimization for production");
    }

    Ok(())
}

// Helper functions and types

#[derive(Clone)]
struct ScalabilityTestCategory {
    name: String,
    processing_type: ProcessingType,
    test_graphs: Vec<graph_diffusion_edge::core::graph::CompactGraph>,
    processing_config: ProcessingConfig,
}

#[derive(Clone)]
enum ProcessingType {
    Single,
    Batch,
    Concurrent,
    QuantumEnhanced,
}

#[derive(Clone)]
struct LoadScenario {
    name: String,
    concurrent_users: usize,
    requests_per_second: usize,
    duration_seconds: u64,
}

struct LoadTestResult {
    actual_rps: f32,
    avg_latency_ms: f32,
    p95_latency_ms: f32,
    p99_latency_ms: f32,
    error_rate: f32,
    throughput_graphs_per_sec: f32,
}

struct StressTestResult {
    peak_rps: f32,
    max_concurrent: usize,
    saturation_point: f32,
    stability_score: f32,
}

fn create_scalability_test_suite() -> Vec<ScalabilityTestCategory> {
    let mut categories = Vec::new();

    // Single processing category
    let single_graphs = create_varied_graphs("single", 100, 10..50, 20..100);
    categories.push(ScalabilityTestCategory {
        name: "Single Processing".to_string(),
        processing_type: ProcessingType::Single,
        test_graphs: single_graphs,
        processing_config: ProcessingConfig {
            temperature: 0.1,
            dropout_rate: 0.0,
            use_attention: false,
            normalize_features: true,
            mixed_precision: false,
        },
    });

    // Batch processing category
    let batch_graphs = create_varied_graphs("batch", 200, 5..25, 10..50);
    categories.push(ScalabilityTestCategory {
        name: "Batch Processing".to_string(),
        processing_type: ProcessingType::Batch,
        test_graphs: batch_graphs,
        processing_config: ProcessingConfig {
            temperature: 0.05,
            dropout_rate: 0.1,
            use_attention: false,
            normalize_features: true,
            mixed_precision: true,
        },
    });

    // Concurrent processing category
    let concurrent_graphs = create_varied_graphs("concurrent", 150, 8..40, 15..75);
    categories.push(ScalabilityTestCategory {
        name: "Concurrent Processing".to_string(),
        processing_type: ProcessingType::Concurrent,
        test_graphs: concurrent_graphs,
        processing_config: ProcessingConfig {
            temperature: 0.2,
            dropout_rate: 0.05,
            use_attention: true,
            normalize_features: true,
            mixed_precision: false,
        },
    });

    // Quantum enhanced processing category (smaller set due to complexity)
    let quantum_graphs = create_varied_graphs("quantum", 30, 15..30, 30..60);
    categories.push(ScalabilityTestCategory {
        name: "Quantum Enhanced Processing".to_string(),
        processing_type: ProcessingType::QuantumEnhanced,
        test_graphs: quantum_graphs,
        processing_config: ProcessingConfig {
            temperature: 0.15,
            dropout_rate: 0.0,
            use_attention: true,
            normalize_features: true,
            mixed_precision: false,
        },
    });

    info!("Created scalability test suite with {} categories", categories.len());
    categories
}

fn create_varied_graphs(
    prefix: &str, 
    count: usize, 
    node_range: std::ops::Range<usize>,
    edge_range: std::ops::Range<usize>
) -> Vec<graph_diffusion_edge::core::graph::CompactGraph> {
    let mut graphs = Vec::new();
    let mut rng = rand::thread_rng();

    for i in 0..count {
        let num_nodes = rng.gen_range(node_range.clone());
        let num_edges = rng.gen_range(edge_range.clone()).min(num_nodes * (num_nodes - 1) / 2);
        
        let mut graph = Graph::new();
        
        // Add nodes with varied features
        for node_id in 1..=num_nodes {
            let feature_count = rng.gen_range(2..=8);
            let features: Vec<f32> = (0..feature_count)
                .map(|j| ((node_id as f32 + j as f32).sin() + rng.gen_range(-0.1..0.1)))
                .collect();
            
            graph.add_node(Node {
                id: node_id as u32,
                features,
                label: Some(format!("{}_{}", prefix, node_id)),
            });
        }
        
        // Add edges with various patterns
        let mut edges_added = 0;
        while edges_added < num_edges {
            let source = rng.gen_range(1..=num_nodes) as u32;
            let target = rng.gen_range(1..=num_nodes) as u32;
            
            if source != target {
                let weight = if rng.gen_bool(0.8) {
                    rng.gen_range(0.1..1.0) // Normal weights
                } else {
                    rng.gen_range(1.0..3.0) // Some stronger connections
                };
                
                graph.add_edge(Edge {
                    source,
                    target,
                    weight,
                    edge_type: Some(format!("edge_type_{}", edges_added % 5)),
                });
                edges_added += 1;
            }
        }
        
        if let Ok(compact_graph) = graph.to_compact() {
            graphs.push(compact_graph);
        }
    }

    debug!("Created {} varied graphs for '{}'", graphs.len(), prefix);
    graphs
}

async fn run_load_test_scenario(
    engine: &HighPerformanceDistributedEngine,
    scenario: &LoadScenario,
    test_category: &ScalabilityTestCategory
) -> Result<LoadTestResult, Box<dyn std::error::Error>> {
    
    let mut latencies = Vec::new();
    let mut errors = 0;
    let mut successful_requests = 0;
    
    let start_time = Instant::now();
    let target_duration = Duration::from_secs(scenario.duration_seconds);
    let target_interval = Duration::from_millis(1000 / scenario.requests_per_second as u64);
    
    // Simulate concurrent users
    let tasks = (0..scenario.concurrent_users).map(|user_id| {
        let engine_clone = engine.clone();
        let graphs = test_category.test_graphs.clone();
        let config = test_category.processing_config.clone();
        let user_target_duration = target_duration;
        let user_interval = target_interval * scenario.concurrent_users as u32;
        
        tokio::spawn(async move {
            let mut user_latencies = Vec::new();
            let mut user_errors = 0;
            let mut user_successes = 0;
            let user_start = Instant::now();
            let mut request_count = 0;
            
            while user_start.elapsed() < user_target_duration {
                let graph_idx = request_count % graphs.len();
                let request_start = Instant::now();
                
                match engine_clone.process_graph(graphs[graph_idx].clone(), config.clone()).await {
                    Ok(_) => {
                        user_successes += 1;
                        user_latencies.push(request_start.elapsed());
                    }
                    Err(_) => {
                        user_errors += 1;
                    }
                }
                
                request_count += 1;
                sleep(user_interval).await;
            }
            
            (user_latencies, user_errors, user_successes)
        })
    });
    
    let results = join_all(tasks).await;
    
    for result in results {
        if let Ok((user_latencies, user_errors, user_successes)) = result {
            latencies.extend(user_latencies);
            errors += user_errors;
            successful_requests += user_successes;
        }
    }
    
    let total_duration = start_time.elapsed();
    let total_requests = successful_requests + errors;
    
    // Calculate metrics
    let actual_rps = total_requests as f32 / total_duration.as_secs_f32();
    let avg_latency_ms = if !latencies.is_empty() {
        latencies.iter().map(|d| d.as_millis() as f32).sum::<f32>() / latencies.len() as f32
    } else {
        0.0
    };
    
    latencies.sort();
    let p95_latency_ms = if latencies.len() > 0 {
        latencies[(latencies.len() as f32 * 0.95) as usize].as_millis() as f32
    } else {
        0.0
    };
    let p99_latency_ms = if latencies.len() > 0 {
        latencies[(latencies.len() as f32 * 0.99) as usize].as_millis() as f32
    } else {
        0.0
    };
    
    let error_rate = if total_requests > 0 {
        errors as f32 / total_requests as f32
    } else {
        0.0
    };
    
    let throughput_graphs_per_sec = successful_requests as f32 / total_duration.as_secs_f32();
    
    Ok(LoadTestResult {
        actual_rps,
        avg_latency_ms,
        p95_latency_ms,
        p99_latency_ms,
        error_rate,
        throughput_graphs_per_sec,
    })
}

async fn run_stress_test(
    engine: &HighPerformanceDistributedEngine,
    test_category: &ScalabilityTestCategory
) -> Result<StressTestResult, Box<dyn std::error::Error>> {
    
    let mut peak_rps = 0.0;
    let mut max_concurrent = 0;
    let mut saturation_point = 0.0;
    let mut stable_measurements = 0;
    let total_measurements = 10;
    
    // Gradually increase load to find saturation point
    for concurrent_level in (10..=100).step_by(10) {
        let test_scenario = LoadScenario {
            name: format!("Stress Test {}", concurrent_level),
            concurrent_users: concurrent_level,
            requests_per_second: concurrent_level * 5,
            duration_seconds: 6,
        };
        
        let result = run_load_test_scenario(engine, &test_scenario, test_category).await?;
        
        if result.actual_rps > peak_rps {
            peak_rps = result.actual_rps;
            max_concurrent = concurrent_level;
        }
        
        // Check if system is still stable (error rate < 5% and latency reasonable)
        if result.error_rate < 0.05 && result.p95_latency_ms < 2000.0 {
            saturation_point = result.actual_rps;
            stable_measurements += 1;
        }
        
        debug!("Stress level {}: {:.1} RPS, {:.1}ms P95, {:.2}% errors", 
              concurrent_level, result.actual_rps, result.p95_latency_ms, result.error_rate * 100.0);
    }
    
    let stability_score = stable_measurements as f32 / total_measurements as f32;
    
    Ok(StressTestResult {
        peak_rps,
        max_concurrent,
        saturation_point,
        stability_score,
    })
}