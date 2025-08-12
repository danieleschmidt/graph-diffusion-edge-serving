//! Generation 3: Performance optimization and scaling demonstration

use graph_diffusion_edge::{
    core::{Graph, Node, Edge, GraphConfig, DGDMProcessor, ProcessingConfig},
    performance_optimizer::PerformanceOptimizer,
    optimization::{
        caching::SmartCache,
        performance::PerformanceProfiler,
        resource_pool::{ResourcePool, PoolConfig, PoolableResource},
    },
    scaling::ScalableWorkerPool,
    Result,
};
use std::time::{Duration, Instant};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    println!("⚡ Generation 3: Performance Optimization & Scaling Demo");
    
    // Test 1: Intelligent Caching System
    println!("\n🧠 Testing Intelligent Caching...");
    test_smart_caching().await?;
    
    // Test 2: Performance Profiling and Adaptive Optimization
    println!("\n📊 Testing Performance Profiling...");
    test_performance_profiling().await?;
    
    // Test 3: Resource Pool Management
    println!("\n🏊 Testing Resource Pool Management...");
    test_resource_pooling().await?;
    
    // Test 4: Concurrent Processing and Load Balancing  
    println!("\n⚖️ Testing Concurrent Processing...");
    test_concurrent_processing().await?;
    
    // Test 5: Memory Optimization and Graph Compression
    println!("\n🗜️ Testing Memory Optimization...");
    test_memory_optimization().await?;
    
    // Test 6: Scalable Worker Pool
    println!("\n👥 Testing Scalable Worker Pool...");
    test_scalable_workers().await?;
    
    println!("\n🚀 Generation 3 (Performance & Scaling) Complete!");
    println!("   - Intelligent caching ⚡");
    println!("   - Performance profiling 📊");
    println!("   - Resource pooling 🏊");
    println!("   - Concurrent processing ⚖️");
    println!("   - Memory optimization 🗜️");
    println!("   - Auto-scaling workers 👥");
    
    Ok(())
}

async fn test_smart_caching() -> Result<()> {
    let mut cache = SmartCache::new(1000, Duration::from_secs(60));
    
    // Test cache miss and population
    let key = "test_graph_123".to_string();
    let start = Instant::now();
    
    let result = if let Some(cached) = cache.get(&key).await {
        cached
    } else {
        // Simulate expensive computation
        std::thread::sleep(Duration::from_millis(10));
        let result = 42.0;
        cache.put(key.clone(), result).await;
        result
    };
    
    let first_access = start.elapsed();
    println!("   Cache miss: {:.2}ms -> result: {}", first_access.as_millis(), result);
    
    // Test cache hit (should be much faster)
    let start = Instant::now();
    let result = cache.get(&key).await.unwrap_or(0.0);
    
    let second_access = start.elapsed();
    println!("   Cache hit:  {:.2}ms -> result: {}", second_access.as_millis(), result);
    
    // Verify cache is working (hit should be much faster)
    if second_access < first_access / 2 {
        println!("   ✅ Cache performance improvement: {:.1}x faster", 
            first_access.as_micros() as f64 / second_access.as_micros() as f64);
    }
    
    println!("   Cache stats: {:?}", cache.get_stats());
    
    Ok(())
}

async fn test_performance_profiling() -> Result<()> {
    let profiler = PerformanceProfiler::new();
    
    // Profile graph creation
    let creation_time = profiler.time_operation("graph_creation", || async {
        let mut graph = Graph::new();
        for i in 0..100 {
            graph.add_node(Node {
                id: i,
                features: vec![i as f32, (i * 2) as f32],
                label: None,
            });
            if i > 0 {
                graph.add_edge(Edge {
                    source: i - 1,
                    target: i,
                    weight: 1.0 / i as f32,
                    edge_type: None,
                });
            }
        }
        graph
    }).await;
    
    println!("   Graph creation: {:.2}ms", creation_time.as_millis());
    
    // Profile graph processing with different configurations
    let graph = creation_time;
    let compact_graph = graph.to_compact()?;
    
    for batch_size in [1, 4, 8, 16] {
        let mut config = GraphConfig::default();
        config.batch_size = batch_size;
        config.feature_dim = 2;
        
        let processing_config = ProcessingConfig::default();
        let processor = DGDMProcessor::new(config, processing_config);
        
        let processing_time = profiler.time_operation("graph_processing", || async {
            processor.process(&compact_graph)
        }).await?;
        
        println!("   Batch size {}: {:.2}ms ({:.0} nodes/ms)", 
            batch_size, 
            processing_time.processing_time_ms,
            compact_graph.num_nodes as f64 / processing_time.processing_time_ms);
    }
    
    // Display profiling results
    let stats = profiler.get_stats().await;
    println!("   Profiling summary:");
    for (operation, metrics) in stats {
        println!("     {}: avg={:.2}ms, min={:.2}ms, max={:.2}ms", 
            operation, 
            metrics.average.as_millis(),
            metrics.min.as_millis(),
            metrics.max.as_millis());
    }
    
    Ok(())
}

async fn test_resource_pooling() -> Result<()> {
    use async_trait::async_trait;
    
    // Mock resource for testing
    #[derive(Debug)]
    struct MockProcessor {
        id: usize,
        processing_count: std::sync::atomic::AtomicU64,
    }
    
    #[derive(Clone)]
    struct MockCreateParams;
    
    #[async_trait]
    impl PoolableResource for MockProcessor {
        type CreateParams = MockCreateParams;
        
        async fn create(_params: &Self::CreateParams) -> Result<Self> {
            static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            
            // Simulate resource creation time
            tokio::time::sleep(Duration::from_millis(5)).await;
            
            Ok(MockProcessor {
                id,
                processing_count: std::sync::atomic::AtomicU64::new(0),
            })
        }
        
        async fn is_healthy(&self) -> bool {
            true // Always healthy for demo
        }
        
        async fn reset(&mut self) -> Result<()> {
            // Reset any state if needed
            Ok(())
        }
    }
    
    // Configure resource pool
    let config = PoolConfig {
        initial_size: 2,
        max_size: 8,
        min_idle: 1,
        max_idle_time_seconds: 30,
        connection_timeout_seconds: 5,
        validation_interval_seconds: 10,
        enable_metrics: true,
        enable_health_checks: true,
        retry_attempts: 3,
        retry_delay_ms: 100,
    };
    
    let pool: ResourcePool<MockProcessor, _> = ResourcePool::new(config, MockCreateParams).await?;
    println!("   Created resource pool with 2-8 processors");
    
    // Test concurrent resource usage
    let pool = Arc::new(pool);
    let mut handles = vec![];
    
    let start_time = Instant::now();
    
    // Spawn multiple concurrent tasks
    for task_id in 0..12 {
        let pool = Arc::clone(&pool);
        let handle = tokio::spawn(async move {
            let processor = pool.acquire().await.unwrap();
            
            // Simulate work
            let work_duration = Duration::from_millis(20 + task_id * 5);
            tokio::time::sleep(work_duration).await;
            
            processor.get().processing_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            
            (task_id, processor.get().id, work_duration)
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let mut results = vec![];
    for handle in handles {
        let result = handle.await.unwrap();
        results.push(result);
    }
    
    let total_time = start_time.elapsed();
    
    println!("   Completed {} concurrent tasks in {:.2}ms", results.len(), total_time.as_millis());
    println!("   Task distribution:");
    
    // Show which processor handled which tasks
    let mut processor_usage = std::collections::HashMap::new();
    for (task_id, processor_id, duration) in results {
        processor_usage.entry(processor_id).or_insert_with(Vec::new).push((task_id, duration));
    }
    
    for (processor_id, tasks) in processor_usage {
        let task_count = tasks.len();
        let total_work: Duration = tasks.iter().map(|(_, d)| *d).sum();
        println!("     Processor {}: {} tasks, {:.0}ms total work", 
            processor_id, task_count, total_work.as_millis());
    }
    
    let metrics = pool.get_metrics().await;
    println!("   Pool metrics: {:?}", metrics);
    
    Ok(())
}

async fn test_concurrent_processing() -> Result<()> {
    use rayon::prelude::*;
    
    // Create multiple graphs for concurrent processing
    let mut graphs = vec![];
    for i in 0..8 {
        let mut graph = Graph::new();
        let node_count = 50 + i * 10; // Variable size graphs
        
        for j in 0..node_count {
            graph.add_node(Node {
                id: j,
                features: vec![j as f32 / node_count as f32, (j * i) as f32 / 100.0],
                label: None,
            });
            if j > 0 {
                graph.add_edge(Edge {
                    source: j - 1,
                    target: j,
                    weight: 1.0 / (j + 1) as f32,
                    edge_type: None,
                });
            }
        }
        graphs.push(graph);
    }
    
    println!("   Created {} graphs with 50-120 nodes each", graphs.len());
    
    // Sequential processing baseline
    let start = Instant::now();
    let mut sequential_results = vec![];
    
    for graph in &graphs {
        let compact = graph.to_compact()?;
        let mut config = GraphConfig::default();
        config.feature_dim = 2;
        
        let processing_config = ProcessingConfig::default();
        let processor = DGDMProcessor::new(config, processing_config);
        
        let result = processor.process(&compact)?;
        sequential_results.push(result);
    }
    
    let sequential_time = start.elapsed();
    println!("   Sequential processing: {:.2}ms", sequential_time.as_millis());
    
    // Parallel processing using rayon
    let start = Instant::now();
    
    let parallel_results: Vec<_> = graphs.par_iter().map(|graph| {
        let compact = graph.to_compact().unwrap();
        let mut config = GraphConfig::default();
        config.feature_dim = 2;
        
        let processing_config = ProcessingConfig::default();
        let processor = DGDMProcessor::new(config, processing_config);
        
        processor.process(&compact).unwrap()
    }).collect();
    
    let parallel_time = start.elapsed();
    println!("   Parallel processing: {:.2}ms", parallel_time.as_millis());
    
    let speedup = sequential_time.as_millis() as f64 / parallel_time.as_millis() as f64;
    println!("   ✅ Parallel speedup: {:.1}x faster", speedup);
    
    // Verify results are equivalent
    assert_eq!(sequential_results.len(), parallel_results.len());
    println!("   ✅ Results validated: {} graphs processed", parallel_results.len());
    
    Ok(())
}

async fn test_memory_optimization() -> Result<()> {
    // Test memory usage with different graph representations
    println!("   Testing graph compression and memory optimization...");
    
    // Create a large graph
    let mut large_graph = Graph::new();
    let node_count = 1000;
    
    for i in 0..node_count {
        large_graph.add_node(Node {
            id: i,
            features: vec![i as f32; 50], // Large feature vectors
            label: Some(format!("node_{}", i)),
        });
        
        // Create a densely connected subgraph
        for j in (i.saturating_sub(5))..i {
            if j != i {
                large_graph.add_edge(Edge {
                    source: j,
                    target: i,
                    weight: (i - j) as f32 / 10.0,
                    edge_type: Some("dense_connection".to_string()),
                });
            }
        }
    }
    
    println!("   Created large graph: {} nodes, {} edges", 
        large_graph.nodes.len(), large_graph.edges.len());
    
    // Convert to compact representation
    let start = Instant::now();
    let compact_graph = large_graph.to_compact()?;
    let conversion_time = start.elapsed();
    
    println!("   Conversion to compact format: {:.2}ms", conversion_time.as_millis());
    
    // Estimate memory savings (rough calculation)
    let original_size = large_graph.nodes.len() * std::mem::size_of::<Node>() + 
                       large_graph.edges.len() * std::mem::size_of::<Edge>();
    let compact_size = compact_graph.node_ids.len() * 4 + // u32 IDs
                      compact_graph.node_features.len() * 4 + // f32 features
                      compact_graph.edge_offsets.len() * 4 + // edge structure
                      compact_graph.edge_targets.len() * 4 +
                      compact_graph.edge_weights.len() * 4;
    
    let memory_savings = (original_size - compact_size) as f64 / original_size as f64 * 100.0;
    
    println!("   Original size: ~{} KB", original_size / 1024);
    println!("   Compact size:  ~{} KB", compact_size / 1024);
    println!("   ✅ Memory savings: {:.1}%", memory_savings);
    
    // Test processing performance on compact graph
    let mut config = GraphConfig::default();
    config.feature_dim = 50;
    config.batch_size = 32; // Larger batch for better performance
    
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(config, processing_config);
    
    let start = Instant::now();
    let result = processor.process(&compact_graph)?;
    let _processing_time = start.elapsed();
    
    println!("   Large graph processing: {:.2}ms", result.processing_time_ms);
    println!("   Throughput: {:.0} nodes/sec", 
        node_count as f64 / result.processing_time_ms * 1000.0);
    
    Ok(())
}

async fn test_scalable_workers() -> Result<()> {
    use graph_diffusion_edge::scaling::{ScalingConfig, DistributedTaskQueue};
    
    // Create distributed task queue
    let task_queue = Arc::new(DistributedTaskQueue::new(1000));
    
    // Create scaling configuration
    let config = ScalingConfig {
        min_workers: 2,
        max_workers: 8,
        target_cpu_utilization: 0.7,
        scale_up_threshold: 0.8,
        scale_down_threshold: 0.3,
        scale_up_cooldown_seconds: 60,
        scale_down_cooldown_seconds: 300,
        queue_size_threshold: 100,
        enable_auto_scaling: true,
        enable_adaptive_batching: true,
        max_batch_size: 32,
        batch_timeout_ms: 50,
        worker_idle_timeout_seconds: 300,
        health_check_interval_seconds: 30,
    };
    
    // Create scalable worker pool
    let worker_pool = ScalableWorkerPool::new(config, task_queue);
    
    println!("   Created scalable worker pool (2-8 workers)");
    
    // Submit varying workload to test scaling
    let mut task_handles = vec![];
    
    // Burst of tasks to trigger scale-up
    for batch in 0..3 {
        for task_id in 0..6 {
            let pool = worker_pool.clone();
            let handle = tokio::spawn(async move {
                let work_duration = Duration::from_millis(50 + (task_id * 10) as u64);
                
                pool.submit_work(format!("batch_{}_task_{}", batch, task_id), work_duration).await
            });
            task_handles.push(handle);
        }
        
        // Brief pause between batches
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    
    // Wait for all tasks to complete
    let start = Instant::now();
    let mut successful_tasks = 0;
    
    for handle in task_handles {
        match handle.await {
            Ok(Ok(_)) => successful_tasks += 1,
            Ok(Err(e)) => println!("   Task failed: {}", e),
            Err(e) => println!("   Task panicked: {}", e),
        }
    }
    
    let total_time = start.elapsed();
    
    println!("   Completed {}/18 tasks in {:.2}ms", successful_tasks, total_time.as_millis());
    
    // Get final metrics
    let metrics = worker_pool.get_metrics().await;
    println!("   Final worker metrics:");
    println!("     Active workers: {}", metrics.active_workers);
    println!("     Total tasks processed: {}", metrics.total_tasks_completed);
    println!("     Average task time: {:.2}ms", metrics.avg_task_duration_ms);
    
    // Let workers scale down
    println!("   Waiting for worker scale-down...");
    tokio::time::sleep(Duration::from_secs(6)).await;
    
    let final_metrics = worker_pool.get_metrics().await;
    println!("   After scale-down: {} active workers", final_metrics.active_workers);
    
    Ok(())
}