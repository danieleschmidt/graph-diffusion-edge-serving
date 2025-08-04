//! Benchmark for graph processing performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use graph_diffusion_edge::{
    core::{Graph, Node, Edge, GraphConfig, ProcessingConfig, DGDMProcessor},
};

fn create_test_graph(num_nodes: usize, num_edges: usize) -> Graph {
    let mut graph = Graph::new();
    
    // Add nodes
    for i in 0..num_nodes {
        graph.add_node(Node {
            id: i as u32,
            features: vec![i as f32 / num_nodes as f32, (i * 2) as f32 / num_nodes as f32],
            label: None,
        });
    }
    
    // Add edges (random connections)
    for i in 0..num_edges {
        let source = (i % num_nodes) as u32;
        let target = ((i + 1) % num_nodes) as u32;
        graph.add_edge(Edge {
            source,
            target,
            weight: (i as f32 / num_edges as f32),
            edge_type: None,
        });
    }
    
    graph
}

fn bench_graph_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_conversion");
    
    let sizes = vec![100, 1000, 10000];
    
    for size in sizes {
        let graph = create_test_graph(size, size * 2);
        
        group.bench_with_input(
            BenchmarkId::new("to_compact", size),
            &graph,
            |b, graph| {
                b.iter(|| black_box(graph.to_compact().unwrap()));
            },
        );
    }
    
    group.finish();
}

fn bench_dgdm_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("dgdm_processing");
    
    let config = GraphConfig::default();
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(config, processing_config);
    
    let sizes = vec![100, 1000, 5000];
    
    for size in sizes {
        let graph = create_test_graph(size, size * 2);
        let compact_graph = graph.to_compact().unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("process", size),
            &compact_graph,
            |b, compact_graph| {
                b.iter(|| black_box(processor.process(compact_graph).unwrap()));
            },
        );
    }
    
    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    
    let config = GraphConfig::default();
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(config, processing_config);
    
    let batch_sizes = vec![1, 5, 10, 20];
    
    for batch_size in batch_sizes {
        let graphs: Vec<_> = (0..batch_size)
            .map(|_| create_test_graph(1000, 2000).to_compact().unwrap())
            .collect();
        let graph_refs: Vec<_> = graphs.iter().collect();
        
        group.bench_with_input(
            BenchmarkId::new("batch_process", batch_size),
            &graph_refs,
            |b, graphs| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async { black_box(processor.process_batch(graphs.clone()).await.unwrap()) });
            },
        );
    }
    
    group.finish();
}

fn bench_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    let config = GraphConfig::default();
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(config, processing_config);
    
    let sizes = vec![1000, 10000, 50000];
    
    for size in sizes {
        let graph = create_test_graph(size, size * 3);
        let compact_graph = graph.to_compact().unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("estimate_memory", size),
            &compact_graph,
            |b, compact_graph| {
                b.iter(|| black_box(processor.estimate_memory_usage(compact_graph)));
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_graph_conversion,
    bench_dgdm_processing,
    bench_batch_processing,
    bench_memory_usage
);
criterion_main!(benches);