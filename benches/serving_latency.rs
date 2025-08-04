//! Benchmark for HTTP serving latency

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use graph_diffusion_edge::{
    core::{Graph, Node, Edge, GraphConfig, ProcessingConfig, DGDMProcessor},
    serving::handlers::{GraphRequest, AppState},
};
use std::sync::Arc;
use prometheus::{Counter, Histogram};

fn create_test_request(num_nodes: usize) -> GraphRequest {
    let mut graph = Graph::new();
    
    for i in 0..num_nodes {
        graph.add_node(Node {
            id: i as u32,
            features: vec![i as f32, (i * 2) as f32],
            label: None,
        });
    }
    
    for i in 0..num_nodes - 1 {
        graph.add_edge(Edge {
            source: i as u32,
            target: (i + 1) as u32,
            weight: 1.0,
            edge_type: None,
        });
    }
    
    GraphRequest {
        graph,
        config: None,
        return_metrics: false,
    }
}

fn create_app_state() -> Arc<AppState> {
    let config = GraphConfig::default();
    let processing_config = ProcessingConfig::default();
    let processor = DGDMProcessor::new(config, processing_config);
    
    Arc::new(AppState {
        processor: Arc::new(processor),
        #[cfg(feature = "tpu")]
        tpu: None,
        request_counter: Counter::new("test_requests", "Test requests").unwrap(),
        processing_histogram: Histogram::with_opts(
            prometheus::HistogramOpts::new("test_duration", "Test duration")
        ).unwrap(),
    })
}

fn bench_request_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("request_processing");
    let app_state = create_app_state();
    
    let sizes = vec![10, 100, 1000];
    
    for size in sizes {
        let request = create_test_request(size);
        
        group.bench_with_input(
            BenchmarkId::new("process_request", size),
            &request,
            |b, request| {
                b.iter(|| {
                    let compact_graph = black_box(request.graph.to_compact().unwrap());
                    black_box(app_state.processor.process(&compact_graph).unwrap())
                });
            },
        );
    }
    
    group.finish();
}

fn bench_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");
    
    let sizes = vec![100, 1000, 5000];
    
    for size in sizes {
        let request = create_test_request(size);
        
        group.bench_with_input(
            BenchmarkId::new("serialize_request", size),
            &request,
            |b, request| {
                b.iter(|| black_box(serde_json::to_string(request).unwrap()));
            },
        );
        
        let json = serde_json::to_string(&request).unwrap();
        group.bench_with_input(
            BenchmarkId::new("deserialize_request", size),
            &json,
            |b, json| {
                b.iter(|| black_box(serde_json::from_str::<GraphRequest>(json).unwrap()));
            },
        );
    }
    
    group.finish();
}

fn bench_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation");
    
    let sizes = vec![100, 1000, 10000];
    
    for size in sizes {
        let request = create_test_request(size);
        
        group.bench_with_input(
            BenchmarkId::new("validate_graph", size),
            &request.graph,
            |b, graph| {
                b.iter(|| black_box(graph.validate().unwrap()));
            },
        );
    }
    
    group.finish();
}

fn bench_memory_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_estimation");
    let app_state = create_app_state();
    
    let sizes = vec![1000, 10000, 50000];
    
    for size in sizes {
        let request = create_test_request(size);
        let compact_graph = request.graph.to_compact().unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("estimate_memory", size),
            &compact_graph,
            |b, compact_graph| {
                b.iter(|| black_box(app_state.processor.estimate_memory_usage(compact_graph)));
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_request_processing,
    bench_serialization,
    bench_validation,
    bench_memory_estimation
);
criterion_main!(benches);