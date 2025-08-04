//! HTTP request handlers for graph diffusion API

use crate::core::{DGDMProcessor, Graph, ProcessingConfig};
use axum::{
    extract::{State, Json},
    http::StatusCode,
    response::{IntoResponse, Json as ResponseJson},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use prometheus::{Counter, Histogram, TextEncoder, Encoder};
use tracing::{info, warn, error, instrument};
use tokio::time::Instant;

#[cfg(feature = "tpu")]
use crate::tpu::EdgeTPU;

#[derive(Clone)]
pub struct AppState {
    pub processor: Arc<DGDMProcessor>,
    #[cfg(feature = "tpu")]
    pub tpu: Option<Arc<EdgeTPU>>,
    pub request_counter: Counter,
    pub processing_histogram: Histogram,
}

#[derive(Debug, Deserialize)]
pub struct GraphRequest {
    pub graph: Graph,
    #[serde(default)]
    pub config: Option<ProcessingConfig>,
    #[serde(default)]
    pub return_metrics: bool,
}

#[derive(Debug, Serialize)]
pub struct DiffusionResponse {
    pub embeddings: Vec<Vec<f32>>,
    pub processing_time_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<ProcessingMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_stats: Option<GraphStats>,
}

#[derive(Debug, Serialize)]
pub struct ProcessingMetrics {
    pub num_iterations: usize,
    pub convergence_score: f32,
    pub memory_usage_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct GraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub avg_degree: f32,
    pub feature_dim: usize,
}

#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    #[cfg(feature = "tpu")]
    pub tpu_available: bool,
    pub cache_size: usize,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub error_type: String,
    pub timestamp: String,
}

pub fn create_routes(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/diffuse", post(diffuse_handler))
        .route("/batch", post(batch_diffuse_handler))
        .route("/metrics", get(metrics_handler))
        // .route("/stream", get(websocket_handler))  // Disabled for now
        .route("/info", get(server_info_handler))
        .with_state(state)
}

#[instrument(skip(state))]
async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let uptime = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let memory_usage = get_memory_usage_mb();

    let health = HealthResponse {
        status: "healthy".to_string(),
        uptime_seconds: uptime,
        memory_usage_mb: memory_usage,
        #[cfg(feature = "tpu")]
        tpu_available: state.tpu.is_some(),
        cache_size: 0, // TODO: Implement cache size tracking
    };

    ResponseJson(health)
}

#[instrument(skip(state, request))]
async fn diffuse_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<GraphRequest>,
) -> Result<impl IntoResponse, AppError> {
    let start_time = Instant::now();
    state.request_counter.inc();

    info!(
        "Processing graph with {} nodes, {} edges",
        request.graph.num_nodes(),
        request.graph.num_edges()
    );

    if let Err(e) = request.graph.validate() {
        warn!("Invalid graph received: {}", e);
        return Err(AppError::Validation(e.to_string()));
    }

    let compact_graph = request.graph.to_compact()
        .map_err(|e| AppError::Processing(e.to_string()))?;

    if !state.processor.can_process(&compact_graph) {
        return Err(AppError::Processing(
            "Graph exceeds processing limits".to_string()
        ));
    }

    let result = state.processor.process(&compact_graph)
        .map_err(|e| AppError::Processing(e.to_string()))?;

    let embeddings: Vec<Vec<f32>> = result.embeddings
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect();

    let processing_time = start_time.elapsed();
    state.processing_histogram.observe(processing_time.as_secs_f64());

    let metrics = if request.return_metrics {
        Some(ProcessingMetrics {
            num_iterations: result.num_iterations,
            convergence_score: result.convergence_score,
            memory_usage_bytes: state.processor.estimate_memory_usage(&compact_graph),
        })
    } else {
        None
    };

    let graph_stats = if request.return_metrics {
        Some(GraphStats {
            num_nodes: compact_graph.num_nodes(),
            num_edges: compact_graph.num_edges(),
            avg_degree: compact_graph.num_edges() as f32 / compact_graph.num_nodes() as f32,
            feature_dim: compact_graph.feature_dim(),
        })
    } else {
        None
    };

    let response = DiffusionResponse {
        embeddings,
        processing_time_ms: result.processing_time_ms,
        metrics,
        graph_stats,
    };

    info!(
        "Graph processed successfully in {:.2}ms",
        result.processing_time_ms
    );

    Ok(ResponseJson(response))
}

#[instrument(skip(state, request))]
async fn batch_diffuse_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<Vec<GraphRequest>>,
) -> Result<impl IntoResponse, AppError> {
    if request.is_empty() {
        return Err(AppError::Validation("Batch cannot be empty".to_string()));
    }

    if request.len() > 100 {
        return Err(AppError::Validation("Batch size exceeds limit of 100".to_string()));
    }

    let start_time = Instant::now();
    state.request_counter.inc_by(request.len() as f64);

    info!("Processing batch of {} graphs", request.len());

    let mut compact_graphs = Vec::new();
    for graph_request in &request {
        if let Err(e) = graph_request.graph.validate() {
            return Err(AppError::Validation(format!("Invalid graph in batch: {}", e)));
        }
        
        let compact = graph_request.graph.to_compact()
            .map_err(|e| AppError::Processing(e.to_string()))?;
        
        if !state.processor.can_process(&compact) {
            return Err(AppError::Processing(
                "One or more graphs in batch exceed processing limits".to_string()
            ));
        }
        
        compact_graphs.push(compact);
    }

    let graph_refs: Vec<&_> = compact_graphs.iter().collect();
    let results = state.processor.process_batch(graph_refs).await
        .map_err(|e| AppError::Processing(e.to_string()))?;

    let responses: Vec<DiffusionResponse> = results
        .into_iter()
        .zip(request.iter())
        .zip(compact_graphs.iter())
        .map(|((result, req), compact_graph)| {
            let embeddings: Vec<Vec<f32>> = result.embeddings
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();

            let metrics = if req.return_metrics {
                Some(ProcessingMetrics {
                    num_iterations: result.num_iterations,
                    convergence_score: result.convergence_score,
                    memory_usage_bytes: state.processor.estimate_memory_usage(compact_graph),
                })
            } else {
                None
            };

            let graph_stats = if req.return_metrics {
                Some(GraphStats {
                    num_nodes: compact_graph.num_nodes(),
                    num_edges: compact_graph.num_edges(),
                    avg_degree: compact_graph.num_edges() as f32 / compact_graph.num_nodes() as f32,
                    feature_dim: compact_graph.feature_dim(),
                })
            } else {
                None
            };

            DiffusionResponse {
                embeddings,
                processing_time_ms: result.processing_time_ms,
                metrics,
                graph_stats,
            }
        })
        .collect();

    let processing_time = start_time.elapsed();
    state.processing_histogram.observe(processing_time.as_secs_f64());

    info!(
        "Batch of {} graphs processed in {:.2}ms",
        request.len(),
        processing_time.as_secs_f64() * 1000.0
    );

    Ok(ResponseJson(responses))
}

async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    
    match encoder.encode_to_string(&metric_families) {
        Ok(metrics) => (StatusCode::OK, metrics),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to encode metrics: {}", e)
        ),
    }
}

// WebSocket support temporarily disabled
// async fn websocket_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
//     ws.on_upgrade(handle_websocket)
// }

// async fn handle_websocket(mut socket: WebSocket) {
//     info!("WebSocket connection established");
    
//     while let Some(msg) = socket.recv().await {
//         match msg {
//             Ok(axum::extract::ws::Message::Text(text)) => {
//                 info!("Received WebSocket message: {}", text);
                
//                 if socket.send(axum::extract::ws::Message::Text(
//                     "Echo: ".to_string() + &text
//                 )).await.is_err() {
//                     break;
//                 }
//             }
//             Ok(axum::extract::ws::Message::Close(_)) => {
//                 info!("WebSocket connection closed");
//                 break;
//             }
//             _ => {}
//         }
//     }
// }

async fn server_info_handler() -> impl IntoResponse {
    let info = serde_json::json!({
        "name": "graph-diffusion-edge-serving",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Ultra-Fast Dynamic Graph Diffusion Models on Edge TPU v5",
        "features": {
            "tpu": cfg!(feature = "tpu"),
            "wasm": cfg!(target_arch = "wasm32"),
        },
        "endpoints": {
            "/health": "Health check",
            "/diffuse": "Single graph diffusion",
            "/batch": "Batch graph diffusion",
            "/metrics": "Prometheus metrics",
            "/stream": "WebSocket streaming",
            "/info": "Server information"
        }
    });

    ResponseJson(info)
}

fn get_memory_usage_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }
    }
    0.0
}

#[derive(Debug)]
pub enum AppError {
    Validation(String),
    Processing(String),
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, error_type, message) = match self {
            AppError::Validation(msg) => (StatusCode::BAD_REQUEST, "validation_error", msg),
            AppError::Processing(msg) => (StatusCode::UNPROCESSABLE_ENTITY, "processing_error", msg),
            AppError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, "internal_error", msg),
        };

        let error_response = ErrorResponse {
            error: message,
            error_type: error_type.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        error!("Request failed: {:?}", error_response);

        (status, ResponseJson(error_response)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Graph, Node, Edge, GraphConfig, ProcessingConfig, DGDMProcessor};
    use axum_test::TestServer;

    async fn create_test_server() -> TestServer {
        let config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        let processor = DGDMProcessor::new(config, processing_config);
        
        let app_state = Arc::new(AppState {
            processor: Arc::new(processor),
            #[cfg(feature = "tpu")]
            tpu: None,
            request_counter: Counter::new("test_requests", "Test requests").unwrap(),
            processing_histogram: Histogram::with_opts(
                prometheus::HistogramOpts::new("test_duration", "Test duration")
            ).unwrap(),
        });

        TestServer::new(create_routes(app_state)).unwrap()
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let server = create_test_server().await;
        let response = server.get("/health").await;
        
        response.assert_status_ok();
        let health: HealthResponse = response.json();
        assert_eq!(health.status, "healthy");
    }

    #[tokio::test]
    async fn test_info_endpoint() {
        let server = create_test_server().await;
        let response = server.get("/info").await;
        
        response.assert_status_ok();
        let info: serde_json::Value = response.json();
        assert_eq!(info["name"], "graph-diffusion-edge-serving");
    }

    #[tokio::test]
    async fn test_diffuse_endpoint() {
        let server = create_test_server().await;
        
        let mut graph = Graph::new();
        graph.add_node(Node { id: 1, features: vec![1.0, 0.0], label: None });
        graph.add_node(Node { id: 2, features: vec![0.0, 1.0], label: None });
        graph.add_edge(Edge { source: 1, target: 2, weight: 1.0, edge_type: None });

        let request = GraphRequest {
            graph,
            config: None,
            return_metrics: true,
        };

        let response = server.post("/diffuse").json(&request).await;
        response.assert_status_ok();
        
        let result: DiffusionResponse = response.json();
        assert_eq!(result.embeddings.len(), 2);
        assert!(result.metrics.is_some());
    }
}