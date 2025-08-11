//! Resilient server with circuit breakers, bulkheads, and health monitoring

use crate::{
    Result,
    core::{Graph, DGDMProcessor, GraphConfig, ProcessingConfig},
    resilience::{
        AdaptiveCircuitBreaker, CircuitBreakerConfig, SmartRetryPolicy,
        ResourceBulkhead, HealthMonitor, GraphProcessingHealthCheck, HealthStatus
    },
    validation::{GraphValidator, SecurityValidator},
    serving::handlers::{GraphRequest, DiffusionResponse},
};

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::{info, warn, error, instrument};
use serde::{Deserialize, Serialize};

/// Resilient server with comprehensive failure handling
pub struct ResilientDGDMServer {
    // Core processing
    processor: Arc<DGDMProcessor>,
    validator: Arc<GraphValidator>,
    security_validator: Arc<SecurityValidator>,
    
    // Resilience patterns
    processing_circuit_breaker: Arc<AdaptiveCircuitBreaker>,
    validation_circuit_breaker: Arc<AdaptiveCircuitBreaker>,
    retry_policy: Arc<SmartRetryPolicy>,
    processing_bulkhead: Arc<ResourceBulkhead>,
    validation_bulkhead: Arc<ResourceBulkhead>,
    
    // Health monitoring
    health_monitor: Arc<RwLock<HealthMonitor>>,
    
    // Metrics
    request_count: Arc<prometheus::Counter>,
    processing_duration: Arc<prometheus::Histogram>,
    error_count: Arc<prometheus::CounterVec>,
}

#[derive(Clone)]
pub struct ServerState {
    server: Arc<ResilientDGDMServer>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub checks: Vec<HealthCheckSummary>,
    pub timestamp: String,
    pub response_time_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthCheckSummary {
    pub name: String,
    pub status: String,
    pub message: String,
    pub response_time_ms: u64,
}

impl ResilientDGDMServer {
    pub async fn new(
        graph_config: GraphConfig,
        processing_config: ProcessingConfig,
    ) -> Result<Self> {
        // Initialize core components
        let processor = Arc::new(DGDMProcessor::new(graph_config.clone(), processing_config));
        let validator = Arc::new(GraphValidator::new());
        let security_validator = Arc::new(SecurityValidator::new());
        
        // Configure circuit breakers
        let processing_circuit_breaker = Arc::new(AdaptiveCircuitBreaker::new(
            CircuitBreakerConfig {
                name: "graph_processing".to_string(),
                failure_threshold: 5,
                success_threshold: 3,
                timeout: Duration::from_secs(30),
                adaptation_factor: 1.5,
            }
        ));
        
        let validation_circuit_breaker = Arc::new(AdaptiveCircuitBreaker::new(
            CircuitBreakerConfig {
                name: "graph_validation".to_string(),
                failure_threshold: 10,
                success_threshold: 5,
                timeout: Duration::from_secs(10),
                adaptation_factor: 1.2,
            }
        ));
        
        // Configure retry policy
        let retry_policy = Arc::new(SmartRetryPolicy::new(
            3, 
            Duration::from_millis(100)
        ));
        
        // Configure bulkheads
        let processing_bulkhead = Arc::new(ResourceBulkhead::new(
            "processing".to_string(),
            graph_config.batch_size.max(10),
            Duration::from_secs(60),
        ));
        
        let validation_bulkhead = Arc::new(ResourceBulkhead::new(
            "validation".to_string(),
            50,
            Duration::from_secs(10),
        ));
        
        // Initialize health monitoring
        let mut health_monitor = HealthMonitor::new(Duration::from_secs(10));
        health_monitor.add_check(GraphProcessingHealthCheck);
        let health_monitor = Arc::new(RwLock::new(health_monitor));
        
        // Initialize metrics
        let request_count = Arc::new(
            prometheus::Counter::new(
                "dgdm_requests_total",
                "Total number of graph processing requests"
            ).map_err(|e| crate::error::Error::configuration(e.to_string(), "metrics"))?
        );
        
        let processing_duration = Arc::new(
            prometheus::Histogram::new(
                prometheus::HistogramOpts::new(
                    "dgdm_processing_duration_seconds",
                    "Graph processing duration distribution"
                )
            ).map_err(|e| crate::error::Error::configuration(e.to_string(), "metrics"))?
        );
        
        let error_count = Arc::new(
            prometheus::CounterVec::new(
                prometheus::Opts::new(
                    "dgdm_errors_total",
                    "Total number of errors by type"
                ),
                &["error_type", "component"]
            ).map_err(|e| crate::error::Error::configuration(e.to_string(), "metrics"))?
        );
        
        Ok(Self {
            processor,
            validator,
            security_validator,
            processing_circuit_breaker,
            validation_circuit_breaker,
            retry_policy,
            processing_bulkhead,
            validation_bulkhead,
            health_monitor,
            request_count,
            processing_duration,
            error_count,
        })
    }
    
    pub fn router(&self) -> Router {
        let state = ServerState {
            server: Arc::new(self.clone()),
        };
        
        Router::new()
            .route("/health", get(health_check))
            .route("/health/detailed", get(detailed_health_check))
            .route("/process", post(process_graph))
            .route("/metrics", get(metrics))
            .with_state(state)
    }
    
    #[instrument(skip(self, request))]
    pub async fn process_graph_resilient(&self, request: GraphRequest) -> Result<DiffusionResponse> {
        let start_time = Instant::now();
        self.request_count.inc();
        
        // Step 1: Secure validation with circuit breaker and bulkhead
        let graph = self.validation_circuit_breaker.call(
            self.validation_bulkhead.execute(async {
                self.validate_request_secure(&request).await
            })
        ).await.map_err(|e| {
            self.error_count.with_label_values(&["validation_failure", "circuit_breaker"]).inc();
            e
        })?;
        
        // Step 2: Process graph with retry logic and circuit breaker
        let result = self.retry_policy.execute(|| {
            let processor = Arc::clone(&self.processor);
            let circuit_breaker = Arc::clone(&self.processing_circuit_breaker);
            let bulkhead = Arc::clone(&self.processing_bulkhead);
            let compact_graph = graph.clone();
            
            async move {
                circuit_breaker.call(
                    bulkhead.execute(async move {
                        processor.process(&compact_graph.to_compact()?)
                    })
                ).await
            }
        }).await.map_err(|e| {
            self.error_count.with_label_values(&["processing_failure", "retry_exhausted"]).inc();
            e
        })?;
        
        // Record metrics
        let processing_time = start_time.elapsed();
        self.processing_duration.observe(processing_time.as_secs_f64());
        
        // Build response
        Ok(DiffusionResponse {
            embeddings: result.embeddings,
            processing_time_ms: result.processing_time_ms,
            num_iterations: result.num_iterations,
            convergence_score: result.convergence_score,
            metadata: std::collections::HashMap::new(),
        })
    }
    
    async fn validate_request_secure(&self, request: &GraphRequest) -> Result<Graph> {
        // Security validation first
        self.security_validator.validate_graph_request(request)?;
        
        // Convert to graph structure
        let mut graph = Graph::new();
        
        for node_data in &request.nodes {
            // Additional security checks on node data
            self.security_validator.validate_string(&node_data.label.as_deref().unwrap_or(""))?;
            
            let node = crate::core::Node {
                id: node_data.id,
                features: node_data.features.clone(),
                label: node_data.label.clone(),
            };
            graph.add_node(node);
        }
        
        for edge_data in &request.edges {
            // Validate edge types for security
            if let Some(edge_type) = &edge_data.edge_type {
                self.security_validator.validate_string(edge_type)?;
            }
            
            let edge = crate::core::Edge {
                source: edge_data.source,
                target: edge_data.target,
                weight: edge_data.weight,
                edge_type: edge_data.edge_type.clone(),
            };
            graph.add_edge(edge);
        }
        
        // Structural validation
        self.validator.validate_graph(&graph)?;
        
        Ok(graph)
    }
    
    pub async fn get_health_status(&self) -> HealthResponse {
        let health_report = self.health_monitor.read().await.check_health().await;
        
        let status = match health_report.status {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Degraded => "degraded", 
            HealthStatus::Unhealthy => "unhealthy",
        };
        
        let checks = health_report.checks.into_iter().map(|check| {
            HealthCheckSummary {
                name: check.name,
                status: match check.status {
                    HealthStatus::Healthy => "healthy".to_string(),
                    HealthStatus::Degraded => "degraded".to_string(),
                    HealthStatus::Unhealthy => "unhealthy".to_string(),
                },
                message: check.message,
                response_time_ms: check.response_time.as_millis() as u64,
            }
        }).collect();
        
        HealthResponse {
            status: status.to_string(),
            checks,
            timestamp: chrono::Utc::now().to_rfc3339(),
            response_time_ms: health_report.total_response_time.as_millis() as u64,
        }
    }
}

impl Clone for ResilientDGDMServer {
    fn clone(&self) -> Self {
        Self {
            processor: Arc::clone(&self.processor),
            validator: Arc::clone(&self.validator),
            security_validator: Arc::clone(&self.security_validator),
            processing_circuit_breaker: Arc::clone(&self.processing_circuit_breaker),
            validation_circuit_breaker: Arc::clone(&self.validation_circuit_breaker),
            retry_policy: Arc::clone(&self.retry_policy),
            processing_bulkhead: Arc::clone(&self.processing_bulkhead),
            validation_bulkhead: Arc::clone(&self.validation_bulkhead),
            health_monitor: Arc::clone(&self.health_monitor),
            request_count: Arc::clone(&self.request_count),
            processing_duration: Arc::clone(&self.processing_duration),
            error_count: Arc::clone(&self.error_count),
        }
    }
}

// HTTP handlers
async fn health_check(State(state): State<ServerState>) -> std::result::Result<Json<HealthResponse>, StatusCode> {
    let health_status = state.server.get_health_status().await;
    
    let status_code = match health_status.status.as_str() {
        "healthy" => StatusCode::OK,
        "degraded" => StatusCode::OK, // Still accepting requests
        "unhealthy" => StatusCode::SERVICE_UNAVAILABLE,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    
    match status_code {
        StatusCode::OK => Ok(Json(health_status)),
        _ => Err(status_code),
    }
}

async fn detailed_health_check(State(state): State<ServerState>) -> Json<HealthResponse> {
    Json(state.server.get_health_status().await)
}

async fn process_graph(
    State(state): State<ServerState>,
    Json(request): Json<GraphRequest>,
) -> std::result::Result<Json<DiffusionResponse>, StatusCode> {
    match state.server.process_graph_resilient(request).await {
        Ok(response) => Ok(Json(response)),
        Err(e) => {
            error!("Graph processing failed: {}", e);
            match e {
                crate::error::Error::Validation { .. } => Err(StatusCode::BAD_REQUEST),
                crate::error::Error::Security { .. } => Err(StatusCode::FORBIDDEN),
                crate::error::Error::ResourceExhausted { .. } => Err(StatusCode::TOO_MANY_REQUESTS),
                crate::error::Error::Timeout { .. } => Err(StatusCode::REQUEST_TIMEOUT),
                _ => Err(StatusCode::INTERNAL_SERVER_ERROR),
            }
        }
    }
}

async fn metrics(State(_state): State<ServerState>) -> std::result::Result<String, StatusCode> {
    use prometheus::TextEncoder;
    
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    
    match encoder.encode_to_string(&metric_families) {
        Ok(metrics) => Ok(metrics),
        Err(e) => {
            error!("Failed to encode metrics: {}", e);
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serving::handlers::{NodeData, EdgeData};
    
    #[tokio::test]
    async fn test_resilient_server_creation() {
        let graph_config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        
        let server = ResilientDGDMServer::new(graph_config, processing_config).await;
        assert!(server.is_ok());
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let graph_config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        
        let server = ResilientDGDMServer::new(graph_config, processing_config).await.unwrap();
        let health = server.get_health_status().await;
        
        // Health check should complete without error
        assert!(!health.status.is_empty());
        assert!(!health.checks.is_empty());
    }
    
    #[tokio::test]
    async fn test_process_simple_graph() {
        let mut graph_config = GraphConfig::default();
        graph_config.feature_dim = 2;
        let processing_config = ProcessingConfig::default();
        
        let server = ResilientDGDMServer::new(graph_config, processing_config).await.unwrap();
        
        let request = GraphRequest {
            nodes: vec![
                NodeData {
                    id: 1,
                    features: vec![1.0, 0.0],
                    label: Some("test1".to_string()),
                },
                NodeData {
                    id: 2,
                    features: vec![0.0, 1.0],
                    label: Some("test2".to_string()),
                },
            ],
            edges: vec![
                EdgeData {
                    source: 1,
                    target: 2,
                    weight: 0.5,
                    edge_type: Some("test".to_string()),
                },
            ],
            config: None,
        };
        
        let result = server.process_graph_resilient(request).await;
        assert!(result.is_ok());
        
        let response = result.unwrap();
        assert_eq!(response.embeddings.nrows(), 2);
        assert!(response.processing_time_ms > 0.0);
    }
}