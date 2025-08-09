//! High-performance HTTP server for graph diffusion serving

use crate::core::{DGDMProcessor, GraphConfig, ProcessingConfig};
use crate::serving::handlers::{create_routes, AppState};
use crate::validation::{GraphValidator, ProcessingConfigValidator, SecurityValidator};
use crate::security::{SecurityMiddleware, AuthConfig, RateLimiter, security_headers_middleware, request_size_limit_middleware, rate_limit_middleware};
use crate::monitoring::{metrics_middleware, MetricsCollector};
use axum::Router;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    compression::CompressionLayer,
    trace::TraceLayer,
};
use tracing::info;

#[cfg(feature = "tpu")]
use crate::tpu::EdgeTPU;

pub struct DGDMServer {
    app_state: Arc<AppState>,
    bind_addr: SocketAddr,
}

impl DGDMServer {
    pub fn new(
        graph_config: GraphConfig, 
        processing_config: ProcessingConfig
    ) -> Self {
        let processor = DGDMProcessor::new(graph_config, processing_config);
        
        let app_state = Arc::new(AppState {
            processor: Arc::new(processor),
            #[cfg(feature = "tpu")]
            tpu: None,
            request_counter: prometheus::Counter::new(
                "dgdm_requests_total", 
                "Total number of requests processed"
            ).unwrap(),
            processing_histogram: prometheus::Histogram::with_opts(
                prometheus::HistogramOpts::new(
                    "dgdm_processing_duration_seconds",
                    "Graph processing duration in seconds"
                )
            ).unwrap(),
            graph_validator: Arc::new(GraphValidator::new()),
            config_validator: Arc::new(ProcessingConfigValidator),
            security_validator: Arc::new(SecurityValidator::new()),
        });

        Self {
            app_state,
            bind_addr: "0.0.0.0:8080".parse().unwrap(),
        }
    }

    #[cfg(feature = "tpu")]
    pub fn with_tpu(mut self, tpu: EdgeTPU) -> Self {
        Arc::get_mut(&mut self.app_state).unwrap().tpu = Some(Arc::new(tpu));
        self
    }

    pub fn bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = addr;
        self
    }

    pub async fn serve(self) -> crate::Result<()> {
        let app = self.create_app();
        
        let listener = TcpListener::bind(&self.bind_addr)
            .await
            .map_err(|e| crate::error::Error::network(
                format!("Failed to bind to {}", self.bind_addr),
                format!("{:?}", self.bind_addr)
            ))?;

        info!("🚀 DGDM Server starting on {}", self.bind_addr);
        info!("📊 Metrics available at /metrics");
        info!("🏥 Health check available at /health");
        info!("📡 WebSocket streaming at /stream");

        axum::serve(listener, app)
            .await
            .map_err(|e| crate::error::Error::network(
                format!("Server error: {}", e),
                format!("{:?}", self.bind_addr)
            ))?;

        Ok(())
    }

    fn create_app(&self) -> Router {
        // Security configuration
        let auth_config = AuthConfig::default();
        let security_middleware = Arc::new(SecurityMiddleware::new(auth_config));
        let rate_limiter = Arc::new(RateLimiter::new(100, 60)); // 100 requests per minute
        
        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any);

        let middleware = ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
            .layer(CompressionLayer::new())
            .layer(axum::middleware::from_fn(metrics_middleware))
            .layer(axum::middleware::from_fn(security_headers_middleware))
            .layer(axum::middleware::from_fn(request_size_limit_middleware))
            .layer(axum::middleware::from_fn_with_state(
                rate_limiter, 
                rate_limit_middleware
            ))
            .layer(cors);

        create_routes(self.app_state.clone())
            .layer(middleware)
    }
}

pub struct ServerBuilder {
    graph_config: GraphConfig,
    processing_config: ProcessingConfig,
    bind_addr: Option<SocketAddr>,
    #[cfg(feature = "tpu")]
    tpu: Option<EdgeTPU>,
}

impl ServerBuilder {
    pub fn new() -> Self {
        Self {
            graph_config: GraphConfig::default(),
            processing_config: ProcessingConfig::default(),
            bind_addr: None,
            #[cfg(feature = "tpu")]
            tpu: None,
        }
    }

    pub fn graph_config(mut self, config: GraphConfig) -> Self {
        self.graph_config = config;
        self
    }

    pub fn processing_config(mut self, config: ProcessingConfig) -> Self {
        self.processing_config = config;
        self
    }

    pub fn bind_addr(mut self, addr: SocketAddr) -> Self {
        self.bind_addr = Some(addr);
        self
    }

    #[cfg(feature = "tpu")]
    pub fn tpu(mut self, tpu: EdgeTPU) -> Self {
        self.tpu = Some(tpu);
        self
    }

    pub fn build(self) -> DGDMServer {
        let mut server = DGDMServer::new(self.graph_config, self.processing_config);
        
        if let Some(addr) = self.bind_addr {
            server = server.bind_addr(addr);
        }

        #[cfg(feature = "tpu")]
        if let Some(tpu) = self.tpu {
            server = server.with_tpu(tpu);
        }

        server
    }
}

impl Default for ServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub async fn start_server_with_config(
    bind_addr: &str,
    graph_config: GraphConfig,
    processing_config: ProcessingConfig,
) -> crate::Result<()> {
    let addr: SocketAddr = bind_addr
        .parse()
        .map_err(|e| crate::error::Error::configuration(
            format!("Invalid bind address '{}': {}", bind_addr, e),
            "bind_address"
        ))?;

    let server = DGDMServer::new(graph_config, processing_config)
        .bind_addr(addr);

    server.serve().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{GraphConfig, ProcessingConfig};

    #[test]
    fn test_server_builder() {
        let server = ServerBuilder::new()
            .bind_addr("127.0.0.1:9090".parse().unwrap())
            .graph_config(GraphConfig {
                max_nodes: 50000,
                ..Default::default()
            })
            .build();

        assert_eq!(server.bind_addr.port(), 9090);
    }

    #[tokio::test]
    async fn test_server_creation() {
        let config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        
        let server = DGDMServer::new(config, processing_config);
        
        // Test that the server can be created without panicking
        assert_eq!(server.bind_addr.port(), 8080);
    }
}