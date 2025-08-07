//! HTTP serving and streaming modules for graph diffusion

pub mod server;
pub mod handlers;
pub mod streaming;
pub mod health;
pub mod middleware;
pub mod circuit_breaker;

pub use server::{DGDMServer, start_server_with_config};
pub use handlers::{GraphRequest, DiffusionResponse};
pub use streaming::{StreamingDGDM, GraphUpdate};
pub use health::{HealthChecker, HealthStatus, ServiceStatus};
pub use middleware::{RateLimiter, rate_limiting_middleware, security_headers_middleware};
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitBreakerRegistry};