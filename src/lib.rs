//! Graph Diffusion Edge Serving
//! 
//! Ultra-fast Dynamic Graph Diffusion Models (DGDM) for Edge TPU v5.
//! Provides high-performance graph processing with sub-10ms inference latency.

pub mod core;
pub mod serving;
pub mod optimization;
pub mod validation;
pub mod security;
pub mod monitoring;
pub mod scaling;
pub mod security_scanner;
pub mod performance_optimizer;
pub mod resilience;
pub mod resilience_patterns;
pub mod distributed_processing;
pub mod quality_gates;
pub mod progressive_quality;
pub mod autonomous_executor;
pub mod adaptive_resilience;
pub mod enhanced_monitoring;
pub mod intelligent_error_recovery;
pub mod intelligent_optimization;
pub mod global_deployment;
pub mod benchmarking;
pub mod adaptive_optimization_engine;
pub mod enhanced_security;
pub mod enhanced_global_deployment;
pub mod research_validation;
pub mod ai_driven_global_orchestration;
pub mod resilience_orchestrator;
pub mod health_monitor;
pub mod scaling_orchestrator;
pub mod autonomous_quality_gates;
pub mod production_deployment_system;
pub mod autonomous_sdlc_executor;
pub mod autonomous_quantum_engine;
pub mod self_modifying_code_generator;
pub mod evolutionary_system_optimizer;

// Minimal working implementation for autonomous SDLC demonstration
pub mod minimal_server;

#[cfg(feature = "tpu")]
pub mod tpu;

#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use core::{
    graph::{Graph, Node, Edge, GraphConfig},
    dgdm::{DGDMProcessor, ProcessingConfig},
    diffusion::{DiffusionEngine, DiffusionConfig},
};

pub use serving::{
    server::DGDMServer,
    handlers::{GraphRequest, DiffusionResponse},
    streaming::{StreamingDGDM, GraphUpdate},
};

#[cfg(feature = "tpu")]
pub use tpu::{EdgeTPU, TpuConfig};

pub type Result<T> = std::result::Result<T, crate::error::Error>;

pub mod error {
    use thiserror::Error;
    use std::fmt;

    #[derive(Error, Debug)]
    pub enum Error {
        #[error("Graph processing failed: {message} (context: {context})")]
        GraphProcessing { message: String, context: String },
        
        #[error("TPU runtime error: {message} (device: {device}, model: {model})")]
        TpuRuntime { message: String, device: String, model: String },
        
        #[error("Serialization error: {0}")]
        Serialization(#[from] serde_json::Error),
        
        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),
        
        #[error("Network error: {message} (endpoint: {endpoint})")]
        Network { message: String, endpoint: String },
        
        #[error("Configuration error: {message} (field: {field})")]
        Configuration { message: String, field: String },
        
        #[error("Validation failed: {message} (input: {input}, constraint: {constraint})")]
        Validation { message: String, input: String, constraint: String },
        
        #[error("Resource exhaustion: {resource} exceeded limit {limit} (current: {current})")]
        ResourceExhausted { resource: String, limit: String, current: String },
        
        #[error("Timeout occurred: {operation} took {duration_ms}ms (limit: {timeout_ms}ms)")]
        Timeout { operation: String, duration_ms: u64, timeout_ms: u64 },
        
        #[error("Dependency failure: {service} is unavailable (reason: {reason})")]
        DependencyFailure { service: String, reason: String },
        
        #[error("Security violation: {violation} (context: {context})")]
        Security { violation: String, context: String },
        
        #[error("Data corruption detected: {description} (checksum_expected: {expected}, checksum_actual: {actual})")]
        DataCorruption { description: String, expected: String, actual: String },
        
        #[error("Rate limit exceeded: {limit} requests per {window} (current: {current})")]
        RateLimit { limit: u32, window: String, current: u32 },
        
        #[error("Prometheus error: {0}")]
        Prometheus(#[from] prometheus::Error),
    }

    // Convenience constructors for common error types
    impl Error {
        pub fn graph_processing<S: Into<String>, C: Into<String>>(message: S, context: C) -> Self {
            Self::GraphProcessing {
                message: message.into(),
                context: context.into(),
            }
        }

        pub fn tpu_runtime<S: Into<String>, D: Into<String>, M: Into<String>>(
            message: S, 
            device: D, 
            model: M
        ) -> Self {
            Self::TpuRuntime {
                message: message.into(),
                device: device.into(),
                model: model.into(),
            }
        }

        pub fn validation<S: Into<String>, I: Into<String>, C: Into<String>>(
            message: S,
            input: I,
            constraint: C,
        ) -> Self {
            Self::Validation {
                message: message.into(),
                input: input.into(),
                constraint: constraint.into(),
            }
        }

        pub fn network<S: Into<String>, E: Into<String>>(message: S, endpoint: E) -> Self {
            Self::Network {
                message: message.into(),
                endpoint: endpoint.into(),
            }
        }

        pub fn configuration<S: Into<String>, F: Into<String>>(message: S, field: F) -> Self {
            Self::Configuration {
                message: message.into(),
                field: field.into(),
            }
        }

        pub fn resource_exhausted<R: fmt::Display, L: fmt::Display, C: fmt::Display>(
            resource: R,
            limit: L,
            current: C,
        ) -> Self {
            Self::ResourceExhausted {
                resource: resource.to_string(),
                limit: limit.to_string(),
                current: current.to_string(),
            }
        }

        pub fn timeout<O: Into<String>>(operation: O, duration_ms: u64, timeout_ms: u64) -> Self {
            Self::Timeout {
                operation: operation.into(),
                duration_ms,
                timeout_ms,
            }
        }

        pub fn dependency_failure<S: Into<String>, R: Into<String>>(service: S, reason: R) -> Self {
            Self::DependencyFailure {
                service: service.into(),
                reason: reason.into(),
            }
        }
    }

    // Error classification for different handling strategies
    #[derive(Debug, Clone, PartialEq)]
    pub enum ErrorCategory {
        /// Errors that can be retried (network issues, temporary resource exhaustion)
        Retriable,
        /// Errors that indicate permanent failures (validation, configuration)
        NonRetriable,
        /// Errors that require immediate attention (security, corruption)
        Critical,
        /// Errors related to rate limiting that require backoff
        RateLimit,
    }

    impl Error {
        pub fn category(&self) -> ErrorCategory {
            match self {
                Error::Network { .. } | Error::DependencyFailure { .. } | Error::Timeout { .. } => {
                    ErrorCategory::Retriable
                }
                Error::RateLimit { .. } => ErrorCategory::RateLimit,
                Error::Security { .. } | Error::DataCorruption { .. } => ErrorCategory::Critical,
                _ => ErrorCategory::NonRetriable,
            }
        }

        pub fn is_retriable(&self) -> bool {
            matches!(self.category(), ErrorCategory::Retriable)
        }

        pub fn is_critical(&self) -> bool {
            matches!(self.category(), ErrorCategory::Critical)
        }
    }
}