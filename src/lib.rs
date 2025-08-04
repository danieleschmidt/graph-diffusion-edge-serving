//! Graph Diffusion Edge Serving
//! 
//! Ultra-fast Dynamic Graph Diffusion Models (DGDM) for Edge TPU v5.
//! Provides high-performance graph processing with sub-10ms inference latency.

pub mod core;
pub mod serving;
pub mod optimization;

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

    #[derive(Error, Debug)]
    pub enum Error {
        #[error("Graph processing error: {0}")]
        GraphProcessing(String),
        
        #[error("TPU runtime error: {0}")]
        TpuRuntime(String),
        
        #[error("Serialization error: {0}")]
        Serialization(#[from] serde_json::Error),
        
        #[error("IO error: {0}")]
        Io(#[from] std::io::Error),
        
        #[error("Network error: {0}")]
        Network(String),
        
        #[error("Configuration error: {0}")]
        Configuration(String),
        
        #[error("Validation error: {0}")]
        Validation(String),
    }
}