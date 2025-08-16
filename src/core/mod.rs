//! Core graph diffusion processing modules

pub mod graph;
pub mod dgdm;
pub mod diffusion;
pub mod adaptive_diffusion;

pub use graph::{Graph, Node, Edge, GraphConfig};
pub use dgdm::{DGDMProcessor, ProcessingConfig};
pub use diffusion::{DiffusionEngine, DiffusionConfig};
pub use adaptive_diffusion::{AdaptiveDiffusionEngine, AdaptiveDiffusionConfig};