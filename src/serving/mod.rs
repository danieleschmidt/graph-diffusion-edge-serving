//! HTTP serving and streaming modules for graph diffusion

pub mod server;
pub mod handlers;
pub mod streaming;

pub use server::{DGDMServer, start_server_with_config};
pub use handlers::{GraphRequest, DiffusionResponse};
pub use streaming::{StreamingDGDM, GraphUpdate};