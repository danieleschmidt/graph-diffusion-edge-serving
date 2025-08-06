//! Edge TPU integration for hardware-accelerated graph diffusion

// Use mock implementation by default for testing
pub mod runtime_mock;
pub mod compiler;
pub mod kernels;

pub use runtime_mock::{EdgeTPU, TpuConfig, TpuDevice, list_available_devices};
pub use compiler::{ModelCompiler, CompilationConfig};

// Real TPU implementation (disabled for testing)
// #[cfg(feature = "tpu")]
// pub mod runtime;
// #[cfg(feature = "tpu")]
// pub use runtime::{EdgeTPU, TpuConfig, TpuDevice};