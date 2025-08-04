//! Edge TPU integration for hardware-accelerated graph diffusion

#[cfg(feature = "tpu")]
pub mod runtime;
#[cfg(feature = "tpu")]
pub mod compiler;
#[cfg(feature = "tpu")]
pub mod kernels;

#[cfg(feature = "tpu")]
pub use runtime::{EdgeTPU, TpuConfig, TpuDevice};
#[cfg(feature = "tpu")]
pub use compiler::{ModelCompiler, CompilationConfig};

// Mock implementations when TPU feature is disabled
#[cfg(not(feature = "tpu"))]
pub mod mock {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct EdgeTPU;

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct TpuConfig;

    impl EdgeTPU {
        pub fn new() -> Self {
            Self
        }
    }

    impl TpuConfig {
        pub fn default() -> Self {
            Self
        }
    }
}

#[cfg(not(feature = "tpu"))]
pub use mock::*;