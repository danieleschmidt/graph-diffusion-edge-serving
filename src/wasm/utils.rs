//! WASM utility functions

use wasm_bindgen::prelude::*;
use web_sys::console;

// Set up better panic messages for debugging
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

// Console logging utility
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

pub fn console_log(s: &str) {
    log(s);
}

// Macro for easier console logging
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => ($crate::wasm::utils::console_log(&format_args!($($t)*).to_string()))
}

// Performance measurement utilities
#[wasm_bindgen]
pub fn performance_now() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now())
        .unwrap_or(0.0)
}

// Memory usage information
#[wasm_bindgen]
pub fn memory_info() -> JsValue {
    let memory = wasm_bindgen::memory();
    let info = serde_json::json!({
        "buffer_length": memory.buffer().byte_length(),
        "used_pages": memory.buffer().byte_length() / 65536, // WASM page size
    });
    
    JsValue::from_serde(&info).unwrap_or(JsValue::NULL)
}

// Feature detection
#[wasm_bindgen]
pub fn get_wasm_features() -> JsValue {
    let features = serde_json::json!({
        "threads": cfg!(target_feature = "atomics"),
        "simd": cfg!(target_feature = "simd128"),
        "bulk_memory": true, // Assume modern WASM environment
        "reference_types": true,
    });
    
    JsValue::from_serde(&features).unwrap_or(JsValue::NULL)
}

// Browser detection utilities
#[wasm_bindgen]
pub fn get_user_agent() -> Option<String> {
    web_sys::window()?
        .navigator()
        .user_agent()
        .ok()
}

#[wasm_bindgen]
pub fn is_worker_context() -> bool {
    web_sys::WorkerGlobalScope::from(js_sys::global()).is_some()
}

// Simple random number utilities (using JavaScript's Math.random)
#[wasm_bindgen]
pub fn random_f32() -> f32 {
    js_sys::Math::random() as f32
}

#[wasm_bindgen]
pub fn random_range(min: f32, max: f32) -> f32 {
    min + (max - min) * random_f32()
}

// Array utilities for JavaScript interop
#[wasm_bindgen]
pub fn f32_array_to_js(data: &[f32]) -> js_sys::Float32Array {
    js_sys::Float32Array::from(data)
}

#[wasm_bindgen]
pub fn js_array_to_f32(js_array: &js_sys::Float32Array) -> Vec<f32> {
    js_array.to_vec()
}

// WASM-specific optimizations
#[wasm_bindgen]
pub fn prefetch_memory(size_mb: usize) {
    // Attempt to grow WASM memory proactively
    let pages_needed = (size_mb * 1024 * 1024) / 65536; // Convert MB to WASM pages
    let current_pages = wasm_bindgen::memory().buffer().byte_length() / 65536;
    
    if pages_needed > current_pages as usize {
        // This is a hint - actual memory growth happens automatically
        console_log(&format!("Estimated memory needed: {} pages", pages_needed));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_range() {
        let val = random_range(0.0, 1.0);
        assert!(val >= 0.0 && val <= 1.0);
    }
}