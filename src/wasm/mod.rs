//! WASM bindings for browser and edge deployment

pub mod client;
pub mod utils;

pub use client::{DGDMClient, WasmGraph, WasmResult};
pub use utils::{set_panic_hook, console_log};

use wasm_bindgen::prelude::*;

// Initialize WASM module
#[wasm_bindgen(start)]
pub fn init() {
    utils::set_panic_hook();
    console_log("DGDM WASM module initialized");
}