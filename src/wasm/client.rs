//! WASM client for graph diffusion processing

use crate::core::{Graph, Node, Edge, GraphConfig, ProcessingConfig, DGDMProcessor};
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use js_sys::Array;
use web_sys::console;

#[wasm_bindgen]
pub struct DGDMClient {
    processor: DGDMProcessor,
}

#[derive(Serialize, Deserialize)]
pub struct WasmNode {
    pub id: u32,
    pub features: Vec<f32>,
    pub label: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct WasmEdge {
    pub source: u32,
    pub target: u32,
    pub weight: f32,
    pub edge_type: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct WasmGraph {
    pub nodes: Vec<WasmNode>,
    pub edges: Vec<WasmEdge>,
}

#[derive(Serialize, Deserialize)]
pub struct WasmConfig {
    pub diffusion_steps: Option<usize>,
    pub temperature: Option<f32>,
    pub normalize_features: Option<bool>,
}

#[derive(Serialize, Deserialize)]
pub struct WasmResult {
    pub embeddings: Vec<Vec<f32>>,
    pub processing_time_ms: f64,
    pub num_iterations: usize,
}

#[wasm_bindgen]
impl DGDMClient {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let graph_config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        let processor = DGDMProcessor::new(graph_config, processing_config);

        Self { processor }
    }

    #[wasm_bindgen]
    pub fn with_config(config_js: &JsValue) -> Result<DGDMClient, JsValue> {
        let wasm_config: WasmConfig = config_js.into_serde()
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

        let mut graph_config = GraphConfig::default();
        let mut processing_config = ProcessingConfig::default();

        if let Some(steps) = wasm_config.diffusion_steps {
            graph_config.diffusion_steps = steps;
        }

        if let Some(temp) = wasm_config.temperature {
            processing_config.temperature = temp;
        }

        if let Some(norm) = wasm_config.normalize_features {
            processing_config.normalize_features = norm;
        }

        let processor = DGDMProcessor::new(graph_config, processing_config);
        Ok(Self { processor })
    }

    #[wasm_bindgen]
    pub fn process_graph(&self, graph_js: &JsValue) -> Result<JsValue, JsValue> {
        let wasm_graph: WasmGraph = graph_js.into_serde()
            .map_err(|e| JsValue::from_str(&format!("Invalid graph: {}", e)))?;

        // Convert WASM graph to internal format
        let mut graph = Graph::new();
        
        for wasm_node in wasm_graph.nodes {
            graph.add_node(Node {
                id: wasm_node.id,
                features: wasm_node.features,
                label: wasm_node.label,
            });
        }

        for wasm_edge in wasm_graph.edges {
            graph.add_edge(Edge {
                source: wasm_edge.source,
                target: wasm_edge.target,
                weight: wasm_edge.weight,
                edge_type: wasm_edge.edge_type,
            });
        }

        // Validate and process
        graph.validate()
            .map_err(|e| JsValue::from_str(&format!("Graph validation failed: {}", e)))?;

        let compact_graph = graph.to_compact()
            .map_err(|e| JsValue::from_str(&format!("Graph conversion failed: {}", e)))?;

        let result = self.processor.process(&compact_graph)
            .map_err(|e| JsValue::from_str(&format!("Processing failed: {}", e)))?;

        // Convert result to WASM format
        let embeddings: Vec<Vec<f32>> = result.embeddings
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        let wasm_result = WasmResult {
            embeddings,
            processing_time_ms: result.processing_time_ms,
            num_iterations: result.num_iterations,
        };

        JsValue::from_serde(&wasm_result)
            .map_err(|e| JsValue::from_str(&format!("Result serialization failed: {}", e)))
    }

    #[wasm_bindgen]
    pub fn create_random_graph(&self, num_nodes: usize, num_edges: usize, feature_dim: usize) -> JsValue {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create random nodes
        for i in 0..num_nodes {
            let features: Vec<f32> = (0..feature_dim)
                .map(|_| js_sys::Math::random() as f32)
                .collect();
            
            nodes.push(WasmNode {
                id: i as u32,
                features,
                label: Some(format!("node_{}", i)),
            });
        }

        // Create random edges
        for _ in 0..num_edges {
            let source = (js_sys::Math::random() * num_nodes as f64) as u32;
            let target = (js_sys::Math::random() * num_nodes as f64) as u32;
            let weight = js_sys::Math::random() as f32;

            if source != target {
                edges.push(WasmEdge {
                    source,
                    target,
                    weight,
                    edge_type: None,
                });
            }
        }

        let graph = WasmGraph { nodes, edges };
        JsValue::from_serde(&graph).unwrap_or(JsValue::NULL)
    }

    #[wasm_bindgen]
    pub fn benchmark(&self, num_graphs: usize, nodes_per_graph: usize) -> Result<JsValue, JsValue> {
        console::time_with_label("DGDM Benchmark");
        
        let mut total_time = 0.0;
        let mut results = Vec::new();

        for i in 0..num_graphs {
            let graph_js = self.create_random_graph(nodes_per_graph, nodes_per_graph * 2, 64);
            let result = self.process_graph(&graph_js)?;
            
            let parsed_result: WasmResult = result.into_serde()
                .map_err(|e| JsValue::from_str(&format!("Benchmark result parsing failed: {}", e)))?;
            
            total_time += parsed_result.processing_time_ms;
            results.push(parsed_result);

            if i % 10 == 0 {
                console::log_1(&format!("Processed {}/{} graphs", i + 1, num_graphs).into());
            }
        }

        console::time_end_with_label("DGDM Benchmark");

        let benchmark_result = serde_json::json!({
            "num_graphs": num_graphs,
            "nodes_per_graph": nodes_per_graph,
            "total_time_ms": total_time,
            "avg_time_ms": total_time / num_graphs as f64,
            "graphs_per_second": (num_graphs as f64) / (total_time / 1000.0),
            "results": results
        });

        JsValue::from_serde(&benchmark_result)
            .map_err(|e| JsValue::from_str(&format!("Benchmark serialization failed: {}", e)))
    }

    #[wasm_bindgen]
    pub fn get_version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    #[wasm_bindgen]
    pub fn get_info(&self) -> JsValue {
        let info = serde_json::json!({
            "name": env!("CARGO_PKG_NAME"),
            "version": env!("CARGO_PKG_VERSION"),
            "description": env!("CARGO_PKG_DESCRIPTION"),
            "platform": "wasm32",
            "features": {
                "tpu": cfg!(feature = "tpu"),
                "benchmark": cfg!(feature = "benchmark")
            }
        });

        JsValue::from_serde(&info).unwrap_or(JsValue::NULL)
    }
}

impl Default for DGDMClient {
    fn default() -> Self {
        Self::new()
    }
}

// Export free functions for JavaScript
#[wasm_bindgen]
pub fn create_dgdm_client() -> DGDMClient {
    DGDMClient::new()
}

#[wasm_bindgen]
pub fn validate_graph(graph_js: &JsValue) -> Result<bool, JsValue> {
    let wasm_graph: WasmGraph = graph_js.into_serde()
        .map_err(|e| JsValue::from_str(&format!("Invalid graph format: {}", e)))?;

    if wasm_graph.nodes.is_empty() {
        return Err(JsValue::from_str("Graph must contain at least one node"));
    }

    // Check for consistent feature dimensions
    if let Some(first_node) = wasm_graph.nodes.first() {
        let feature_dim = first_node.features.len();
        for node in &wasm_graph.nodes {
            if node.features.len() != feature_dim {
                return Err(JsValue::from_str("All nodes must have the same feature dimension"));
            }
        }
    }

    // Check edge validity
    let node_ids: std::collections::HashSet<u32> = wasm_graph.nodes.iter().map(|n| n.id).collect();
    for edge in &wasm_graph.edges {
        if !node_ids.contains(&edge.source) {
            return Err(JsValue::from_str(&format!("Edge references non-existent source node: {}", edge.source)));
        }
        if !node_ids.contains(&edge.target) {
            return Err(JsValue::from_str(&format!("Edge references non-existent target node: {}", edge.target)));
        }
    }

    Ok(true)
}