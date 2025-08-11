//! Comprehensive input validation and sanitization

use crate::core::{Graph, Node, Edge, ProcessingConfig};
use crate::Result;
use regex::Regex;
use std::collections::HashSet;
use tracing::{warn, error};

const MAX_NODES: usize = 10_000_000;
const MAX_EDGES: usize = 100_000_000; 
const MAX_FEATURE_DIM: usize = 10_000;
const MAX_REQUEST_SIZE_MB: usize = 500;
const MIN_TEMPERATURE: f32 = 1e-8;
const MAX_TEMPERATURE: f32 = 10.0;
const MAX_DROPOUT: f32 = 0.99;
const MAX_STRING_LENGTH: usize = 1000;

pub struct SecurityValidator {
    malicious_patterns: Vec<Regex>,
    allowed_file_extensions: HashSet<String>,
}

impl Default for SecurityValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl SecurityValidator {
    pub fn new() -> Self {
        let malicious_patterns = vec![
            // SQL injection patterns
            Regex::new(r"(?i)(union|select|insert|update|delete|drop|create|alter)\s").unwrap(),
            // JavaScript injection patterns  
            Regex::new(r"(?i)<script|javascript:|data:text/html|vbscript:").unwrap(),
            // Path traversal patterns
            Regex::new(r"\.\.\/|\.\.\\|%2e%2e%2f|%2e%2e\\").unwrap(),
            // Command injection patterns
            Regex::new(r"(?i)(;|&&|\|\||\||`|\$\(|<\(|>\()").unwrap(),
            // LDAP injection patterns
            Regex::new(r"[()&|=*!<>]+").unwrap(),
        ];
        
        let mut allowed_extensions = HashSet::new();
        allowed_extensions.insert("json".to_string());
        allowed_extensions.insert("txt".to_string());
        allowed_extensions.insert("csv".to_string());
        
        Self {
            malicious_patterns,
            allowed_file_extensions: allowed_extensions,
        }
    }

    pub fn validate_string(&self, input: &str, field_name: &str) -> Result<()> {
        // Length validation
        if input.len() > MAX_STRING_LENGTH {
            return Err(crate::error::Error::validation(
                format!("String too long for field '{}'", field_name),
                format!("length={}", input.len()),
                format!("<= {}", MAX_STRING_LENGTH)
            ));
        }

        // Security pattern validation
        for pattern in &self.malicious_patterns {
            if pattern.is_match(input) {
                warn!("Potential security violation detected in field '{}': pattern match", field_name);
                return Err(crate::error::Error::Security {
                    violation: "Potentially malicious pattern detected".to_string(),
                    context: format!("field: {}, pattern_matched: true", field_name),
                });
            }
        }

        // Control character validation
        if input.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r') {
            return Err(crate::error::Error::Security {
                violation: "Control characters detected".to_string(),
                context: format!("field: {}", field_name),
            });
        }

        // Unicode normalization check
        use unicode_normalization::{UnicodeNormalization, is_nfc};
        if !is_nfc(input) {
            warn!("Non-NFC Unicode detected in field '{}'", field_name);
        }

        Ok(())
    }

    pub fn validate_filename(&self, filename: &str) -> Result<()> {
        self.validate_string(filename, "filename")?;
        
        if let Some(extension) = std::path::Path::new(filename).extension() {
            if let Some(ext_str) = extension.to_str() {
                if !self.allowed_file_extensions.contains(ext_str) {
                    return Err(crate::error::Error::Security {
                        violation: "Disallowed file extension".to_string(),
                        context: format!("extension: {}", ext_str),
                    });
                }
            }
        }

        Ok(())
    }
}

pub struct GraphValidator {
    security_validator: SecurityValidator,
}

impl Default for GraphValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphValidator {
    pub fn new() -> Self {
        Self {
            security_validator: SecurityValidator::new(),
        }
    }

    pub fn validate_graph(&self, graph: &Graph) -> Result<()> {
        // Size validation
        if graph.nodes.len() > MAX_NODES {
            return Err(crate::error::Error::validation(
                "Too many nodes in graph",
                format!("{}", graph.nodes.len()),
                format!("<= {}", MAX_NODES)
            ));
        }

        if graph.edges.len() > MAX_EDGES {
            return Err(crate::error::Error::validation(
                "Too many edges in graph",
                format!("{}", graph.edges.len()),
                format!("<= {}", MAX_EDGES)
            ));
        }

        // Memory size estimation (rough)
        let estimated_size = self.estimate_graph_memory_size(graph);
        let max_size_bytes = MAX_REQUEST_SIZE_MB * 1024 * 1024;
        if estimated_size > max_size_bytes {
            return Err(crate::error::Error::validation(
                "Graph too large",
                format!("{} bytes", estimated_size),
                format!("<= {} MB", MAX_REQUEST_SIZE_MB)
            ));
        }

        // Node validation
        self.validate_nodes(&graph.nodes)?;
        
        // Edge validation
        self.validate_edges(&graph.edges, &graph.nodes)?;
        
        // Graph structure validation
        self.validate_graph_structure(graph)?;

        Ok(())
    }

    fn validate_nodes(&self, nodes: &[Node]) -> Result<()> {
        if nodes.is_empty() {
            return Err(crate::error::Error::validation(
                "Graph cannot be empty",
                "0",
                "> 0"
            ));
        }

        let mut node_ids = HashSet::new();
        let expected_feature_dim = nodes[0].features.len();
        
        if expected_feature_dim == 0 {
            return Err(crate::error::Error::validation(
                "Node features cannot be empty",
                "0",
                "> 0"
            ));
        }

        if expected_feature_dim > MAX_FEATURE_DIM {
            return Err(crate::error::Error::validation(
                "Feature dimension too large",
                format!("{}", expected_feature_dim),
                format!("<= {}", MAX_FEATURE_DIM)
            ));
        }

        for (idx, node) in nodes.iter().enumerate() {
            // Duplicate ID check
            if !node_ids.insert(node.id) {
                return Err(crate::error::Error::validation(
                    format!("Duplicate node ID at index {}", idx),
                    format!("id={}", node.id),
                    "unique"
                ));
            }

            // Feature dimension consistency
            if node.features.len() != expected_feature_dim {
                return Err(crate::error::Error::validation(
                    format!("Inconsistent feature dimension at node {}", idx),
                    format!("{}", node.features.len()),
                    format!("{}", expected_feature_dim)
                ));
            }

            // Feature value validation
            for (feat_idx, &feature_val) in node.features.iter().enumerate() {
                if !feature_val.is_finite() {
                    return Err(crate::error::Error::validation(
                        format!("Non-finite feature value at node {} feature {}", idx, feat_idx),
                        format!("{}", feature_val),
                        "finite number"
                    ));
                }

                if feature_val.abs() > 1e6 {
                    warn!("Very large feature value detected: {} at node {} feature {}", 
                          feature_val, idx, feat_idx);
                }
            }

            // Label validation if present
            if let Some(ref label) = node.label {
                self.security_validator.validate_string(label, &format!("node_{}_label", node.id))?;
            }
        }

        Ok(())
    }

    fn validate_edges(&self, edges: &[Edge], nodes: &[Node]) -> Result<()> {
        let node_ids: HashSet<u32> = nodes.iter().map(|n| n.id).collect();
        let mut edge_pairs = HashSet::new();

        for (idx, edge) in edges.iter().enumerate() {
            // Node existence validation
            if !node_ids.contains(&edge.source) {
                return Err(crate::error::Error::validation(
                    format!("Edge {} references non-existent source node", idx),
                    format!("source={}", edge.source),
                    "valid node id"
                ));
            }

            if !node_ids.contains(&edge.target) {
                return Err(crate::error::Error::validation(
                    format!("Edge {} references non-existent target node", idx),
                    format!("target={}", edge.target),
                    "valid node id"
                ));
            }

            // Self-loop check (optional - depends on your use case)
            if edge.source == edge.target {
                warn!("Self-loop detected at edge {} (node {})", idx, edge.source);
            }

            // Duplicate edge check
            let edge_pair = (edge.source, edge.target);
            if !edge_pairs.insert(edge_pair) {
                return Err(crate::error::Error::validation(
                    format!("Duplicate edge at index {}", idx),
                    format!("source={}, target={}", edge.source, edge.target),
                    "unique"
                ));
            }

            // Weight validation
            if !edge.weight.is_finite() {
                return Err(crate::error::Error::validation(
                    format!("Non-finite edge weight at edge {}", idx),
                    format!("{}", edge.weight),
                    "finite number"
                ));
            }

            if edge.weight < 0.0 {
                warn!("Negative edge weight detected at edge {}: {}", idx, edge.weight);
            }

            if edge.weight.abs() > 1e6 {
                warn!("Very large edge weight detected: {} at edge {}", edge.weight, idx);
            }

            // Edge type validation if present
            if let Some(ref edge_type) = edge.edge_type {
                self.security_validator.validate_string(edge_type, &format!("edge_{}_type", idx))?;
            }
        }

        Ok(())
    }

    fn validate_graph_structure(&self, graph: &Graph) -> Result<()> {
        // Check graph connectivity (warning only)
        let node_count = graph.nodes.len();
        let edge_count = graph.edges.len();
        
        if edge_count == 0 && node_count > 1 {
            warn!("Graph has {} nodes but no edges - all nodes are isolated", node_count);
        }

        // Density check
        let max_edges = node_count * (node_count - 1);
        if max_edges > 0 {
            let density = edge_count as f64 / max_edges as f64;
            if density > 0.5 {
                warn!("Very dense graph detected: density = {:.3}", density);
            }
        }

        // Degree distribution analysis
        let mut in_degrees = vec![0usize; node_count];
        let mut out_degrees = vec![0usize; node_count];
        let node_id_to_idx: std::collections::HashMap<u32, usize> = 
            graph.nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();

        for edge in &graph.edges {
            if let (Some(&src_idx), Some(&tgt_idx)) = 
                (node_id_to_idx.get(&edge.source), node_id_to_idx.get(&edge.target)) {
                out_degrees[src_idx] += 1;
                in_degrees[tgt_idx] += 1;
            }
        }

        let max_in_degree = *in_degrees.iter().max().unwrap_or(&0);
        let max_out_degree = *out_degrees.iter().max().unwrap_or(&0);

        if max_in_degree > node_count / 2 {
            warn!("High fan-in detected: max in-degree = {}", max_in_degree);
        }
        
        if max_out_degree > node_count / 2 {
            warn!("High fan-out detected: max out-degree = {}", max_out_degree);
        }

        Ok(())
    }

    fn estimate_graph_memory_size(&self, graph: &Graph) -> usize {
        let node_size = std::mem::size_of::<Node>() + 
                       graph.nodes.first().map_or(0, |n| n.features.len() * 4);
        let edge_size = std::mem::size_of::<Edge>();
        
        graph.nodes.len() * node_size + graph.edges.len() * edge_size
    }
}

pub struct ProcessingConfigValidator;

impl ProcessingConfigValidator {
    pub fn validate(&self, config: &ProcessingConfig) -> Result<()> {
        if config.temperature < MIN_TEMPERATURE || config.temperature > MAX_TEMPERATURE {
            return Err(crate::error::Error::validation(
                "Invalid temperature value",
                format!("{}", config.temperature),
                format!("{} <= temperature <= {}", MIN_TEMPERATURE, MAX_TEMPERATURE)
            ));
        }

        if config.dropout_rate < 0.0 || config.dropout_rate > MAX_DROPOUT {
            return Err(crate::error::Error::validation(
                "Invalid dropout rate",
                format!("{}", config.dropout_rate),
                format!("0.0 <= dropout <= {}", MAX_DROPOUT)
            ));
        }

        Ok(())
    }
}

// Rate limiting validation
pub struct RateLimitValidator {
    max_requests_per_second: f64,
    max_requests_per_minute: f64,
    max_requests_per_hour: f64,
}

impl Default for RateLimitValidator {
    fn default() -> Self {
        Self {
            max_requests_per_second: 10.0,
            max_requests_per_minute: 300.0,
            max_requests_per_hour: 10000.0,
        }
    }
}

impl RateLimitValidator {
    pub fn validate_rate(&self, requests_per_second: f64) -> Result<()> {
        if requests_per_second > self.max_requests_per_second {
            return Err(crate::error::Error::RateLimit {
                limit: self.max_requests_per_second as u32,
                window: "second".to_string(),
                current: requests_per_second as u32,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Node, Edge, Graph};

    #[test]
    fn test_valid_graph() {
        let validator = GraphValidator::new();
        let graph = Graph {
            nodes: vec![
                Node { id: 1, features: vec![1.0, 2.0], label: None },
                Node { id: 2, features: vec![3.0, 4.0], label: None },
            ],
            edges: vec![
                Edge { source: 1, target: 2, weight: 0.5, edge_type: None },
            ],
            metadata: std::collections::HashMap::new(),
        };
        
        assert!(validator.validate_graph(&graph).is_ok());
    }

    #[test]
    fn test_malicious_string() {
        let validator = SecurityValidator::new();
        let malicious_input = "<script>alert('xss')</script>";
        assert!(validator.validate_string(malicious_input, "test").is_err());
    }

    #[test]
    fn test_processing_config_validation() {
        let validator = ProcessingConfigValidator;
        
        let valid_config = ProcessingConfig {
            temperature: 0.1,
            dropout_rate: 0.1,
            use_attention: false,
            normalize_features: true,
            mixed_precision: false,
        };
        assert!(validator.validate(&valid_config).is_ok());
        
        let invalid_config = ProcessingConfig {
            temperature: -1.0,  // Invalid
            dropout_rate: 0.1,
            use_attention: false,
            normalize_features: true,
            mixed_precision: false,
        };
        assert!(validator.validate(&invalid_config).is_err());
    }
}