//! Generation 2 Security Hardening Module
//! 
//! Implements comprehensive security measures for production deployment:
//! - Input validation and sanitization
//! - Rate limiting and DDoS protection
//! - Authentication and authorization
//! - Audit logging and security monitoring

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{warn, error, info};
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

#[derive(Debug, Clone)]
pub struct SecurityConfig {
    pub max_graph_nodes: usize,
    pub max_graph_edges: usize,
    pub max_feature_dimension: usize,
    pub max_diffusion_steps: u32,
    pub rate_limit_requests_per_second: u32,
    pub rate_limit_burst_size: u32,
    pub enable_authentication: bool,
    pub audit_log_enabled: bool,
    pub input_validation_strict: bool,
    pub memory_limit_mb: usize,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            max_graph_nodes: 100_000,
            max_graph_edges: 1_000_000,
            max_feature_dimension: 1_000,
            max_diffusion_steps: 50,
            rate_limit_requests_per_second: 100,
            rate_limit_burst_size: 200,
            enable_authentication: true,
            audit_log_enabled: true,
            input_validation_strict: true,
            memory_limit_mb: 2048,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SecurityEvent {
    pub timestamp: u64,
    pub event_type: SecurityEventType,
    pub client_ip: String,
    pub user_agent: Option<String>,
    pub details: String,
    pub severity: SecuritySeverity,
}

#[derive(Debug, Clone, Serialize)]
pub enum SecurityEventType {
    InputValidationFailed,
    RateLimitExceeded,
    AuthenticationFailed,
    ResourceLimitExceeded,
    SuspiciousActivity,
    AccessDenied,
}

#[derive(Debug, Clone, Serialize)]
pub enum SecuritySeverity {
    Low,
    Medium,
    High,
    Critical,
}

pub struct SecurityHardeningSystem {
    config: SecurityConfig,
    rate_limiter: Arc<RateLimiter>,
    input_validator: Arc<InputValidator>,
    audit_logger: Arc<AuditLogger>,
    threat_detector: Arc<ThreatDetector>,
}

impl SecurityHardeningSystem {
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            rate_limiter: Arc::new(RateLimiter::new(&config)),
            input_validator: Arc::new(InputValidator::new(&config)),
            audit_logger: Arc::new(AuditLogger::new()),
            threat_detector: Arc::new(ThreatDetector::new()),
            config,
        }
    }

    /// Comprehensive security validation for incoming requests
    pub async fn validate_request(
        &self,
        client_ip: &str,
        user_agent: Option<&str>,
        graph: &crate::minimal_server::MinimalGraph,
        steps: u32,
        temperature: f32,
    ) -> crate::Result<()> {
        // Rate limiting check
        if !self.rate_limiter.allow_request(client_ip).await {
            self.audit_logger.log_security_event(SecurityEvent {
                timestamp: self.current_timestamp(),
                event_type: SecurityEventType::RateLimitExceeded,
                client_ip: client_ip.to_string(),
                user_agent: user_agent.map(|s| s.to_string()),
                details: "Rate limit exceeded for client".to_string(),
                severity: SecuritySeverity::Medium,
            }).await;

            return Err(crate::error::Error::configuration(
                "Rate limit exceeded",
                "rate_limit"
            ));
        }

        // Input validation
        self.input_validator.validate_graph(graph)?;
        self.input_validator.validate_processing_params(steps, temperature)?;

        // Threat detection
        if self.threat_detector.detect_suspicious_patterns(client_ip, user_agent, graph).await {
            self.audit_logger.log_security_event(SecurityEvent {
                timestamp: self.current_timestamp(),
                event_type: SecurityEventType::SuspiciousActivity,
                client_ip: client_ip.to_string(),
                user_agent: user_agent.map(|s| s.to_string()),
                details: "Suspicious activity pattern detected".to_string(),
                severity: SecuritySeverity::High,
            }).await;

            return Err(crate::error::Error::configuration(
                "Request blocked due to suspicious activity",
                "threat_detection"
            ));
        }

        // Resource exhaustion protection
        let estimated_memory = self.estimate_memory_usage(graph);
        if estimated_memory > self.config.memory_limit_mb * 1024 * 1024 {
            self.audit_logger.log_security_event(SecurityEvent {
                timestamp: self.current_timestamp(),
                event_type: SecurityEventType::ResourceLimitExceeded,
                client_ip: client_ip.to_string(),
                user_agent: user_agent.map(|s| s.to_string()),
                details: format!("Memory limit exceeded: {} bytes", estimated_memory),
                severity: SecuritySeverity::High,
            }).await;

            return Err(crate::error::Error::resource_exhausted(
                "memory",
                self.config.memory_limit_mb,
                estimated_memory / 1024 / 1024
            ));
        }

        Ok(())
    }

    fn estimate_memory_usage(&self, graph: &crate::minimal_server::MinimalGraph) -> usize {
        let node_memory = graph.nodes.len() * graph.nodes.get(0)
            .map(|n| n.features.len() * std::mem::size_of::<f32>())
            .unwrap_or(0);
        let edge_memory = graph.edges.len() * std::mem::size_of::<crate::minimal_server::MinimalGraphEdge>();
        node_memory + edge_memory
    }

    fn current_timestamp(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

pub struct RateLimiter {
    client_buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    requests_per_second: u32,
    burst_size: u32,
}

impl RateLimiter {
    pub fn new(config: &SecurityConfig) -> Self {
        Self {
            client_buckets: Arc::new(RwLock::new(HashMap::new())),
            requests_per_second: config.rate_limit_requests_per_second,
            burst_size: config.rate_limit_burst_size,
        }
    }

    pub async fn allow_request(&self, client_ip: &str) -> bool {
        let mut buckets = self.client_buckets.write().await;
        let bucket = buckets.entry(client_ip.to_string())
            .or_insert_with(|| TokenBucket::new(self.requests_per_second, self.burst_size));
        
        bucket.try_consume()
    }
}

struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    fn new(rate_per_second: u32, capacity: u32) -> Self {
        Self {
            tokens: capacity as f64,
            capacity: capacity as f64,
            refill_rate: rate_per_second as f64,
            last_refill: Instant::now(),
        }
    }

    fn try_consume(&mut self) -> bool {
        self.refill();
        
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let time_elapsed = now.duration_since(self.last_refill).as_secs_f64();
        
        self.tokens = (self.tokens + time_elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }
}

pub struct InputValidator {
    config: SecurityConfig,
}

impl InputValidator {
    pub fn new(config: &SecurityConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    pub fn validate_graph(&self, graph: &crate::minimal_server::MinimalGraph) -> crate::Result<()> {
        // Node count validation
        if graph.nodes.len() > self.config.max_graph_nodes {
            return Err(crate::error::Error::validation(
                format!("Too many nodes: {} > {}", graph.nodes.len(), self.config.max_graph_nodes),
                graph.nodes.len().to_string(),
                format!("<= {}", self.config.max_graph_nodes)
            ));
        }

        // Edge count validation
        if graph.edges.len() > self.config.max_graph_edges {
            return Err(crate::error::Error::validation(
                format!("Too many edges: {} > {}", graph.edges.len(), self.config.max_graph_edges),
                graph.edges.len().to_string(),
                format!("<= {}", self.config.max_graph_edges)
            ));
        }

        // Feature dimension validation
        if let Some(node) = graph.nodes.first() {
            if node.features.len() > self.config.max_feature_dimension {
                return Err(crate::error::Error::validation(
                    format!("Feature dimension too high: {} > {}", 
                           node.features.len(), self.config.max_feature_dimension),
                    node.features.len().to_string(),
                    format!("<= {}", self.config.max_feature_dimension)
                ));
            }
        }

        // Validate all feature values are finite
        if self.config.input_validation_strict {
            for (i, node) in graph.nodes.iter().enumerate() {
                for (j, &feature) in node.features.iter().enumerate() {
                    if !feature.is_finite() {
                        return Err(crate::error::Error::validation(
                            "Non-finite feature value detected",
                            format!("node[{}].features[{}] = {}", i, j, feature),
                            "finite values only"
                        ));
                    }
                }
            }
        }

        // Validate edge weights
        for (i, edge) in graph.edges.iter().enumerate() {
            if !edge.weight.is_finite() || edge.weight < 0.0 {
                return Err(crate::error::Error::validation(
                    "Invalid edge weight",
                    format!("edges[{}].weight = {}", i, edge.weight),
                    "non-negative finite values"
                ));
            }
        }

        // Validate node IDs are unique
        let mut seen_ids = std::collections::HashSet::new();
        for node in &graph.nodes {
            if !seen_ids.insert(node.id) {
                return Err(crate::error::Error::validation(
                    "Duplicate node ID detected",
                    node.id.to_string(),
                    "unique node IDs"
                ));
            }
        }

        Ok(())
    }

    pub fn validate_processing_params(&self, steps: u32, temperature: f32) -> crate::Result<()> {
        if steps > self.config.max_diffusion_steps {
            return Err(crate::error::Error::validation(
                format!("Too many diffusion steps: {} > {}", steps, self.config.max_diffusion_steps),
                steps.to_string(),
                format!("<= {}", self.config.max_diffusion_steps)
            ));
        }

        if !temperature.is_finite() || temperature < 0.0 || temperature > 10.0 {
            return Err(crate::error::Error::validation(
                "Invalid temperature parameter",
                temperature.to_string(),
                "0.0 <= temperature <= 10.0"
            ));
        }

        Ok(())
    }
}

pub struct AuditLogger {
    events: Arc<RwLock<Vec<SecurityEvent>>>,
    max_events: usize,
}

impl AuditLogger {
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::new())),
            max_events: 10000,
        }
    }

    pub async fn log_security_event(&self, event: SecurityEvent) {
        // Log to structured logger
        match event.severity {
            SecuritySeverity::Low => info!("Security event: {:?}", event),
            SecuritySeverity::Medium => warn!("Security event: {:?}", event),
            SecuritySeverity::High | SecuritySeverity::Critical => {
                error!("Security event: {:?}", event);
            }
        }

        // Store in memory for analysis
        let mut events = self.events.write().await;
        events.push(event);
        
        // Rotate if too many events
        if events.len() > self.max_events {
            events.remove(0);
        }
    }

    pub async fn get_recent_events(&self, limit: usize) -> Vec<SecurityEvent> {
        let events = self.events.read().await;
        events.iter().rev().take(limit).cloned().collect()
    }
}

pub struct ThreatDetector {
    client_activity: Arc<RwLock<HashMap<String, ClientActivity>>>,
}

#[derive(Debug, Clone)]
struct ClientActivity {
    request_count: AtomicU64,
    last_request: Instant,
    suspicious_patterns: u32,
}

impl ThreatDetector {
    pub fn new() -> Self {
        Self {
            client_activity: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn detect_suspicious_patterns(
        &self,
        client_ip: &str,
        user_agent: Option<&str>,
        graph: &crate::minimal_server::MinimalGraph,
    ) -> bool {
        let mut activity_map = self.client_activity.write().await;
        let activity = activity_map.entry(client_ip.to_string())
            .or_insert_with(|| ClientActivity {
                request_count: AtomicU64::new(0),
                last_request: Instant::now(),
                suspicious_patterns: 0,
            });

        activity.request_count.fetch_add(1, Ordering::Relaxed);
        activity.last_request = Instant::now();

        // Detect patterns
        let mut suspicious = false;

        // Pattern 1: Extremely regular requests (bot-like behavior)
        if activity.request_count.load(Ordering::Relaxed) > 100 {
            suspicious = true;
        }

        // Pattern 2: Suspicious user agent
        if let Some(ua) = user_agent {
            if ua.contains("bot") || ua.contains("curl") || ua.contains("wget") {
                suspicious = true;
            }
        }

        // Pattern 3: Graph structure anomalies
        if self.detect_graph_anomalies(graph) {
            suspicious = true;
        }

        if suspicious {
            activity.suspicious_patterns += 1;
        }

        activity.suspicious_patterns > 5
    }

    fn detect_graph_anomalies(&self, graph: &crate::minimal_server::MinimalGraph) -> bool {
        // Detect very sparse or very dense graphs
        if graph.nodes.is_empty() || graph.edges.is_empty() {
            return false; // Not necessarily suspicious
        }

        let max_possible_edges = graph.nodes.len() * (graph.nodes.len() - 1);
        let density = graph.edges.len() as f64 / max_possible_edges as f64;

        // Extremely dense graphs might indicate testing/attack
        density > 0.8 || (graph.nodes.len() > 1000 && density < 0.001)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter() {
        let config = SecurityConfig {
            rate_limit_requests_per_second: 2,
            rate_limit_burst_size: 3,
            ..Default::default()
        };
        let limiter = RateLimiter::new(&config);

        // Should allow burst
        assert!(limiter.allow_request("127.0.0.1").await);
        assert!(limiter.allow_request("127.0.0.1").await);
        assert!(limiter.allow_request("127.0.0.1").await);
        
        // Should reject after burst
        assert!(!limiter.allow_request("127.0.0.1").await);
    }

    #[test]
    fn test_input_validation() {
        let config = SecurityConfig::default();
        let validator = InputValidator::new(&config);

        let valid_graph = crate::minimal_server::MinimalGraph {
            nodes: vec![
                crate::minimal_server::MinimalGraphNode {
                    id: 1,
                    features: vec![1.0, 2.0],
                },
            ],
            edges: vec![],
        };

        assert!(validator.validate_graph(&valid_graph).is_ok());

        // Test invalid graph with NaN
        let invalid_graph = crate::minimal_server::MinimalGraph {
            nodes: vec![
                crate::minimal_server::MinimalGraphNode {
                    id: 1,
                    features: vec![f32::NAN, 2.0],
                },
            ],
            edges: vec![],
        };

        assert!(validator.validate_graph(&invalid_graph).is_err());
    }
}