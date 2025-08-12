//! Resilience patterns: Circuit breakers, retries, bulkheads, and failsafes

use crate::{Result, error::Error};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tracing::{warn, info, debug};

/// Enhanced circuit breaker with multiple failure types and adaptive thresholds
#[derive(Debug)]
pub struct AdaptiveCircuitBreaker {
    name: String,
    failure_threshold: AtomicU64,
    success_threshold: AtomicU64,
    timeout: Duration,
    
    // State tracking
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_failure_time: Arc<parking_lot::RwLock<Option<Instant>>>,
    state: Arc<parking_lot::RwLock<CircuitState>>,
    
    // Adaptive parameters
    base_failure_threshold: u64,
    max_failure_threshold: u64,
    adaptation_factor: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,    // Normal operation
    Open,      // Failing fast
    HalfOpen,  // Testing recovery
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub name: String,
    pub failure_threshold: u64,
    pub success_threshold: u64,
    pub timeout: Duration,
    pub adaptation_factor: f64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            failure_threshold: 5,
            success_threshold: 3,
            timeout: Duration::from_secs(60),
            adaptation_factor: 1.5,
        }
    }
}

impl AdaptiveCircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        let max_failure_threshold = (config.failure_threshold as f64 * 5.0) as u64;
        
        Self {
            name: config.name,
            failure_threshold: AtomicU64::new(config.failure_threshold),
            success_threshold: AtomicU64::new(config.success_threshold),
            timeout: config.timeout,
            
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            last_failure_time: Arc::new(parking_lot::RwLock::new(None)),
            state: Arc::new(parking_lot::RwLock::new(CircuitState::Closed)),
            
            base_failure_threshold: config.failure_threshold,
            max_failure_threshold,
            adaptation_factor: config.adaptation_factor,
        }
    }
    
    pub async fn call<F, T, E>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = std::result::Result<T, E>> + Send,
        E: Into<Error> + std::fmt::Debug,
    {
        // Check circuit state
        if self.should_reject_request().await {
            return Err(Error::dependency_failure(
                &self.name,
                "Circuit breaker is OPEN"
            ));
        }
        
        // Execute operation
        match operation.await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(error) => {
                let error = error.into();
                self.on_failure(&error).await;
                Err(error)
            }
        }
    }
    
    async fn should_reject_request(&self) -> bool {
        let state = self.state.read().clone();
        
        match state {
            CircuitState::Closed => false,
            CircuitState::Open => {
                // Check if timeout has passed
                let last_failure = self.last_failure_time.read();
                if let Some(last_failure) = *last_failure {
                    if last_failure.elapsed() >= self.timeout {
                        // Transition to half-open
                        drop(last_failure);
                        *self.state.write() = CircuitState::HalfOpen;
                        info!("Circuit breaker {} transitioning to HALF_OPEN", self.name);
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            }
            CircuitState::HalfOpen => false, // Allow limited requests
        }
    }
    
    async fn on_success(&self) {
        let current_state = self.state.read().clone();
        
        match current_state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                let threshold = self.success_threshold.load(Ordering::Relaxed);
                
                if success_count >= threshold {
                    // Close the circuit
                    *self.state.write() = CircuitState::Closed;
                    self.success_count.store(0, Ordering::Relaxed);
                    self.failure_count.store(0, Ordering::Relaxed);
                    
                    // Adapt thresholds back towards baseline
                    self.adapt_thresholds_on_recovery();
                    
                    info!("Circuit breaker {} is now CLOSED", self.name);
                }
            }
            CircuitState::Open => {
                // This shouldn't happen, but reset success count
                self.success_count.store(0, Ordering::Relaxed);
            }
        }
    }
    
    async fn on_failure(&self, error: &Error) {
        let severity = self.assess_failure_severity(error);
        let failure_weight = match severity {
            FailureSeverity::Low => 1,
            FailureSeverity::Medium => 2,
            FailureSeverity::High => 3,
            FailureSeverity::Critical => 5,
        };
        
        let failure_count = self.failure_count.fetch_add(failure_weight, Ordering::Relaxed) + failure_weight;
        let threshold = self.failure_threshold.load(Ordering::Relaxed);
        
        *self.last_failure_time.write() = Some(Instant::now());
        
        let current_state = self.state.read().clone();
        
        match current_state {
            CircuitState::Closed | CircuitState::HalfOpen => {
                if failure_count >= threshold {
                    // Open the circuit
                    *self.state.write() = CircuitState::Open;
                    self.success_count.store(0, Ordering::Relaxed);
                    
                    // Adapt thresholds based on failure pattern
                    self.adapt_thresholds_on_failure(severity);
                    
                    warn!("Circuit breaker {} is now OPEN after {} failures", 
                        self.name, failure_count);
                }
            }
            CircuitState::Open => {
                // Already open, just log
                debug!("Additional failure in OPEN circuit: {}", self.name);
            }
        }
    }
    
    fn assess_failure_severity(&self, error: &Error) -> FailureSeverity {
        match error {
            Error::Network { .. } => FailureSeverity::Medium,
            Error::Timeout { .. } => FailureSeverity::Medium,
            Error::ResourceExhausted { .. } => FailureSeverity::High,
            Error::Security { .. } => FailureSeverity::Critical,
            Error::DataCorruption { .. } => FailureSeverity::Critical,
            Error::DependencyFailure { .. } => FailureSeverity::High,
            _ => FailureSeverity::Low,
        }
    }
    
    fn adapt_thresholds_on_failure(&self, severity: FailureSeverity) {
        let current_threshold = self.failure_threshold.load(Ordering::Relaxed);
        let adaptation_factor = match severity {
            FailureSeverity::Critical => self.adaptation_factor * 2.0,
            FailureSeverity::High => self.adaptation_factor * 1.5,
            _ => self.adaptation_factor,
        };
        
        let new_threshold = ((current_threshold as f64) * adaptation_factor) as u64;
        let capped_threshold = new_threshold.min(self.max_failure_threshold);
        
        self.failure_threshold.store(capped_threshold, Ordering::Relaxed);
        
        info!("Adapted failure threshold for {} from {} to {}", 
            self.name, current_threshold, capped_threshold);
    }
    
    fn adapt_thresholds_on_recovery(&self) {
        let current_threshold = self.failure_threshold.load(Ordering::Relaxed);
        let recovery_factor = 0.8; // Gradually return to baseline
        
        let new_threshold = ((current_threshold as f64) * recovery_factor) as u64;
        let final_threshold = new_threshold.max(self.base_failure_threshold);
        
        self.failure_threshold.store(final_threshold, Ordering::Relaxed);
        
        debug!("Recovered failure threshold for {} from {} to {}", 
            self.name, current_threshold, final_threshold);
    }
    
    pub fn get_state(&self) -> CircuitState {
        self.state.read().clone()
    }
    
    pub fn get_metrics(&self) -> CircuitBreakerMetrics {
        CircuitBreakerMetrics {
            name: self.name.clone(),
            state: self.get_state(),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            current_failure_threshold: self.failure_threshold.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
enum FailureSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerMetrics {
    pub name: String,
    pub state: CircuitState,
    pub failure_count: u64,
    pub success_count: u64,
    pub current_failure_threshold: u64,
}

/// Enhanced retry mechanism with exponential backoff and jitter
pub struct SmartRetryPolicy {
    max_attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
    backoff_multiplier: f64,
    jitter_factor: f64,
}

impl Default for SmartRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}

impl SmartRetryPolicy {
    pub fn new(max_attempts: u32, base_delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay,
            ..Default::default()
        }
    }
    
    pub async fn execute<F, T, E>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = std::result::Result<T, E>> + Send>>,
        E: Into<Error> + std::fmt::Debug,
    {
        let mut last_error = None;
        
        for attempt in 0..self.max_attempts {
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Operation succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(error) => {
                    let error = error.into();
                    
                    // Check if error is retryable
                    if !self.is_retryable(&error) {
                        debug!("Non-retryable error encountered: {:?}", error);
                        return Err(error);
                    }
                    
                    last_error = Some(error);
                    
                    // Don't sleep after the last attempt
                    if attempt < self.max_attempts - 1 {
                        let delay = self.calculate_delay(attempt);
                        debug!("Retrying after {:.2}s (attempt {}/{})", 
                            delay.as_secs_f64(), attempt + 1, self.max_attempts);
                        sleep(delay).await;
                    }
                }
            }
        }
        
        warn!("All {} retry attempts failed", self.max_attempts);
        Err(last_error.unwrap())
    }
    
    fn is_retryable(&self, error: &Error) -> bool {
        match error {
            Error::Network { .. } => true,
            Error::Timeout { .. } => true,
            Error::ResourceExhausted { .. } => true,
            Error::DependencyFailure { .. } => true,
            Error::TpuRuntime { .. } => true,
            Error::Security { .. } => false,
            Error::Validation { .. } => false,
            Error::Configuration { .. } => false,
            Error::DataCorruption { .. } => false,
            _ => false,
        }
    }
    
    fn calculate_delay(&self, attempt: u32) -> Duration {
        let exponential_delay = self.base_delay.as_secs_f64() * 
            self.backoff_multiplier.powi(attempt as i32);
        
        // Add jitter to prevent thundering herd
        let jitter = 1.0 + (rand::random::<f64>() - 0.5) * 2.0 * self.jitter_factor;
        let final_delay = exponential_delay * jitter;
        
        Duration::from_secs_f64(final_delay.min(self.max_delay.as_secs_f64()))
    }
}

/// Bulkhead pattern for resource isolation
pub struct ResourceBulkhead {
    name: String,
    permits: Arc<tokio::sync::Semaphore>,
    max_permits: usize,
    active_requests: AtomicU64,
    timeout: Duration,
}

impl ResourceBulkhead {
    pub fn new(name: String, max_concurrent: usize, timeout: Duration) -> Self {
        Self {
            name,
            permits: Arc::new(tokio::sync::Semaphore::new(max_concurrent)),
            max_permits: max_concurrent,
            active_requests: AtomicU64::new(0),
            timeout,
        }
    }
    
    pub async fn execute<F, T>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = Result<T>> + Send,
    {
        // Try to acquire permit with timeout
        let permit = match tokio::time::timeout(self.timeout, self.permits.acquire()).await {
            Ok(Ok(permit)) => permit,
            Ok(Err(_)) => {
                return Err(Error::resource_exhausted(
                    "semaphore", 
                    self.max_permits, 
                    self.max_permits
                ));
            }
            Err(_) => {
                return Err(Error::timeout(
                    format!("acquiring {} bulkhead permit", self.name),
                    self.timeout.as_millis() as u64,
                    self.timeout.as_millis() as u64
                ));
            }
        };
        
        // Track active request
        let active = self.active_requests.fetch_add(1, Ordering::Relaxed) + 1;
        debug!("Bulkhead {}: {} active requests", self.name, active);
        
        // Execute operation
        let result = operation.await;
        
        // Clean up
        self.active_requests.fetch_sub(1, Ordering::Relaxed);
        drop(permit);
        
        result
    }
    
    pub fn get_metrics(&self) -> BulkheadMetrics {
        BulkheadMetrics {
            name: self.name.clone(),
            max_permits: self.max_permits,
            available_permits: self.permits.available_permits(),
            active_requests: self.active_requests.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BulkheadMetrics {
    pub name: String,
    pub max_permits: usize,
    pub available_permits: usize,
    pub active_requests: u64,
}

/// Comprehensive health monitoring
pub struct HealthMonitor {
    checks: Vec<Box<dyn HealthCheck + Send + Sync>>,
    overall_timeout: Duration,
}

#[async_trait::async_trait]
pub trait HealthCheck: Send + Sync {
    async fn check(&self) -> HealthCheckResult;
    fn name(&self) -> &str;
}

#[derive(Debug, Clone)]
pub struct HealthCheckResult {
    pub name: String,
    pub status: HealthStatus,
    pub message: String,
    pub details: std::collections::HashMap<String, String>,
    pub response_time: Duration,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl HealthMonitor {
    pub fn new(timeout: Duration) -> Self {
        Self {
            checks: Vec::new(),
            overall_timeout: timeout,
        }
    }
    
    pub fn add_check<T: HealthCheck + Send + Sync + 'static>(&mut self, check: T) {
        self.checks.push(Box::new(check));
    }
    
    pub async fn check_health(&self) -> HealthReport {
        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut overall_status = HealthStatus::Healthy;
        
        // Run all health checks concurrently
        let futures: Vec<_> = self.checks.iter().map(|check| {
            let timeout = self.overall_timeout;
            async move {
                tokio::time::timeout(timeout, check.check()).await
                    .unwrap_or_else(|_| HealthCheckResult {
                        name: check.name().to_string(),
                        status: HealthStatus::Unhealthy,
                        message: "Health check timed out".to_string(),
                        details: std::collections::HashMap::new(),
                        response_time: timeout,
                    })
            }
        }).collect();
        
        results = futures::future::join_all(futures).await;
        
        // Determine overall status
        for result in &results {
            match result.status {
                HealthStatus::Unhealthy => {
                    overall_status = HealthStatus::Unhealthy;
                    break;
                }
                HealthStatus::Degraded if overall_status == HealthStatus::Healthy => {
                    overall_status = HealthStatus::Degraded;
                }
                _ => {}
            }
        }
        
        HealthReport {
            status: overall_status,
            checks: results,
            total_response_time: start_time.elapsed(),
            timestamp: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HealthReport {
    pub status: HealthStatus,
    pub checks: Vec<HealthCheckResult>,
    pub total_response_time: Duration,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// Example health checks
pub struct GraphProcessingHealthCheck;

#[async_trait::async_trait]
impl HealthCheck for GraphProcessingHealthCheck {
    async fn check(&self) -> HealthCheckResult {
        let start = Instant::now();
        
        // Simulate a basic graph processing test
        let result = tokio::time::timeout(Duration::from_secs(5), async {
            // Create a minimal test graph
            use crate::core::{Graph, Node, Edge, DGDMProcessor, GraphConfig, ProcessingConfig};
            
            let mut graph = Graph::new();
            graph.add_node(Node { id: 1, features: vec![1.0], label: None });
            graph.add_edge(Edge { source: 1, target: 1, weight: 1.0, edge_type: None });
            
            let compact = graph.to_compact()?;
            
            let mut config = GraphConfig::default();
            config.feature_dim = 1;
            let processing = ProcessingConfig::default();
            
            let processor = DGDMProcessor::new(config, processing);
            processor.process(&compact)
        }).await;
        
        let response_time = start.elapsed();
        let mut details = std::collections::HashMap::new();
        details.insert("response_time_ms".to_string(), response_time.as_millis().to_string());
        
        match result {
            Ok(Ok(_)) => HealthCheckResult {
                name: "graph_processing".to_string(),
                status: HealthStatus::Healthy,
                message: "Graph processing is operational".to_string(),
                details,
                response_time,
            },
            Ok(Err(e)) => HealthCheckResult {
                name: "graph_processing".to_string(),
                status: HealthStatus::Unhealthy,
                message: format!("Graph processing failed: {}", e),
                details,
                response_time,
            },
            Err(_) => HealthCheckResult {
                name: "graph_processing".to_string(),
                status: HealthStatus::Unhealthy,
                message: "Graph processing health check timed out".to_string(),
                details,
                response_time,
            },
        }
    }
    
    fn name(&self) -> &str {
        "graph_processing"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;
    
    #[tokio::test]
    async fn test_circuit_breaker_normal_operation() {
        let config = CircuitBreakerConfig {
            name: "test".to_string(),
            failure_threshold: 3,
            success_threshold: 2,
            timeout: Duration::from_millis(100),
            adaptation_factor: 1.5,
        };
        
        let breaker = AdaptiveCircuitBreaker::new(config);
        
        // Successful operation
        let result = breaker.call(async { Ok::<i32, Error>(42) }).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(breaker.get_state(), CircuitState::Closed);
    }
    
    #[tokio::test]
    async fn test_retry_policy_success_after_failure() {
        let policy = SmartRetryPolicy::new(3, Duration::from_millis(10));
        let mut attempt_count = 0;
        
        let result = policy.execute(|| {
            attempt_count += 1;
            Box::pin(async move {
                if attempt_count < 2 {
                    Err(Error::network("Connection refused", "localhost:8080"))
                } else {
                    Ok(42)
                }
            })
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
        assert_eq!(attempt_count, 2);
    }
    
    #[tokio::test]
    async fn test_bulkhead_permits() {
        let bulkhead = ResourceBulkhead::new(
            "test".to_string(),
            2,
            Duration::from_secs(1)
        );
        
        // Should succeed within permit limit
        let result = bulkhead.execute(async { Ok::<i32, Error>(42) }).await;
        assert!(result.is_ok());
        
        let metrics = bulkhead.get_metrics();
        assert_eq!(metrics.max_permits, 2);
        assert_eq!(metrics.available_permits, 2);
    }
}