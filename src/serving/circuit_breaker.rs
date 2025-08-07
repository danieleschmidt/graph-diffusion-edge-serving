//! Circuit breaker pattern for fault tolerance and graceful degradation

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Circuit open, failing fast
    HalfOpen, // Testing if service recovered
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub success_threshold: usize,
    pub timeout_duration: Duration,
    pub max_requests_half_open: usize,
    pub slow_call_threshold: Duration,
    pub slow_call_rate_threshold: f64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_duration: Duration::from_secs(60),
            max_requests_half_open: 3,
            slow_call_threshold: Duration::from_secs(5),
            slow_call_rate_threshold: 0.5, // 50%
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CircuitBreakerMetrics {
    pub state: CircuitState,
    pub failure_count: usize,
    pub success_count: usize,
    pub total_requests: u64,
    pub failed_requests: u64,
    pub slow_requests: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_failure_time: Option<u64>, // Unix timestamp in seconds
    pub uptime_seconds: u64,
    pub failure_rate: f64,
    pub slow_call_rate: f64,
}

pub struct CircuitBreaker {
    name: String,
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    failure_count: AtomicUsize,
    success_count: AtomicUsize,
    total_requests: AtomicU64,
    failed_requests: AtomicU64,
    slow_requests: AtomicU64,
    last_failure_time: Arc<RwLock<Option<u64>>>,
    half_open_requests: AtomicUsize,
    creation_time: Instant,
}

impl CircuitBreaker {
    pub fn new<S: Into<String>>(name: S, config: CircuitBreakerConfig) -> Self {
        let name_str = name.into();
        info!("Creating circuit breaker '{}' with config: {:?}", name_str, config);
        
        Self {
            name: name_str,
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: AtomicUsize::new(0),
            success_count: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
            slow_requests: AtomicU64::new(0),
            last_failure_time: Arc::new(RwLock::new(None)),
            half_open_requests: AtomicUsize::new(0),
            creation_time: Instant::now(),
        }
    }

    pub async fn call<F, T, E>(&self, operation: F) -> Result<T, crate::error::Error>
    where
        F: std::future::Future<Output = Result<T, E>>,
        E: std::fmt::Display + std::fmt::Debug,
    {
        let state = *self.state.read().await;
        
        match state {
            CircuitState::Open => {
                if self.should_attempt_reset().await {
                    debug!("Circuit breaker '{}' transitioning from Open to HalfOpen", self.name);
                    *self.state.write().await = CircuitState::HalfOpen;
                    self.half_open_requests.store(0, Ordering::Relaxed);
                } else {
                    return Err(crate::error::Error::dependency_failure(
                        &self.name,
                        "Circuit breaker is open"
                    ));
                }
            }
            CircuitState::HalfOpen => {
                let current_half_open = self.half_open_requests.load(Ordering::Relaxed);
                if current_half_open >= self.config.max_requests_half_open {
                    return Err(crate::error::Error::dependency_failure(
                        &self.name,
                        "Circuit breaker half-open request limit exceeded"
                    ));
                }
                self.half_open_requests.fetch_add(1, Ordering::Relaxed);
            }
            CircuitState::Closed => {
                // Normal operation
            }
        }

        let start_time = Instant::now();
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        let result = operation.await;
        let duration = start_time.elapsed();

        match result {
            Ok(value) => {
                self.on_success(duration).await;
                Ok(value)
            }
            Err(error) => {
                self.on_failure(duration).await;
                Err(crate::error::Error::dependency_failure(
                    &self.name,
                    format!("Operation failed: {}", error)
                ))
            }
        }
    }

    async fn on_success(&self, duration: Duration) {
        let is_slow = duration >= self.config.slow_call_threshold;
        
        if is_slow {
            self.slow_requests.fetch_add(1, Ordering::Relaxed);
            debug!(
                "Circuit breaker '{}': Slow call detected ({}ms > {}ms)",
                self.name,
                duration.as_millis(),
                self.config.slow_call_threshold.as_millis()
            );
        }

        let state = *self.state.read().await;
        
        match state {
            CircuitState::HalfOpen => {
                let success_count = self.success_count.fetch_add(1, Ordering::Relaxed) + 1;
                
                if success_count >= self.config.success_threshold {
                    info!("Circuit breaker '{}' transitioning from HalfOpen to Closed", self.name);
                    *self.state.write().await = CircuitState::Closed;
                    self.reset_counters();
                }
            }
            CircuitState::Closed => {
                // Reset failure count on successful call
                self.failure_count.store(0, Ordering::Relaxed);
            }
            CircuitState::Open => {
                // This shouldn't happen if circuit is properly managed
                warn!("Circuit breaker '{}': Unexpected success in Open state", self.name);
            }
        }
    }

    async fn on_failure(&self, duration: Duration) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
        
        let is_slow = duration >= self.config.slow_call_threshold;
        if is_slow {
            self.slow_requests.fetch_add(1, Ordering::Relaxed);
        }

        let state = *self.state.read().await;
        
        match state {
            CircuitState::Closed => {
                let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
                
                if failure_count >= self.config.failure_threshold || self.should_open_due_to_slow_calls() {
                    warn!(
                        "Circuit breaker '{}' transitioning from Closed to Open (failures: {}, threshold: {})",
                        self.name, failure_count, self.config.failure_threshold
                    );
                    *self.state.write().await = CircuitState::Open;
                    *self.last_failure_time.write().await = Some(Instant::now());
                }
            }
            CircuitState::HalfOpen => {
                warn!("Circuit breaker '{}' transitioning from HalfOpen back to Open", self.name);
                *self.state.write().await = CircuitState::Open;
                *self.last_failure_time.write().await = Some(Instant::now());
                self.reset_counters();
            }
            CircuitState::Open => {
                // Already open, update last failure time
                *self.last_failure_time.write().await = Some(Instant::now());
            }
        }
    }

    fn should_open_due_to_slow_calls(&self) -> bool {
        let total = self.total_requests.load(Ordering::Relaxed);
        if total == 0 {
            return false;
        }

        let slow = self.slow_requests.load(Ordering::Relaxed);
        let slow_rate = slow as f64 / total as f64;
        
        slow_rate >= self.config.slow_call_rate_threshold
    }

    async fn should_attempt_reset(&self) -> bool {
        if let Some(last_failure) = *self.last_failure_time.read().await {
            last_failure.elapsed() >= self.config.timeout_duration
        } else {
            false
        }
    }

    fn reset_counters(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.half_open_requests.store(0, Ordering::Relaxed);
    }

    pub async fn get_metrics(&self) -> CircuitBreakerMetrics {
        let total = self.total_requests.load(Ordering::Relaxed);
        let failed = self.failed_requests.load(Ordering::Relaxed);
        let slow = self.slow_requests.load(Ordering::Relaxed);

        let failure_rate = if total > 0 { failed as f64 / total as f64 } else { 0.0 };
        let slow_call_rate = if total > 0 { slow as f64 / total as f64 } else { 0.0 };

        CircuitBreakerMetrics {
            state: *self.state.read().await,
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            total_requests: total,
            failed_requests: failed,
            slow_requests: slow,
            last_failure_time: *self.last_failure_time.read().await,
            uptime_seconds: self.creation_time.elapsed().as_secs(),
            failure_rate,
            slow_call_rate,
        }
    }

    pub async fn get_state(&self) -> CircuitState {
        *self.state.read().await
    }

    pub async fn force_open(&self) {
        warn!("Circuit breaker '{}' forced to Open state", self.name);
        *self.state.write().await = CircuitState::Open;
        *self.last_failure_time.write().await = Some(Instant::now());
    }

    pub async fn force_closed(&self) {
        info!("Circuit breaker '{}' forced to Closed state", self.name);
        *self.state.write().await = CircuitState::Closed;
        self.reset_counters();
    }

    pub async fn force_half_open(&self) {
        info!("Circuit breaker '{}' forced to HalfOpen state", self.name);
        *self.state.write().await = CircuitState::HalfOpen;
        self.half_open_requests.store(0, Ordering::Relaxed);
    }
}

// Circuit breaker registry for managing multiple circuit breakers
pub struct CircuitBreakerRegistry {
    breakers: Arc<RwLock<std::collections::HashMap<String, Arc<CircuitBreaker>>>>,
}

impl CircuitBreakerRegistry {
    pub fn new() -> Self {
        Self {
            breakers: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }

    pub async fn register<S: Into<String>>(
        &self,
        name: S,
        config: CircuitBreakerConfig,
    ) -> Arc<CircuitBreaker> {
        let name = name.into();
        let breaker = Arc::new(CircuitBreaker::new(name.clone(), config));
        
        let mut breakers = self.breakers.write().await;
        breakers.insert(name.clone(), breaker.clone());
        
        info!("Registered circuit breaker '{}'", name);
        breaker
    }

    pub async fn get<S: AsRef<str>>(&self, name: S) -> Option<Arc<CircuitBreaker>> {
        let breakers = self.breakers.read().await;
        breakers.get(name.as_ref()).cloned()
    }

    pub async fn get_all_metrics(&self) -> std::collections::HashMap<String, CircuitBreakerMetrics> {
        let breakers = self.breakers.read().await;
        let mut metrics = std::collections::HashMap::new();
        
        for (name, breaker) in breakers.iter() {
            metrics.insert(name.clone(), breaker.get_metrics().await);
        }
        
        metrics
    }

    pub async fn health_check(&self) -> (usize, usize) {
        let breakers = self.breakers.read().await;
        let mut healthy = 0;
        let mut total = 0;
        
        for (_, breaker) in breakers.iter() {
            total += 1;
            if matches!(breaker.get_state().await, CircuitState::Closed) {
                healthy += 1;
            }
        }
        
        (healthy, total)
    }
}

impl Default for CircuitBreakerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_circuit_breaker_closed_to_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            ..Default::default()
        };
        
        let breaker = CircuitBreaker::new("test", config);
        
        // First failure
        let result = breaker.call(async { Err::<(), _>("error") }).await;
        assert!(result.is_err());
        assert_eq!(breaker.get_state().await, CircuitState::Closed);
        
        // Second failure should open circuit
        let result = breaker.call(async { Err::<(), _>("error") }).await;
        assert!(result.is_err());
        assert_eq!(breaker.get_state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            success_threshold: 2,
            timeout_duration: Duration::from_millis(100),
            ..Default::default()
        };
        
        let breaker = CircuitBreaker::new("test", config);
        
        // Cause failure to open circuit
        let _ = breaker.call(async { Err::<(), _>("error") }).await;
        assert_eq!(breaker.get_state().await, CircuitState::Open);
        
        // Wait for timeout
        sleep(Duration::from_millis(150)).await;
        
        // First success should transition to half-open
        let result = breaker.call(async { Ok::<_, &str>(()) }).await;
        assert!(result.is_ok());
        
        // Second success should close circuit
        let result = breaker.call(async { Ok::<_, &str>(()) }).await;
        assert!(result.is_ok());
        assert_eq!(breaker.get_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_slow_call_detection() {
        let config = CircuitBreakerConfig {
            slow_call_threshold: Duration::from_millis(50),
            slow_call_rate_threshold: 0.5,
            failure_threshold: 10, // High threshold so we test slow calls
            ..Default::default()
        };
        
        let breaker = CircuitBreaker::new("test", config);
        
        // Fast call
        let _ = breaker.call(async { 
            sleep(Duration::from_millis(10)).await;
            Ok::<_, &str>(()) 
        }).await;
        
        // Slow call
        let _ = breaker.call(async { 
            sleep(Duration::from_millis(100)).await;
            Ok::<_, &str>(()) 
        }).await;
        
        let metrics = breaker.get_metrics().await;
        assert_eq!(metrics.slow_requests, 1);
        assert!(metrics.slow_call_rate > 0.0);
    }

    #[tokio::test]
    async fn test_circuit_breaker_registry() {
        let registry = CircuitBreakerRegistry::new();
        
        let breaker = registry.register("test-service", CircuitBreakerConfig::default()).await;
        
        let retrieved = registry.get("test-service").await;
        assert!(retrieved.is_some());
        
        let (healthy, total) = registry.health_check().await;
        assert_eq!(healthy, 1);
        assert_eq!(total, 1);
    }
}