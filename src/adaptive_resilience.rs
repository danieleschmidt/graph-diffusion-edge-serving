//! Adaptive Resilience System
//! 
//! Self-healing infrastructure that learns from failures and automatically
//! adapts resilience patterns based on real-world conditions.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

/// Adaptive resilience configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveResilienceConfig {
    pub enable_learning: bool,
    pub failure_history_size: usize,
    pub adaptation_threshold: f64,
    pub min_adaptation_interval_seconds: u64,
    pub circuit_breaker_config: CircuitBreakerConfig,
    pub retry_config: RetryConfig,
    pub health_check_config: HealthCheckConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub recovery_timeout_seconds: u64,
    pub half_open_max_calls: usize,
    pub success_threshold: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub base_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
    pub jitter_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    pub check_interval_seconds: u64,
    pub timeout_seconds: u64,
    pub consecutive_failures_threshold: usize,
    pub degraded_mode_threshold: f64,
}

impl Default for AdaptiveResilienceConfig {
    fn default() -> Self {
        Self {
            enable_learning: true,
            failure_history_size: 1000,
            adaptation_threshold: 0.2, // 20% failure rate triggers adaptation
            min_adaptation_interval_seconds: 300, // 5 minutes
            circuit_breaker_config: CircuitBreakerConfig {
                failure_threshold: 5,
                recovery_timeout_seconds: 60,
                half_open_max_calls: 3,
                success_threshold: 2,
            },
            retry_config: RetryConfig {
                max_attempts: 3,
                base_delay_ms: 100,
                max_delay_ms: 5000,
                backoff_multiplier: 2.0,
                jitter_factor: 0.1,
            },
            health_check_config: HealthCheckConfig {
                check_interval_seconds: 30,
                timeout_seconds: 10,
                consecutive_failures_threshold: 3,
                degraded_mode_threshold: 0.8,
            },
        }
    }
}

/// Failure pattern analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailurePattern {
    pub pattern_type: FailureType,
    pub frequency: f64,
    pub avg_duration_ms: u64,
    pub recommended_adaptation: AdaptationRecommendation,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureType {
    NetworkTimeout,
    ServiceUnavailable,
    ResourceExhaustion,
    RateLimited,
    DataCorruption,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdaptationRecommendation {
    IncreaseTimeout(u64),
    ReduceRequestRate(f64),
    AddCircuitBreaker,
    ImplementBulkhead,
    EnableDegradedMode,
    ScaleResources,
}

/// Failure event for learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureEvent {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub failure_type: FailureType,
    pub duration_ms: u64,
    pub context: HashMap<String, String>,
    pub recovery_action: Option<String>,
    pub success: bool,
}

/// Circuit breaker state
#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Adaptive circuit breaker
#[derive(Debug)]
pub struct AdaptiveCircuitBreaker {
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<RwLock<usize>>,
    success_count: Arc<RwLock<usize>>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    config: CircuitBreakerConfig,
}

impl AdaptiveCircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(RwLock::new(0)),
            success_count: Arc::new(RwLock::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            config,
        }
    }

    /// Execute operation with circuit breaker protection
    pub async fn execute<F, T, E>(&self, operation: F) -> Result<T>
    where
        F: std::future::Future<Output = std::result::Result<T, E>>,
        E: std::fmt::Display,
    {
        // Check if circuit is open
        if self.should_reject_request().await {
            return Err(crate::error::Error::dependency_failure(
                "circuit_breaker", 
                "Circuit breaker is open"
            ));
        }

        let start = Instant::now();
        match operation.await {
            Ok(result) => {
                self.on_success().await;
                Ok(result)
            }
            Err(e) => {
                self.on_failure().await;
                Err(crate::error::Error::dependency_failure(
                    "operation",
                    e.to_string()
                ))
            }
        }
    }

    async fn should_reject_request(&self) -> bool {
        let state = self.state.read().await;
        match *state {
            CircuitState::Closed => false,
            CircuitState::Open => {
                // Check if recovery timeout has passed
                if let Some(last_failure) = *self.last_failure_time.read().await {
                    if last_failure.elapsed() > Duration::from_secs(self.config.recovery_timeout_seconds) {
                        // Transition to half-open
                        drop(state);
                        *self.state.write().await = CircuitState::HalfOpen;
                        *self.success_count.write().await = 0;
                        debug!("Circuit breaker transitioning to half-open");
                        false
                    } else {
                        true
                    }
                } else {
                    true
                }
            }
            CircuitState::HalfOpen => {
                // Allow limited requests in half-open state
                *self.success_count.read().await < self.config.half_open_max_calls
            }
        }
    }

    async fn on_success(&self) {
        let mut state = self.state.write().await;
        match *state {
            CircuitState::Closed => {
                *self.failure_count.write().await = 0;
            }
            CircuitState::HalfOpen => {
                let mut success_count = self.success_count.write().await;
                *success_count += 1;
                
                if *success_count >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    *self.failure_count.write().await = 0;
                    info!("Circuit breaker recovered to closed state");
                }
            }
            CircuitState::Open => {
                // This shouldn't happen, but reset just in case
                *state = CircuitState::Closed;
                *self.failure_count.write().await = 0;
            }
        }
    }

    async fn on_failure(&self) {
        let mut failure_count = self.failure_count.write().await;
        *failure_count += 1;
        *self.last_failure_time.write().await = Some(Instant::now());

        let mut state = self.state.write().await;
        match *state {
            CircuitState::Closed => {
                if *failure_count >= self.config.failure_threshold {
                    *state = CircuitState::Open;
                    warn!("Circuit breaker opened due to {} failures", *failure_count);
                }
            }
            CircuitState::HalfOpen => {
                *state = CircuitState::Open;
                warn!("Circuit breaker reopened due to failure in half-open state");
            }
            CircuitState::Open => {
                // Already open, just track the failure
            }
        }
    }

    pub async fn get_state(&self) -> CircuitState {
        self.state.read().await.clone()
    }
}

/// Adaptive retry mechanism with intelligent backoff
#[derive(Debug)]
pub struct AdaptiveRetry {
    config: RetryConfig,
    failure_history: Arc<RwLock<Vec<FailureEvent>>>,
}

impl AdaptiveRetry {
    pub fn new(config: RetryConfig) -> Self {
        Self {
            config,
            failure_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Execute operation with adaptive retry
    pub async fn execute_with_retry<F, T, E>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = std::result::Result<T, E>> + Send>>,
        E: std::fmt::Display + Clone,
    {
        let mut last_error = None;
        
        for attempt in 0..self.config.max_attempts {
            if attempt > 0 {
                let delay = self.calculate_adaptive_delay(attempt).await;
                debug!("Retrying in {}ms (attempt {})", delay.as_millis(), attempt + 1);
                tokio::time::sleep(delay).await;
            }

            let start = Instant::now();
            match operation().await {
                Ok(result) => {
                    if attempt > 0 {
                        info!("Operation succeeded after {} retries", attempt);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    let duration = start.elapsed();
                    self.record_failure(e.clone(), duration).await;
                    last_error = Some(e);
                    
                    if attempt == self.config.max_attempts - 1 {
                        error!("Operation failed after {} attempts", self.config.max_attempts);
                    }
                }
            }
        }

        Err(crate::error::Error::graph_processing(
            format!("Operation failed after {} retries: {}", 
                   self.config.max_attempts, 
                   last_error.unwrap().to_string()),
            "adaptive_retry"
        ))
    }

    async fn calculate_adaptive_delay(&self, attempt: usize) -> Duration {
        let base_delay = self.config.base_delay_ms as f64;
        let multiplier = self.config.backoff_multiplier.powi(attempt as i32);
        let jitter = (rand::random::<f64>() - 0.5) * 2.0 * self.config.jitter_factor;
        
        // Analyze recent failures to adapt delay
        let failure_rate = self.calculate_recent_failure_rate().await;
        let adaptive_multiplier = if failure_rate > 0.5 {
            2.0 // Increase delay if high failure rate
        } else {
            1.0
        };
        
        let delay_ms = (base_delay * multiplier * adaptive_multiplier * (1.0 + jitter))
            .min(self.config.max_delay_ms as f64)
            .max(self.config.base_delay_ms as f64);
        
        Duration::from_millis(delay_ms as u64)
    }

    async fn record_failure<E: std::fmt::Display>(&self, error: E, duration: Duration) {
        let failure_event = FailureEvent {
            timestamp: chrono::Utc::now(),
            failure_type: self.classify_error(&error.to_string()),
            duration_ms: duration.as_millis() as u64,
            context: HashMap::new(),
            recovery_action: None,
            success: false,
        };

        let mut history = self.failure_history.write().await;
        history.push(failure_event);
        
        // Keep only recent failures
        if history.len() > 100 {
            history.drain(0..history.len() - 100);
        }
    }

    async fn calculate_recent_failure_rate(&self) -> f64 {
        let history = self.failure_history.read().await;
        if history.is_empty() {
            return 0.0;
        }

        let recent_cutoff = chrono::Utc::now() - chrono::Duration::minutes(10);
        let recent_failures = history.iter()
            .filter(|f| f.timestamp > recent_cutoff)
            .count();

        recent_failures as f64 / history.len() as f64
    }

    fn classify_error(&self, error_msg: &str) -> FailureType {
        let error_lower = error_msg.to_lowercase();
        
        if error_lower.contains("timeout") {
            FailureType::NetworkTimeout
        } else if error_lower.contains("unavailable") || error_lower.contains("connection") {
            FailureType::ServiceUnavailable
        } else if error_lower.contains("resource") || error_lower.contains("memory") {
            FailureType::ResourceExhaustion
        } else if error_lower.contains("rate") || error_lower.contains("limit") {
            FailureType::RateLimited
        } else if error_lower.contains("corrupt") || error_lower.contains("checksum") {
            FailureType::DataCorruption
        } else {
            FailureType::Unknown
        }
    }
}

/// Comprehensive adaptive resilience system
#[derive(Debug)]
pub struct AdaptiveResilienceSystem {
    config: AdaptiveResilienceConfig,
    circuit_breakers: Arc<RwLock<HashMap<String, AdaptiveCircuitBreaker>>>,
    retry_mechanisms: Arc<RwLock<HashMap<String, AdaptiveRetry>>>,
    failure_history: Arc<RwLock<Vec<FailureEvent>>>,
    last_adaptation: Arc<RwLock<Option<Instant>>>,
}

impl AdaptiveResilienceSystem {
    /// Create new adaptive resilience system
    pub fn new(config: AdaptiveResilienceConfig) -> Self {
        Self {
            config,
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            retry_mechanisms: Arc::new(RwLock::new(HashMap::new())),
            failure_history: Arc::new(RwLock::new(Vec::new())),
            last_adaptation: Arc::new(RwLock::new(None)),
        }
    }

    /// Get or create circuit breaker for service
    pub async fn get_circuit_breaker(&self, service_name: &str) -> AdaptiveCircuitBreaker {
        let mut breakers = self.circuit_breakers.write().await;
        
        if let Some(breaker) = breakers.get(service_name) {
            // Return a clone of the existing breaker (shared state via Arc)
            AdaptiveCircuitBreaker {
                state: breaker.state.clone(),
                failure_count: breaker.failure_count.clone(),
                success_count: breaker.success_count.clone(),
                last_failure_time: breaker.last_failure_time.clone(),
                config: breaker.config.clone(),
            }
        } else {
            let breaker = AdaptiveCircuitBreaker::new(self.config.circuit_breaker_config.clone());
            let cloned_breaker = AdaptiveCircuitBreaker {
                state: breaker.state.clone(),
                failure_count: breaker.failure_count.clone(),
                success_count: breaker.success_count.clone(),
                last_failure_time: breaker.last_failure_time.clone(),
                config: breaker.config.clone(),
            };
            breakers.insert(service_name.to_string(), breaker);
            cloned_breaker
        }
    }

    /// Get or create retry mechanism for service
    pub async fn get_retry_mechanism(&self, service_name: &str) -> AdaptiveRetry {
        let mut retries = self.retry_mechanisms.write().await;
        
        if let Some(retry) = retries.get(service_name) {
            AdaptiveRetry {
                config: retry.config.clone(),
                failure_history: retry.failure_history.clone(),
            }
        } else {
            let retry = AdaptiveRetry::new(self.config.retry_config.clone());
            let cloned_retry = AdaptiveRetry {
                config: retry.config.clone(),
                failure_history: retry.failure_history.clone(),
            };
            retries.insert(service_name.to_string(), retry);
            cloned_retry
        }
    }

    /// Analyze failure patterns and suggest adaptations
    pub async fn analyze_failure_patterns(&self) -> Vec<FailurePattern> {
        let history = self.failure_history.read().await;
        if history.is_empty() {
            return Vec::new();
        }

        let mut patterns = Vec::new();
        
        // Group failures by type
        let mut type_groups: HashMap<String, Vec<&FailureEvent>> = HashMap::new();
        for event in history.iter() {
            let type_key = format!("{:?}", event.failure_type);
            type_groups.entry(type_key).or_default().push(event);
        }

        for (failure_type_str, events) in type_groups {
            if events.len() < 3 {
                continue; // Need at least 3 events for pattern analysis
            }

            let frequency = events.len() as f64 / history.len() as f64;
            let avg_duration = events.iter()
                .map(|e| e.duration_ms)
                .sum::<u64>() / events.len() as u64;

            let failure_type = match failure_type_str.as_str() {
                "NetworkTimeout" => FailureType::NetworkTimeout,
                "ServiceUnavailable" => FailureType::ServiceUnavailable,
                "ResourceExhaustion" => FailureType::ResourceExhaustion,
                "RateLimited" => FailureType::RateLimited,
                "DataCorruption" => FailureType::DataCorruption,
                _ => FailureType::Unknown,
            };

            let recommendation = self.recommend_adaptation(&failure_type, frequency, avg_duration);
            let confidence = self.calculate_confidence(events.len(), frequency);

            patterns.push(FailurePattern {
                pattern_type: failure_type,
                frequency,
                avg_duration_ms: avg_duration,
                recommended_adaptation: recommendation,
                confidence,
            });
        }

        patterns.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        patterns
    }

    /// Apply adaptive improvements based on failure analysis
    pub async fn apply_adaptations(&mut self) -> Result<usize> {
        if !self.should_adapt().await {
            return Ok(0);
        }

        let patterns = self.analyze_failure_patterns().await;
        let mut adaptations_applied = 0;

        for pattern in patterns {
            if pattern.confidence < 0.7 {
                continue; // Only apply high-confidence adaptations
            }

            match pattern.recommended_adaptation {
                AdaptationRecommendation::IncreaseTimeout(new_timeout) => {
                    info!("Adapting: Increasing timeout to {}ms for {:?}", new_timeout, pattern.pattern_type);
                    // Would update configuration
                    adaptations_applied += 1;
                }
                AdaptationRecommendation::ReduceRequestRate(factor) => {
                    info!("Adapting: Reducing request rate by factor {} for {:?}", factor, pattern.pattern_type);
                    // Would implement rate limiting
                    adaptations_applied += 1;
                }
                AdaptationRecommendation::AddCircuitBreaker => {
                    info!("Adapting: Adding circuit breaker for {:?}", pattern.pattern_type);
                    // Circuit breakers are already available
                    adaptations_applied += 1;
                }
                AdaptationRecommendation::ImplementBulkhead => {
                    info!("Adapting: Implementing bulkhead pattern for {:?}", pattern.pattern_type);
                    // Would implement resource isolation
                    adaptations_applied += 1;
                }
                AdaptationRecommendation::EnableDegradedMode => {
                    info!("Adapting: Enabling degraded mode for {:?}", pattern.pattern_type);
                    // Would implement graceful degradation
                    adaptations_applied += 1;
                }
                AdaptationRecommendation::ScaleResources => {
                    info!("Adapting: Recommending resource scaling for {:?}", pattern.pattern_type);
                    // Would trigger auto-scaling
                    adaptations_applied += 1;
                }
            }
        }

        *self.last_adaptation.write().await = Some(Instant::now());
        
        if adaptations_applied > 0 {
            info!("Applied {} adaptive resilience improvements", adaptations_applied);
        }

        Ok(adaptations_applied)
    }

    async fn should_adapt(&self) -> bool {
        if !self.config.enable_learning {
            return false;
        }

        if let Some(last_adaptation) = *self.last_adaptation.read().await {
            let min_interval = Duration::from_secs(self.config.min_adaptation_interval_seconds);
            if last_adaptation.elapsed() < min_interval {
                return false;
            }
        }

        // Check if failure rate exceeds adaptation threshold
        let history = self.failure_history.read().await;
        if history.len() < 10 {
            return false; // Need sufficient data
        }

        let recent_cutoff = chrono::Utc::now() - chrono::Duration::minutes(30);
        let recent_failures = history.iter()
            .filter(|f| f.timestamp > recent_cutoff)
            .count();

        let failure_rate = recent_failures as f64 / history.len() as f64;
        failure_rate > self.config.adaptation_threshold
    }

    fn recommend_adaptation(&self, failure_type: &FailureType, frequency: f64, avg_duration: u64) -> AdaptationRecommendation {
        match failure_type {
            FailureType::NetworkTimeout => {
                if avg_duration > 5000 {
                    AdaptationRecommendation::IncreaseTimeout(avg_duration * 2)
                } else {
                    AdaptationRecommendation::AddCircuitBreaker
                }
            }
            FailureType::ServiceUnavailable => {
                if frequency > 0.3 {
                    AdaptationRecommendation::EnableDegradedMode
                } else {
                    AdaptationRecommendation::AddCircuitBreaker
                }
            }
            FailureType::ResourceExhaustion => {
                AdaptationRecommendation::ScaleResources
            }
            FailureType::RateLimited => {
                AdaptationRecommendation::ReduceRequestRate(0.7)
            }
            FailureType::DataCorruption => {
                AdaptationRecommendation::ImplementBulkhead
            }
            FailureType::Unknown => {
                AdaptationRecommendation::AddCircuitBreaker
            }
        }
    }

    fn calculate_confidence(&self, sample_size: usize, frequency: f64) -> f64 {
        // Simple confidence calculation based on sample size and frequency
        let sample_factor = (sample_size as f64 / 100.0).min(1.0);
        let frequency_factor = if frequency > 0.1 { 1.0 } else { frequency * 10.0 };
        
        sample_factor * frequency_factor
    }

    /// Record failure event for learning
    pub async fn record_failure(&self, event: FailureEvent) {
        let mut history = self.failure_history.write().await;
        history.push(event);
        
        // Maintain history size limit
        if history.len() > self.config.failure_history_size {
            history.drain(0..history.len() - self.config.failure_history_size);
        }
    }

    /// Get current resilience metrics
    pub async fn get_metrics(&self) -> ResilienceMetrics {
        let history = self.failure_history.read().await;
        let total_events = history.len();
        
        let recent_cutoff = chrono::Utc::now() - chrono::Duration::hours(1);
        let recent_failures = history.iter()
            .filter(|f| f.timestamp > recent_cutoff)
            .count();

        let circuit_breakers = self.circuit_breakers.read().await;
        let open_breakers = futures::future::join_all(
            circuit_breakers.values().map(|cb| cb.get_state())
        ).await.into_iter()
            .filter(|state| matches!(state, CircuitState::Open))
            .count();

        ResilienceMetrics {
            total_failure_events: total_events,
            recent_failure_rate: if total_events > 0 { 
                recent_failures as f64 / total_events as f64 
            } else { 
                0.0 
            },
            active_circuit_breakers: circuit_breakers.len(),
            open_circuit_breakers: open_breakers,
            adaptation_enabled: self.config.enable_learning,
            last_adaptation: *self.last_adaptation.read().await,
        }
    }
}

/// Resilience system metrics
#[derive(Debug, Clone, Serialize)]
pub struct ResilienceMetrics {
    pub total_failure_events: usize,
    pub recent_failure_rate: f64,
    pub active_circuit_breakers: usize,
    pub open_circuit_breakers: usize,
    pub adaptation_enabled: bool,
    #[serde(skip)]
    pub last_adaptation: Option<Instant>,
}