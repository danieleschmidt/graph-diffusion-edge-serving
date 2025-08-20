//! Advanced resilience patterns for robust graph diffusion serving
//! 
//! Implements circuit breakers, bulkheads, retry policies, and adaptive rate limiting

use std::sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use std::collections::{HashMap, VecDeque};
use tracing::{warn, info, error, debug, instrument};
use tokio::time::{sleep, timeout};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout_duration: Duration,
    pub reset_timeout: Duration,
    pub slow_call_threshold: Duration,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 3,
            timeout_duration: Duration::from_secs(10),
            reset_timeout: Duration::from_secs(60),
            slow_call_threshold: Duration::from_millis(1000),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitState {
    Closed,   // Normal operation
    Open,     // Failing, rejecting calls
    HalfOpen, // Testing if service recovered
}

pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitState>>,
    failure_count: Arc<AtomicU64>,
    success_count: Arc<AtomicU64>,
    last_failure_time: Arc<RwLock<Option<Instant>>>,
    call_history: Arc<Mutex<VecDeque<(Instant, Duration, bool)>>>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(CircuitState::Closed)),
            failure_count: Arc::new(AtomicU64::new(0)),
            success_count: Arc::new(AtomicU64::new(0)),
            last_failure_time: Arc::new(RwLock::new(None)),
            call_history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    #[instrument(skip(self, operation))]
    pub async fn call<F, R, E>(&self, operation: F) -> Result<R, CircuitBreakerError<E>>
    where
        F: std::future::Future<Output = Result<R, E>> + Send,
        E: std::fmt::Debug,
    {
        // Check if circuit should reject the call
        if !self.should_allow_call() {
            warn!("Circuit breaker is OPEN, rejecting call");
            return Err(CircuitBreakerError::CircuitOpen);
        }

        let start_time = Instant::now();
        
        // Execute with timeout
        let result = timeout(self.config.timeout_duration, operation).await;
        
        let duration = start_time.elapsed();
        let success = matches!(result, Ok(Ok(_)));

        // Record call result
        self.record_call(duration, success).await;

        match result {
            Ok(Ok(value)) => {
                self.on_success().await;
                Ok(value)
            }
            Ok(Err(e)) => {
                self.on_failure().await;
                Err(CircuitBreakerError::CallFailed(e))
            }
            Err(_) => {
                self.on_failure().await;
                Err(CircuitBreakerError::Timeout)
            }
        }
    }

    fn should_allow_call(&self) -> bool {
        let state = self.state.read();
        match *state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if reset timeout has elapsed
                if let Some(last_failure) = *self.last_failure_time.read() {
                    if last_failure.elapsed() > self.config.reset_timeout {
                        drop(state);
                        self.transition_to_half_open();
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    async fn record_call(&self, duration: Duration, success: bool) {
        let mut history = self.call_history.lock();
        let now = Instant::now();
        
        // Add new record
        history.push_back((now, duration, success));
        
        // Remove old records (keep last 100 calls or 5 minutes)
        let cutoff = now - Duration::from_secs(300);
        while history.len() > 100 || 
              (history.front().map_or(false, |(time, _, _)| *time < cutoff)) {
            history.pop_front();
        }
    }

    async fn on_success(&self) {
        let state = self.state.read().clone();
        self.success_count.fetch_add(1, Ordering::Relaxed);

        match state {
            CircuitState::HalfOpen => {
                if self.success_count.load(Ordering::Relaxed) >= self.config.success_threshold as u64 {
                    drop(state);
                    self.transition_to_closed();
                }
            }
            _ => {}
        }
    }

    async fn on_failure(&self) {
        let failure_count = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        *self.last_failure_time.write() = Some(Instant::now());

        let state = self.state.read().clone();
        match state {
            CircuitState::Closed => {
                if failure_count >= self.config.failure_threshold as u64 {
                    drop(state);
                    self.transition_to_open();
                }
            }
            CircuitState::HalfOpen => {
                drop(state);
                self.transition_to_open();
            }
            _ => {}
        }
    }

    fn transition_to_open(&self) {
        *self.state.write() = CircuitState::Open;
        warn!("Circuit breaker transitioned to OPEN state");
    }

    fn transition_to_half_open(&self) {
        *self.state.write() = CircuitState::HalfOpen;
        self.success_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker transitioned to HALF-OPEN state");
    }

    fn transition_to_closed(&self) {
        *self.state.write() = CircuitState::Closed;
        self.failure_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        info!("Circuit breaker transitioned to CLOSED state");
    }

    pub fn get_stats(&self) -> CircuitBreakerStats {
        let history = self.call_history.lock();
        let now = Instant::now();
        
        let recent_calls: Vec<_> = history.iter()
            .filter(|(time, _, _)| now.duration_since(*time) < Duration::from_secs(60))
            .collect();
        
        let total_calls = recent_calls.len();
        let successful_calls = recent_calls.iter().filter(|(_, _, success)| *success).count();
        let avg_duration = if !recent_calls.is_empty() {
            recent_calls.iter().map(|(_, duration, _)| duration.as_millis() as f64).sum::<f64>() 
                / recent_calls.len() as f64
        } else {
            0.0
        };

        CircuitBreakerStats {
            state: self.state.read().clone(),
            failure_count: self.failure_count.load(Ordering::Relaxed),
            success_count: self.success_count.load(Ordering::Relaxed),
            total_calls: total_calls as u64,
            success_rate: if total_calls > 0 { 
                successful_calls as f64 / total_calls as f64 
            } else { 
                1.0 
            },
            avg_response_time_ms: avg_duration,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CircuitBreakerStats {
    pub state: CircuitState,
    pub failure_count: u64,
    pub success_count: u64,
    pub total_calls: u64,
    pub success_rate: f64,
    pub avg_response_time_ms: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum CircuitBreakerError<E> {
    #[error("Circuit breaker is open")]
    CircuitOpen,
    
    #[error("Operation timed out")]
    Timeout,
    
    #[error("Call failed: {0:?}")]
    CallFailed(E),
}

// Adaptive Rate Limiter with sliding window
pub struct AdaptiveRateLimiter {
    max_requests: Arc<AtomicU64>,
    window_size: Duration,
    request_log: Arc<Mutex<VecDeque<Instant>>>,
    adaptive_enabled: AtomicBool,
    load_factor: Arc<RwLock<f64>>,
    last_adjustment: Arc<Mutex<Instant>>,
}

impl AdaptiveRateLimiter {
    pub fn new(initial_max: u64, window_size: Duration) -> Self {
        Self {
            max_requests: Arc::new(AtomicU64::new(initial_max)),
            window_size,
            request_log: Arc::new(Mutex::new(VecDeque::new())),
            adaptive_enabled: AtomicBool::new(true),
            load_factor: Arc::new(RwLock::new(1.0)),
            last_adjustment: Arc::new(Mutex::new(Instant::now())),
        }
    }

    pub async fn try_acquire(&self) -> bool {
        let now = Instant::now();
        let mut log = self.request_log.lock();
        
        // Remove old requests outside the window
        let cutoff = now - self.window_size;
        while log.front().map_or(false, |&time| time < cutoff) {
            log.pop_front();
        }
        
        let current_count = log.len() as u64;
        let max_allowed = self.max_requests.load(Ordering::Relaxed);
        
        if current_count < max_allowed {
            log.push_back(now);
            
            // Adaptive adjustment based on usage patterns
            if self.adaptive_enabled.load(Ordering::Relaxed) {
                self.adjust_rate_limit(current_count, max_allowed).await;
            }
            
            true
        } else {
            debug!("Rate limit exceeded: {}/{}", current_count, max_allowed);
            false
        }
    }

    async fn adjust_rate_limit(&self, current: u64, max_allowed: u64) {
        let mut last_adjustment = self.last_adjustment.lock();
        if last_adjustment.elapsed() < Duration::from_secs(30) {
            return; // Don't adjust too frequently
        }
        *last_adjustment = Instant::now();
        drop(last_adjustment);

        let utilization = current as f64 / max_allowed as f64;
        let load_factor = *self.load_factor.read();
        
        // Adjust based on system load and utilization
        let new_limit = if utilization > 0.9 && load_factor > 1.5 {
            // High utilization and load - decrease limit
            (max_allowed as f64 * 0.9) as u64
        } else if utilization < 0.5 && load_factor < 0.8 {
            // Low utilization and load - increase limit
            (max_allowed as f64 * 1.1) as u64
        } else {
            max_allowed
        };
        
        if new_limit != max_allowed {
            self.max_requests.store(new_limit, Ordering::Relaxed);
            info!("Adaptive rate limit adjusted: {} -> {}", max_allowed, new_limit);
        }
    }

    pub fn update_load_factor(&self, factor: f64) {
        *self.load_factor.write() = factor;
    }

    pub fn get_stats(&self) -> RateLimiterStats {
        let log = self.request_log.lock();
        let now = Instant::now();
        let window_start = now - self.window_size;
        
        let current_requests = log.iter().filter(|&&time| time >= window_start).count();
        let max_requests = self.max_requests.load(Ordering::Relaxed);
        
        RateLimiterStats {
            current_requests: current_requests as u64,
            max_requests,
            utilization: current_requests as f64 / max_requests as f64,
            window_size_secs: self.window_size.as_secs(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RateLimiterStats {
    pub current_requests: u64,
    pub max_requests: u64,
    pub utilization: f64,
    pub window_size_secs: u64,
}

// Bulkhead Pattern for resource isolation
pub struct BulkheadExecutor {
    pools: HashMap<String, Arc<Mutex<ResourcePool>>>,
}

pub struct ResourcePool {
    name: String,
    max_concurrent: usize,
    current_active: usize,
    queue: VecDeque<tokio::sync::oneshot::Sender<()>>,
    timeout: Duration,
}

impl BulkheadExecutor {
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    pub fn add_pool(&mut self, name: String, max_concurrent: usize, timeout: Duration) {
        let pool = ResourcePool {
            name: name.clone(),
            max_concurrent,
            current_active: 0,
            queue: VecDeque::new(),
            timeout,
        };
        self.pools.insert(name, Arc::new(Mutex::new(pool)));
    }

    pub async fn execute<F, R>(&self, pool_name: &str, operation: F) -> Result<R, BulkheadError>
    where
        F: std::future::Future<Output = R> + Send,
        R: Send,
    {
        let pool = self.pools.get(pool_name)
            .ok_or(BulkheadError::PoolNotFound(pool_name.to_string()))?
            .clone();

        // Try to acquire slot
        let permit = self.acquire_slot(pool.clone()).await?;
        
        // Execute operation
        let result = operation.await;
        
        // Release slot
        self.release_slot(pool).await;
        
        Ok(result)
    }

    async fn acquire_slot(&self, pool: Arc<Mutex<ResourcePool>>) -> Result<(), BulkheadError> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        
        {
            let mut pool_guard = pool.lock();
            if pool_guard.current_active < pool_guard.max_concurrent {
                pool_guard.current_active += 1;
                return Ok(());
            } else {
                pool_guard.queue.push_back(sender);
            }
        }
        
        // Wait for slot to become available
        timeout(Duration::from_secs(30), receiver).await
            .map_err(|_| BulkheadError::Timeout)?
            .map_err(|_| BulkheadError::Cancelled)?;
        
        Ok(())
    }

    async fn release_slot(&self, pool: Arc<Mutex<ResourcePool>>) {
        let mut pool_guard = pool.lock();
        pool_guard.current_active -= 1;
        
        if let Some(waiter) = pool_guard.queue.pop_front() {
            pool_guard.current_active += 1;
            let _ = waiter.send(()); // Ignore if receiver dropped
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BulkheadError {
    #[error("Pool '{0}' not found")]
    PoolNotFound(String),
    
    #[error("Operation timed out waiting for resource")]
    Timeout,
    
    #[error("Operation was cancelled")]
    Cancelled,
}

// Retry Policy with exponential backoff and jitter
pub struct RetryPolicy {
    max_attempts: u32,
    base_delay: Duration,
    max_delay: Duration,
    backoff_multiplier: f64,
    jitter: bool,
}

impl RetryPolicy {
    pub fn exponential(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }

    pub fn fixed(max_attempts: u32, delay: Duration) -> Self {
        Self {
            max_attempts,
            base_delay: delay,
            max_delay: delay,
            backoff_multiplier: 1.0,
            jitter: false,
        }
    }

    pub async fn execute<F, R, E>(&self, mut operation: F) -> Result<R, E>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<R, E>> + Send>> + Send,
        E: std::fmt::Debug,
    {
        let mut last_error = None;
        
        for attempt in 1..=self.max_attempts {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = Some(e);
                    
                    if attempt < self.max_attempts {
                        let delay = self.calculate_delay(attempt);
                        debug!("Retry attempt {}/{} failed, waiting {:?}", attempt, self.max_attempts, delay);
                        sleep(delay).await;
                    }
                }
            }
        }
        
        Err(last_error.unwrap())
    }

    fn calculate_delay(&self, attempt: u32) -> Duration {
        let delay = self.base_delay.as_millis() as f64 
            * self.backoff_multiplier.powi((attempt - 1) as i32);
        
        let delay = Duration::from_millis(delay as u64);
        let delay = std::cmp::min(delay, self.max_delay);
        
        if self.jitter {
            let jitter_factor = 1.0 + (rand::random::<f64>() - 0.5) * 0.1; // ±5% jitter
            Duration::from_millis((delay.as_millis() as f64 * jitter_factor) as u64)
        } else {
            delay
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[tokio::test]
    async fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let cb = CircuitBreaker::new("test_circuit", config);
        
        // Simulate failures
        for _ in 0..3 {
            let result = cb.call(async { Err::<(), &str>("test error") }).await;
            assert!(result.is_err());
        }
        
        // Circuit should be open now
        let result = cb.call(async { Ok::<(), &str>(()) }).await;
        assert!(matches!(result, Err(CircuitBreakerError::CircuitOpen)));
    }

    #[tokio::test]
    async fn test_adaptive_rate_limiter() {
        let limiter = AdaptiveRateLimiter::new(5, Duration::from_secs(1));
        
        // Should allow up to 5 requests
        for _ in 0..5 {
            assert!(limiter.try_acquire().await);
        }
        
        // 6th request should be rejected
        assert!(!limiter.try_acquire().await);
        
        // Wait for window to reset
        tokio::time::sleep(Duration::from_secs(1)).await;
        
        // Should allow requests again
        assert!(limiter.try_acquire().await);
    }

    #[tokio::test]
    async fn test_retry_policy() {
        let policy = RetryPolicy::exponential(3);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();
        
        let result = policy.execute(move || {
            let counter = counter_clone.clone();
            Box::pin(async move {
                let count = counter.fetch_add(1, Ordering::Relaxed);
                if count < 2 {
                    Err("fail")
                } else {
                    Ok("success")
                }
            })
        }).await;
        
        assert_eq!(result, Ok("success"));
        assert_eq!(counter.load(Ordering::Relaxed), 3);
    }
}