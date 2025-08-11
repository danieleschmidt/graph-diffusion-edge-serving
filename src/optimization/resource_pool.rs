//! Resource pooling for TPU connections, database connections, and compute resources

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use tokio::sync::{RwLock, Semaphore, Notify};
use tracing::{debug, info, warn, error};
use serde::{Serialize, Deserialize};
use async_trait::async_trait;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub initial_size: usize,
    pub max_size: usize,
    pub min_idle: usize,
    pub max_idle_time_seconds: u64,
    pub connection_timeout_seconds: u64,
    pub validation_interval_seconds: u64,
    pub enable_metrics: bool,
    pub enable_health_checks: bool,
    pub retry_attempts: usize,
    pub retry_delay_ms: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 5,
            max_size: 20,
            min_idle: 2,
            max_idle_time_seconds: 300, // 5 minutes
            connection_timeout_seconds: 30,
            validation_interval_seconds: 60,
            enable_metrics: true,
            enable_health_checks: true,
            retry_attempts: 3,
            retry_delay_ms: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PoolMetrics {
    pub total_connections: usize,
    pub active_connections: usize,
    pub idle_connections: usize,
    pub failed_connections: usize,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_wait_time_ms: f64,
    pub peak_active_connections: usize,
    pub connection_creation_errors: u64,
    pub validation_failures: u64,
}

#[derive(Debug)]
struct PooledResource<T> {
    resource: T,
    created_at: Instant,
    last_used: Instant,
    use_count: u64,
    is_valid: bool,
}

impl<T> PooledResource<T> {
    fn new(resource: T) -> Self {
        let now = Instant::now();
        Self {
            resource,
            created_at: now,
            last_used: now,
            use_count: 0,
            is_valid: true,
        }
    }

    fn touch(&mut self) {
        self.last_used = Instant::now();
        self.use_count += 1;
    }

    fn is_idle_too_long(&self, max_idle_time: Duration) -> bool {
        self.last_used.elapsed() > max_idle_time
    }

    fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

#[async_trait]
pub trait PoolableResource: Send + Sync + 'static {
    type CreateParams: Send + Sync;
    
    async fn create(params: &Self::CreateParams) -> crate::Result<Self>
    where 
        Self: Sized;
    
    async fn validate(&self) -> bool;
    
    async fn reset(&mut self) -> crate::Result<()>;
    
    fn resource_type(&self) -> &'static str;
}

pub struct ResourcePool<T, P> 
where
    T: PoolableResource<CreateParams = P> + Send + Sync,
    P: Send + Sync,
{
    config: PoolConfig,
    creation_params: P,
    available: Arc<RwLock<VecDeque<PooledResource<T>>>>,
    active_count: Arc<RwLock<usize>>,
    semaphore: Arc<Semaphore>,
    metrics: Arc<RwLock<PoolMetrics>>,
    shutdown_notify: Arc<Notify>,
    is_shutdown: Arc<RwLock<bool>>,
}

impl<T, P> ResourcePool<T, P> 
where
    T: PoolableResource<CreateParams = P> + Send + Sync,
    P: Send + Sync + Clone,
{
    pub async fn new(config: PoolConfig, creation_params: P) -> crate::Result<Self> {
        let pool = Self {
            semaphore: Arc::new(Semaphore::new(config.max_size)),
            available: Arc::new(RwLock::new(VecDeque::new())),
            active_count: Arc::new(RwLock::new(0)),
            metrics: Arc::new(RwLock::new(PoolMetrics {
                total_connections: 0,
                active_connections: 0,
                idle_connections: 0,
                failed_connections: 0,
                total_requests: 0,
                successful_requests: 0,
                failed_requests: 0,
                average_wait_time_ms: 0.0,
                peak_active_connections: 0,
                connection_creation_errors: 0,
                validation_failures: 0,
            })),
            config,
            creation_params,
            shutdown_notify: Arc::new(Notify::new()),
            is_shutdown: Arc::new(RwLock::new(false)),
        };

        // Pre-populate with initial connections
        pool.initialize_pool().await?;

        // Start background maintenance task
        if pool.config.enable_health_checks {
            pool.start_maintenance_task().await;
        }

        info!(
            "Resource pool initialized with {} connections (type: {})",
            pool.config.initial_size,
            std::any::type_name::<T>()
        );

        Ok(pool)
    }

    pub async fn acquire(&self) -> crate::Result<PooledConnection<T, P>> {
        if *self.is_shutdown.read().await {
            return Err(crate::error::Error::dependency_failure(
                "resource_pool",
                "Pool is shutdown"
            ));
        }

        let start_time = Instant::now();
        let mut metrics = self.metrics.write().await;
        metrics.total_requests += 1;
        drop(metrics);

        // Wait for an available slot
        let _permit = self.semaphore
            .acquire()
            .await
            .map_err(|_| crate::error::Error::dependency_failure(
                "resource_pool",
                "Failed to acquire semaphore permit"
            ))?;

        // Try to get an existing connection from the pool
        let mut connection = self.try_acquire_existing().await;

        // If no valid existing connection, create a new one
        if connection.is_none() {
            connection = Some(self.create_new_connection().await?);
        }

        let pooled_connection = connection.unwrap();
        
        // Update metrics
        let wait_time = start_time.elapsed();
        let mut metrics = self.metrics.write().await;
        let mut active_count = self.active_count.write().await;
        
        *active_count += 1;
        metrics.active_connections = *active_count;
        if *active_count > metrics.peak_active_connections {
            metrics.peak_active_connections = *active_count;
        }
        
        // Update average wait time
        let total_wait_time = metrics.average_wait_time_ms * (metrics.total_requests - 1) as f64;
        metrics.average_wait_time_ms = (total_wait_time + wait_time.as_secs_f64() * 1000.0) / metrics.total_requests as f64;
        
        metrics.successful_requests += 1;
        
        debug!(
            "Acquired resource from pool (active: {}, wait_time: {:.2}ms)",
            *active_count,
            wait_time.as_secs_f64() * 1000.0
        );

        Ok(PooledConnection {
            resource: Some(pooled_connection),
            pool: self,
            _permit: _permit,
        })
    }

    pub async fn get_metrics(&self) -> PoolMetrics {
        let metrics = self.metrics.read().await;
        let available = self.available.read().await;
        let active_count = self.active_count.read().await;

        let mut updated_metrics = metrics.clone();
        updated_metrics.idle_connections = available.len();
        updated_metrics.active_connections = *active_count;
        updated_metrics.total_connections = available.len() + *active_count;

        updated_metrics
    }

    pub async fn health_check(&self) -> bool {
        let metrics = self.get_metrics().await;
        
        // Consider pool healthy if:
        // 1. Not shutdown
        // 2. Has some connections available or can create new ones
        // 3. Recent requests have been mostly successful
        
        if *self.is_shutdown.read().await {
            return false;
        }

        if metrics.total_connections == 0 && metrics.total_requests == 0 {
            return true; // New pool, not yet used
        }

        if metrics.total_requests > 0 {
            let success_rate = metrics.successful_requests as f64 / metrics.total_requests as f64;
            success_rate >= 0.8 // 80% success rate threshold
        } else {
            true
        }
    }

    pub async fn shutdown(&self) {
        info!("Shutting down resource pool");
        
        *self.is_shutdown.write().await = true;
        self.shutdown_notify.notify_waiters();

        // Close all available connections
        let mut available = self.available.write().await;
        available.clear();

        // Note: Active connections will be returned and closed when dropped
    }

    async fn try_acquire_existing(&self) -> Option<PooledResource<T>> {
        let mut available = self.available.write().await;
        
        while let Some(mut pooled_resource) = available.pop_front() {
            // Validate the resource if health checks are enabled
            if self.config.enable_health_checks {
                if !pooled_resource.resource.validate().await {
                    let mut metrics = self.metrics.write().await;
                    metrics.validation_failures += 1;
                    metrics.failed_connections += 1;
                    
                    debug!("Resource failed validation, discarding");
                    continue;
                }
            }

            // Check if resource has been idle too long
            let max_idle = Duration::from_secs(self.config.max_idle_time_seconds);
            if pooled_resource.is_idle_too_long(max_idle) {
                debug!("Resource idle too long, discarding");
                continue;
            }

            pooled_resource.touch();
            return Some(pooled_resource);
        }

        None
    }

    async fn create_new_connection(&self) -> crate::Result<PooledResource<T>> {
        debug!("Creating new resource for pool");
        
        let mut retry_count = 0;
        let max_retries = self.config.retry_attempts;
        let retry_delay = Duration::from_millis(self.config.retry_delay_ms);

        loop {
            match T::create(&self.creation_params).await {
                Ok(resource) => {
                    let mut metrics = self.metrics.write().await;
                    metrics.total_connections += 1;
                    
                    return Ok(PooledResource::new(resource));
                }
                Err(e) => {
                    retry_count += 1;
                    let mut metrics = self.metrics.write().await;
                    metrics.connection_creation_errors += 1;
                    
                    if retry_count >= max_retries {
                        error!("Failed to create resource after {} retries: {}", max_retries, e);
                        return Err(e);
                    }
                    
                    warn!("Failed to create resource (attempt {}/{}): {}", retry_count, max_retries, e);
                    drop(metrics);
                    tokio::time::sleep(retry_delay).await;
                }
            }
        }
    }

    async fn return_connection(&self, mut connection: PooledResource<T>) {
        let mut active_count = self.active_count.write().await;
        *active_count -= 1;
        drop(active_count);

        // Try to reset the connection
        match connection.resource.reset().await {
            Ok(()) => {
                let mut available = self.available.write().await;
                let mut metrics = self.metrics.write().await;
                
                // Only keep if we haven't exceeded max connections and pool isn't shutdown
                if available.len() < self.config.max_size && !*self.is_shutdown.read().await {
                    available.push_back(connection);
                    metrics.idle_connections = available.len();
                } else {
                    // Pool is full or shutdown, discard connection
                    metrics.total_connections -= 1;
                }
            }
            Err(e) => {
                warn!("Failed to reset connection, discarding: {}", e);
                let mut metrics = self.metrics.write().await;
                metrics.failed_connections += 1;
                metrics.total_connections -= 1;
            }
        }

        debug!("Returned resource to pool (active: {})", *self.active_count.read().await);
    }

    async fn initialize_pool(&self) -> crate::Result<()> {
        let mut available = self.available.write().await;
        
        for i in 0..self.config.initial_size {
            match self.create_new_connection().await {
                Ok(connection) => {
                    available.push_back(connection);
                    debug!("Pre-created resource {}/{}", i + 1, self.config.initial_size);
                }
                Err(e) => {
                    warn!("Failed to create initial resource {}: {}", i + 1, e);
                    // Continue trying to create other connections
                }
            }
        }

        info!("Pool initialized with {}/{} connections", available.len(), self.config.initial_size);
        Ok(())
    }

    async fn start_maintenance_task(&self) {
        let available = self.available.clone();
        let metrics = self.metrics.clone();
        let config = self.config.clone();
        let shutdown_notify = self.shutdown_notify.clone();
        let is_shutdown = self.is_shutdown.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                Duration::from_secs(config.validation_interval_seconds)
            );

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if *is_shutdown.read().await {
                            break;
                        }
                        
                        Self::maintenance_cycle(&available, &metrics, &config).await;
                    }
                    _ = shutdown_notify.notified() => {
                        debug!("Maintenance task shutting down");
                        break;
                    }
                }
            }
        });
    }

    async fn maintenance_cycle(
        available: &Arc<RwLock<VecDeque<PooledResource<T>>>>,
        metrics: &Arc<RwLock<PoolMetrics>>,
        config: &PoolConfig,
    ) {
        let mut available_guard = available.write().await;
        let mut to_remove = Vec::new();
        let max_idle = Duration::from_secs(config.max_idle_time_seconds);

        // Identify connections that need to be removed
        for (index, connection) in available_guard.iter().enumerate() {
            if connection.is_idle_too_long(max_idle) || !connection.is_valid {
                to_remove.push(index);
            }
        }

        // Remove stale connections (in reverse order to maintain indices)
        for &index in to_remove.iter().rev() {
            if let Some(_) = available_guard.remove(index) {
                let mut metrics_guard = metrics.write().await;
                metrics_guard.total_connections -= 1;
                debug!("Removed stale connection during maintenance");
            }
        }

        // Ensure minimum idle connections
        let current_idle = available_guard.len();
        if current_idle < config.min_idle {
            debug!("Pool below minimum idle connections ({} < {}), maintenance will handle this", 
                   current_idle, config.min_idle);
        }

        debug!("Maintenance cycle completed: {} idle connections", available_guard.len());
    }
}

pub struct PooledConnection<'a, T, P> 
where
    T: PoolableResource<CreateParams = P> + Send + Sync,
    P: Send + Sync + Clone,
{
    resource: Option<PooledResource<T>>,
    pool: &'a ResourcePool<T, P>,
    _permit: tokio::sync::SemaphorePermit<'a>,
}

impl<'a, T, P> PooledConnection<'a, T, P> 
where
    T: PoolableResource<CreateParams = P> + Send + Sync,
    P: Send + Sync + Clone,
{
    pub fn get(&self) -> &T {
        &self.resource.as_ref().unwrap().resource
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.resource.as_mut().unwrap().resource
    }

    pub fn usage_stats(&self) -> (Duration, u64) {
        let resource = self.resource.as_ref().unwrap();
        (resource.age(), resource.use_count)
    }
}

impl<'a, T, P> Drop for PooledConnection<'a, T, P> 
where
    T: PoolableResource<CreateParams = P> + Send + Sync,
    P: Send + Sync + Clone,
{
    fn drop(&mut self) {
        if let Some(connection) = self.resource.take() {
            // Use a blocking runtime-agnostic approach for returning connections
            // In production, you'd want to handle this more elegantly
            let available = self.pool.available.clone();
            let _ = std::thread::spawn(move || {
                let rt = tokio::runtime::Handle::try_current();
                if let Ok(handle) = rt {
                    handle.spawn(async move {
                        let mut available_guard = available.write().await;
                        available_guard.push_back(connection);
                    });
                }
            });
        }
    }
}

// Example implementation for TPU resources
#[cfg(feature = "tpu")]
pub struct TpuResource {
    tpu: crate::tpu::EdgeTPU,
    device_path: String,
}

#[cfg(feature = "tpu")]
pub struct TpuCreateParams {
    pub device_path: String,
    pub model_path: String,
    pub power_limit: f32,
}

#[cfg(feature = "tpu")]
#[async_trait]
impl PoolableResource for TpuResource {
    type CreateParams = TpuCreateParams;

    async fn create(params: &Self::CreateParams) -> crate::Result<Self> {
        let tpu = crate::tpu::EdgeTPU::new()
            .device_path(&params.device_path)
            .model_path(&params.model_path)
            .power_limit(params.power_limit)
            .build()?;

        Ok(Self {
            tpu,
            device_path: params.device_path.clone(),
        })
    }

    async fn validate(&self) -> bool {
        match self.tpu.get_stats() {
            Ok(stats) => {
                stats.temperature_celsius < 90.0 && 
                stats.power_consumption_watts < 6.0
            }
            Err(_) => false,
        }
    }

    async fn reset(&mut self) -> crate::Result<()> {
        // TPU connections don't typically need reset
        Ok(())
    }

    fn resource_type(&self) -> &'static str {
        "tpu"
    }
}

#[cfg(feature = "tpu")]
impl TpuResource {
    pub fn inference(&self, input: &[f32]) -> crate::Result<Vec<f32>> {
        self.tpu.inference(input)
    }

    pub fn batch_inference(&self, batch: &[&[f32]]) -> crate::Result<Vec<Vec<f32>>> {
        self.tpu.batch_inference(batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Mock resource for testing
    struct MockResource {
        id: usize,
        is_valid: bool,
    }

    #[derive(Clone)]
    struct MockCreateParams {
        should_fail: bool,
    }

    static MOCK_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

    #[async_trait]
    impl PoolableResource for MockResource {
        type CreateParams = MockCreateParams;

        async fn create(params: &Self::CreateParams) -> crate::Result<Self> {
            if params.should_fail {
                return Err(crate::error::Error::dependency_failure("mock", "Simulated failure"));
            }

            let id = MOCK_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
            Ok(Self { id, is_valid: true })
        }

        async fn validate(&self) -> bool {
            self.is_valid
        }

        async fn reset(&mut self) -> crate::Result<()> {
            Ok(())
        }

        fn resource_type(&self) -> &'static str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_pool_creation_and_acquisition() {
        let config = PoolConfig {
            initial_size: 2,
            max_size: 5,
            ..Default::default()
        };

        let params = MockCreateParams { should_fail: false };
        let pool: ResourcePool<MockResource, _> = ResourcePool::new(config, params).await.unwrap();

        let conn1 = pool.acquire().await.unwrap();
        let conn2 = pool.acquire().await.unwrap();

        assert_eq!(conn1.get().id, 0);
        assert_eq!(conn2.get().id, 1);

        let metrics = pool.get_metrics().await;
        assert_eq!(metrics.active_connections, 2);
        assert!(metrics.total_connections >= 2);
    }

    #[tokio::test]
    async fn test_pool_return_and_reuse() {
        let config = PoolConfig {
            initial_size: 1,
            max_size: 2,
            ..Default::default()
        };

        let params = MockCreateParams { should_fail: false };
        let pool: ResourcePool<MockResource, _> = ResourcePool::new(config, params).await.unwrap();

        let first_id = {
            let conn = pool.acquire().await.unwrap();
            conn.get().id
        }; // Connection returned here

        let second_conn = pool.acquire().await.unwrap();
        assert_eq!(second_conn.get().id, first_id); // Should reuse the same resource
    }

    #[tokio::test]
    async fn test_pool_health_check() {
        let config = PoolConfig {
            initial_size: 1,
            max_size: 2,
            enable_health_checks: true,
            ..Default::default()
        };

        let params = MockCreateParams { should_fail: false };
        let pool: ResourcePool<MockResource, _> = ResourcePool::new(config, params).await.unwrap();

        assert!(pool.health_check().await);

        pool.shutdown().await;
        assert!(!pool.health_check().await);
    }

    #[tokio::test]
    async fn test_pool_metrics() {
        let config = PoolConfig {
            initial_size: 1,
            max_size: 3,
            ..Default::default()
        };

        let params = MockCreateParams { should_fail: false };
        let pool: ResourcePool<MockResource, _> = ResourcePool::new(config, params).await.unwrap();

        // Acquire and return a connection
        {
            let _conn = pool.acquire().await.unwrap();
        }

        let metrics = pool.get_metrics().await;
        assert_eq!(metrics.total_requests, 1);
        assert_eq!(metrics.successful_requests, 1);
        assert!(metrics.average_wait_time_ms >= 0.0);
    }

    #[tokio::test] 
    async fn test_concurrent_access() {
        let config = PoolConfig {
            initial_size: 2,
            max_size: 5,
            ..Default::default()
        };

        let params = MockCreateParams { should_fail: false };
        let pool: Arc<ResourcePool<MockResource, _>> = Arc::new(ResourcePool::new(config, params).await.unwrap());

        let mut handles = vec![];

        // Spawn multiple concurrent tasks
        for _ in 0..10 {
            let pool_clone = pool.clone();
            let handle = tokio::spawn(async move {
                let _conn = pool_clone.acquire().await.unwrap();
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let metrics = pool.get_metrics().await;
        assert_eq!(metrics.total_requests, 10);
        assert_eq!(metrics.successful_requests, 10);
    }
}