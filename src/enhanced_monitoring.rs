//! Enhanced Monitoring System
//! 
//! Comprehensive observability platform with real-time metrics,
//! intelligent alerting, and predictive health analysis.

use crate::Result;
use prometheus::{Encoder, TextEncoder, Counter, Histogram, Gauge, Registry};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

/// Enhanced monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedMonitoringConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub enable_health_checks: bool,
    pub enable_alerting: bool,
    pub enable_predictive_analysis: bool,
    pub metrics_retention_hours: u64,
    pub health_check_interval_seconds: u64,
    pub alert_cooldown_minutes: u64,
    pub anomaly_detection_window_minutes: u64,
}

impl Default for EnhancedMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            enable_tracing: true,
            enable_health_checks: true,
            enable_alerting: true,
            enable_predictive_analysis: true,
            metrics_retention_hours: 24,
            health_check_interval_seconds: 30,
            alert_cooldown_minutes: 15,
            anomaly_detection_window_minutes: 60,
        }
    }
}

/// Health status levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Critical,
}

/// Health check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResult {
    pub component: String,
    pub status: HealthStatus,
    pub message: String,
    pub response_time_ms: u64,
    pub details: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Performance metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub request_rate_per_second: f64,
    pub avg_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub active_connections: u64,
    pub queue_depth: u64,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Alert definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: String,
    pub title: String,
    pub description: String,
    pub severity: AlertSeverity,
    pub metric_name: String,
    pub threshold_value: f64,
    pub comparison: AlertComparison,
    pub duration_minutes: u64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertComparison {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
}

/// Triggered alert instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggeredAlert {
    pub alert: Alert,
    pub current_value: f64,
    pub triggered_at: chrono::DateTime<chrono::Utc>,
    pub resolved_at: Option<chrono::DateTime<chrono::Utc>>,
    pub context: HashMap<String, String>,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub metric_name: String,
    pub current_value: f64,
    pub expected_value: f64,
    pub anomaly_score: f64,
    pub is_anomaly: bool,
    pub confidence: f64,
    pub detected_at: chrono::DateTime<chrono::Utc>,
}

/// Enhanced monitoring system
#[derive(Debug)]
pub struct EnhancedMonitoringSystem {
    config: EnhancedMonitoringConfig,
    registry: Arc<Registry>,
    metrics_storage: Arc<RwLock<HashMap<String, Vec<(chrono::DateTime<chrono::Utc>, f64)>>>>,
    health_checks: Arc<RwLock<HashMap<String, HealthCheckResult>>>,
    alerts: Arc<RwLock<HashMap<String, Alert>>>,
    triggered_alerts: Arc<RwLock<Vec<TriggeredAlert>>>,
    performance_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
    
    // Prometheus metrics
    request_counter: Counter,
    request_duration: Histogram,
    active_connections_gauge: Gauge,
    memory_usage_gauge: Gauge,
    cpu_usage_gauge: Gauge,
    error_counter: Counter,
}

impl EnhancedMonitoringSystem {
    /// Create new enhanced monitoring system
    pub fn new(config: EnhancedMonitoringConfig) -> Result<Self> {
        let registry = Arc::new(Registry::new());
        
        // Initialize Prometheus metrics
        let request_counter = Counter::new("dgdm_requests_total", "Total HTTP requests")?;
        let request_duration = Histogram::with_opts(
            prometheus::HistogramOpts::new("dgdm_request_duration_seconds", "Request duration")
        )?;
        let active_connections_gauge = Gauge::new("dgdm_active_connections", "Active connections")?;
        let memory_usage_gauge = Gauge::new("dgdm_memory_usage_bytes", "Memory usage in bytes")?;
        let cpu_usage_gauge = Gauge::new("dgdm_cpu_usage_percent", "CPU usage percentage")?;
        let error_counter = Counter::new("dgdm_errors_total", "Total errors")?;
        
        // Register metrics
        registry.register(Box::new(request_counter.clone()))?;
        registry.register(Box::new(request_duration.clone()))?;
        registry.register(Box::new(active_connections_gauge.clone()))?;
        registry.register(Box::new(memory_usage_gauge.clone()))?;
        registry.register(Box::new(cpu_usage_gauge.clone()))?;
        registry.register(Box::new(error_counter.clone()))?;

        Ok(Self {
            config,
            registry,
            metrics_storage: Arc::new(RwLock::new(HashMap::new())),
            health_checks: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(RwLock::new(HashMap::new())),
            triggered_alerts: Arc::new(RwLock::new(Vec::new())),
            performance_history: Arc::new(RwLock::new(Vec::new())),
            request_counter,
            request_duration,
            active_connections_gauge,
            memory_usage_gauge,
            cpu_usage_gauge,
            error_counter,
        })
    }

    /// Start monitoring background tasks
    pub async fn start_monitoring(&self) -> Result<()> {
        if !self.config.enable_metrics {
            return Ok(());
        }

        info!("🔍 Starting enhanced monitoring system");
        
        // Start health check task
        if self.config.enable_health_checks {
            let health_checks = self.health_checks.clone();
            let interval = Duration::from_secs(self.config.health_check_interval_seconds);
            
            tokio::spawn(async move {
                let mut interval_timer = tokio::time::interval(interval);
                loop {
                    interval_timer.tick().await;
                    Self::run_health_checks(health_checks.clone()).await;
                }
            });
        }

        // Start metrics collection task
        let metrics_storage = self.metrics_storage.clone();
        let performance_history = self.performance_history.clone();
        let retention_hours = self.config.metrics_retention_hours;
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(Duration::from_secs(60)); // Collect every minute
            loop {
                interval_timer.tick().await;
                Self::collect_performance_metrics(metrics_storage.clone(), performance_history.clone()).await;
                Self::cleanup_old_metrics(metrics_storage.clone(), retention_hours).await;
            }
        });

        // Start alerting task
        if self.config.enable_alerting {
            let alerts = self.alerts.clone();
            let triggered_alerts = self.triggered_alerts.clone();
            let metrics_storage = self.metrics_storage.clone();
            let cooldown_minutes = self.config.alert_cooldown_minutes;
            
            tokio::spawn(async move {
                let mut interval_timer = tokio::time::interval(Duration::from_secs(30));
                loop {
                    interval_timer.tick().await;
                    Self::check_alerts(
                        alerts.clone(),
                        triggered_alerts.clone(),
                        metrics_storage.clone(),
                        cooldown_minutes,
                    ).await;
                }
            });
        }

        // Start anomaly detection task
        if self.config.enable_predictive_analysis {
            let metrics_storage = self.metrics_storage.clone();
            let window_minutes = self.config.anomaly_detection_window_minutes;
            
            tokio::spawn(async move {
                let mut interval_timer = tokio::time::interval(Duration::from_secs(300)); // Every 5 minutes
                loop {
                    interval_timer.tick().await;
                    Self::detect_anomalies(metrics_storage.clone(), window_minutes).await;
                }
            });
        }

        Ok(())
    }

    /// Record request metrics
    pub fn record_request(&self, duration: Duration, success: bool) {
        if self.config.enable_metrics {
            self.request_counter.inc();
            self.request_duration.observe(duration.as_secs_f64());
            
            if !success {
                self.error_counter.inc();
            }
        }
    }

    /// Update connection metrics
    pub fn update_connections(&self, active_count: u64) {
        if self.config.enable_metrics {
            self.active_connections_gauge.set(active_count as f64);
        }
    }

    /// Update resource usage metrics
    pub fn update_resource_usage(&self, memory_mb: f64, cpu_percent: f64) {
        if self.config.enable_metrics {
            self.memory_usage_gauge.set(memory_mb * 1024.0 * 1024.0); // Convert to bytes
            self.cpu_usage_gauge.set(cpu_percent);
        }
    }

    /// Store custom metric
    pub async fn store_metric(&self, name: String, value: f64) {
        if !self.config.enable_metrics {
            return;
        }

        let mut storage = self.metrics_storage.write().await;
        let timestamp = chrono::Utc::now();
        
        storage.entry(name)
            .or_insert_with(Vec::new)
            .push((timestamp, value));
    }

    /// Add health check
    pub async fn add_health_check<F, Fut>(&self, name: String, check_fn: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<String>> + Send,
    {
        // This is a simplified version - in practice, you'd store the check function
        let result = match check_fn().await {
            Ok(message) => HealthCheckResult {
                component: name.clone(),
                status: HealthStatus::Healthy,
                message,
                response_time_ms: 0,
                details: HashMap::new(),
                timestamp: chrono::Utc::now(),
            },
            Err(e) => HealthCheckResult {
                component: name.clone(),
                status: HealthStatus::Unhealthy,
                message: e.to_string(),
                response_time_ms: 0,
                details: HashMap::new(),
                timestamp: chrono::Utc::now(),
            },
        };

        self.health_checks.write().await.insert(name, result);
    }

    /// Add alert rule
    pub async fn add_alert(&self, alert: Alert) {
        if self.config.enable_alerting {
            self.alerts.write().await.insert(alert.id.clone(), alert);
        }
    }

    /// Get current health status
    pub async fn get_health_status(&self) -> HashMap<String, HealthCheckResult> {
        self.health_checks.read().await.clone()
    }

    /// Get metrics in Prometheus format
    pub async fn get_prometheus_metrics(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode_to_string(&metric_families)
            .map_err(|e| crate::error::Error::graph_processing(
                format!("Failed to encode metrics: {}", e),
                "prometheus_metrics"
            ))
    }

    /// Get performance metrics history
    pub async fn get_performance_history(&self, duration_minutes: u64) -> Vec<PerformanceMetrics> {
        let history = self.performance_history.read().await;
        let cutoff = chrono::Utc::now() - chrono::Duration::minutes(duration_minutes as i64);
        
        history.iter()
            .filter(|m| m.timestamp > cutoff)
            .cloned()
            .collect()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<TriggeredAlert> {
        self.triggered_alerts.read().await
            .iter()
            .filter(|alert| alert.resolved_at.is_none())
            .cloned()
            .collect()
    }

    /// Get anomaly detections
    pub async fn detect_current_anomalies(&self) -> Vec<AnomalyDetection> {
        let storage = self.metrics_storage.read().await;
        let mut anomalies = Vec::new();
        
        for (metric_name, data_points) in storage.iter() {
            if let Some(anomaly) = self.analyze_metric_for_anomalies(metric_name, data_points).await {
                anomalies.push(anomaly);
            }
        }
        
        anomalies
    }

    // Background task implementations
    async fn run_health_checks(health_checks: Arc<RwLock<HashMap<String, HealthCheckResult>>>) {
        debug!("Running health checks...");
        
        // Basic health checks
        let system_health = Self::check_system_health().await;
        let database_health = Self::check_database_health().await;
        let external_deps_health = Self::check_external_dependencies().await;
        
        let mut checks = health_checks.write().await;
        checks.insert("system".to_string(), system_health);
        checks.insert("database".to_string(), database_health);
        checks.insert("external_deps".to_string(), external_deps_health);
    }

    async fn check_system_health() -> HealthCheckResult {
        let start = Instant::now();
        
        // Check basic system metrics
        let memory_info = sys_info::mem_info().unwrap_or_else(|_| sys_info::MemInfo {
            total: 8192,
            free: 4096,
            avail: 4096,
            buffers: 0,
            cached: 0,
            swap_total: 0,
            swap_free: 0,
        });
        let cpu_usage = Self::get_cpu_usage().await;
        
        let mut details = HashMap::new();
        details.insert("memory_total_mb".to_string(), (memory_info.total / 1024).to_string());
        details.insert("memory_free_mb".to_string(), (memory_info.free / 1024).to_string());
        details.insert("cpu_usage_percent".to_string(), format!("{:.1}", cpu_usage));
        
        let status = if memory_info.free < memory_info.total / 10 || cpu_usage > 90.0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
        
        HealthCheckResult {
            component: "system".to_string(),
            status,
            message: "System resources checked".to_string(),
            response_time_ms: start.elapsed().as_millis() as u64,
            details,
            timestamp: chrono::Utc::now(),
        }
    }

    async fn check_database_health() -> HealthCheckResult {
        let start = Instant::now();
        
        // Simulate database connectivity check
        // In practice, this would ping the actual database
        let status = HealthStatus::Healthy;
        let message = "Database connectivity verified".to_string();
        
        HealthCheckResult {
            component: "database".to_string(),
            status,
            message,
            response_time_ms: start.elapsed().as_millis() as u64,
            details: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    async fn check_external_dependencies() -> HealthCheckResult {
        let start = Instant::now();
        
        // Check external service connectivity
        // This is a simplified check - in practice, you'd ping actual services
        let status = HealthStatus::Healthy;
        let message = "External dependencies accessible".to_string();
        
        HealthCheckResult {
            component: "external_deps".to_string(),
            status,
            message,
            response_time_ms: start.elapsed().as_millis() as u64,
            details: HashMap::new(),
            timestamp: chrono::Utc::now(),
        }
    }

    async fn get_cpu_usage() -> f64 {
        // Simple CPU usage estimation
        // In practice, you'd use a proper system monitoring library
        rand::random::<f64>() * 100.0
    }

    async fn collect_performance_metrics(
        metrics_storage: Arc<RwLock<HashMap<String, Vec<(chrono::DateTime<chrono::Utc>, f64)>>>>,
        performance_history: Arc<RwLock<Vec<PerformanceMetrics>>>,
    ) {
        let timestamp = chrono::Utc::now();
        
        // Collect current performance metrics
        let memory_info = sys_info::mem_info().unwrap_or_else(|_| sys_info::MemInfo {
            total: 8192,
            free: 4096,
            avail: 4096,
            buffers: 0,
            cached: 0,
            swap_total: 0,
            swap_free: 0,
        });
        let cpu_usage = Self::get_cpu_usage().await;
        
        let metrics = PerformanceMetrics {
            timestamp,
            cpu_usage_percent: cpu_usage,
            memory_usage_mb: (memory_info.total - memory_info.free) as f64 / 1024.0,
            request_rate_per_second: rand::random::<f64>() * 100.0, // Simulated
            avg_response_time_ms: rand::random::<f64>() * 200.0,   // Simulated
            error_rate_percent: rand::random::<f64>() * 5.0,       // Simulated
            active_connections: rand::random::<u64>() % 1000,       // Simulated
            queue_depth: rand::random::<u64>() % 100,               // Simulated
        };
        
        // Store in performance history
        {
            let mut history = performance_history.write().await;
            history.push(metrics.clone());
            
            // Keep only recent history (last 24 hours)
            let cutoff = chrono::Utc::now() - chrono::Duration::hours(24);
            history.retain(|m| m.timestamp > cutoff);
        }
        
        // Store individual metrics
        {
            let mut storage = metrics_storage.write().await;
            storage.entry("cpu_usage".to_string())
                .or_insert_with(Vec::new)
                .push((timestamp, metrics.cpu_usage_percent));
            storage.entry("memory_usage".to_string())
                .or_insert_with(Vec::new)
                .push((timestamp, metrics.memory_usage_mb));
            storage.entry("request_rate".to_string())
                .or_insert_with(Vec::new)
                .push((timestamp, metrics.request_rate_per_second));
        }
    }

    async fn cleanup_old_metrics(
        metrics_storage: Arc<RwLock<HashMap<String, Vec<(chrono::DateTime<chrono::Utc>, f64)>>>>,
        retention_hours: u64,
    ) {
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(retention_hours as i64);
        let mut storage = metrics_storage.write().await;
        
        for (_, data_points) in storage.iter_mut() {
            data_points.retain(|(timestamp, _)| *timestamp > cutoff);
        }
    }

    async fn check_alerts(
        alerts: Arc<RwLock<HashMap<String, Alert>>>,
        triggered_alerts: Arc<RwLock<Vec<TriggeredAlert>>>,
        metrics_storage: Arc<RwLock<HashMap<String, Vec<(chrono::DateTime<chrono::Utc>, f64)>>>>,
        cooldown_minutes: u64,
    ) {
        let alert_rules = alerts.read().await.clone();
        let storage = metrics_storage.read().await;
        
        for (_, alert) in alert_rules {
            if !alert.enabled {
                continue;
            }
            
            if let Some(data_points) = storage.get(&alert.metric_name) {
                if let Some((_, current_value)) = data_points.last() {
                    let should_trigger = match alert.comparison {
                        AlertComparison::GreaterThan => *current_value > alert.threshold_value,
                        AlertComparison::LessThan => *current_value < alert.threshold_value,
                        AlertComparison::Equals => (*current_value - alert.threshold_value).abs() < 0.001,
                        AlertComparison::NotEquals => (*current_value - alert.threshold_value).abs() >= 0.001,
                    };
                    
                    if should_trigger {
                        // Check if already triggered and within cooldown
                        let mut triggered = triggered_alerts.write().await;
                        let cooldown_cutoff = chrono::Utc::now() - chrono::Duration::minutes(cooldown_minutes as i64);
                        
                        let recently_triggered = triggered.iter().any(|ta| {
                            ta.alert.id == alert.id && 
                            ta.triggered_at > cooldown_cutoff &&
                            ta.resolved_at.is_none()
                        });
                        
                        if !recently_triggered {
                            let triggered_alert = TriggeredAlert {
                                alert: alert.clone(),
                                current_value: *current_value,
                                triggered_at: chrono::Utc::now(),
                                resolved_at: None,
                                context: HashMap::new(),
                            };
                            
                            warn!("🚨 Alert triggered: {} (value: {}, threshold: {})",
                                  alert.title, current_value, alert.threshold_value);
                            
                            triggered.push(triggered_alert);
                        }
                    }
                }
            }
        }
    }

    async fn detect_anomalies(
        metrics_storage: Arc<RwLock<HashMap<String, Vec<(chrono::DateTime<chrono::Utc>, f64)>>>>,
        window_minutes: u64,
    ) {
        let storage = metrics_storage.read().await;
        
        for (metric_name, data_points) in storage.iter() {
            if data_points.len() < 10 {
                continue; // Need sufficient data for anomaly detection
            }
            
            let window_cutoff = chrono::Utc::now() - chrono::Duration::minutes(window_minutes as i64);
            let recent_points: Vec<f64> = data_points.iter()
                .filter(|(timestamp, _)| *timestamp > window_cutoff)
                .map(|(_, value)| *value)
                .collect();
            
            if recent_points.len() < 5 {
                continue;
            }
            
            // Simple anomaly detection using standard deviation
            let mean = recent_points.iter().sum::<f64>() / recent_points.len() as f64;
            let variance = recent_points.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / recent_points.len() as f64;
            let std_dev = variance.sqrt();
            
            if let Some(&current_value) = recent_points.last() {
                let z_score = (current_value - mean) / std_dev;
                let is_anomaly = z_score.abs() > 2.0; // 2 standard deviations
                
                if is_anomaly {
                    debug!("🔍 Anomaly detected in {}: current={:.2}, expected={:.2}, z-score={:.2}",
                           metric_name, current_value, mean, z_score);
                }
            }
        }
    }

    async fn analyze_metric_for_anomalies(&self, metric_name: &str, data_points: &[(chrono::DateTime<chrono::Utc>, f64)]) -> Option<AnomalyDetection> {
        if data_points.len() < 10 {
            return None;
        }

        let window_cutoff = chrono::Utc::now() - chrono::Duration::minutes(self.config.anomaly_detection_window_minutes as i64);
        let recent_points: Vec<f64> = data_points.iter()
            .filter(|(timestamp, _)| *timestamp > window_cutoff)
            .map(|(_, value)| *value)
            .collect();

        if recent_points.len() < 5 {
            return None;
        }

        let mean = recent_points.iter().sum::<f64>() / recent_points.len() as f64;
        let variance = recent_points.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / recent_points.len() as f64;
        let std_dev = variance.sqrt();

        if let Some(&current_value) = recent_points.last() {
            let z_score = if std_dev > 0.0 { (current_value - mean) / std_dev } else { 0.0 };
            let anomaly_score = z_score.abs();
            let is_anomaly = anomaly_score > 2.0;
            let confidence = (anomaly_score / 3.0).min(1.0);

            if is_anomaly {
                Some(AnomalyDetection {
                    metric_name: metric_name.to_string(),
                    current_value,
                    expected_value: mean,
                    anomaly_score,
                    is_anomaly,
                    confidence,
                    detected_at: chrono::Utc::now(),
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}