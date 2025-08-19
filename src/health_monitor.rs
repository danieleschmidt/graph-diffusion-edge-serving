//! Advanced Health Monitoring System for Generation 2 Robustness
//! 
//! Provides comprehensive health checks, anomaly detection, and 
//! automated recovery recommendations.

use crate::{Result, error::Error};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::collections::{HashMap, VecDeque};
use tokio::sync::{RwLock, mpsc};
use tracing::{info, warn, error, debug};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub timestamp: u64,
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_latency_ms: f64,
    pub active_connections: usize,
    pub error_rate_percent: f64,
    pub response_time_p95_ms: f64,
    pub queue_depth: usize,
    pub cache_hit_rate_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub message: String,
    pub metrics: Option<HealthMetrics>,
    pub last_check: u64,
    pub check_duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthAlert {
    pub severity: AlertSeverity,
    pub component: String,
    pub message: String,
    pub timestamp: u64,
    pub metrics: HealthMetrics,
    pub recommended_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

pub struct HealthMonitor {
    checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
    metrics_history: Arc<RwLock<VecDeque<HealthMetrics>>>,
    alert_tx: mpsc::UnboundedSender<HealthAlert>,
    alert_rx: Arc<RwLock<Option<mpsc::UnboundedReceiver<HealthAlert>>>>,
    thresholds: HealthThresholds,
    config: HealthMonitorConfig,
}

#[derive(Debug, Clone)]
pub struct HealthThresholds {
    pub cpu_warning: f64,
    pub cpu_critical: f64,
    pub memory_warning: f64,
    pub memory_critical: f64,
    pub error_rate_warning: f64,
    pub error_rate_critical: f64,
    pub response_time_warning: f64,
    pub response_time_critical: f64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            cpu_warning: 70.0,
            cpu_critical: 90.0,
            memory_warning: 80.0,
            memory_critical: 95.0,
            error_rate_warning: 5.0,
            error_rate_critical: 15.0,
            response_time_warning: 1000.0,
            response_time_critical: 5000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    pub check_interval: Duration,
    pub metrics_retention_count: usize,
    pub anomaly_detection_enabled: bool,
    pub auto_recovery_enabled: bool,
    pub alert_cooldown: Duration,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            metrics_retention_count: 1440, // 12 hours at 30s intervals
            anomaly_detection_enabled: true,
            auto_recovery_enabled: true,
            alert_cooldown: Duration::from_secs(300), // 5 minutes
        }
    }
}

impl HealthMonitor {
    pub fn new(config: HealthMonitorConfig, thresholds: HealthThresholds) -> Self {
        let (alert_tx, alert_rx) = mpsc::unbounded_channel();
        
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
            metrics_history: Arc::new(RwLock::new(VecDeque::with_capacity(config.metrics_retention_count))),
            alert_tx,
            alert_rx: Arc::new(RwLock::new(Some(alert_rx))),
            thresholds,
            config,
        }
    }

    /// Start the health monitoring loop
    pub async fn start_monitoring(&self) -> Result<()> {
        info!("Starting health monitoring system");
        
        let checks_clone = Arc::clone(&self.checks);
        let metrics_history_clone = Arc::clone(&self.metrics_history);
        let alert_tx_clone = self.alert_tx.clone();
        let thresholds_clone = self.thresholds.clone();
        let config_clone = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config_clone.check_interval);
            
            loop {
                interval.tick().await;
                
                match Self::perform_health_checks().await {
                    Ok(metrics) => {
                        Self::update_metrics_history(&metrics_history_clone, metrics.clone()).await;
                        
                        if let Err(e) = Self::analyze_and_alert(
                            &metrics,
                            &metrics_history_clone,
                            &thresholds_clone,
                            &alert_tx_clone,
                            &config_clone,
                        ).await {
                            error!("Failed to analyze health metrics: {}", e);
                        }
                    }
                    Err(e) => {
                        error!("Health check failed: {}", e);
                    }
                }
            }
        });

        Ok(())
    }

    /// Perform comprehensive system health checks
    async fn perform_health_checks() -> Result<HealthMetrics> {
        let start_time = Instant::now();
        
        // System metrics collection
        let cpu_usage = Self::get_cpu_usage().await?;
        let memory_usage = Self::get_memory_usage().await?;
        let disk_usage = Self::get_disk_usage().await?;
        let network_latency = Self::check_network_latency().await?;
        
        // Application metrics
        let active_connections = Self::get_active_connections().await?;
        let error_rate = Self::calculate_error_rate().await?;
        let response_time_p95 = Self::get_response_time_p95().await?;
        let queue_depth = Self::get_queue_depth().await?;
        let cache_hit_rate = Self::get_cache_hit_rate().await?;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Ok(HealthMetrics {
            timestamp,
            cpu_usage_percent: cpu_usage,
            memory_usage_percent: memory_usage,
            disk_usage_percent: disk_usage,
            network_latency_ms: network_latency,
            active_connections,
            error_rate_percent: error_rate,
            response_time_p95_ms: response_time_p95,
            queue_depth,
            cache_hit_rate_percent: cache_hit_rate,
        })
    }

    /// Analyze metrics and generate alerts
    async fn analyze_and_alert(
        metrics: &HealthMetrics,
        metrics_history: &Arc<RwLock<VecDeque<HealthMetrics>>>,
        thresholds: &HealthThresholds,
        alert_tx: &mpsc::UnboundedSender<HealthAlert>,
        config: &HealthMonitorConfig,
    ) -> Result<()> {
        let mut alerts = Vec::new();

        // CPU usage analysis
        if metrics.cpu_usage_percent > thresholds.cpu_critical {
            alerts.push(HealthAlert {
                severity: AlertSeverity::Critical,
                component: "CPU".to_string(),
                message: format!("Critical CPU usage: {:.1}%", metrics.cpu_usage_percent),
                timestamp: metrics.timestamp,
                metrics: metrics.clone(),
                recommended_actions: vec![
                    "Scale up compute resources".to_string(),
                    "Enable request throttling".to_string(),
                    "Investigate high CPU processes".to_string(),
                ],
            });
        } else if metrics.cpu_usage_percent > thresholds.cpu_warning {
            alerts.push(HealthAlert {
                severity: AlertSeverity::Warning,
                component: "CPU".to_string(),
                message: format!("High CPU usage: {:.1}%", metrics.cpu_usage_percent),
                timestamp: metrics.timestamp,
                metrics: metrics.clone(),
                recommended_actions: vec![
                    "Monitor CPU trends".to_string(),
                    "Consider scaling preparation".to_string(),
                ],
            });
        }

        // Memory usage analysis
        if metrics.memory_usage_percent > thresholds.memory_critical {
            alerts.push(HealthAlert {
                severity: AlertSeverity::Critical,
                component: "Memory".to_string(),
                message: format!("Critical memory usage: {:.1}%", metrics.memory_usage_percent),
                timestamp: metrics.timestamp,
                metrics: metrics.clone(),
                recommended_actions: vec![
                    "Trigger garbage collection".to_string(),
                    "Clear caches".to_string(),
                    "Scale up memory".to_string(),
                    "Enable memory pressure relief".to_string(),
                ],
            });
        }

        // Error rate analysis
        if metrics.error_rate_percent > thresholds.error_rate_critical {
            alerts.push(HealthAlert {
                severity: AlertSeverity::Critical,
                component: "ErrorRate".to_string(),
                message: format!("Critical error rate: {:.1}%", metrics.error_rate_percent),
                timestamp: metrics.timestamp,
                metrics: metrics.clone(),
                recommended_actions: vec![
                    "Enable circuit breakers".to_string(),
                    "Reduce traffic load".to_string(),
                    "Investigate error patterns".to_string(),
                    "Enable graceful degradation".to_string(),
                ],
            });
        }

        // Response time analysis
        if metrics.response_time_p95_ms > thresholds.response_time_critical {
            alerts.push(HealthAlert {
                severity: AlertSeverity::Critical,
                component: "ResponseTime".to_string(),
                message: format!("Critical response time: {:.1}ms", metrics.response_time_p95_ms),
                timestamp: metrics.timestamp,
                metrics: metrics.clone(),
                recommended_actions: vec![
                    "Enable request prioritization".to_string(),
                    "Increase cache utilization".to_string(),
                    "Scale processing capacity".to_string(),
                ],
            });
        }

        // Anomaly detection
        if config.anomaly_detection_enabled {
            if let Some(anomalies) = Self::detect_anomalies(metrics, metrics_history).await? {
                alerts.extend(anomalies);
            }
        }

        // Send alerts
        for alert in alerts {
            if let Err(e) = alert_tx.send(alert) {
                error!("Failed to send health alert: {}", e);
            }
        }

        Ok(())
    }

    /// Detect anomalies using statistical analysis
    async fn detect_anomalies(
        current: &HealthMetrics,
        history: &Arc<RwLock<VecDeque<HealthMetrics>>>,
    ) -> Result<Option<Vec<HealthAlert>>> {
        let history_guard = history.read().await;
        
        if history_guard.len() < 30 {
            return Ok(None); // Need sufficient history for anomaly detection
        }

        let mut alerts = Vec::new();

        // Calculate moving averages and standard deviations
        let recent_metrics: Vec<&HealthMetrics> = history_guard.iter().rev().take(30).collect();
        
        let avg_response_time: f64 = recent_metrics.iter()
            .map(|m| m.response_time_p95_ms)
            .sum::<f64>() / recent_metrics.len() as f64;
        
        let response_time_variance: f64 = recent_metrics.iter()
            .map(|m| (m.response_time_p95_ms - avg_response_time).powi(2))
            .sum::<f64>() / recent_metrics.len() as f64;
        
        let response_time_stddev = response_time_variance.sqrt();

        // Detect response time anomalies (more than 3 standard deviations)
        if (current.response_time_p95_ms - avg_response_time).abs() > 3.0 * response_time_stddev {
            alerts.push(HealthAlert {
                severity: AlertSeverity::Warning,
                component: "AnomalyDetection".to_string(),
                message: format!(
                    "Anomalous response time detected: {:.1}ms (avg: {:.1}ms, stddev: {:.1}ms)",
                    current.response_time_p95_ms, avg_response_time, response_time_stddev
                ),
                timestamp: current.timestamp,
                metrics: current.clone(),
                recommended_actions: vec![
                    "Investigate recent changes".to_string(),
                    "Check for unusual load patterns".to_string(),
                    "Review system logs".to_string(),
                ],
            });
        }

        Ok(if alerts.is_empty() { None } else { Some(alerts) })
    }

    /// Update metrics history with size limit
    async fn update_metrics_history(
        history: &Arc<RwLock<VecDeque<HealthMetrics>>>,
        metrics: HealthMetrics,
    ) {
        let mut history_guard = history.write().await;
        
        if history_guard.len() >= history_guard.capacity() {
            history_guard.pop_front();
        }
        
        history_guard.push_back(metrics);
    }

    /// Get current health status
    pub async fn get_health_status(&self) -> Result<Vec<HealthCheck>> {
        let checks = self.checks.read().await;
        Ok(checks.values().cloned().collect())
    }

    /// Get metrics history
    pub async fn get_metrics_history(&self, limit: Option<usize>) -> Result<Vec<HealthMetrics>> {
        let history = self.metrics_history.read().await;
        let limit = limit.unwrap_or(history.len());
        
        Ok(history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect())
    }

    // Mock implementations for system metrics (would use real system APIs in production)
    async fn get_cpu_usage() -> Result<f64> {
        // Mock implementation - would use system APIs
        Ok(rand::random::<f64>() * 100.0)
    }

    async fn get_memory_usage() -> Result<f64> {
        // Mock implementation - would use system APIs
        Ok(rand::random::<f64>() * 100.0)
    }

    async fn get_disk_usage() -> Result<f64> {
        // Mock implementation - would check filesystem
        Ok(rand::random::<f64>() * 100.0)
    }

    async fn check_network_latency() -> Result<f64> {
        // Mock implementation - would ping external services
        Ok(rand::random::<f64>() * 50.0 + 10.0)
    }

    async fn get_active_connections() -> Result<usize> {
        // Mock implementation - would check network connections
        Ok((rand::random::<f64>() * 1000.0) as usize)
    }

    async fn calculate_error_rate() -> Result<f64> {
        // Mock implementation - would calculate from metrics
        Ok(rand::random::<f64>() * 10.0)
    }

    async fn get_response_time_p95() -> Result<f64> {
        // Mock implementation - would calculate from histogram
        Ok(rand::random::<f64>() * 1000.0 + 50.0)
    }

    async fn get_queue_depth() -> Result<usize> {
        // Mock implementation - would check queue sizes
        Ok((rand::random::<f64>() * 100.0) as usize)
    }

    async fn get_cache_hit_rate() -> Result<f64> {
        // Mock implementation - would calculate from cache metrics
        Ok(rand::random::<f64>() * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor_creation() {
        let config = HealthMonitorConfig::default();
        let thresholds = HealthThresholds::default();
        let monitor = HealthMonitor::new(config, thresholds);
        
        let status = monitor.get_health_status().await.unwrap();
        assert!(status.is_empty()); // No checks initially
    }

    #[tokio::test]
    async fn test_health_checks() {
        let metrics = HealthMonitor::perform_health_checks().await.unwrap();
        assert!(metrics.cpu_usage_percent >= 0.0);
        assert!(metrics.memory_usage_percent >= 0.0);
    }
}