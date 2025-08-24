//! Intelligent Health Monitoring System
//!
//! Advanced health monitoring with predictive analytics, anomaly detection,
//! and proactive alerting for all system components.

use crate::robust_error_recovery_system::{ComponentHealth, ComponentStatus};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::interval;
use tracing::{info, warn, error, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub timestamp: u64,
    pub cpu_usage_percent: f32,
    pub memory_usage_bytes: u64,
    pub memory_usage_percent: f32,
    pub disk_usage_bytes: u64,
    pub network_latency_ms: f32,
    pub active_connections: usize,
    pub request_rate: f32,
    pub error_rate: f32,
    pub response_time_ms: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemAlert {
    pub id: String,
    pub severity: AlertSeverity,
    pub component: String,
    pub message: String,
    pub timestamp: u64,
    pub metadata: HashMap<String, String>,
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PartialOrd)]
pub enum AlertSeverity {
    Info,
    Warning, 
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthThresholds {
    pub cpu_usage_warning: f32,
    pub cpu_usage_critical: f32,
    pub memory_usage_warning: f32,
    pub memory_usage_critical: f32,
    pub response_time_warning_ms: f32,
    pub response_time_critical_ms: f32,
    pub error_rate_warning: f32,
    pub error_rate_critical: f32,
    pub network_latency_warning_ms: f32,
    pub network_latency_critical_ms: f32,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_warning: 70.0,
            cpu_usage_critical: 90.0,
            memory_usage_warning: 75.0,
            memory_usage_critical: 90.0,
            response_time_warning_ms: 1000.0,
            response_time_critical_ms: 5000.0,
            error_rate_warning: 0.05,
            error_rate_critical: 0.15,
            network_latency_warning_ms: 100.0,
            network_latency_critical_ms: 500.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HealthMonitorConfig {
    pub collection_interval_ms: u64,
    pub retention_period_hours: u64,
    pub anomaly_detection_window: usize,
    pub predictive_analysis_enabled: bool,
    pub auto_scaling_enabled: bool,
    pub alert_cooldown_minutes: u64,
}

impl Default for HealthMonitorConfig {
    fn default() -> Self {
        Self {
            collection_interval_ms: 5000,
            retention_period_hours: 24,
            anomaly_detection_window: 20,
            predictive_analysis_enabled: true,
            auto_scaling_enabled: false,
            alert_cooldown_minutes: 5,
        }
    }
}

pub struct IntelligentHealthMonitor {
    config: HealthMonitorConfig,
    thresholds: HealthThresholds,
    metrics_history: Arc<Mutex<VecDeque<HealthMetrics>>>,
    component_health: Arc<Mutex<HashMap<String, ComponentHealth>>>,
    active_alerts: Arc<Mutex<HashMap<String, SystemAlert>>>,
    alert_history: Arc<Mutex<VecDeque<SystemAlert>>>,
    baseline_metrics: Arc<Mutex<Option<HealthMetrics>>>,
    anomaly_scores: Arc<Mutex<HashMap<String, f32>>>,
    is_running: Arc<Mutex<bool>>,
}

impl IntelligentHealthMonitor {
    pub fn new(config: HealthMonitorConfig, thresholds: HealthThresholds) -> Self {
        Self {
            config,
            thresholds,
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            component_health: Arc::new(Mutex::new(HashMap::new())),
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(VecDeque::new())),
            baseline_metrics: Arc::new(Mutex::new(None)),
            anomaly_scores: Arc::new(Mutex::new(HashMap::new())),
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the health monitoring system
    pub async fn start(&self) -> crate::Result<()> {
        {
            let mut running = self.is_running.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire running lock", "monitor_start")
            })?;
            
            if *running {
                return Err(crate::error::Error::configuration(
                    "Health monitor is already running",
                    "monitor_state"
                ));
            }
            
            *running = true;
        }

        info!("Starting Intelligent Health Monitor");

        // Collect baseline metrics
        self.collect_baseline_metrics().await?;

        // Start monitoring loops
        let monitor_clone = self.clone();
        tokio::spawn(async move {
            monitor_clone.metrics_collection_loop().await;
        });

        let monitor_clone = self.clone();
        tokio::spawn(async move {
            monitor_clone.health_analysis_loop().await;
        });

        let monitor_clone = self.clone();
        tokio::spawn(async move {
            monitor_clone.alert_management_loop().await;
        });

        if self.config.predictive_analysis_enabled {
            let monitor_clone = self.clone();
            tokio::spawn(async move {
                monitor_clone.predictive_analysis_loop().await;
            });
        }

        info!("Health monitoring started successfully");
        Ok(())
    }

    /// Stop the health monitoring system
    pub async fn stop(&self) -> crate::Result<()> {
        {
            let mut running = self.is_running.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire running lock", "monitor_stop")
            })?;
            
            *running = false;
        }

        info!("Health monitor stopped");
        Ok(())
    }

    async fn collect_baseline_metrics(&self) -> crate::Result<()> {
        info!("Collecting baseline metrics");
        
        let metrics = self.collect_current_metrics().await?;
        
        {
            let mut baseline = self.baseline_metrics.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire baseline lock", "baseline_collection")
            })?;
            
            *baseline = Some(metrics);
        }

        Ok(())
    }

    async fn metrics_collection_loop(&self) {
        let mut interval = interval(Duration::from_millis(self.config.collection_interval_ms));
        
        while self.is_running() {
            interval.tick().await;
            
            if let Err(e) = self.collect_and_store_metrics().await {
                error!("Failed to collect metrics: {}", e);
            }
        }
    }

    async fn collect_and_store_metrics(&self) -> crate::Result<()> {
        let metrics = self.collect_current_metrics().await?;
        
        {
            let mut history = self.metrics_history.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire history lock", "metrics_collection")
            })?;
            
            history.push_back(metrics);
            
            // Clean up old metrics
            let max_entries = (self.config.retention_period_hours * 3600) / (self.config.collection_interval_ms / 1000);
            while history.len() > max_entries as usize {
                history.pop_front();
            }
        }

        Ok(())
    }

    async fn collect_current_metrics(&self) -> crate::Result<HealthMetrics> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // In a real implementation, these would collect actual system metrics
        // For now, we'll simulate with realistic values
        Ok(HealthMetrics {
            timestamp,
            cpu_usage_percent: self.get_cpu_usage(),
            memory_usage_bytes: self.get_memory_usage_bytes(),
            memory_usage_percent: self.get_memory_usage_percent(),
            disk_usage_bytes: self.get_disk_usage(),
            network_latency_ms: self.get_network_latency().await,
            active_connections: self.get_active_connections(),
            request_rate: self.get_request_rate(),
            error_rate: self.get_error_rate(),
            response_time_ms: self.get_response_time(),
        })
    }

    async fn health_analysis_loop(&self) {
        let mut interval = interval(Duration::from_millis(self.config.collection_interval_ms * 2));
        
        while self.is_running() {
            interval.tick().await;
            
            if let Err(e) = self.analyze_system_health().await {
                error!("Failed to analyze system health: {}", e);
            }
        }
    }

    async fn analyze_system_health(&self) -> crate::Result<()> {
        let current_metrics = self.get_latest_metrics()?;
        
        // Threshold-based analysis
        self.check_thresholds(&current_metrics).await?;
        
        // Anomaly detection
        if self.config.anomaly_detection_enabled() {
            self.detect_anomalies(&current_metrics).await?;
        }
        
        // Trend analysis
        self.analyze_trends().await?;
        
        Ok(())
    }

    async fn check_thresholds(&self, metrics: &HealthMetrics) -> crate::Result<()> {
        // CPU usage check
        if metrics.cpu_usage_percent > self.thresholds.cpu_usage_critical {
            self.create_alert(
                "cpu_critical",
                AlertSeverity::Critical,
                "system",
                &format!("CPU usage critical: {:.1}%", metrics.cpu_usage_percent),
                HashMap::new(),
            ).await?;
        } else if metrics.cpu_usage_percent > self.thresholds.cpu_usage_warning {
            self.create_alert(
                "cpu_warning", 
                AlertSeverity::Warning,
                "system",
                &format!("CPU usage high: {:.1}%", metrics.cpu_usage_percent),
                HashMap::new(),
            ).await?;
        }

        // Memory usage check
        if metrics.memory_usage_percent > self.thresholds.memory_usage_critical {
            self.create_alert(
                "memory_critical",
                AlertSeverity::Critical,
                "system", 
                &format!("Memory usage critical: {:.1}%", metrics.memory_usage_percent),
                HashMap::new(),
            ).await?;
        } else if metrics.memory_usage_percent > self.thresholds.memory_usage_warning {
            self.create_alert(
                "memory_warning",
                AlertSeverity::Warning,
                "system",
                &format!("Memory usage high: {:.1}%", metrics.memory_usage_percent),
                HashMap::new(),
            ).await?;
        }

        // Response time check
        if metrics.response_time_ms > self.thresholds.response_time_critical_ms {
            self.create_alert(
                "response_time_critical",
                AlertSeverity::Critical,
                "performance",
                &format!("Response time critical: {:.1}ms", metrics.response_time_ms),
                HashMap::new(),
            ).await?;
        }

        // Error rate check
        if metrics.error_rate > self.thresholds.error_rate_critical {
            self.create_alert(
                "error_rate_critical",
                AlertSeverity::Critical,
                "reliability",
                &format!("Error rate critical: {:.2}%", metrics.error_rate * 100.0),
                HashMap::new(),
            ).await?;
        }

        Ok(())
    }

    async fn detect_anomalies(&self, current_metrics: &HealthMetrics) -> crate::Result<()> {
        if let Some(baseline) = self.get_baseline_metrics()? {
            let mut anomaly_scores = HashMap::new();
            
            // Calculate anomaly scores for each metric
            anomaly_scores.insert("cpu_usage".to_string(), 
                self.calculate_anomaly_score(current_metrics.cpu_usage_percent, baseline.cpu_usage_percent));
            
            anomaly_scores.insert("memory_usage".to_string(),
                self.calculate_anomaly_score(current_metrics.memory_usage_percent, baseline.memory_usage_percent));
            
            anomaly_scores.insert("response_time".to_string(),
                self.calculate_anomaly_score(current_metrics.response_time_ms, baseline.response_time_ms));
            
            anomaly_scores.insert("error_rate".to_string(),
                self.calculate_anomaly_score(current_metrics.error_rate, baseline.error_rate));

            // Store anomaly scores
            {
                let mut scores = self.anomaly_scores.lock().map_err(|_| {
                    crate::error::Error::graph_processing("Failed to acquire anomaly scores lock", "anomaly_detection")
                })?;
                
                *scores = anomaly_scores.clone();
            }

            // Check for high anomaly scores
            for (metric, score) in &anomaly_scores {
                if *score > 3.0 { // 3 standard deviations
                    self.create_alert(
                        &format!("anomaly_{}", metric),
                        AlertSeverity::Warning,
                        "anomaly_detection",
                        &format!("Anomaly detected in {}: score {:.2}", metric, score),
                        [("anomaly_score".to_string(), score.to_string())].iter().cloned().collect(),
                    ).await?;
                }
            }
        }

        Ok(())
    }

    async fn analyze_trends(&self) -> crate::Result<()> {
        let history = self.get_recent_metrics_history(10)?; // Last 10 data points
        
        if history.len() < 5 {
            return Ok(()); // Not enough data for trend analysis
        }

        // Analyze CPU usage trend
        let cpu_trend = self.calculate_trend(&history, |m| m.cpu_usage_percent);
        if cpu_trend > 5.0 { // Increasing by more than 5% per measurement
            self.create_alert(
                "cpu_trend_warning",
                AlertSeverity::Warning,
                "trend_analysis",
                &format!("CPU usage trending upward: +{:.1}% per measurement", cpu_trend),
                HashMap::new(),
            ).await?;
        }

        // Analyze memory usage trend
        let memory_trend = self.calculate_trend(&history, |m| m.memory_usage_percent);
        if memory_trend > 3.0 {
            self.create_alert(
                "memory_trend_warning",
                AlertSeverity::Warning,
                "trend_analysis", 
                &format!("Memory usage trending upward: +{:.1}% per measurement", memory_trend),
                HashMap::new(),
            ).await?;
        }

        // Analyze error rate trend
        let error_trend = self.calculate_trend(&history, |m| m.error_rate);
        if error_trend > 0.01 { // 1% increase per measurement
            self.create_alert(
                "error_trend_warning",
                AlertSeverity::Critical,
                "trend_analysis",
                &format!("Error rate trending upward: +{:.2}% per measurement", error_trend * 100.0),
                HashMap::new(),
            ).await?;
        }

        Ok(())
    }

    async fn predictive_analysis_loop(&self) {
        let mut interval = interval(Duration::from_millis(self.config.collection_interval_ms * 10));
        
        while self.is_running() {
            interval.tick().await;
            
            if let Err(e) = self.run_predictive_analysis().await {
                error!("Failed to run predictive analysis: {}", e);
            }
        }
    }

    async fn run_predictive_analysis(&self) -> crate::Result<()> {
        let history = self.get_recent_metrics_history(50)?; // More data for prediction
        
        if history.len() < 20 {
            return Ok(()); // Not enough data for prediction
        }

        // Predict future resource usage
        let cpu_prediction = self.predict_metric_value(&history, |m| m.cpu_usage_percent, 5)?;
        let memory_prediction = self.predict_metric_value(&history, |m| m.memory_usage_percent, 5)?;

        // Check if predictions exceed thresholds
        if cpu_prediction > self.thresholds.cpu_usage_critical {
            self.create_alert(
                "cpu_prediction_critical",
                AlertSeverity::Warning,
                "predictive_analysis",
                &format!("CPU usage predicted to reach {:.1}% in 5 measurements", cpu_prediction),
                [("predicted_value".to_string(), cpu_prediction.to_string())].iter().cloned().collect(),
            ).await?;
        }

        if memory_prediction > self.thresholds.memory_usage_critical {
            self.create_alert(
                "memory_prediction_critical",
                AlertSeverity::Warning,
                "predictive_analysis",
                &format!("Memory usage predicted to reach {:.1}% in 5 measurements", memory_prediction),
                [("predicted_value".to_string(), memory_prediction.to_string())].iter().cloned().collect(),
            ).await?;
        }

        Ok(())
    }

    async fn alert_management_loop(&self) {
        let mut interval = interval(Duration::from_millis(self.config.collection_interval_ms));
        
        while self.is_running() {
            interval.tick().await;
            
            if let Err(e) = self.manage_alerts().await {
                error!("Failed to manage alerts: {}", e);
            }
        }
    }

    async fn manage_alerts(&self) -> crate::Result<()> {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let cooldown_seconds = self.config.alert_cooldown_minutes * 60;
        
        {
            let mut active_alerts = self.active_alerts.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire active alerts lock", "alert_management")
            })?;
            
            // Remove expired alerts
            active_alerts.retain(|_, alert| {
                now - alert.timestamp < cooldown_seconds
            });
        }

        Ok(())
    }

    async fn create_alert(
        &self,
        id: &str,
        severity: AlertSeverity,
        component: &str,
        message: &str,
        metadata: HashMap<String, String>,
    ) -> crate::Result<()> {
        let alert_id = format!("{}_{}", component, id);
        
        // Check if alert is already active (for cooldown)
        {
            let active_alerts = self.active_alerts.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire active alerts lock", "create_alert")
            })?;
            
            if active_alerts.contains_key(&alert_id) {
                return Ok(()); // Alert already active, skip
            }
        }

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        let alert = SystemAlert {
            id: alert_id.clone(),
            severity,
            component: component.to_string(),
            message: message.to_string(),
            timestamp,
            metadata,
            acknowledged: false,
        };

        // Add to active alerts
        {
            let mut active_alerts = self.active_alerts.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire active alerts lock", "create_alert")
            })?;
            
            active_alerts.insert(alert_id, alert.clone());
        }

        // Add to history
        {
            let mut history = self.alert_history.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire alert history lock", "create_alert")
            })?;
            
            history.push_back(alert.clone());
            
            // Keep only recent alerts in history
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        // Log alert
        match alert.severity {
            AlertSeverity::Emergency | AlertSeverity::Critical => {
                error!("ALERT [{}]: {}", alert.severity, alert.message);
            }
            AlertSeverity::Warning => {
                warn!("ALERT [{}]: {}", alert.severity, alert.message);
            }
            AlertSeverity::Info => {
                info!("ALERT [{}]: {}", alert.severity, alert.message);
            }
        }

        Ok(())
    }

    // Utility methods for metric collection (simplified implementations)
    fn get_cpu_usage(&self) -> f32 {
        // Simulate CPU usage - in real implementation would use system APIs
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(10.0..80.0)
    }

    fn get_memory_usage_bytes(&self) -> u64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(1_000_000_000..8_000_000_000) // 1GB to 8GB
    }

    fn get_memory_usage_percent(&self) -> f32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(20.0..75.0)
    }

    fn get_disk_usage(&self) -> u64 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(10_000_000_000..100_000_000_000) // 10GB to 100GB
    }

    async fn get_network_latency(&self) -> f32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(1.0..50.0)
    }

    fn get_active_connections(&self) -> usize {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(10..500)
    }

    fn get_request_rate(&self) -> f32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(1.0..100.0)
    }

    fn get_error_rate(&self) -> f32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(0.001..0.05)
    }

    fn get_response_time(&self) -> f32 {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(50.0..500.0)
    }

    // Helper methods
    fn is_running(&self) -> bool {
        self.is_running.lock().map(|r| *r).unwrap_or(false)
    }

    fn get_latest_metrics(&self) -> crate::Result<HealthMetrics> {
        let history = self.metrics_history.lock().map_err(|_| {
            crate::error::Error::graph_processing("Failed to acquire metrics history lock", "get_latest_metrics")
        })?;
        
        history.back().cloned().ok_or_else(|| {
            crate::error::Error::graph_processing("No metrics available", "get_latest_metrics")
        })
    }

    fn get_baseline_metrics(&self) -> crate::Result<Option<HealthMetrics>> {
        let baseline = self.baseline_metrics.lock().map_err(|_| {
            crate::error::Error::graph_processing("Failed to acquire baseline lock", "get_baseline_metrics")
        })?;
        
        Ok(baseline.clone())
    }

    fn get_recent_metrics_history(&self, count: usize) -> crate::Result<Vec<HealthMetrics>> {
        let history = self.metrics_history.lock().map_err(|_| {
            crate::error::Error::graph_processing("Failed to acquire metrics history lock", "get_recent_metrics_history")
        })?;
        
        Ok(history.iter().rev().take(count).rev().cloned().collect())
    }

    fn calculate_anomaly_score(&self, current: f32, baseline: f32) -> f32 {
        if baseline == 0.0 {
            return 0.0;
        }
        
        (current - baseline).abs() / baseline
    }

    fn calculate_trend<F>(&self, history: &[HealthMetrics], extractor: F) -> f32
    where
        F: Fn(&HealthMetrics) -> f32,
    {
        if history.len() < 2 {
            return 0.0;
        }

        let values: Vec<f32> = history.iter().map(extractor).collect();
        let n = values.len() as f32;
        let sum_x: f32 = (0..values.len()).map(|i| i as f32).sum();
        let sum_y: f32 = values.iter().sum();
        let sum_xy: f32 = values.iter().enumerate().map(|(i, &y)| i as f32 * y).sum();
        let sum_x2: f32 = (0..values.len()).map(|i| (i as f32).powi(2)).sum();

        // Linear regression slope
        (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2))
    }

    fn predict_metric_value<F>(&self, history: &[HealthMetrics], extractor: F, steps_ahead: usize) -> crate::Result<f32>
    where
        F: Fn(&HealthMetrics) -> f32,
    {
        let trend = self.calculate_trend(history, &extractor);
        let latest_value = extractor(history.last().ok_or_else(|| {
            crate::error::Error::graph_processing("No history available for prediction", "predict_metric_value")
        })?);

        Ok(latest_value + trend * steps_ahead as f32)
    }

    // Public API methods
    pub fn get_current_health_summary(&self) -> HashMap<String, ComponentHealth> {
        if let Ok(health) = self.component_health.lock() {
            health.clone()
        } else {
            HashMap::new()
        }
    }

    pub fn get_active_alerts(&self) -> Vec<SystemAlert> {
        if let Ok(alerts) = self.active_alerts.lock() {
            alerts.values().cloned().collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_alert_history(&self, limit: Option<usize>) -> Vec<SystemAlert> {
        if let Ok(history) = self.alert_history.lock() {
            match limit {
                Some(n) => history.iter().rev().take(n).rev().cloned().collect(),
                None => history.iter().cloned().collect(),
            }
        } else {
            Vec::new()
        }
    }

    pub fn acknowledge_alert(&self, alert_id: &str) -> crate::Result<()> {
        let mut alerts = self.active_alerts.lock().map_err(|_| {
            crate::error::Error::graph_processing("Failed to acquire active alerts lock", "acknowledge_alert")
        })?;

        if let Some(alert) = alerts.get_mut(alert_id) {
            alert.acknowledged = true;
            info!("Alert acknowledged: {}", alert_id);
            Ok(())
        } else {
            Err(crate::error::Error::validation(
                "Alert not found",
                alert_id,
                "existing_alert_id"
            ))
        }
    }

    pub fn get_anomaly_scores(&self) -> HashMap<String, f32> {
        if let Ok(scores) = self.anomaly_scores.lock() {
            scores.clone()
        } else {
            HashMap::new()
        }
    }

    pub fn update_component_health(&self, component: String, health: ComponentHealth) {
        if let Ok(mut health_map) = self.component_health.lock() {
            health_map.insert(component, health);
        }
    }
}

impl Clone for IntelligentHealthMonitor {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            thresholds: self.thresholds.clone(),
            metrics_history: Arc::clone(&self.metrics_history),
            component_health: Arc::clone(&self.component_health),
            active_alerts: Arc::clone(&self.active_alerts),
            alert_history: Arc::clone(&self.alert_history),
            baseline_metrics: Arc::clone(&self.baseline_metrics),
            anomaly_scores: Arc::clone(&self.anomaly_scores),
            is_running: Arc::clone(&self.is_running),
        }
    }
}

impl HealthMonitorConfig {
    fn anomaly_detection_enabled(&self) -> bool {
        true // Always enabled for this implementation
    }
}

impl std::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertSeverity::Info => write!(f, "INFO"),
            AlertSeverity::Warning => write!(f, "WARNING"),
            AlertSeverity::Critical => write!(f, "CRITICAL"),
            AlertSeverity::Emergency => write!(f, "EMERGENCY"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor_creation() {
        let config = HealthMonitorConfig::default();
        let thresholds = HealthThresholds::default();
        let monitor = IntelligentHealthMonitor::new(config, thresholds);
        
        assert!(!monitor.is_running());
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = HealthMonitorConfig::default();
        let thresholds = HealthThresholds::default();
        let monitor = IntelligentHealthMonitor::new(config, thresholds);
        
        let metrics = monitor.collect_current_metrics().await.unwrap();
        
        assert!(metrics.cpu_usage_percent >= 0.0);
        assert!(metrics.memory_usage_percent >= 0.0);
        assert!(metrics.timestamp > 0);
    }

    #[tokio::test]
    async fn test_alert_creation() {
        let config = HealthMonitorConfig::default();
        let thresholds = HealthThresholds::default();
        let monitor = IntelligentHealthMonitor::new(config, thresholds);
        
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());
        
        monitor.create_alert(
            "test_alert",
            AlertSeverity::Warning,
            "test_component",
            "Test alert message",
            metadata,
        ).await.unwrap();
        
        let active_alerts = monitor.get_active_alerts();
        assert_eq!(active_alerts.len(), 1);
        assert_eq!(active_alerts[0].message, "Test alert message");
    }
}