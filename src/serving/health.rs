//! Comprehensive health checking and monitoring with circuit breaker integration

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::serving::circuit_breaker::{CircuitBreakerRegistry, CircuitState};
use crate::core::DGDMProcessor;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: ServiceStatus,
    pub checks: HashMap<String, HealthCheck>,
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub status: CheckStatus,
    pub message: String,
    pub last_check: String,
    pub duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

pub struct HealthChecker {
    checks: Arc<RwLock<HashMap<String, Box<dyn HealthCheckTrait + Send + Sync>>>>,
    start_time: std::time::Instant,
    circuit_breaker_registry: Arc<CircuitBreakerRegistry>,
    processor: Option<Arc<DGDMProcessor>>,
}

#[async_trait::async_trait]
pub trait HealthCheckTrait {
    async fn check(&self) -> HealthCheck;
    fn name(&self) -> &str;
}

impl HealthChecker {
    pub fn new() -> Self {
        Self {
            checks: Arc::new(RwLock::new(HashMap::new())),
            start_time: std::time::Instant::now(),
            circuit_breaker_registry: Arc::new(CircuitBreakerRegistry::new()),
            processor: None,
        }
    }

    pub fn with_circuit_breaker_registry(mut self, registry: Arc<CircuitBreakerRegistry>) -> Self {
        self.circuit_breaker_registry = registry;
        self
    }

    pub fn with_processor(mut self, processor: Arc<DGDMProcessor>) -> Self {
        self.processor = Some(processor);
        self
    }

    pub async fn add_check(&self, check: Box<dyn HealthCheckTrait + Send + Sync>) {
        let mut checks = self.checks.write().await;
        checks.insert(check.name().to_string(), check);
    }

    pub async fn check_health(&self) -> HealthStatus {
        let _start = std::time::Instant::now();
        let mut all_checks = HashMap::new();
        let mut overall_status = ServiceStatus::Healthy;

        // Run regular health checks
        let checks = self.checks.read().await;
        for (name, checker) in checks.iter() {
            let check_result = checker.check().await;
            
            match check_result.status {
                CheckStatus::Warn if matches!(overall_status, ServiceStatus::Healthy) => {
                    overall_status = ServiceStatus::Degraded;
                }
                CheckStatus::Fail => {
                    overall_status = ServiceStatus::Unhealthy;
                }
                _ => {}
            }
            
            all_checks.insert(name.clone(), check_result);
        }

        // Check circuit breakers
        let cb_check = self.check_circuit_breakers().await;
        match cb_check.status {
            CheckStatus::Warn if matches!(overall_status, ServiceStatus::Healthy) => {
                overall_status = ServiceStatus::Degraded;
            }
            CheckStatus::Fail => {
                overall_status = ServiceStatus::Unhealthy;
            }
            _ => {}
        }
        all_checks.insert("circuit_breakers".to_string(), cb_check);

        // Check processor health if available
        if let Some(processor_check) = self.check_processor().await {
            match processor_check.status {
                CheckStatus::Warn if matches!(overall_status, ServiceStatus::Healthy) => {
                    overall_status = ServiceStatus::Degraded;
                }
                CheckStatus::Fail => {
                    overall_status = ServiceStatus::Unhealthy;
                }
                _ => {}
            }
            all_checks.insert("processor".to_string(), processor_check);
        }

        HealthStatus {
            status: overall_status,
            checks: all_checks,
            uptime_seconds: self.start_time.elapsed().as_secs(),
            memory_usage_mb: get_memory_usage_mb(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    async fn check_circuit_breakers(&self) -> HealthCheck {
        let start = std::time::Instant::now();
        
        let (healthy, total) = self.circuit_breaker_registry.health_check().await;
        let metrics = self.circuit_breaker_registry.get_all_metrics().await;
        
        let duration = start.elapsed();
        
        let (status, message) = if total == 0 {
            (CheckStatus::Pass, "No circuit breakers configured".to_string())
        } else if healthy == total {
            (CheckStatus::Pass, format!("All {} circuit breakers healthy", total))
        } else if healthy == 0 {
            let mut details = format!("All {} circuit breakers are open: ", total);
            for (name, metric) in metrics.iter() {
                if metric.state != CircuitState::Closed {
                    details.push_str(&format!("{} ({:?}) ", name, metric.state));
                }
            }
            (CheckStatus::Fail, details)
        } else {
            let unhealthy = total - healthy;
            let mut details = format!("{} of {} circuit breakers unhealthy: ", unhealthy, total);
            for (name, metric) in metrics.iter() {
                if metric.state != CircuitState::Closed {
                    details.push_str(&format!("{} ({:?}) ", name, metric.state));
                }
            }
            (CheckStatus::Warn, details)
        };

        HealthCheck {
            status,
            message,
            last_check: chrono::Utc::now().to_rfc3339(),
            duration_ms: duration.as_secs_f64() * 1000.0,
        }
    }

    async fn check_processor(&self) -> Option<HealthCheck> {
        let processor = self.processor.as_ref()?;
        let start = std::time::Instant::now();
        
        // Create a simple test graph to validate processor health
        let mut test_graph = crate::core::graph::Graph::new();
        test_graph.add_node(crate::core::graph::Node {
            id: 1,
            features: vec![1.0, 0.0],
            label: None,
        });
        test_graph.add_node(crate::core::graph::Node {
            id: 2,
            features: vec![0.0, 1.0],
            label: None,
        });
        test_graph.add_edge(crate::core::graph::Edge {
            source: 1,
            target: 2,
            weight: 1.0,
            edge_type: None,
        });

        let (status, message) = match test_graph.to_compact() {
            Ok(compact_graph) => {
                if processor.can_process(&compact_graph) {
                    match processor.process(&compact_graph) {
                        Ok(_) => (CheckStatus::Pass, "Processor functioning normally".to_string()),
                        Err(e) => (CheckStatus::Fail, format!("Processor failed test: {}", e)),
                    }
                } else {
                    (CheckStatus::Warn, "Processor cannot handle test graph".to_string())
                }
            }
            Err(e) => (CheckStatus::Fail, format!("Test graph creation failed: {}", e)),
        };

        let duration = start.elapsed();

        Some(HealthCheck {
            status,
            message,
            last_check: chrono::Utc::now().to_rfc3339(),
            duration_ms: duration.as_secs_f64() * 1000.0,
        })
    }
}

// Memory health check
pub struct MemoryHealthCheck {
    max_memory_mb: f64,
}

impl MemoryHealthCheck {
    pub fn new(max_memory_mb: f64) -> Self {
        Self { max_memory_mb }
    }
}

#[async_trait::async_trait]
impl HealthCheckTrait for MemoryHealthCheck {
    async fn check(&self) -> HealthCheck {
        let start = std::time::Instant::now();
        let current_memory = get_memory_usage_mb();
        let duration = start.elapsed();

        let (status, message) = if current_memory > self.max_memory_mb {
            (CheckStatus::Fail, format!("Memory usage {:.1}MB exceeds limit {:.1}MB", current_memory, self.max_memory_mb))
        } else if current_memory > self.max_memory_mb * 0.8 {
            (CheckStatus::Warn, format!("Memory usage {:.1}MB is high (limit: {:.1}MB)", current_memory, self.max_memory_mb))
        } else {
            (CheckStatus::Pass, format!("Memory usage {:.1}MB is normal", current_memory))
        };

        HealthCheck {
            status,
            message,
            last_check: chrono::Utc::now().to_rfc3339(),
            duration_ms: duration.as_secs_f64() * 1000.0,
        }
    }

    fn name(&self) -> &str {
        "memory"
    }
}

// Disk space health check
pub struct DiskSpaceHealthCheck {
    min_free_gb: f64,
}

impl DiskSpaceHealthCheck {
    pub fn new(min_free_gb: f64) -> Self {
        Self { min_free_gb }
    }
}

#[async_trait::async_trait]
impl HealthCheckTrait for DiskSpaceHealthCheck {
    async fn check(&self) -> HealthCheck {
        let start = std::time::Instant::now();
        let duration = start.elapsed();

        // Simplified disk check (could be enhanced with actual disk space monitoring)
        let (status, message) = (
            CheckStatus::Pass,
            "Disk space check not implemented".to_string(),
        );

        HealthCheck {
            status,
            message,
            last_check: chrono::Utc::now().to_rfc3339(),
            duration_ms: duration.as_secs_f64() * 1000.0,
        }
    }

    fn name(&self) -> &str {
        "disk_space"
    }
}

#[cfg(feature = "tpu")]
pub struct TpuHealthCheck {
    tpu: Arc<crate::tpu::EdgeTPU>,
}

#[cfg(feature = "tpu")]
impl TpuHealthCheck {
    pub fn new(tpu: Arc<crate::tpu::EdgeTPU>) -> Self {
        Self { tpu }
    }
}

#[cfg(feature = "tpu")]
#[async_trait::async_trait]
impl HealthCheckTrait for TpuHealthCheck {
    async fn check(&self) -> HealthCheck {
        let start = std::time::Instant::now();
        
        let (status, message) = match self.tpu.get_stats() {
            Ok(stats) => {
                if stats.temperature_celsius > 85.0 {
                    (CheckStatus::Warn, format!("TPU temperature high: {:.1}°C", stats.temperature_celsius))
                } else if stats.power_consumption_watts > 5.0 {
                    (CheckStatus::Warn, format!("TPU power consumption high: {:.1}W", stats.power_consumption_watts))
                } else {
                    (CheckStatus::Pass, format!("TPU operating normally: {:.1}°C, {:.1}W", stats.temperature_celsius, stats.power_consumption_watts))
                }
            }
            Err(e) => (CheckStatus::Fail, format!("TPU check failed: {}", e)),
        };

        let duration = start.elapsed();

        HealthCheck {
            status,
            message,
            last_check: chrono::Utc::now().to_rfc3339(),
            duration_ms: duration.as_secs_f64() * 1000.0,
        }
    }

    fn name(&self) -> &str {
        "tpu"
    }
}

fn get_memory_usage_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<f64>() {
                            return kb / 1024.0; // Convert KB to MB
                        }
                    }
                }
            }
        }
    }
    0.0
}