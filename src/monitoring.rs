//! Comprehensive monitoring, metrics, and observability for DGDM server

use prometheus::{
    Counter, Histogram, Gauge, IntGauge, IntCounter, HistogramOpts, 
    TextEncoder, Encoder, register_counter, register_histogram, 
    register_gauge, register_int_gauge, register_int_counter,
};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

lazy_static::lazy_static! {
    // Request metrics
    pub static ref HTTP_REQUESTS_TOTAL: Counter = register_counter!(
        "dgdm_http_requests_total",
        "Total number of HTTP requests received"
    ).unwrap();

    pub static ref HTTP_REQUEST_DURATION: Histogram = register_histogram!(
        HistogramOpts::new(
            "dgdm_http_request_duration_seconds",
            "Duration of HTTP requests in seconds"
        ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    ).unwrap();

    pub static ref HTTP_REQUESTS_IN_FLIGHT: IntGauge = register_int_gauge!(
        "dgdm_http_requests_in_flight",
        "Current number of HTTP requests being processed"
    ).unwrap();

    // Graph processing metrics
    pub static ref GRAPHS_PROCESSED_TOTAL: Counter = register_counter!(
        "dgdm_graphs_processed_total",
        "Total number of graphs processed"
    ).unwrap();

    pub static ref GRAPH_PROCESSING_DURATION: Histogram = register_histogram!(
        HistogramOpts::new(
            "dgdm_graph_processing_duration_seconds",
            "Duration of graph processing in seconds"
        ).buckets(vec![0.001, 0.01, 0.1, 1.0, 10.0, 30.0, 60.0, 300.0])
    ).unwrap();

    pub static ref GRAPH_NODES_PROCESSED: Histogram = register_histogram!(
        HistogramOpts::new(
            "dgdm_graph_nodes_total",
            "Number of nodes in processed graphs"
        ).buckets(vec![1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0])
    ).unwrap();

    pub static ref GRAPH_EDGES_PROCESSED: Histogram = register_histogram!(
        HistogramOpts::new(
            "dgdm_graph_edges_total",
            "Number of edges in processed graphs"
        ).buckets(vec![1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0])
    ).unwrap();

    // Error metrics
    pub static ref VALIDATION_ERRORS_TOTAL: Counter = register_counter!(
        "dgdm_validation_errors_total",
        "Total number of validation errors"
    ).unwrap();

    pub static ref PROCESSING_ERRORS_TOTAL: Counter = register_counter!(
        "dgdm_processing_errors_total",
        "Total number of processing errors"
    ).unwrap();

    pub static ref AUTHENTICATION_FAILURES_TOTAL: Counter = register_counter!(
        "dgdm_auth_failures_total",
        "Total number of authentication failures"
    ).unwrap();

    pub static ref RATE_LIMIT_VIOLATIONS_TOTAL: Counter = register_counter!(
        "dgdm_rate_limit_violations_total",
        "Total number of rate limit violations"
    ).unwrap();

    // System resource metrics
    pub static ref MEMORY_USAGE_BYTES: IntGauge = register_int_gauge!(
        "dgdm_memory_usage_bytes",
        "Current memory usage in bytes"
    ).unwrap();

    pub static ref CPU_USAGE_PERCENT: Gauge = register_gauge!(
        "dgdm_cpu_usage_percent",
        "Current CPU usage percentage"
    ).unwrap();

    pub static ref ACTIVE_CONNECTIONS: IntGauge = register_int_gauge!(
        "dgdm_active_connections",
        "Number of active connections"
    ).unwrap();

    // Cache metrics
    pub static ref CACHE_HITS_TOTAL: Counter = register_counter!(
        "dgdm_cache_hits_total",
        "Total number of cache hits"
    ).unwrap();

    pub static ref CACHE_MISSES_TOTAL: Counter = register_counter!(
        "dgdm_cache_misses_total",
        "Total number of cache misses"
    ).unwrap();

    pub static ref CACHE_SIZE_BYTES: IntGauge = register_int_gauge!(
        "dgdm_cache_size_bytes",
        "Current size of cache in bytes"
    ).unwrap();
}

// TPU-specific metrics (conditional compilation)
#[cfg(feature = "tpu")]
lazy_static::lazy_static! {
    pub static ref TPU_INFERENCE_DURATION: Histogram = register_histogram!(
        HistogramOpts::new(
            "dgdm_tpu_inference_duration_seconds",
            "Duration of TPU inference in seconds"
        ).buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    ).unwrap();

    pub static ref TPU_TEMPERATURE_CELSIUS: Gauge = register_gauge!(
        "dgdm_tpu_temperature_celsius",
        "Current TPU temperature in Celsius"
    ).unwrap();

    pub static ref TPU_POWER_CONSUMPTION_WATTS: Gauge = register_gauge!(
        "dgdm_tpu_power_consumption_watts",
        "Current TPU power consumption in watts"
    ).unwrap();

    pub static ref TPU_UTILIZATION_PERCENT: Gauge = register_gauge!(
        "dgdm_tpu_utilization_percent",
        "Current TPU utilization percentage"
    ).unwrap();
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: u64,
    pub uptime_seconds: u64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub active_connections: i64,
    pub total_requests: f64,
    pub error_rate: f64,
    pub average_response_time_ms: f64,
    pub graphs_processed: f64,
    pub cache_hit_rate: f64,
    #[cfg(feature = "tpu")]
    pub tpu_metrics: Option<TpuMetrics>,
}

#[cfg(feature = "tpu")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpuMetrics {
    pub temperature_celsius: f64,
    pub power_consumption_watts: f64,
    pub utilization_percent: f64,
    pub inference_count: u64,
    pub average_inference_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub avg_processing_time_ms: f64,
    pub p95_processing_time_ms: f64,
    pub p99_processing_time_ms: f64,
    pub throughput_requests_per_second: f64,
    pub error_rate_percent: f64,
    pub memory_usage_mb: f64,
    pub measured_at: u64,
}

pub struct MetricsCollector {
    baselines: Arc<RwLock<HashMap<String, PerformanceBaseline>>>,
    startup_time: Instant,
    last_collection: Arc<RwLock<Instant>>,
    collection_interval: Duration,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            baselines: Arc::new(RwLock::new(HashMap::new())),
            startup_time: Instant::now(),
            last_collection: Arc::new(RwLock::new(Instant::now())),
            collection_interval: Duration::from_secs(60),
        }
    }

    pub async fn collect_system_metrics(&self) -> SystemMetrics {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let uptime_seconds = self.startup_time.elapsed().as_secs();
        let memory_usage_mb = self.get_memory_usage_mb();
        let cpu_usage_percent = self.get_cpu_usage_percent();

        // Get Prometheus metrics
        let total_requests = HTTP_REQUESTS_TOTAL.get();
        let graphs_processed = GRAPHS_PROCESSED_TOTAL.get();
        
        // Calculate rates
        let error_rate = self.calculate_error_rate();
        let cache_hit_rate = self.calculate_cache_hit_rate();
        let average_response_time_ms = self.calculate_average_response_time_ms();

        SystemMetrics {
            timestamp,
            uptime_seconds,
            memory_usage_mb,
            cpu_usage_percent,
            active_connections: ACTIVE_CONNECTIONS.get(),
            total_requests,
            error_rate,
            average_response_time_ms,
            graphs_processed,
            cache_hit_rate,
            #[cfg(feature = "tpu")]
            tpu_metrics: self.collect_tpu_metrics().await,
        }
    }

    #[cfg(feature = "tpu")]
    async fn collect_tpu_metrics(&self) -> Option<TpuMetrics> {
        Some(TpuMetrics {
            temperature_celsius: TPU_TEMPERATURE_CELSIUS.get(),
            power_consumption_watts: TPU_POWER_CONSUMPTION_WATTS.get(),
            utilization_percent: TPU_UTILIZATION_PERCENT.get(),
            inference_count: 0, // TODO: Implement inference counter
            average_inference_time_ms: 0.0, // TODO: Calculate from histogram
        })
    }

    fn get_memory_usage_mb(&self) -> f64 {
        if let Ok(info) = sys_info::mem_info() {
            let used_kb = info.total - info.avail;
            (used_kb as f64) / 1024.0
        } else {
            0.0
        }
    }

    fn get_cpu_usage_percent(&self) -> f64 {
        // Simplified CPU usage - in production, use a proper system monitoring library
        if let Ok(loadavg) = sys_info::loadavg() {
            (loadavg.one * 100.0) / num_cpus::get() as f64
        } else {
            0.0
        }
    }

    fn calculate_error_rate(&self) -> f64 {
        let total_requests = HTTP_REQUESTS_TOTAL.get();
        if total_requests == 0.0 {
            return 0.0;
        }
        
        let validation_errors = VALIDATION_ERRORS_TOTAL.get();
        let processing_errors = PROCESSING_ERRORS_TOTAL.get();
        let total_errors = validation_errors + processing_errors;
        
        (total_errors / total_requests) * 100.0
    }

    fn calculate_cache_hit_rate(&self) -> f64 {
        let hits = CACHE_HITS_TOTAL.get();
        let misses = CACHE_MISSES_TOTAL.get();
        let total = hits + misses;
        
        if total == 0.0 {
            0.0
        } else {
            (hits / total) * 100.0
        }
    }

    fn calculate_average_response_time_ms(&self) -> f64 {
        // Calculate from histogram - simplified version
        let histogram = &HTTP_REQUEST_DURATION;
        let count = histogram.get_sample_count();
        if count == 0 {
            return 0.0;
        }
        
        let sum = histogram.get_sample_sum();
        (sum / count as f64) * 1000.0 // Convert to milliseconds
    }

    pub async fn set_baseline(&self, name: &str, baseline: PerformanceBaseline) {
        let mut baselines = self.baselines.write().await;
        baselines.insert(name.to_string(), baseline);
        info!("Performance baseline '{}' set", name);
    }

    pub async fn check_performance_degradation(&self, threshold_percent: f64) -> Vec<String> {
        let mut alerts = Vec::new();
        let current_metrics = self.collect_system_metrics().await;
        let baselines = self.baselines.read().await;

        for (name, baseline) in baselines.iter() {
            // Check processing time degradation
            if current_metrics.average_response_time_ms > 
               baseline.avg_processing_time_ms * (1.0 + threshold_percent / 100.0) {
                alerts.push(format!(
                    "Performance degradation detected for {}: current {}ms vs baseline {}ms",
                    name, current_metrics.average_response_time_ms, baseline.avg_processing_time_ms
                ));
            }

            // Check error rate increase
            if current_metrics.error_rate > 
               baseline.error_rate_percent * (1.0 + threshold_percent / 100.0) {
                alerts.push(format!(
                    "Error rate increase detected for {}: current {}% vs baseline {}%",
                    name, current_metrics.error_rate, baseline.error_rate_percent
                ));
            }

            // Check memory usage increase
            if current_metrics.memory_usage_mb > 
               baseline.memory_usage_mb * (1.0 + threshold_percent / 100.0) {
                alerts.push(format!(
                    "Memory usage increase detected for {}: current {}MB vs baseline {}MB",
                    name, current_metrics.memory_usage_mb, baseline.memory_usage_mb
                ));
            }
        }

        if !alerts.is_empty() {
            warn!("Performance alerts: {:?}", alerts);
        }

        alerts
    }
}

// Request metrics middleware
pub async fn metrics_middleware(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let start_time = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();
    
    HTTP_REQUESTS_TOTAL.inc();
    HTTP_REQUESTS_IN_FLIGHT.inc();

    let response = next.run(request).await;
    
    let duration = start_time.elapsed();
    HTTP_REQUEST_DURATION.observe(duration.as_secs_f64());
    HTTP_REQUESTS_IN_FLIGHT.dec();

    // Log slow requests
    if duration > Duration::from_millis(1000) {
        warn!(
            "Slow request detected: {} {} took {}ms",
            method,
            uri,
            duration.as_millis()
        );
    }

    response
}

// Graph processing metrics wrapper
pub struct ProcessingTimer {
    start_time: Instant,
}

impl ProcessingTimer {
    pub fn start() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    pub fn finish(&self, nodes: usize, edges: usize) {
        let duration = self.start_time.elapsed();
        
        GRAPHS_PROCESSED_TOTAL.inc();
        GRAPH_PROCESSING_DURATION.observe(duration.as_secs_f64());
        GRAPH_NODES_PROCESSED.observe(nodes as f64);
        GRAPH_EDGES_PROCESSED.observe(edges as f64);

        info!(
            "Graph processed: {} nodes, {} edges in {}ms",
            nodes,
            edges,
            duration.as_millis()
        );
    }

    pub fn error(&self, error_type: &str) {
        let duration = self.start_time.elapsed();
        
        match error_type {
            "validation" => VALIDATION_ERRORS_TOTAL.inc(),
            "processing" => PROCESSING_ERRORS_TOTAL.inc(),
            _ => error!("Unknown error type: {}", error_type),
        }

        error!(
            "Graph processing failed after {}ms with error type: {}",
            duration.as_millis(),
            error_type
        );
    }
}

// Health check with detailed metrics
#[derive(Debug, Serialize)]
pub struct DetailedHealthCheck {
    pub status: String,
    pub timestamp: u64,
    pub uptime_seconds: u64,
    pub version: String,
    pub system: SystemHealthInfo,
    pub performance: PerformanceHealthInfo,
    pub dependencies: DependencyHealthInfo,
}

#[derive(Debug, Serialize)]
pub struct SystemHealthInfo {
    pub memory_usage_mb: f64,
    pub memory_available_mb: f64,
    pub cpu_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub load_average: f64,
}

#[derive(Debug, Serialize)]
pub struct PerformanceHealthInfo {
    pub active_requests: i64,
    pub requests_per_second: f64,
    pub average_response_time_ms: f64,
    pub p95_response_time_ms: f64,
    pub error_rate_percent: f64,
    pub cache_hit_rate_percent: f64,
}

#[derive(Debug, Serialize)]
pub struct DependencyHealthInfo {
    #[cfg(feature = "tpu")]
    pub tpu_available: bool,
    #[cfg(feature = "tpu")]
    pub tpu_temperature_celsius: f64,
    pub prometheus_metrics_enabled: bool,
    pub tracing_enabled: bool,
}

pub async fn get_detailed_health_check() -> DetailedHealthCheck {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let memory_info = sys_info::mem_info().unwrap_or(sys_info::MemInfo {
        total: 0,
        free: 0,
        avail: 0,
        buffers: 0,
        cached: 0,
        swap_total: 0,
        swap_free: 0,
    });
    let loadavg = sys_info::loadavg().unwrap_or(sys_info::LoadAvg {
        one: 0.0,
        five: 0.0,
        fifteen: 0.0,
    });

    DetailedHealthCheck {
        status: "healthy".to_string(),
        timestamp,
        uptime_seconds: 0, // TODO: Track from startup
        version: env!("CARGO_PKG_VERSION").to_string(),
        system: SystemHealthInfo {
            memory_usage_mb: ((memory_info.total - memory_info.avail) as f64) / 1024.0,
            memory_available_mb: (memory_info.avail as f64) / 1024.0,
            cpu_usage_percent: (loadavg.one * 100.0) / num_cpus::get() as f64,
            disk_usage_percent: 0.0, // TODO: Implement disk usage
            load_average: loadavg.one,
        },
        performance: PerformanceHealthInfo {
            active_requests: HTTP_REQUESTS_IN_FLIGHT.get(),
            requests_per_second: 0.0, // TODO: Calculate RPS
            average_response_time_ms: 0.0, // TODO: Calculate from histogram
            p95_response_time_ms: 0.0, // TODO: Get P95 from histogram
            error_rate_percent: 0.0, // TODO: Calculate error rate
            cache_hit_rate_percent: 0.0, // TODO: Calculate cache hit rate
        },
        dependencies: DependencyHealthInfo {
            #[cfg(feature = "tpu")]
            tpu_available: false, // TODO: Check TPU availability
            #[cfg(feature = "tpu")]
            tpu_temperature_celsius: 0.0, // TODO: Get TPU temperature
            prometheus_metrics_enabled: true,
            tracing_enabled: true,
        },
    }
}

// Export metrics in Prometheus format
pub fn export_prometheus_metrics() -> Result<String, Box<dyn std::error::Error>> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        let metrics = collector.collect_system_metrics().await;
        
        assert!(metrics.timestamp > 0);
        assert!(metrics.uptime_seconds >= 0);
    }

    #[test]
    fn test_processing_timer() {
        let timer = ProcessingTimer::start();
        std::thread::sleep(Duration::from_millis(10));
        timer.finish(100, 200);
        
        // Verify metrics were recorded
        assert!(GRAPHS_PROCESSED_TOTAL.get() >= 1.0);
    }

    #[test]
    fn test_prometheus_export() {
        HTTP_REQUESTS_TOTAL.inc();
        let metrics = export_prometheus_metrics().unwrap();
        assert!(metrics.contains("dgdm_http_requests_total"));
    }
}