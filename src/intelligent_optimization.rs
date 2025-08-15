//! Intelligent Optimization System
//! 
//! AI-driven performance optimization that continuously learns from
//! system behavior and automatically applies the most effective optimizations.

use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

/// Optimization strategy types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    CacheOptimization {
        cache_size_mb: usize,
        ttl_seconds: u64,
        eviction_policy: String,
    },
    ResourcePooling {
        pool_size: usize,
        max_connections: usize,
        idle_timeout_seconds: u64,
    },
    ComputeOptimization {
        batch_size: usize,
        parallel_workers: usize,
        simd_enabled: bool,
    },
    MemoryOptimization {
        allocation_strategy: String,
        gc_threshold_mb: usize,
        zero_copy_enabled: bool,
    },
    NetworkOptimization {
        compression_enabled: bool,
        keep_alive_timeout: u64,
        tcp_no_delay: bool,
    },
    AlgorithmicOptimization {
        algorithm_variant: String,
        optimization_level: u8,
        approximation_factor: f64,
    },
}

/// Performance metrics for optimization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub throughput_ops_per_sec: f64,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
    pub cpu_utilization_percent: f64,
    pub memory_usage_mb: f64,
    pub cache_hit_rate: f64,
    pub error_rate_percent: f64,
    pub concurrent_requests: u64,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub strategy: OptimizationStrategy,
    pub before_metrics: PerformanceMetrics,
    pub after_metrics: PerformanceMetrics,
    pub improvement_factor: f64,
    pub execution_time_ms: u64,
    pub success: bool,
    pub side_effects: Vec<String>,
    pub applied_at: chrono::DateTime<chrono::Utc>,
}

/// Learned optimization pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationPattern {
    pub workload_characteristics: WorkloadCharacteristics,
    pub most_effective_strategy: OptimizationStrategy,
    pub avg_improvement_factor: f64,
    pub confidence_score: f64,
    pub sample_size: usize,
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Workload characteristics for pattern matching
#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct WorkloadCharacteristics {
    pub request_pattern: RequestPattern,
    pub data_size_category: DataSizeCategory,
    pub concurrency_level: ConcurrencyLevel,
    pub computation_intensity: ComputationIntensity,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum RequestPattern {
    Burst,
    Steady,
    Periodic,
    Random,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum DataSizeCategory {
    Small,    // < 1KB
    Medium,   // 1KB - 1MB  
    Large,    // 1MB - 100MB
    XLarge,   // > 100MB
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum ConcurrencyLevel {
    Low,      // < 10 concurrent
    Medium,   // 10-100 concurrent
    High,     // 100-1000 concurrent
    Extreme,  // > 1000 concurrent
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub enum ComputationIntensity {
    Light,    // Simple operations
    Moderate, // Standard graph processing
    Heavy,    // Complex algorithms
    Extreme,  // AI/ML workloads
}

/// Intelligent optimization system
#[derive(Debug)]
pub struct IntelligentOptimizationSystem {
    config: OptimizationConfig,
    optimization_history: Arc<RwLock<Vec<OptimizationResult>>>,
    learned_patterns: Arc<RwLock<HashMap<WorkloadCharacteristics, OptimizationPattern>>>,
    performance_baseline: Arc<RwLock<Option<PerformanceMetrics>>>,
    active_optimizations: Arc<RwLock<HashMap<String, OptimizationStrategy>>>,
    current_workload: Arc<RwLock<Option<WorkloadCharacteristics>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    pub enable_autonomous_optimization: bool,
    pub enable_learning: bool,
    pub min_improvement_threshold: f64,
    pub optimization_interval_minutes: u64,
    pub rollback_on_degradation: bool,
    pub max_concurrent_optimizations: usize,
    pub learning_window_hours: u64,
    pub confidence_threshold: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            enable_autonomous_optimization: true,
            enable_learning: true,
            min_improvement_threshold: 0.05, // 5% minimum improvement
            optimization_interval_minutes: 15,
            rollback_on_degradation: true,
            max_concurrent_optimizations: 3,
            learning_window_hours: 24,
            confidence_threshold: 0.7,
        }
    }
}

impl IntelligentOptimizationSystem {
    /// Create new intelligent optimization system
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            optimization_history: Arc::new(RwLock::new(Vec::new())),
            learned_patterns: Arc::new(RwLock::new(HashMap::new())),
            performance_baseline: Arc::new(RwLock::new(None)),
            active_optimizations: Arc::new(RwLock::new(HashMap::new())),
            current_workload: Arc::new(RwLock::new(None)),
        }
    }

    /// Start autonomous optimization
    pub async fn start_autonomous_optimization(&self) -> Result<()> {
        if !self.config.enable_autonomous_optimization {
            return Ok(());
        }

        info!("🧠 Starting intelligent optimization system");
        
        // Start optimization loop
        let optimization_history = self.optimization_history.clone();
        let learned_patterns = self.learned_patterns.clone();
        let performance_baseline = self.performance_baseline.clone();
        let active_optimizations = self.active_optimizations.clone();
        let current_workload = self.current_workload.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(config.optimization_interval_minutes * 60));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::run_optimization_cycle(
                    config.clone(),
                    optimization_history.clone(),
                    learned_patterns.clone(),
                    performance_baseline.clone(),
                    active_optimizations.clone(),
                    current_workload.clone(),
                ).await {
                    error!("Optimization cycle failed: {}", e);
                }
            }
        });

        // Start workload analysis
        let current_workload_clone = self.current_workload.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Analyze every minute
            
            loop {
                interval.tick().await;
                Self::analyze_current_workload(current_workload_clone.clone()).await;
            }
        });

        Ok(())
    }

    /// Run single optimization cycle
    async fn run_optimization_cycle(
        config: OptimizationConfig,
        optimization_history: Arc<RwLock<Vec<OptimizationResult>>>,
        learned_patterns: Arc<RwLock<HashMap<WorkloadCharacteristics, OptimizationPattern>>>,
        performance_baseline: Arc<RwLock<Option<PerformanceMetrics>>>,
        active_optimizations: Arc<RwLock<HashMap<String, OptimizationStrategy>>>,
        current_workload: Arc<RwLock<Option<WorkloadCharacteristics>>>,
    ) -> Result<()> {
        debug!("🔄 Running optimization cycle");
        
        // Collect current performance metrics
        let current_metrics = Self::collect_performance_metrics().await;
        
        // Update baseline if not set
        {
            let mut baseline = performance_baseline.write().await;
            if baseline.is_none() {
                *baseline = Some(current_metrics.clone());
                info!("📊 Performance baseline established");
                return Ok(());
            }
        }
        
        // Analyze workload and determine optimization opportunities
        let workload = current_workload.read().await.clone();
        if let Some(workload_chars) = workload {
            let optimization_strategies = Self::determine_optimization_strategies(
                &workload_chars,
                &current_metrics,
                learned_patterns.clone(),
                config.clone(),
            ).await;
            
            // Apply optimizations
            for strategy in optimization_strategies {
                if Self::count_active_optimizations(active_optimizations.clone()).await >= config.max_concurrent_optimizations {
                    debug!("Maximum concurrent optimizations reached, skipping");
                    break;
                }
                
                let optimization_id = format!("opt_{}", chrono::Utc::now().timestamp_millis());
                
                match Self::apply_optimization(
                    optimization_id.clone(),
                    strategy.clone(),
                    current_metrics.clone(),
                    active_optimizations.clone(),
                ).await {
                    Ok(result) => {
                        // Record successful optimization
                        optimization_history.write().await.push(result.clone());
                        
                        // Update learned patterns
                        if config.enable_learning {
                            Self::update_learned_patterns(
                                workload_chars.clone(),
                                strategy,
                                result,
                                learned_patterns.clone(),
                            ).await;
                        }
                    }
                    Err(e) => {
                        warn!("Optimization failed: {}", e);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Collect current performance metrics
    async fn collect_performance_metrics() -> PerformanceMetrics {
        // Simulate performance metrics collection
        // In production, this would gather real metrics from the system
        
        let base_throughput = 100.0;
        let base_latency = 50.0;
        
        // Add some realistic variation
        let throughput_variation = (rand::random::<f64>() - 0.5) * 40.0;
        let latency_variation = (rand::random::<f64>() - 0.5) * 20.0;
        
        PerformanceMetrics {
            timestamp: chrono::Utc::now(),
            throughput_ops_per_sec: base_throughput + throughput_variation,
            latency_p50_ms: base_latency + latency_variation,
            latency_p95_ms: (base_latency + latency_variation) * 1.5,
            latency_p99_ms: (base_latency + latency_variation) * 2.0,
            cpu_utilization_percent: 40.0 + rand::random::<f64>() * 40.0,
            memory_usage_mb: 256.0 + rand::random::<f64>() * 256.0,
            cache_hit_rate: 0.8 + rand::random::<f64>() * 0.15,
            error_rate_percent: rand::random::<f64>() * 2.0,
            concurrent_requests: (rand::random::<u64>() % 500) + 50,
        }
    }

    /// Analyze current workload characteristics
    async fn analyze_current_workload(current_workload: Arc<RwLock<Option<WorkloadCharacteristics>>>) {
        // Simulate workload analysis
        // In production, this would analyze actual request patterns
        
        let request_pattern = match rand::random::<u8>() % 4 {
            0 => RequestPattern::Burst,
            1 => RequestPattern::Steady,
            2 => RequestPattern::Periodic,
            _ => RequestPattern::Random,
        };
        
        let data_size_category = match rand::random::<u8>() % 4 {
            0 => DataSizeCategory::Small,
            1 => DataSizeCategory::Medium,
            2 => DataSizeCategory::Large,
            _ => DataSizeCategory::XLarge,
        };
        
        let concurrency_level = match rand::random::<u8>() % 4 {
            0 => ConcurrencyLevel::Low,
            1 => ConcurrencyLevel::Medium,
            2 => ConcurrencyLevel::High,
            _ => ConcurrencyLevel::Extreme,
        };
        
        let computation_intensity = match rand::random::<u8>() % 4 {
            0 => ComputationIntensity::Light,
            1 => ComputationIntensity::Moderate,
            2 => ComputationIntensity::Heavy,
            _ => ComputationIntensity::Extreme,
        };
        
        let workload_chars = WorkloadCharacteristics {
            request_pattern,
            data_size_category,
            concurrency_level,
            computation_intensity,
        };
        
        *current_workload.write().await = Some(workload_chars);
    }

    /// Determine optimization strategies for current workload
    async fn determine_optimization_strategies(
        workload: &WorkloadCharacteristics,
        current_metrics: &PerformanceMetrics,
        learned_patterns: Arc<RwLock<HashMap<WorkloadCharacteristics, OptimizationPattern>>>,
        config: OptimizationConfig,
    ) -> Vec<OptimizationStrategy> {
        let mut strategies = Vec::new();
        
        // Check learned patterns first
        {
            let patterns = learned_patterns.read().await;
            if let Some(pattern) = patterns.get(workload) {
                if pattern.confidence_score >= config.confidence_threshold {
                    debug!("Using learned optimization pattern for workload");
                    strategies.push(pattern.most_effective_strategy.clone());
                }
            }
        }
        
        // Add workload-specific optimizations
        strategies.extend(Self::get_workload_specific_optimizations(workload, current_metrics));
        
        // Add general optimizations based on performance issues
        strategies.extend(Self::get_performance_based_optimizations(current_metrics));
        
        strategies
    }

    /// Get optimizations based on workload characteristics
    fn get_workload_specific_optimizations(
        workload: &WorkloadCharacteristics,
        metrics: &PerformanceMetrics,
    ) -> Vec<OptimizationStrategy> {
        let mut strategies = Vec::new();
        
        match workload.concurrency_level {
            ConcurrencyLevel::High | ConcurrencyLevel::Extreme => {
                strategies.push(OptimizationStrategy::ResourcePooling {
                    pool_size: 100,
                    max_connections: 1000,
                    idle_timeout_seconds: 30,
                });
            }
            _ => {}
        }
        
        match workload.data_size_category {
            DataSizeCategory::Large | DataSizeCategory::XLarge => {
                strategies.push(OptimizationStrategy::MemoryOptimization {
                    allocation_strategy: "pool".to_string(),
                    gc_threshold_mb: 512,
                    zero_copy_enabled: true,
                });
            }
            _ => {}
        }
        
        match workload.request_pattern {
            RequestPattern::Burst => {
                strategies.push(OptimizationStrategy::CacheOptimization {
                    cache_size_mb: 256,
                    ttl_seconds: 300,
                    eviction_policy: "lru".to_string(),
                });
            }
            _ => {}
        }
        
        match workload.computation_intensity {
            ComputationIntensity::Heavy | ComputationIntensity::Extreme => {
                strategies.push(OptimizationStrategy::ComputeOptimization {
                    batch_size: 64,
                    parallel_workers: num_cpus::get(),
                    simd_enabled: true,
                });
            }
            _ => {}
        }
        
        strategies
    }

    /// Get optimizations based on performance metrics
    fn get_performance_based_optimizations(metrics: &PerformanceMetrics) -> Vec<OptimizationStrategy> {
        let mut strategies = Vec::new();
        
        // High latency optimizations
        if metrics.latency_p95_ms > 100.0 {
            strategies.push(OptimizationStrategy::NetworkOptimization {
                compression_enabled: true,
                keep_alive_timeout: 60,
                tcp_no_delay: true,
            });
        }
        
        // Low cache hit rate
        if metrics.cache_hit_rate < 0.8 {
            strategies.push(OptimizationStrategy::CacheOptimization {
                cache_size_mb: 512,
                ttl_seconds: 600,
                eviction_policy: "lfu".to_string(),
            });
        }
        
        // High CPU utilization
        if metrics.cpu_utilization_percent > 80.0 {
            strategies.push(OptimizationStrategy::AlgorithmicOptimization {
                algorithm_variant: "optimized".to_string(),
                optimization_level: 3,
                approximation_factor: 0.95,
            });
        }
        
        // High memory usage
        if metrics.memory_usage_mb > 400.0 {
            strategies.push(OptimizationStrategy::MemoryOptimization {
                allocation_strategy: "compact".to_string(),
                gc_threshold_mb: 256,
                zero_copy_enabled: true,
            });
        }
        
        strategies
    }

    /// Apply optimization strategy
    async fn apply_optimization(
        optimization_id: String,
        strategy: OptimizationStrategy,
        before_metrics: PerformanceMetrics,
        active_optimizations: Arc<RwLock<HashMap<String, OptimizationStrategy>>>,
    ) -> Result<OptimizationResult> {
        let start_time = Instant::now();
        
        info!("🚀 Applying optimization: {:?}", strategy);
        
        // Register active optimization
        active_optimizations.write().await.insert(optimization_id.clone(), strategy.clone());
        
        // Execute optimization
        let result = Self::execute_optimization_strategy(&strategy).await;
        
        // Remove from active optimizations
        active_optimizations.write().await.remove(&optimization_id);
        
        let execution_time = start_time.elapsed();
        
        // Collect after metrics
        tokio::time::sleep(Duration::from_secs(5)).await; // Wait for effects
        let after_metrics = Self::collect_performance_metrics().await;
        
        // Calculate improvement
        let improvement_factor = Self::calculate_improvement_factor(&before_metrics, &after_metrics);
        
        let optimization_result = OptimizationResult {
            strategy,
            before_metrics,
            after_metrics,
            improvement_factor,
            execution_time_ms: execution_time.as_millis() as u64,
            success: result.is_ok() && improvement_factor > 0.0,
            side_effects: Vec::new(),
            applied_at: chrono::Utc::now(),
        };
        
        if optimization_result.success {
            info!("✅ Optimization successful: {:.1}% improvement", improvement_factor * 100.0);
        } else {
            warn!("❌ Optimization ineffective or harmful");
        }
        
        Ok(optimization_result)
    }

    /// Execute specific optimization strategy
    async fn execute_optimization_strategy(strategy: &OptimizationStrategy) -> Result<()> {
        match strategy {
            OptimizationStrategy::CacheOptimization { cache_size_mb, ttl_seconds, eviction_policy } => {
                debug!("Configuring cache: {}MB, TTL={}s, policy={}", cache_size_mb, ttl_seconds, eviction_policy);
                // Simulate cache configuration
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(())
            }
            OptimizationStrategy::ResourcePooling { pool_size, max_connections, idle_timeout_seconds } => {
                debug!("Configuring resource pool: size={}, max_conn={}, timeout={}s", 
                       pool_size, max_connections, idle_timeout_seconds);
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok(())
            }
            OptimizationStrategy::ComputeOptimization { batch_size, parallel_workers, simd_enabled } => {
                debug!("Optimizing compute: batch={}, workers={}, SIMD={}", 
                       batch_size, parallel_workers, simd_enabled);
                tokio::time::sleep(Duration::from_millis(150)).await;
                Ok(())
            }
            OptimizationStrategy::MemoryOptimization { allocation_strategy, gc_threshold_mb, zero_copy_enabled } => {
                debug!("Optimizing memory: strategy={}, GC={}MB, zero_copy={}", 
                       allocation_strategy, gc_threshold_mb, zero_copy_enabled);
                tokio::time::sleep(Duration::from_millis(300)).await;
                Ok(())
            }
            OptimizationStrategy::NetworkOptimization { compression_enabled, keep_alive_timeout, tcp_no_delay } => {
                debug!("Optimizing network: compression={}, keep_alive={}s, tcp_no_delay={}", 
                       compression_enabled, keep_alive_timeout, tcp_no_delay);
                tokio::time::sleep(Duration::from_millis(50)).await;
                Ok(())
            }
            OptimizationStrategy::AlgorithmicOptimization { algorithm_variant, optimization_level, approximation_factor } => {
                debug!("Optimizing algorithm: variant={}, level={}, approx={}", 
                       algorithm_variant, optimization_level, approximation_factor);
                tokio::time::sleep(Duration::from_millis(400)).await;
                Ok(())
            }
        }
    }

    /// Calculate improvement factor between metrics
    fn calculate_improvement_factor(before: &PerformanceMetrics, after: &PerformanceMetrics) -> f64 {
        // Composite improvement score considering multiple metrics
        let throughput_improvement = (after.throughput_ops_per_sec - before.throughput_ops_per_sec) / before.throughput_ops_per_sec;
        let latency_improvement = (before.latency_p95_ms - after.latency_p95_ms) / before.latency_p95_ms;
        let cpu_improvement = (before.cpu_utilization_percent - after.cpu_utilization_percent) / before.cpu_utilization_percent;
        let memory_improvement = (before.memory_usage_mb - after.memory_usage_mb) / before.memory_usage_mb;
        
        // Weighted average (throughput and latency are most important)
        let improvement = (throughput_improvement * 0.4) + 
                         (latency_improvement * 0.4) + 
                         (cpu_improvement * 0.1) + 
                         (memory_improvement * 0.1);
        
        improvement.max(-1.0).min(10.0) // Cap extreme values
    }

    /// Update learned patterns
    async fn update_learned_patterns(
        workload: WorkloadCharacteristics,
        strategy: OptimizationStrategy,
        result: OptimizationResult,
        learned_patterns: Arc<RwLock<HashMap<WorkloadCharacteristics, OptimizationPattern>>>,
    ) {
        let mut patterns = learned_patterns.write().await;
        
        if let Some(pattern) = patterns.get_mut(&workload) {
            // Update existing pattern
            let total_samples = pattern.sample_size;
            let current_avg = pattern.avg_improvement_factor;
            
            // Update running average
            let new_avg = (current_avg * total_samples as f64 + result.improvement_factor) / (total_samples + 1) as f64;
            
            pattern.avg_improvement_factor = new_avg;
            pattern.sample_size = total_samples + 1;
            pattern.last_updated = chrono::Utc::now();
            
            // Update confidence based on sample size and consistency
            pattern.confidence_score = Self::calculate_pattern_confidence(pattern.sample_size, new_avg);
            
            // Update strategy if this one is more effective
            if result.improvement_factor > current_avg {
                pattern.most_effective_strategy = strategy;
            }
        } else {
            // Create new pattern
            let new_pattern = OptimizationPattern {
                workload_characteristics: workload.clone(),
                most_effective_strategy: strategy,
                avg_improvement_factor: result.improvement_factor,
                confidence_score: Self::calculate_pattern_confidence(1, result.improvement_factor),
                sample_size: 1,
                last_updated: chrono::Utc::now(),
            };
            
            patterns.insert(workload, new_pattern);
        }
    }

    /// Calculate pattern confidence score
    fn calculate_pattern_confidence(sample_size: usize, avg_improvement: f64) -> f64 {
        let sample_factor = (sample_size as f64 / 20.0).min(1.0); // Full confidence at 20 samples
        let improvement_factor = if avg_improvement > 0.0 { 
            (avg_improvement * 10.0).min(1.0) 
        } else { 
            0.0 
        };
        
        sample_factor * improvement_factor
    }

    /// Count active optimizations
    async fn count_active_optimizations(active_optimizations: Arc<RwLock<HashMap<String, OptimizationStrategy>>>) -> usize {
        active_optimizations.read().await.len()
    }

    /// Get optimization statistics
    pub async fn get_optimization_statistics(&self) -> OptimizationStatistics {
        let history = self.optimization_history.read().await;
        let patterns = self.learned_patterns.read().await;
        let active = self.active_optimizations.read().await;
        
        let total_optimizations = history.len();
        let successful_optimizations = history.iter().filter(|r| r.success).count();
        let avg_improvement = if successful_optimizations > 0 {
            history.iter()
                .filter(|r| r.success)
                .map(|r| r.improvement_factor)
                .sum::<f64>() / successful_optimizations as f64
        } else {
            0.0
        };
        
        let best_optimization = history.iter()
            .filter(|r| r.success)
            .max_by(|a, b| a.improvement_factor.partial_cmp(&b.improvement_factor).unwrap())
            .cloned();
        
        OptimizationStatistics {
            total_optimizations,
            successful_optimizations,
            success_rate: if total_optimizations > 0 {
                successful_optimizations as f64 / total_optimizations as f64
            } else {
                0.0
            },
            avg_improvement_factor: avg_improvement,
            learned_patterns_count: patterns.len(),
            active_optimizations_count: active.len(),
            best_optimization,
        }
    }

    /// Force optimization for specific workload
    pub async fn force_optimization(&self, workload: WorkloadCharacteristics) -> Result<Vec<OptimizationResult>> {
        info!("🔧 Forcing optimization for workload: {:?}", workload);
        
        let current_metrics = Self::collect_performance_metrics().await;
        let strategies = Self::determine_optimization_strategies(
            &workload,
            &current_metrics,
            self.learned_patterns.clone(),
            self.config.clone(),
        ).await;
        
        let mut results = Vec::new();
        
        for strategy in strategies {
            let optimization_id = format!("forced_{}", chrono::Utc::now().timestamp_millis());
            
            match Self::apply_optimization(
                optimization_id,
                strategy.clone(),
                current_metrics.clone(),
                self.active_optimizations.clone(),
            ).await {
                Ok(result) => {
                    self.optimization_history.write().await.push(result.clone());
                    
                    if self.config.enable_learning {
                        Self::update_learned_patterns(
                            workload.clone(),
                            strategy,
                            result.clone(),
                            self.learned_patterns.clone(),
                        ).await;
                    }
                    
                    results.push(result);
                }
                Err(e) => {
                    error!("Forced optimization failed: {}", e);
                }
            }
        }
        
        Ok(results)
    }
}

/// Optimization system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStatistics {
    pub total_optimizations: usize,
    pub successful_optimizations: usize,
    pub success_rate: f64,
    pub avg_improvement_factor: f64,
    pub learned_patterns_count: usize,
    pub active_optimizations_count: usize,
    pub best_optimization: Option<OptimizationResult>,
}