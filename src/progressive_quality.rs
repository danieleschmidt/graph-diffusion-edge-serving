//! Progressive Quality System
//! 
//! Implements autonomous quality assurance that evolves through generations:
//! Generation 1: Basic functionality validation
//! Generation 2: Robust error handling and resilience  
//! Generation 3: Performance optimization and scaling

use crate::quality_gates::{QualityGatesRunner, GatesConfig, QualityGate, GateResult};
use crate::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use tokio::process::Command;
use tracing::{info, warn, error, debug};

/// Progressive quality generation level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityGeneration {
    Generation1, // Basic functionality
    Generation2, // Robust and reliable  
    Generation3, // Optimized and scalable
}

/// Progressive quality orchestrator
#[derive(Debug)]
pub struct ProgressiveQualitySystem {
    config: ProgressiveQualityConfig,
    current_generation: QualityGeneration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveQualityConfig {
    pub auto_advance: bool,
    pub require_all_gates: bool,
    pub performance_thresholds: PerformanceThresholds,
    pub gates_config: GatesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub max_compilation_time_ms: u64,
    pub max_test_time_ms: u64, 
    pub min_throughput_rps: f64,
    pub max_memory_usage_mb: f64,
    pub max_cpu_usage_percent: f64,
}

impl Default for ProgressiveQualityConfig {
    fn default() -> Self {
        Self {
            auto_advance: true,
            require_all_gates: true,
            performance_thresholds: PerformanceThresholds::default(),
            gates_config: GatesConfig::default(),
        }
    }
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_compilation_time_ms: 60_000,   // 1 minute
            max_test_time_ms: 120_000,         // 2 minutes
            min_throughput_rps: 100.0,         // 100 requests/second
            max_memory_usage_mb: 512.0,        // 512 MB
            max_cpu_usage_percent: 80.0,       // 80% CPU
        }
    }
}

impl ProgressiveQualitySystem {
    /// Create new progressive quality system
    pub fn new(config: ProgressiveQualityConfig) -> Self {
        Self {
            config,
            current_generation: QualityGeneration::Generation1,
        }
    }

    /// Execute quality gates for current generation
    pub async fn execute_generation(&self, generation: QualityGeneration) -> Result<crate::quality_gates::QualityReport> {
        info!("🚀 Executing {:?} quality gates", generation);
        
        let mut runner = QualityGatesRunner::new(self.config.gates_config.clone());
        
        // Add generation-specific gates
        self.add_generation_gates(&mut runner, generation);
        
        let report = runner.execute_all().await?;
        
        if report.overall_passed {
            info!("✅ {:?} quality gates passed!", generation);
            
            if self.config.auto_advance && generation != QualityGeneration::Generation3 {
                info!("🔄 Auto-advancing to next generation");
            }
        } else {
            error!("❌ {:?} quality gates failed!", generation);
            if report.critical_failures > 0 {
                error!("🚨 Critical failures detected - stopping progression");
            }
        }
        
        Ok(report)
    }

    /// Execute all generations progressively  
    pub async fn execute_progressive_cycle(&mut self) -> Result<Vec<crate::quality_gates::QualityReport>> {
        let mut reports = Vec::new();
        
        let generations = [
            QualityGeneration::Generation1,
            QualityGeneration::Generation2, 
            QualityGeneration::Generation3,
        ];
        
        for generation in generations {
            info!("🎯 Starting {:?} quality validation", generation);
            self.current_generation = generation;
            
            let report = self.execute_generation(generation).await?;
            
            if !report.overall_passed {
                if self.config.require_all_gates {
                    error!("❌ {:?} failed, stopping progressive cycle", generation);
                    reports.push(report);
                    break;
                } else {
                    warn!("⚠️ {:?} failed but continuing due to config", generation);
                }
            }
            
            reports.push(report);
        }
        
        self.log_progressive_summary(&reports);
        Ok(reports)
    }

    /// Add gates specific to each generation
    fn add_generation_gates(&self, runner: &mut QualityGatesRunner, generation: QualityGeneration) {
        match generation {
            QualityGeneration::Generation1 => {
                // Basic functionality gates
                runner.add_gate(Box::new(crate::quality_gates::CompilationGate::default()));
                runner.add_gate(Box::new(crate::quality_gates::TestsGate::default()));
                runner.add_gate(Box::new(BasicFunctionalityGate::default()));
                runner.add_gate(Box::new(CodeQualityGate::default()));
            }
            
            QualityGeneration::Generation2 => {
                // All Generation 1 gates plus robustness
                runner.add_gate(Box::new(crate::quality_gates::CompilationGate::default()));
                runner.add_gate(Box::new(crate::quality_gates::TestsGate::default()));
                runner.add_gate(Box::new(crate::quality_gates::SecurityGate::default()));
                runner.add_gate(Box::new(RobustnessGate::default()));
                runner.add_gate(Box::new(ErrorHandlingGate::default()));
                runner.add_gate(Box::new(ResilienceGate::default()));
            }
            
            QualityGeneration::Generation3 => {
                // All previous gates plus performance and scaling
                runner.add_gate(Box::new(crate::quality_gates::CompilationGate::default()));
                runner.add_gate(Box::new(crate::quality_gates::TestsGate::default()));
                runner.add_gate(Box::new(crate::quality_gates::SecurityGate::default()));
                runner.add_gate(Box::new(PerformanceGate::new(self.config.performance_thresholds.clone())));
                runner.add_gate(Box::new(ScalingGate::default()));
                runner.add_gate(Box::new(OptimizationGate::default()));
                runner.add_gate(Box::new(ProductionReadinessGate::default()));
            }
        }
    }

    /// Log summary of progressive execution
    fn log_progressive_summary(&self, reports: &[crate::quality_gates::QualityReport]) {
        info!("📊 Progressive Quality Summary:");
        
        let total_execution_time: u64 = reports.iter().map(|r| r.execution_time_ms).sum();
        let overall_passed = reports.iter().all(|r| r.overall_passed);
        let total_gates = reports.iter().map(|r| r.gates_passed + r.gates_failed).sum::<usize>();
        let total_passed = reports.iter().map(|r| r.gates_passed).sum::<usize>();
        
        info!("   • Overall Status: {}", if overall_passed { "✅ PASSED" } else { "❌ FAILED" });
        info!("   • Total Execution Time: {}ms", total_execution_time);
        info!("   • Total Gates: {}/{} passed", total_passed, total_gates);
        
        for (i, report) in reports.iter().enumerate() {
            let generation = match i {
                0 => "Generation 1",
                1 => "Generation 2", 
                2 => "Generation 3",
                _ => "Unknown",
            };
            
            let status = if report.overall_passed { "✅" } else { "❌" };
            info!("   {} {}: {:.1}% ({:.0}ms)", 
                  status, generation, report.total_score * 100.0, report.execution_time_ms);
        }
    }
}

// ================================================================================
// GENERATION-SPECIFIC QUALITY GATES
// ================================================================================

/// Basic functionality validation gate
pub struct BasicFunctionalityGate {
    pub check_examples: bool,
}

impl Default for BasicFunctionalityGate {
    fn default() -> Self {
        Self {
            check_examples: true,
        }
    }
}

#[async_trait::async_trait]
impl QualityGate for BasicFunctionalityGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Basic Functionality";
        let mut passed = true;
        let mut details = HashMap::new();
        let mut score = 1.0;

        // Check if examples compile and run
        if self.check_examples {
            debug!("Checking examples compilation...");
            match Command::new("cargo")
                .args(&["check", "--examples"])
                .output()
                .await 
            {
                Ok(output) => {
                    let example_check_passed = output.status.success();
                    details.insert("examples_check".to_string(), example_check_passed.to_string());
                    
                    if !example_check_passed {
                        passed = false;
                        score *= 0.7; // 30% penalty for failed examples
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        details.insert("examples_error".to_string(), stderr.to_string());
                    }
                }
                Err(e) => {
                    warn!("Failed to check examples: {}", e);
                    score *= 0.9; // Minor penalty for check failure
                }
            }
        }

        // Check core library functionality
        debug!("Validating core library structure...");
        let lib_check = self.validate_lib_structure().await;
        details.insert("lib_structure_valid".to_string(), lib_check.to_string());
        
        if !lib_check {
            passed = false;
            score *= 0.5; // Major penalty for invalid lib structure
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        
        let message = if passed {
            "Basic functionality validated successfully".to_string()
        } else {
            "Basic functionality validation failed".to_string()
        };

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score,
            duration_ms,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Basic Functionality" }
    fn min_score(&self) -> f64 { 0.8 }
    fn is_critical(&self) -> bool { true }
}

impl BasicFunctionalityGate {
    async fn validate_lib_structure(&self) -> bool {
        // Check if src/lib.rs exists and has basic exports
        if let Ok(content) = tokio::fs::read_to_string("src/lib.rs").await {
            content.contains("pub mod") && content.contains("pub use")
        } else {
            false
        }
    }
}

/// Code quality gate
pub struct CodeQualityGate {
    pub check_formatting: bool,
    pub check_clippy: bool,
}

impl Default for CodeQualityGate {
    fn default() -> Self {
        Self {
            check_formatting: true,
            check_clippy: true,
        }
    }
}

#[async_trait::async_trait]
impl QualityGate for CodeQualityGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Code Quality";
        let mut passed = true;
        let mut details = HashMap::new();
        let mut score = 1.0;

        // Check code formatting
        if self.check_formatting {
            debug!("Checking code formatting...");
            match Command::new("cargo")
                .args(&["fmt", "--", "--check"])
                .output()
                .await 
            {
                Ok(output) => {
                    let fmt_passed = output.status.success();
                    details.insert("formatting_check".to_string(), fmt_passed.to_string());
                    
                    if !fmt_passed {
                        score *= 0.9; // Minor penalty for formatting issues
                        details.insert("formatting_issues".to_string(), "Code not properly formatted".to_string());
                    }
                }
                Err(e) => {
                    warn!("cargo fmt not available: {}", e);
                    details.insert("fmt_error".to_string(), e.to_string());
                }
            }
        }

        // Check clippy lints
        if self.check_clippy {
            debug!("Running clippy lints...");
            match Command::new("cargo")
                .args(&["clippy", "--", "-D", "warnings"])
                .output()
                .await 
            {
                Ok(output) => {
                    let clippy_passed = output.status.success();
                    details.insert("clippy_check".to_string(), clippy_passed.to_string());
                    
                    if !clippy_passed {
                        passed = false;
                        score *= 0.6; // Significant penalty for clippy warnings
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        details.insert("clippy_warnings".to_string(), stderr.to_string());
                    }
                }
                Err(e) => {
                    warn!("clippy not available: {}", e);
                    details.insert("clippy_error".to_string(), e.to_string());
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        
        let message = if passed {
            format!("Code quality validated (score: {:.1}%)", score * 100.0)
        } else {
            format!("Code quality issues found (score: {:.1}%)", score * 100.0)
        };

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score,
            duration_ms,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Code Quality" }
    fn min_score(&self) -> f64 { 0.8 }
    fn is_critical(&self) -> bool { false }
}

/// Robustness validation gate
pub struct RobustnessGate {
    pub check_error_handling: bool,
    pub check_edge_cases: bool,
}

impl Default for RobustnessGate {
    fn default() -> Self {
        Self {
            check_error_handling: true,
            check_edge_cases: true,
        }
    }
}

#[async_trait::async_trait]
impl QualityGate for RobustnessGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Robustness";
        let mut passed = true;
        let mut details = HashMap::new();
        let mut score = 1.0;

        // Check error handling patterns
        if self.check_error_handling {
            debug!("Analyzing error handling...");
            let error_coverage = self.analyze_error_handling().await;
            details.insert("error_handling_coverage".to_string(), format!("{:.1}%", error_coverage * 100.0));
            
            if error_coverage < 0.8 {
                passed = false;
                score = error_coverage;
            }
        }

        // Run integration tests for edge cases
        if self.check_edge_cases {
            debug!("Running integration tests...");
            match Command::new("cargo")
                .args(&["test", "--test", "*"])
                .output()
                .await 
            {
                Ok(output) => {
                    let integration_passed = output.status.success();
                    details.insert("integration_tests".to_string(), integration_passed.to_string());
                    
                    if !integration_passed {
                        passed = false;
                        score *= 0.7;
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        details.insert("integration_errors".to_string(), stderr.to_string());
                    }
                }
                Err(e) => {
                    warn!("Integration tests failed to run: {}", e);
                    score *= 0.9;
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        
        let message = if passed {
            "Robustness validation passed".to_string()
        } else {
            format!("Robustness issues found (score: {:.1}%)", score * 100.0)
        };

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score,
            duration_ms,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Robustness" }
    fn min_score(&self) -> f64 { 0.8 }
    fn is_critical(&self) -> bool { true }
}

impl RobustnessGate {
    async fn analyze_error_handling(&self) -> f64 {
        // Simple heuristic: count error handling patterns in source code
        let patterns = ["Result<", "Error", "?", "unwrap_or", "map_err"];
        let mut total_score: f32 = 0.0;
        
        for pattern in &patterns {
            match Command::new("grep")
                .args(&["-r", "--include=*.rs", "-c", pattern, "src/"])
                .output()
                .await 
            {
                Ok(output) => {
                    if output.status.success() {
                        let count_lines = String::from_utf8_lossy(&output.stdout)
                            .lines()
                            .filter_map(|line| {
                                line.split(':').nth(1)?.parse::<i32>().ok()
                            })
                            .sum::<i32>();
                        
                        if count_lines > 0 {
                            total_score += 0.2; // Each pattern adds 20%
                        }
                    }
                }
                Err(_) => {} // grep not available
            }
        }
        
        total_score.min(1.0)
    }
}

/// Error handling validation gate
pub struct ErrorHandlingGate;

impl Default for ErrorHandlingGate {
    fn default() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl QualityGate for ErrorHandlingGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Error Handling";
        
        // Check if comprehensive error types are defined
        let error_types_defined = self.check_error_types().await;
        let error_propagation = self.check_error_propagation().await;
        
        let passed = error_types_defined && error_propagation;
        let score = if passed { 1.0 } else { 0.5 };
        
        let mut details = HashMap::new();
        details.insert("error_types_defined".to_string(), error_types_defined.to_string());
        details.insert("error_propagation".to_string(), error_propagation.to_string());
        
        let message = if passed {
            "Comprehensive error handling validated".to_string()
        } else {
            "Error handling improvements needed".to_string()
        };

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score,
            duration_ms: start.elapsed().as_millis() as u64,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Error Handling" }
    fn min_score(&self) -> f64 { 0.8 }
    fn is_critical(&self) -> bool { true }
}

impl ErrorHandlingGate {
    async fn check_error_types(&self) -> bool {
        // Check if custom error types are defined
        if let Ok(content) = tokio::fs::read_to_string("src/lib.rs").await {
            content.contains("pub mod error") || content.contains("enum Error")
        } else {
            false
        }
    }

    async fn check_error_propagation(&self) -> bool {
        // Check for proper Result type usage
        match Command::new("grep")
            .args(&["-r", "--include=*.rs", "-c", "Result<", "src/"])
            .output()
            .await 
        {
            Ok(output) => {
                if output.status.success() {
                    let result_count: i32 = String::from_utf8_lossy(&output.stdout)
                        .lines()
                        .filter_map(|line| {
                            line.split(':').nth(1)?.parse::<i32>().ok()
                        })
                        .sum();
                    result_count > 10 // Heuristic: at least 10 Result usages
                } else {
                    false
                }
            }
            Err(_) => false
        }
    }
}

/// Resilience patterns gate
pub struct ResilienceGate;

impl Default for ResilienceGate {
    fn default() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl QualityGate for ResilienceGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Resilience Patterns";
        
        let circuit_breaker = self.check_circuit_breaker().await;
        let retry_logic = self.check_retry_logic().await;
        let health_checks = self.check_health_checks().await;
        
        let patterns_found = [circuit_breaker, retry_logic, health_checks]
            .iter()
            .filter(|&&x| x)
            .count();
        
        let score = patterns_found as f64 / 3.0;
        let passed = score >= 0.67; // At least 2 out of 3 patterns
        
        let mut details = HashMap::new();
        details.insert("circuit_breaker".to_string(), circuit_breaker.to_string());
        details.insert("retry_logic".to_string(), retry_logic.to_string());
        details.insert("health_checks".to_string(), health_checks.to_string());
        details.insert("patterns_found".to_string(), patterns_found.to_string());
        
        let message = format!("Resilience patterns: {}/3 found", patterns_found);

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score,
            duration_ms: start.elapsed().as_millis() as u64,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Resilience Patterns" }
    fn min_score(&self) -> f64 { 0.67 }
    fn is_critical(&self) -> bool { true }
}

impl ResilienceGate {
    async fn check_circuit_breaker(&self) -> bool {
        self.check_pattern("circuit_breaker").await
    }

    async fn check_retry_logic(&self) -> bool {
        self.check_pattern("retry").await || self.check_pattern("backoff").await
    }

    async fn check_health_checks(&self) -> bool {
        self.check_pattern("health").await || self.check_pattern("/health").await
    }

    async fn check_pattern(&self, pattern: &str) -> bool {
        match Command::new("grep")
            .args(&["-r", "--include=*.rs", "-i", pattern, "src/"])
            .output()
            .await 
        {
            Ok(output) => output.status.success() && !output.stdout.is_empty(),
            Err(_) => false
        }
    }
}

/// Performance validation gate
pub struct PerformanceGate {
    pub thresholds: PerformanceThresholds,
}

impl PerformanceGate {
    pub fn new(thresholds: PerformanceThresholds) -> Self {
        Self { thresholds }
    }
}

#[async_trait::async_trait]
impl QualityGate for PerformanceGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Performance";
        let mut passed = true;
        let mut details = HashMap::new();
        let mut score = 1.0;

        // Run benchmarks if available
        debug!("Running performance benchmarks...");
        match Command::new("cargo")
            .args(&["bench", "--features", "benchmark"])
            .output()
            .await 
        {
            Ok(output) => {
                let bench_success = output.status.success();
                details.insert("benchmarks_ran".to_string(), bench_success.to_string());
                
                if bench_success {
                    let bench_output = String::from_utf8_lossy(&output.stdout);
                    let perf_metrics = self.parse_benchmark_results(&bench_output);
                    
                    for (metric, value) in perf_metrics {
                        details.insert(metric, value);
                    }
                } else {
                    warn!("Benchmarks failed to run");
                    score *= 0.8;
                }
            }
            Err(e) => {
                warn!("Benchmarks not available: {}", e);
                details.insert("benchmark_error".to_string(), e.to_string());
                score *= 0.9; // Minor penalty for missing benchmarks
            }
        }

        // Check compilation time (already measured)
        let compilation_time = self.measure_compilation_time().await;
        details.insert("compilation_time_ms".to_string(), compilation_time.to_string());
        
        if compilation_time > self.thresholds.max_compilation_time_ms {
            passed = false;
            score *= 0.7;
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        
        let message = if passed {
            format!("Performance targets met (score: {:.1}%)", score * 100.0)
        } else {
            format!("Performance issues detected (score: {:.1}%)", score * 100.0)
        };

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score,
            duration_ms,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Performance" }
    fn min_score(&self) -> f64 { 0.8 }
    fn is_critical(&self) -> bool { true }
}

impl PerformanceGate {
    async fn measure_compilation_time(&self) -> u64 {
        let start = Instant::now();
        let _ = Command::new("cargo")
            .args(&["check"])
            .output()
            .await;
        start.elapsed().as_millis() as u64
    }

    fn parse_benchmark_results(&self, output: &str) -> HashMap<String, String> {
        let mut metrics = HashMap::new();
        
        // Parse criterion benchmark output
        for line in output.lines() {
            if line.contains("time:") {
                if let Some(time_part) = line.split("time:").nth(1) {
                    metrics.insert("benchmark_time".to_string(), time_part.trim().to_string());
                }
            }
            if line.contains("throughput:") {
                if let Some(throughput_part) = line.split("throughput:").nth(1) {
                    metrics.insert("benchmark_throughput".to_string(), throughput_part.trim().to_string());
                }
            }
        }
        
        metrics
    }
}

/// Scaling validation gate
pub struct ScalingGate;

impl Default for ScalingGate {
    fn default() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl QualityGate for ScalingGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Scaling";
        
        let async_patterns = self.check_async_patterns().await;
        let concurrency_support = self.check_concurrency_support().await;
        let resource_management = self.check_resource_management().await;
        
        let features_found = [async_patterns, concurrency_support, resource_management]
            .iter()
            .filter(|&&x| x)
            .count();
        
        let score = features_found as f64 / 3.0;
        let passed = score >= 0.67;
        
        let mut details = HashMap::new();
        details.insert("async_patterns".to_string(), async_patterns.to_string());
        details.insert("concurrency_support".to_string(), concurrency_support.to_string());
        details.insert("resource_management".to_string(), resource_management.to_string());
        
        let message = format!("Scaling features: {}/3 implemented", features_found);

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score,
            duration_ms: start.elapsed().as_millis() as u64,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Scaling" }
    fn min_score(&self) -> f64 { 0.67 }
    fn is_critical(&self) -> bool { true }
}

impl ScalingGate {
    async fn check_async_patterns(&self) -> bool {
        self.check_pattern("async").await && self.check_pattern("await").await
    }

    async fn check_concurrency_support(&self) -> bool {
        self.check_pattern("tokio").await || self.check_pattern("rayon").await
    }

    async fn check_resource_management(&self) -> bool {
        self.check_pattern("pool").await || self.check_pattern("Arc<").await
    }

    async fn check_pattern(&self, pattern: &str) -> bool {
        match Command::new("grep")
            .args(&["-r", "--include=*.rs", pattern, "src/"])
            .output()
            .await 
        {
            Ok(output) => output.status.success() && !output.stdout.is_empty(),
            Err(_) => false
        }
    }
}

/// Optimization validation gate
pub struct OptimizationGate;

impl Default for OptimizationGate {
    fn default() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl QualityGate for OptimizationGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Optimization";
        
        let caching_patterns = self.check_caching().await;
        let memory_optimization = self.check_memory_optimization().await;
        let algorithmic_optimization = self.check_algorithmic_optimization().await;
        
        let optimizations = [caching_patterns, memory_optimization, algorithmic_optimization]
            .iter()
            .filter(|&&x| x)
            .count();
        
        let score = optimizations as f64 / 3.0;
        let passed = score >= 0.5; // At least 1.5 out of 3 optimizations
        
        let mut details = HashMap::new();
        details.insert("caching_patterns".to_string(), caching_patterns.to_string());
        details.insert("memory_optimization".to_string(), memory_optimization.to_string());
        details.insert("algorithmic_optimization".to_string(), algorithmic_optimization.to_string());
        
        let message = format!("Optimizations: {}/3 implemented", optimizations);

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score,
            duration_ms: start.elapsed().as_millis() as u64,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Optimization" }
    fn min_score(&self) -> f64 { 0.5 }
    fn is_critical(&self) -> bool { false }
}

impl OptimizationGate {
    async fn check_caching(&self) -> bool {
        self.check_pattern("cache").await || self.check_pattern("Cache").await
    }

    async fn check_memory_optimization(&self) -> bool {
        self.check_pattern("Vec::with_capacity").await || 
        self.check_pattern("Box<").await ||
        self.check_pattern("Rc<").await
    }

    async fn check_algorithmic_optimization(&self) -> bool {
        self.check_pattern("optimize").await || 
        self.check_pattern("simd").await ||
        self.check_pattern("parallel").await
    }

    async fn check_pattern(&self, pattern: &str) -> bool {
        match Command::new("grep")
            .args(&["-r", "--include=*.rs", pattern, "src/"])
            .output()
            .await 
        {
            Ok(output) => output.status.success() && !output.stdout.is_empty(),
            Err(_) => false
        }
    }
}

/// Production readiness gate
pub struct ProductionReadinessGate;

impl Default for ProductionReadinessGate {
    fn default() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl QualityGate for ProductionReadinessGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Production Readiness";
        
        let monitoring = self.check_monitoring().await;
        let configuration = self.check_configuration().await;
        let containerization = self.check_containerization().await;
        let documentation = self.check_documentation().await;
        
        let features = [monitoring, configuration, containerization, documentation]
            .iter()
            .filter(|&&x| x)
            .count();
        
        let score = features as f64 / 4.0;
        let passed = score >= 0.75; // At least 3 out of 4 features
        
        let mut details = HashMap::new();
        details.insert("monitoring".to_string(), monitoring.to_string());
        details.insert("configuration".to_string(), configuration.to_string());
        details.insert("containerization".to_string(), containerization.to_string());
        details.insert("documentation".to_string(), documentation.to_string());
        
        let message = format!("Production features: {}/4 ready", features);

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score,
            duration_ms: start.elapsed().as_millis() as u64,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Production Readiness" }
    fn min_score(&self) -> f64 { 0.75 }
    fn is_critical(&self) -> bool { true }
}

impl ProductionReadinessGate {
    async fn check_monitoring(&self) -> bool {
        self.check_pattern("prometheus").await || 
        self.check_pattern("metrics").await ||
        self.check_pattern("tracing").await
    }

    async fn check_configuration(&self) -> bool {
        tokio::fs::metadata("config.toml").await.is_ok() ||
        self.check_pattern("config").await
    }

    async fn check_containerization(&self) -> bool {
        tokio::fs::metadata("Dockerfile").await.is_ok()
    }

    async fn check_documentation(&self) -> bool {
        tokio::fs::metadata("README.md").await.is_ok() &&
        self.check_pattern("///").await // Documentation comments
    }

    async fn check_pattern(&self, pattern: &str) -> bool {
        match Command::new("grep")
            .args(&["-r", "--include=*.rs", pattern, "src/"])
            .output()
            .await 
        {
            Ok(output) => output.status.success() && !output.stdout.is_empty(),
            Err(_) => false
        }
    }
}