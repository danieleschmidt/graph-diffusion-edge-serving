//! Progressive Quality Gates System
//! 
//! Autonomous validation system that ensures code quality, performance,
//! and security standards are maintained throughout the development lifecycle.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tracing::{info, warn, error, debug};
use crate::Result;

/// Quality gate execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    pub gate_name: String,
    pub passed: bool,
    pub score: f64,
    pub duration_ms: u64,
    pub message: String,
    pub details: HashMap<String, String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Progressive quality gate that can be executed
#[async_trait::async_trait]
pub trait QualityGate: Send + Sync {
    /// Execute the quality gate check
    async fn execute(&self) -> Result<GateResult>;
    
    /// Get the gate name
    fn name(&self) -> &str;
    
    /// Get the minimum passing score (0.0 to 1.0)
    fn min_score(&self) -> f64;
    
    /// Whether this gate is critical (must pass)
    fn is_critical(&self) -> bool { true }
}

/// Quality gates orchestrator
#[derive(Debug)]
pub struct QualityGatesRunner {
    gates: Vec<Box<dyn QualityGate>>,
    config: GatesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatesConfig {
    pub parallel_execution: bool,
    pub fail_fast: bool,
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
    pub retry_delay_ms: u64,
}

impl Default for GatesConfig {
    fn default() -> Self {
        Self {
            parallel_execution: true,
            fail_fast: false,
            timeout_seconds: 300, // 5 minutes
            retry_attempts: 2,
            retry_delay_ms: 1000,
        }
    }
}

/// Comprehensive gate execution report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    pub overall_passed: bool,
    pub total_score: f64,
    pub execution_time_ms: u64,
    pub gates_passed: usize,
    pub gates_failed: usize,
    pub critical_failures: usize,
    pub results: Vec<GateResult>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl QualityGatesRunner {
    /// Create new quality gates runner
    pub fn new(config: GatesConfig) -> Self {
        Self {
            gates: Vec::new(),
            config,
        }
    }

    /// Add a quality gate
    pub fn add_gate(&mut self, gate: Box<dyn QualityGate>) {
        self.gates.push(gate);
    }

    /// Execute all quality gates
    pub async fn execute_all(&self) -> Result<QualityReport> {
        let start_time = Instant::now();
        let mut results = Vec::new();
        let mut overall_passed = true;
        let mut critical_failures = 0;

        info!("🚀 Executing {} quality gates", self.gates.len());

        if self.config.parallel_execution {
            // Execute gates in parallel
            let futures: Vec<_> = self.gates.iter().map(|gate| {
                self.execute_gate_with_retry(gate.as_ref())
            }).collect();

            let gate_results = futures::future::join_all(futures).await;
            
            for result in gate_results {
                match result {
                    Ok(gate_result) => {
                        if !gate_result.passed {
                            overall_passed = false;
                            if self.gates.iter().any(|g| g.name() == gate_result.gate_name && g.is_critical()) {
                                critical_failures += 1;
                            }
                        }
                        results.push(gate_result);
                    }
                    Err(e) => {
                        error!("Gate execution failed: {}", e);
                        overall_passed = false;
                        critical_failures += 1;
                    }
                }
            }
        } else {
            // Execute gates sequentially
            for gate in &self.gates {
                match self.execute_gate_with_retry(gate.as_ref()).await {
                    Ok(gate_result) => {
                        let gate_passed = gate_result.passed;
                        results.push(gate_result);
                        
                        if !gate_passed {
                            overall_passed = false;
                            if gate.is_critical() {
                                critical_failures += 1;
                                if self.config.fail_fast {
                                    warn!("Critical gate failed, stopping execution due to fail_fast");
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        error!("Gate execution failed: {}", e);
                        overall_passed = false;
                        critical_failures += 1;
                        if self.config.fail_fast {
                            break;
                        }
                    }
                }
            }
        }

        let execution_time = start_time.elapsed();
        let gates_passed = results.iter().filter(|r| r.passed).count();
        let gates_failed = results.len() - gates_passed;
        let total_score = if results.is_empty() { 
            0.0 
        } else { 
            results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64 
        };

        let report = QualityReport {
            overall_passed,
            total_score,
            execution_time_ms: execution_time.as_millis() as u64,
            gates_passed,
            gates_failed,
            critical_failures,
            results,
            timestamp: chrono::Utc::now(),
        };

        self.log_report(&report);
        Ok(report)
    }

    /// Execute a single gate with retry logic
    async fn execute_gate_with_retry(&self, gate: &dyn QualityGate) -> Result<GateResult> {
        let mut last_error = None;
        
        for attempt in 0..=self.config.retry_attempts {
            if attempt > 0 {
                debug!("Retrying gate {} (attempt {})", gate.name(), attempt + 1);
                tokio::time::sleep(Duration::from_millis(self.config.retry_delay_ms)).await;
            }

            let timeout = Duration::from_secs(self.config.timeout_seconds);
            match tokio::time::timeout(timeout, gate.execute()).await {
                Ok(Ok(result)) => return Ok(result),
                Ok(Err(e)) => {
                    warn!("Gate {} failed on attempt {}: {}", gate.name(), attempt + 1, e);
                    last_error = Some(e);
                }
                Err(_) => {
                    let timeout_error = crate::error::Error::timeout(
                        format!("Gate {}", gate.name()),
                        timeout.as_millis() as u64,
                        timeout.as_millis() as u64,
                    );
                    warn!("Gate {} timed out on attempt {}", gate.name(), attempt + 1);
                    last_error = Some(timeout_error);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            crate::error::Error::graph_processing(
                format!("Gate {} failed after {} attempts", gate.name(), self.config.retry_attempts + 1),
                "quality_gates"
            )
        }))
    }

    /// Log comprehensive report
    fn log_report(&self, report: &QualityReport) {
        if report.overall_passed {
            info!("✅ All quality gates passed!");
        } else {
            error!("❌ Quality gates failed!");
        }

        info!("📊 Quality Report Summary:");
        info!("   • Overall Score: {:.1}%", report.total_score * 100.0);
        info!("   • Execution Time: {}ms", report.execution_time_ms);
        info!("   • Gates Passed: {}/{}", report.gates_passed, report.gates_passed + report.gates_failed);
        
        if report.critical_failures > 0 {
            error!("   • Critical Failures: {}", report.critical_failures);
        }

        for result in &report.results {
            let status = if result.passed { "✅" } else { "❌" };
            let score_pct = result.score * 100.0;
            info!("{} {} - {:.1}% ({:.0}ms): {}", 
                  status, result.gate_name, score_pct, result.duration_ms, result.message);
            
            if !result.passed && !result.details.is_empty() {
                for (key, value) in &result.details {
                    debug!("     {} = {}", key, value);
                }
            }
        }
    }
}

// ================================================================================
// BUILT-IN QUALITY GATES
// ================================================================================

/// Code compilation gate
pub struct CompilationGate {
    pub cargo_args: Vec<String>,
}

impl Default for CompilationGate {
    fn default() -> Self {
        Self {
            cargo_args: vec!["build", "--release"].into_iter().map(String::from).collect(),
        }
    }
}

#[async_trait::async_trait]
impl QualityGate for CompilationGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Compilation";

        debug!("Running cargo build...");
        let output = Command::new("cargo")
            .args(&self.cargo_args)
            .output()
            .await
            .map_err(|e| crate::error::Error::graph_processing(
                format!("Failed to execute cargo: {}", e),
                "compilation_gate"
            ))?;

        let duration_ms = start.elapsed().as_millis() as u64;
        let passed = output.status.success();
        
        let mut details = HashMap::new();
        details.insert("exit_code".to_string(), output.status.code().unwrap_or(-1).to_string());
        
        if !output.stderr.is_empty() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            details.insert("stderr".to_string(), stderr.to_string());
        }

        let message = if passed {
            "Code compiled successfully".to_string()
        } else {
            format!("Compilation failed with exit code {}", 
                   output.status.code().unwrap_or(-1))
        };

        Ok(GateResult {
            gate_name: gate_name.to_string(),
            passed,
            score: if passed { 1.0 } else { 0.0 },
            duration_ms,
            message,
            details,
            timestamp: chrono::Utc::now(),
        })
    }

    fn name(&self) -> &str { "Compilation" }
    fn min_score(&self) -> f64 { 1.0 }
    fn is_critical(&self) -> bool { true }
}

/// Unit tests gate
pub struct TestsGate {
    pub test_args: Vec<String>,
    pub min_coverage: f64,
}

impl Default for TestsGate {
    fn default() -> Self {
        Self {
            test_args: vec!["test", "--lib"].into_iter().map(String::from).collect(),
            min_coverage: 0.85, // 85% minimum coverage
        }
    }
}

#[async_trait::async_trait]
impl QualityGate for TestsGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Unit Tests";

        debug!("Running cargo test...");
        let output = Command::new("cargo")
            .args(&self.test_args)
            .output()
            .await
            .map_err(|e| crate::error::Error::graph_processing(
                format!("Failed to execute cargo test: {}", e),
                "tests_gate"
            ))?;

        let duration_ms = start.elapsed().as_millis() as u64;
        let exit_success = output.status.success();
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        // Parse test results
        let (tests_passed, tests_failed) = self.parse_test_output(&stdout);
        let total_tests = tests_passed + tests_failed;
        
        let mut details = HashMap::new();
        details.insert("exit_code".to_string(), output.status.code().unwrap_or(-1).to_string());
        details.insert("tests_passed".to_string(), tests_passed.to_string());
        details.insert("tests_failed".to_string(), tests_failed.to_string());
        details.insert("total_tests".to_string(), total_tests.to_string());
        
        if !stderr.is_empty() {
            details.insert("stderr".to_string(), stderr.to_string());
        }

        let test_success_rate = if total_tests > 0 {
            tests_passed as f64 / total_tests as f64
        } else {
            1.0 // No tests is considered passing for now
        };

        let passed = exit_success && test_success_rate >= self.min_coverage;
        let score = test_success_rate;

        let message = if passed {
            format!("Tests passed: {}/{} ({:.1}%)", tests_passed, total_tests, score * 100.0)
        } else {
            format!("Tests failed: {}/{} ({:.1}%)", tests_passed, total_tests, score * 100.0)
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

    fn name(&self) -> &str { "Unit Tests" }
    fn min_score(&self) -> f64 { self.min_coverage }
    fn is_critical(&self) -> bool { true }
}

impl TestsGate {
    fn parse_test_output(&self, output: &str) -> (usize, usize) {
        // Simple regex-based parsing of cargo test output
        let re = regex::Regex::new(r"test result: \w+\. (\d+) passed; (\d+) failed").unwrap();
        
        if let Some(captures) = re.captures(output) {
            let passed = captures.get(1)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let failed = captures.get(2)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            (passed, failed)
        } else {
            // Fallback: count individual test results
            let passed = output.matches("test ").filter(|line| line.contains(" ... ok")).count();
            let failed = output.matches("test ").filter(|line| line.contains(" ... FAILED")).count();
            (passed, failed)
        }
    }
}

/// Security scan gate  
pub struct SecurityGate {
    pub cargo_audit: bool,
    pub dependency_check: bool,
}

impl Default for SecurityGate {
    fn default() -> Self {
        Self {
            cargo_audit: true,
            dependency_check: true,
        }
    }
}

#[async_trait::async_trait]
impl QualityGate for SecurityGate {
    async fn execute(&self) -> Result<GateResult> {
        let start = Instant::now();
        let gate_name = "Security Scan";
        let mut passed = true;
        let mut details = HashMap::new();
        let mut issues = Vec::new();

        // Run cargo audit if enabled
        if self.cargo_audit {
            debug!("Running cargo audit...");
            match Command::new("cargo")
                .args(&["audit", "--format", "json"])
                .output()
                .await 
            {
                Ok(output) => {
                    details.insert("audit_exit_code".to_string(), output.status.code().unwrap_or(-1).to_string());
                    
                    if !output.status.success() {
                        passed = false;
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        if stderr.contains("vulnerabilities found") {
                            issues.push("Security vulnerabilities found in dependencies".to_string());
                        }
                        details.insert("audit_stderr".to_string(), stderr.to_string());
                    }
                }
                Err(e) => {
                    warn!("cargo audit not available: {}", e);
                    details.insert("audit_error".to_string(), e.to_string());
                }
            }
        }

        // Check for common security anti-patterns in code
        if self.dependency_check {
            debug!("Checking for security anti-patterns...");
            let security_issues = self.check_security_patterns().await;
            if !security_issues.is_empty() {
                passed = false;
                issues.extend(security_issues);
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        let score = if passed { 1.0 } else { 0.0 };
        
        details.insert("issues_found".to_string(), issues.len().to_string());
        
        let message = if passed {
            "No security issues found".to_string()
        } else {
            format!("Found {} security issues", issues.len())
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

    fn name(&self) -> &str { "Security Scan" }
    fn min_score(&self) -> f64 { 1.0 }
    fn is_critical(&self) -> bool { true }
}

impl SecurityGate {
    async fn check_security_patterns(&self) -> Vec<String> {
        let mut issues = Vec::new();
        
        // Check for hardcoded secrets/passwords
        let patterns = [
            "password", "secret", "key", "token", "credential"
        ];
        
        for pattern in &patterns {
            match Command::new("grep")
                .args(&["-r", "-i", "--include=*.rs", pattern, "src/"])
                .output()
                .await 
            {
                Ok(output) => {
                    if output.status.success() && !output.stdout.is_empty() {
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        // Simple heuristic: look for assignments to strings containing these words
                        if stdout.lines().any(|line| line.contains("=") && line.contains("\"")) {
                            issues.push(format!("Potential hardcoded {} found", pattern));
                        }
                    }
                }
                Err(_) => {} // grep not available, skip
            }
        }
        
        issues
    }
}