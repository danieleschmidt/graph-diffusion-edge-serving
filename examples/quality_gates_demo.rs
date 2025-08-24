//! Quality Gates Demo
//!
//! Demonstrates comprehensive quality assurance with automated testing,
//! security scanning, performance validation, and code quality checks.

use graph_diffusion_edge::comprehensive_quality_gates::{
    ComprehensiveQualityGates, QualityGateConfig, VulnerabilitySeverity, 
    PerformanceThresholds, CodeQualityThresholds
};
use std::collections::HashMap;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info,graph_diffusion_edge=debug")
        .init();

    println!("🛡️ Comprehensive Quality Gates Demo");
    println!("====================================");
    
    // Configure quality gates with production-ready standards
    let mut max_vulnerabilities = HashMap::new();
    max_vulnerabilities.insert(VulnerabilitySeverity::Critical, 0);
    max_vulnerabilities.insert(VulnerabilitySeverity::High, 1);
    max_vulnerabilities.insert(VulnerabilitySeverity::Medium, 3);
    max_vulnerabilities.insert(VulnerabilitySeverity::Low, 8);

    let quality_config = QualityGateConfig {
        min_code_coverage: 80.0,
        max_cyclomatic_complexity: 12,
        security_scanning_enabled: true,
        performance_validation_enabled: true,
        static_analysis_enabled: true,
        max_vulnerabilities,
        performance_thresholds: PerformanceThresholds {
            max_memory_usage_mb: 1024,
            max_cpu_usage_percent: 85.0,
            max_test_duration_seconds: 600,
            min_throughput_ops_per_sec: 50.0,
            max_response_time_ms: 2000,
        },
        quality_thresholds: CodeQualityThresholds {
            max_lines_per_function: 75,
            max_function_parameters: 8,
            max_nesting_depth: 5,
            min_documentation_coverage: 75.0,
            max_duplicate_code_percentage: 5.0,
        },
    };

    println!("Quality Gate Configuration:");
    println!("  • Minimum code coverage: {:.1}%", quality_config.min_code_coverage);
    println!("  • Maximum cyclomatic complexity: {}", quality_config.max_cyclomatic_complexity);
    println!("  • Security scanning: {}", if quality_config.security_scanning_enabled { "✅ Enabled" } else { "❌ Disabled" });
    println!("  • Performance validation: {}", if quality_config.performance_validation_enabled { "✅ Enabled" } else { "❌ Disabled" });
    println!("  • Static analysis: {}", if quality_config.static_analysis_enabled { "✅ Enabled" } else { "❌ Disabled" });

    let mut quality_gates = ComprehensiveQualityGates::new(quality_config);

    // Execute all quality gates
    info!("Starting comprehensive quality gate execution");
    
    match quality_gates.execute_all_gates().await {
        Ok(summary) => {
            println!("\n📊 Quality Gates Execution Summary");
            println!("==================================");
            println!("Overall Status: {}", if summary.overall_passed { "✅ PASSED" } else { "❌ FAILED" });
            println!("Gates Passed: {}/{}", summary.gates_passed, summary.gates_total);
            println!("Success Rate: {:.1}%", summary.success_rate * 100.0);
            println!("Total Execution Time: {:.2}s", summary.execution_time.as_secs_f32());
            println!("Timestamp: {}", summary.timestamp);

            // Detailed gate results
            println!("\n📋 Detailed Gate Results:");
            println!("=========================");
            
            for result in &summary.gate_results {
                let status_icon = if result.passed { "✅" } else { "❌" };
                println!("\n{} {} (Score: {:.1}%)", status_icon, result.gate_name, result.score);
                println!("   Execution Time: {:.2}s", result.execution_time.as_secs_f32());
                
                // Show specific details based on gate type
                match &result.details {
                    graph_diffusion_edge::comprehensive_quality_gates::QualityGateDetails::TestResults { 
                        total_tests, passed_tests, failed_tests, coverage_percentage 
                    } => {
                        println!("   Tests: {} total, {} passed, {} failed", total_tests, passed_tests, failed_tests);
                        println!("   Coverage: {:.1}%", coverage_percentage);
                    }
                    graph_diffusion_edge::comprehensive_quality_gates::QualityGateDetails::SecurityScan { 
                        vulnerabilities_found, security_score, scan_duration 
                    } => {
                        println!("   Security Score: {:.1}%", security_score);
                        println!("   Scan Duration: {:.2}s", scan_duration.as_secs_f32());
                        for (severity, count) in vulnerabilities_found {
                            if *count > 0 {
                                println!("   {:?}: {} vulnerabilities", severity, count);
                            }
                        }
                    }
                    graph_diffusion_edge::comprehensive_quality_gates::QualityGateDetails::PerformanceValidation { 
                        throughput_ops_per_sec, avg_response_time_ms, peak_memory_usage_mb, cpu_usage_percent 
                    } => {
                        println!("   Throughput: {:.1} ops/sec", throughput_ops_per_sec);
                        println!("   Avg Response Time: {}ms", avg_response_time_ms);
                        println!("   Peak Memory: {}MB", peak_memory_usage_mb);
                        println!("   CPU Usage: {:.1}%", cpu_usage_percent);
                    }
                    graph_diffusion_edge::comprehensive_quality_gates::QualityGateDetails::StaticAnalysis { 
                        cyclomatic_complexity, code_smells, maintainability_index, technical_debt_minutes 
                    } => {
                        println!("   Cyclomatic Complexity: {}", cyclomatic_complexity);
                        println!("   Code Smells: {}", code_smells);
                        println!("   Maintainability Index: {:.1}", maintainability_index);
                        println!("   Technical Debt: {} minutes", technical_debt_minutes);
                    }
                    graph_diffusion_edge::comprehensive_quality_gates::QualityGateDetails::CodeQuality { 
                        documentation_coverage, duplicate_code_percentage, code_quality_score, violations 
                    } => {
                        println!("   Documentation Coverage: {:.1}%", documentation_coverage);
                        println!("   Duplicate Code: {:.1}%", duplicate_code_percentage);
                        println!("   Quality Score: {:.1}", code_quality_score);
                        println!("   Violations: {}", violations);
                    }
                }

                // Show recommendations if any
                if !result.recommendations.is_empty() {
                    println!("   📝 Recommendations:");
                    for (i, recommendation) in result.recommendations.iter().enumerate() {
                        println!("      {}. {}", i + 1, recommendation);
                    }
                }
            }

            // Generate and display quality report
            println!("\n📄 Quality Report");
            println!("=================");
            let report = quality_gates.generate_quality_report();
            println!("{}", report);

            // Quality assessment
            println!("🎯 Quality Assessment:");
            if summary.success_rate == 1.0 {
                println!("   🏆 EXCELLENT - All quality gates passed!");
                println!("   ✅ Production deployment ready");
            } else if summary.success_rate >= 0.8 {
                println!("   ⚠️  GOOD - Most quality gates passed");
                println!("   🔧 Some improvements needed before deployment");
            } else if summary.success_rate >= 0.6 {
                println!("   ❗ NEEDS WORK - Several quality gates failed");
                println!("   🚧 Significant improvements required");
            } else {
                println!("   ❌ POOR - Major quality issues detected");
                println!("   🛑 Deployment not recommended");
            }

            // Performance analysis
            let avg_execution_time = summary.execution_time.as_secs_f32() / summary.gates_total as f32;
            println!("\n⚡ Performance Analysis:");
            println!("   Average gate execution time: {:.2}s", avg_execution_time);
            
            if summary.execution_time.as_secs() < 60 {
                println!("   ✅ Fast execution - suitable for CI/CD pipeline");
            } else if summary.execution_time.as_secs() < 300 {
                println!("   ⚠️ Moderate execution time - acceptable for CI/CD");
            } else {
                println!("   ❗ Slow execution - consider optimizing quality gates");
            }

            // CI/CD integration recommendations
            println!("\n🔄 CI/CD Integration Recommendations:");
            println!("   • Run quality gates on every pull request");
            println!("   • Block deployment if critical gates fail");
            println!("   • Generate quality reports for code review");
            println!("   • Set up automated notifications for gate failures");
            
            if summary.overall_passed {
                println!("   ✅ READY FOR CI/CD INTEGRATION");
            } else {
                println!("   ⚠️ IMPROVE QUALITY BEFORE CI/CD INTEGRATION");
            }

        }
        Err(error) => {
            warn!("Quality gates execution failed: {}", error);
            println!("❌ Quality gates execution failed!");
            println!("Error: {}", error);
            
            // Show partial results if available
            let partial_results = quality_gates.get_gate_results();
            if !partial_results.is_empty() {
                println!("\n🔍 Partial Results Available:");
                for result in partial_results {
                    let status = if result.passed { "✅" } else { "❌" };
                    println!("   {} {} (Score: {:.1}%)", status, result.gate_name, result.score);
                }
            }
        }
    }

    println!("\n✨ Quality Gates Demo Complete!");
    println!("The system now has comprehensive quality assurance in place.");

    Ok(())
}