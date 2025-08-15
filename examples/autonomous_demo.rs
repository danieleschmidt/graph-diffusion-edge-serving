//! Autonomous SDLC Demonstration
//! 
//! Demonstrates the complete autonomous Software Development Life Cycle
//! with progressive quality gates and self-improving systems.

use graph_diffusion_edge::{
    autonomous_executor::{AutonomousExecutor, AutonomousConfig},
    progressive_quality::{ProgressiveQualityConfig, PerformanceThresholds},
    quality_gates::GatesConfig,
};
use tracing::{info, warn, error, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(false)
        .init();

    info!("🤖 Terragon Autonomous SDLC Demo");
    info!("═══════════════════════════════════════");
    
    // Configure autonomous execution
    let config = AutonomousConfig {
        enable_auto_commit: false, // Safety: disabled for demo
        enable_auto_deploy: false, // Safety: disabled for demo
        enable_auto_scaling: true,
        max_execution_time_minutes: 30,
        quality_config: ProgressiveQualityConfig {
            auto_advance: true,
            require_all_gates: false, // Allow progression with warnings
            performance_thresholds: PerformanceThresholds {
                max_compilation_time_ms: 120_000, // 2 minutes for demo
                max_test_time_ms: 180_000,        // 3 minutes for demo
                min_throughput_rps: 50.0,         // Relaxed for demo
                max_memory_usage_mb: 1024.0,      // 1GB limit
                max_cpu_usage_percent: 90.0,      // 90% CPU
            },
            gates_config: GatesConfig {
                parallel_execution: true,
                fail_fast: false,
                timeout_seconds: 300,
                retry_attempts: 1, // Reduce retries for demo
                retry_delay_ms: 500,
            },
        },
        research_mode: true,
        global_deployment: true,
    };

    info!("⚙️  Autonomous Configuration:");
    info!("   • Research Mode: {}", config.research_mode);
    info!("   • Global Deployment: {}", config.global_deployment);
    info!("   • Max Execution: {}min", config.max_execution_time_minutes);
    info!("   • Auto Commit: {} (disabled for safety)", config.enable_auto_commit);
    info!("   • Auto Deploy: {} (disabled for safety)", config.enable_auto_deploy);

    // Create autonomous executor
    let mut executor = AutonomousExecutor::new(config);

    info!("\n🚀 Starting Autonomous SDLC Cycle...");
    
    // Execute autonomous cycle
    match executor.execute_autonomous_cycle().await {
        Ok(result) => {
            info!("\n✅ Autonomous execution completed!");
            
            info!("📊 Results Summary:");
            info!("   • Success: {}", result.success);
            info!("   • Execution Time: {:.2} minutes", result.execution_time_minutes);
            info!("   • Generations Completed: {}", result.generations_completed.len());
            info!("   • Quality Reports: {}", result.quality_reports.len());
            info!("   • Commits Made: {}", result.commits_made);
            info!("   • Deployments: {}", result.deployments_made);
            
            if !result.errors.is_empty() {
                warn!("⚠️  Errors encountered:");
                for error in &result.errors {
                    warn!("   • {}", error);
                }
            }
            
            if !result.recommendations.is_empty() {
                info!("💡 Recommendations:");
                for rec in &result.recommendations {
                    info!("   • {}", rec);
                }
            }

            // Detailed quality analysis
            if !result.quality_reports.is_empty() {
                info!("\n📈 Quality Analysis:");
                
                for (i, report) in result.quality_reports.iter().enumerate() {
                    let generation = match i {
                        0 => "Generation 1 (Basic)",
                        1 => "Generation 2 (Robust)", 
                        2 => "Generation 3 (Optimized)",
                        _ => "Unknown Generation",
                    };
                    
                    let status = if report.overall_passed { "✅" } else { "❌" };
                    info!("   {} {}", status, generation);
                    info!("      • Score: {:.1}%", report.total_score * 100.0);
                    info!("      • Time: {}ms", report.execution_time_ms);
                    info!("      • Gates: {}/{} passed", 
                          report.gates_passed, 
                          report.gates_passed + report.gates_failed);
                    
                    if report.critical_failures > 0 {
                        warn!("      • Critical failures: {}", report.critical_failures);
                    }
                }
                
                // Show best performing gates
                let mut all_results = Vec::new();
                for report in &result.quality_reports {
                    all_results.extend(report.results.iter());
                }
                
                if !all_results.is_empty() {
                    info!("\n🏆 Top Performing Gates:");
                    let mut sorted_results = all_results.clone();
                    sorted_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
                    
                    for result in sorted_results.iter().take(5) {
                        let status = if result.passed { "✅" } else { "❌" };
                        info!("   {} {} - {:.1}% ({:.0}ms)", 
                              status, result.gate_name, result.score * 100.0, result.duration_ms);
                    }
                }
            }

            info!("\n🎯 Autonomous SDLC Demonstration Complete!");
            info!("Next steps:");
            info!("   • Review quality gate results");
            info!("   • Implement recommended improvements");
            info!("   • Enable auto-commit/deploy for production use");
            info!("   • Scale to multi-environment deployments");
            
        }
        Err(e) => {
            error!("❌ Autonomous execution failed: {}", e);
            return Err(e.into());
        }
    }

    Ok(())
}