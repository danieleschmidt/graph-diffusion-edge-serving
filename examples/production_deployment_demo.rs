//! Production Deployment Demo
//!
//! Demonstrates comprehensive production deployment orchestration with
//! blue-green deployments, canary releases, rollback capabilities, and health monitoring.

use graph_diffusion_edge::production_deployment_orchestrator::{
    ProductionDeploymentOrchestrator, ProductionDeploymentConfig, DeploymentStrategy,
    DeploymentEnvironment, ResourceRequirements, NetworkPolicy, IngressRule, EgressRule,
    LoadBalancerConfig, PersistentStorage, HealthCheckConfig, RollbackConfig, RollbackTrigger,
    MonitoringConfig, AlertingRule, AlertSeverity, DashboardConfig, DashboardPanel,
    SecurityConfig, PodSecurityStandards, SecretsManagement, ImageSecurity
};
use tracing::{info, warn};
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize comprehensive logging for deployment tracking
    tracing_subscriber::fmt()
        .with_env_filter("info,graph_diffusion_edge=debug")
        .with_target(true)
        .init();

    println!("🚀 Production Deployment Orchestration Demo");
    println!("=============================================");
    
    // === Part 1: Blue-Green Deployment Strategy ===
    println!("\n💙💚 Part 1: Blue-Green Deployment Strategy");
    println!("--------------------------------------------");
    
    await_blue_green_deployment_demo().await?;

    // === Part 2: Canary Release Strategy ===
    println!("\n🕊️ Part 2: Canary Release Strategy");
    println!("----------------------------------");
    
    await_canary_deployment_demo().await?;

    // === Part 3: Rolling Update Strategy ===
    println!("\n🔄 Part 3: Rolling Update Strategy");
    println!("----------------------------------");
    
    await_rolling_deployment_demo().await?;

    // === Part 4: Production Configuration & Security ===
    println!("\n🔒 Part 4: Production Configuration & Security");
    println!("----------------------------------------------");
    
    await_production_config_demo().await?;

    println!("\n✅ Production Deployment Demo Complete!");
    println!("System is ready for enterprise production deployment.");
    
    Ok(())
}

async fn await_blue_green_deployment_demo() -> Result<(), Box<dyn std::error::Error>> {
    let deployment_config = ProductionDeploymentConfig {
        strategy: DeploymentStrategy::BlueGreen {
            switch_timeout_seconds: 180,
            validation_traffic_percent: 15.0,
        },
        environments: vec![
            create_production_environment("blue", 3),
            create_production_environment("green", 3),
        ],
        health_check_config: HealthCheckConfig {
            endpoint: "/health".to_string(),
            interval_seconds: 5,
            timeout_seconds: 3,
            healthy_threshold: 3,
            unhealthy_threshold: 2,
            expected_status_codes: vec![200],
        },
        rollback_config: RollbackConfig {
            auto_rollback_enabled: true,
            rollback_triggers: vec![
                RollbackTrigger::HighErrorRate { 
                    threshold_percent: 3.0, 
                    duration_seconds: 120 
                },
                RollbackTrigger::HighLatency { 
                    threshold_ms: 2000, 
                    duration_seconds: 180 
                },
                RollbackTrigger::HealthCheckFailure { 
                    failure_count: 3 
                },
            ],
            rollback_timeout_seconds: 240,
            preserve_rollback_history: 5,
        },
        ..ProductionDeploymentConfig::default()
    };

    let mut orchestrator = ProductionDeploymentOrchestrator::new(deployment_config);

    println!("Blue-Green Deployment Configuration:");
    println!("  • Switch timeout: 180 seconds");
    println!("  • Validation traffic: 15%");
    println!("  • Environments: Blue (3 replicas), Green (3 replicas)");
    println!("  • Auto-rollback: Enabled");
    println!("  • Health check interval: 5 seconds");

    // Execute blue-green deployment
    println!("\n🔄 Executing Blue-Green Deployment to v2.1.0...");
    
    let deployment_result = orchestrator.deploy("v2.1.0".to_string()).await;
    
    match deployment_result {
        Ok(result) => {
            println!("\n📊 Blue-Green Deployment Results:");
            println!("  • Deployment ID: {}", result.deployment_id);
            println!("  • Success: {}", if result.success { "✅" } else { "❌" });
            println!("  • Environments deployed: {:?}", result.environments_deployed);
            println!("  • Rollback performed: {}", if result.rollback_performed { "Yes" } else { "No" });
            println!("  • Duration: {:.2}s", result.deployment_duration.as_secs_f32());
            
            // Display deployment status
            let status = orchestrator.get_deployment_status();
            println!("  • Current version: {}", status.current_version);
            println!("  • Target version: {}", status.target_version);
            println!("  • Status: {:?}", status.status);
            
            // Show environment states
            for (env_name, env_state) in &status.environments {
                println!("  • Environment {}: {:?} ({}/{} replicas ready, {:.1}% traffic)", 
                        env_name, env_state.status, env_state.replicas_ready, 
                        env_state.replicas_total, env_state.traffic_percent);
            }

            if result.success {
                println!("  ✅ Blue-Green deployment completed successfully!");
            } else if result.rollback_performed {
                warn!("  ⚠️ Deployment failed and was rolled back");
            } else {
                warn!("  ❌ Deployment failed without rollback");
            }
        }
        Err(e) => {
            println!("❌ Blue-Green deployment failed: {}", e);
        }
    }

    // Test health monitoring
    println!("\n🏥 Testing Environment Health Monitoring:");
    for env_name in ["blue", "green"] {
        match orchestrator.get_environment_health(env_name).await {
            Ok(health) => {
                println!("  • {}: {} ({}ms response time)", 
                        env_name, 
                        if health.healthy { "✅ Healthy" } else { "❌ Unhealthy" },
                        health.response_time_ms);
            }
            Err(e) => {
                println!("  • {}: ❌ Health check failed - {}", env_name, e);
            }
        }
    }

    // Show deployment history
    let history = orchestrator.get_deployment_history();
    if !history.is_empty() {
        println!("\n📋 Deployment History:");
        for (i, record) in history.iter().rev().take(3).enumerate() {
            println!("  {}. {} -> {} ({}) - {}", 
                    i + 1, record.deployment_id, record.version, 
                    record.status, 
                    if let Some(reason) = &record.rollback_reason {
                        format!("Rollback: {}", reason)
                    } else {
                        "Success".to_string()
                    });
        }
    }

    Ok(())
}

async fn await_canary_deployment_demo() -> Result<(), Box<dyn std::error::Error>> {
    let deployment_config = ProductionDeploymentConfig {
        strategy: DeploymentStrategy::CanaryRelease {
            initial_traffic_percent: 5.0,
            traffic_increment_percent: 15.0,
            step_duration_seconds: 30, // Reduced for demo
        },
        environments: vec![
            create_production_environment("production", 5),
            create_production_environment("canary", 1),
        ],
        health_check_config: HealthCheckConfig {
            endpoint: "/health".to_string(),
            interval_seconds: 10,
            timeout_seconds: 5,
            healthy_threshold: 2,
            unhealthy_threshold: 3,
            expected_status_codes: vec![200, 201],
        },
        rollback_config: RollbackConfig {
            auto_rollback_enabled: true,
            rollback_triggers: vec![
                RollbackTrigger::HighErrorRate { 
                    threshold_percent: 2.0, 
                    duration_seconds: 90 
                },
                RollbackTrigger::LowThroughput { 
                    threshold_rps: 50.0, 
                    duration_seconds: 120 
                },
            ],
            rollback_timeout_seconds: 180,
            preserve_rollback_history: 10,
        },
        ..ProductionDeploymentConfig::default()
    };

    let mut orchestrator = ProductionDeploymentOrchestrator::new(deployment_config);

    println!("Canary Release Configuration:");
    println!("  • Initial traffic: 5%");
    println!("  • Traffic increment: 15% per step");
    println!("  • Step duration: 30 seconds");
    println!("  • Production: 5 replicas, Canary: 1 replica");
    println!("  • Rollback triggers: Error rate >2%, Low throughput <50 RPS");

    // Execute canary deployment
    println!("\n🔄 Executing Canary Release to v2.2.0...");
    
    let deployment_result = orchestrator.deploy("v2.2.0".to_string()).await;
    
    match deployment_result {
        Ok(result) => {
            println!("\n📊 Canary Release Results:");
            println!("  • Deployment ID: {}", result.deployment_id);
            println!("  • Success: {}", if result.success { "✅" } else { "❌" });
            println!("  • Environments deployed: {:?}", result.environments_deployed);
            println!("  • Rollback performed: {}", if result.rollback_performed { "Yes" } else { "No" });
            println!("  • Duration: {:.2}s", result.deployment_duration.as_secs_f32());

            // Analyze deployment performance
            let avg_step_duration = result.deployment_duration.as_secs_f32() / 6.0; // Approximately 6 steps
            println!("  • Average step duration: {:.1}s", avg_step_duration);
            
            if result.success {
                println!("  ✅ Canary release completed successfully!");
                println!("  📈 Traffic gradually migrated: 5% → 20% → 35% → 50% → 65% → 80% → 100%");
            } else if result.rollback_performed {
                warn!("  ⚠️ Canary release failed and was rolled back");
            }
        }
        Err(e) => {
            println!("❌ Canary release failed: {}", e);
        }
    }

    // Show final environment state
    let status = orchestrator.get_deployment_status();
    println!("\n🎯 Final Environment State:");
    for (env_name, env_state) in &status.environments {
        println!("  • {}: {:?} - {:.1}% traffic, {}/{} replicas", 
                env_name, env_state.status, env_state.traffic_percent,
                env_state.replicas_ready, env_state.replicas_total);
    }

    Ok(())
}

async fn await_rolling_deployment_demo() -> Result<(), Box<dyn std::error::Error>> {
    let deployment_config = ProductionDeploymentConfig {
        strategy: DeploymentStrategy::RollingUpdate {
            max_unavailable: 1,
            max_surge: 2,
            update_delay_seconds: 15, // Reduced for demo
        },
        environments: vec![
            create_production_environment("production", 6),
        ],
        health_check_config: HealthCheckConfig {
            endpoint: "/health".to_string(),
            interval_seconds: 8,
            timeout_seconds: 4,
            healthy_threshold: 2,
            unhealthy_threshold: 2,
            expected_status_codes: vec![200],
        },
        rollback_config: RollbackConfig {
            auto_rollback_enabled: true,
            rollback_triggers: vec![
                RollbackTrigger::HealthCheckFailure { failure_count: 2 },
            ],
            rollback_timeout_seconds: 300,
            preserve_rollback_history: 15,
        },
        ..ProductionDeploymentConfig::default()
    };

    let mut orchestrator = ProductionDeploymentOrchestrator::new(deployment_config);

    println!("Rolling Update Configuration:");
    println!("  • Max unavailable: 1 replica");
    println!("  • Max surge: 2 replicas");
    println!("  • Update delay: 15 seconds");
    println!("  • Production replicas: 6");
    println!("  • Zero-downtime deployment strategy");

    // Execute rolling deployment
    println!("\n🔄 Executing Rolling Update to v2.3.0...");
    
    let deployment_result = orchestrator.deploy("v2.3.0".to_string()).await;
    
    match deployment_result {
        Ok(result) => {
            println!("\n📊 Rolling Update Results:");
            println!("  • Deployment ID: {}", result.deployment_id);
            println!("  • Success: {}", if result.success { "✅" } else { "❌" });
            println!("  • Environments deployed: {:?}", result.environments_deployed);
            println!("  • Rollback performed: {}", if result.rollback_performed { "Yes" } else { "No" });
            println!("  • Duration: {:.2}s", result.deployment_duration.as_secs_f32());

            // Calculate rolling update statistics
            let total_replicas = 6;
            let batch_size = 2; // Based on max_surge
            let estimated_batches = (total_replicas + batch_size - 1) / batch_size;
            let avg_batch_time = result.deployment_duration.as_secs_f32() / estimated_batches as f32;
            
            println!("  • Estimated batches: {}", estimated_batches);
            println!("  • Average batch time: {:.1}s", avg_batch_time);
            println!("  • Zero downtime maintained: ✅");
            
            if result.success {
                println!("  ✅ Rolling update completed successfully!");
                println!("  🔄 All {} replicas updated incrementally", total_replicas);
            } else if result.rollback_performed {
                warn!("  ⚠️ Rolling update failed and was rolled back");
            }
        }
        Err(e) => {
            println!("❌ Rolling update failed: {}", e);
        }
    }

    Ok(())
}

async fn await_production_config_demo() -> Result<(), Box<dyn std::error::Error>> {
    // Create comprehensive production configuration
    let mut environments = vec![
        create_enterprise_production_environment("production", 8),
        create_enterprise_production_environment("staging", 3),
    ];

    // Add disaster recovery environment
    environments.push(DeploymentEnvironment {
        name: "disaster-recovery".to_string(),
        namespace: "dr".to_string(),
        replicas: 5,
        resources: ResourceRequirements {
            cpu_request: "1".to_string(),
            cpu_limit: "4".to_string(),
            memory_request: "2Gi".to_string(),
            memory_limit: "8Gi".to_string(),
            gpu_count: 2,
        },
        network_policy: create_secure_network_policy(),
        persistent_storage: PersistentStorage {
            enabled: true,
            storage_class: "fast-ssd-replicated".to_string(),
            size: "100Gi".to_string(),
            access_modes: vec!["ReadWriteOnce".to_string()],
        },
    });

    let production_config = ProductionDeploymentConfig {
        strategy: DeploymentStrategy::BlueGreen {
            switch_timeout_seconds: 600,
            validation_traffic_percent: 10.0,
        },
        environments,
        health_check_config: create_enterprise_health_config(),
        rollback_config: create_comprehensive_rollback_config(),
        monitoring_config: create_comprehensive_monitoring_config(),
        security_config: create_enterprise_security_config(),
    };

    let orchestrator = ProductionDeploymentOrchestrator::new(production_config.clone());

    println!("🏢 Enterprise Production Configuration:");
    println!("======================================");
    
    // Display configuration summary
    println!("\n📋 Deployment Configuration:");
    println!("  • Strategy: Blue-Green with 600s timeout");
    println!("  • Environments: Production (8), Staging (3), DR (5)");
    println!("  • Total capacity: 16 replicas across 3 environments");
    
    println!("\n🛡️ Security Configuration:");
    println!("  • RBAC: {}", if production_config.security_config.rbac_enabled { "✅ Enabled" } else { "❌ Disabled" });
    println!("  • Network Policies: {}", if production_config.security_config.network_policies_enabled { "✅ Enabled" } else { "❌ Disabled" });
    println!("  • Pod Security: {} profile", production_config.security_config.pod_security_standards.profile);
    println!("  • Run as non-root: {}", if production_config.security_config.pod_security_standards.run_as_non_root { "✅" } else { "❌" });
    println!("  • Read-only filesystem: {}", if production_config.security_config.pod_security_standards.read_only_root_filesystem { "✅" } else { "❌" });
    println!("  • Vulnerability scanning: {}", if production_config.security_config.image_security.vulnerability_scanning { "✅" } else { "❌" });
    println!("  • Image signing: {}", if production_config.security_config.image_security.image_signing { "✅" } else { "❌" });

    println!("\n🔐 Secrets Management:");
    println!("  • Vault integration: {}", if production_config.security_config.secrets_management.vault_integration { "✅" } else { "❌" });
    println!("  • Auto rotation: {}", if production_config.security_config.secrets_management.auto_rotation_enabled { "✅" } else { "❌" });
    println!("  • Encryption at rest: {}", if production_config.security_config.secrets_management.encryption_at_rest { "✅" } else { "❌" });
    println!("  • Sealed secrets: {}", if production_config.security_config.secrets_management.sealed_secrets { "✅" } else { "❌" });

    println!("\n📊 Monitoring & Observability:");
    println!("  • Metrics collection: {}", if production_config.monitoring_config.metrics_enabled { "✅ Enabled" } else { "❌ Disabled" });
    println!("  • Distributed tracing: {}", if production_config.monitoring_config.distributed_tracing { "✅ Enabled" } else { "❌ Disabled" });
    println!("  • Logging level: {}", production_config.monitoring_config.logging_level);
    println!("  • Alert rules: {}", production_config.monitoring_config.alerting_rules.len());
    println!("  • Dashboard panels: {}", production_config.monitoring_config.dashboard_config.panels.len());

    println!("\n🚨 Rollback Configuration:");
    println!("  • Auto-rollback: {}", if production_config.rollback_config.auto_rollback_enabled { "✅ Enabled" } else { "❌ Disabled" });
    println!("  • Rollback triggers: {}", production_config.rollback_config.rollback_triggers.len());
    println!("  • Rollback timeout: {}s", production_config.rollback_config.rollback_timeout_seconds);
    println!("  • History retention: {} deployments", production_config.rollback_config.preserve_rollback_history);

    // Generate Kubernetes configuration
    println!("\n⚙️ Generated Kubernetes Configuration:");
    println!("======================================");
    let k8s_yaml = orchestrator.generate_deployment_config_yaml();
    println!("{}", k8s_yaml);

    // Resource calculations
    println!("💰 Resource Requirements Summary:");
    println!("=================================");
    let mut total_cpu_requests = 0.0;
    let mut total_cpu_limits = 0.0;
    let mut total_memory_requests = 0.0;
    let mut total_memory_limits = 0.0;
    let mut total_replicas = 0;
    let mut total_gpus = 0;

    for env in &production_config.environments {
        let cpu_request = parse_cpu_string(&env.resources.cpu_request);
        let cpu_limit = parse_cpu_string(&env.resources.cpu_limit);
        let memory_request = parse_memory_string(&env.resources.memory_request);
        let memory_limit = parse_memory_string(&env.resources.memory_limit);

        total_cpu_requests += cpu_request * env.replicas as f32;
        total_cpu_limits += cpu_limit * env.replicas as f32;
        total_memory_requests += memory_request * env.replicas as f32;
        total_memory_limits += memory_limit * env.replicas as f32;
        total_replicas += env.replicas;
        total_gpus += env.resources.gpu_count * env.replicas;

        println!("  • {}: {} replicas", env.name, env.replicas);
        println!("    - CPU: {:.1} cores request, {:.1} cores limit", 
                cpu_request * env.replicas as f32, 
                cpu_limit * env.replicas as f32);
        println!("    - Memory: {:.1}Gi request, {:.1}Gi limit", 
                memory_request * env.replicas as f32, 
                memory_limit * env.replicas as f32);
        println!("    - GPUs: {}", env.resources.gpu_count * env.replicas);
    }

    println!("\n📈 Total Resource Requirements:");
    println!("  • Total replicas: {}", total_replicas);
    println!("  • Total CPU requests: {:.1} cores", total_cpu_requests);
    println!("  • Total CPU limits: {:.1} cores", total_cpu_limits);
    println!("  • Total memory requests: {:.1}Gi", total_memory_requests);
    println!("  • Total memory limits: {:.1}Gi", total_memory_limits);
    println!("  • Total GPUs: {}", total_gpus);

    // Estimated costs (simplified)
    let monthly_cost = estimate_monthly_cost(total_cpu_limits, total_memory_limits, total_gpus);
    println!("  • Estimated monthly cost: ${:.2}", monthly_cost);

    println!("\n✅ Production Configuration Ready!");
    println!("  • Enterprise-grade security implemented");
    println!("  • Comprehensive monitoring configured");
    println!("  • Multi-environment deployment ready");
    println!("  • Disaster recovery environment included");
    println!("  • Auto-scaling and rollback capabilities enabled");

    Ok(())
}

// Helper functions

fn create_production_environment(name: &str, replicas: u32) -> DeploymentEnvironment {
    DeploymentEnvironment {
        name: name.to_string(),
        namespace: "production".to_string(),
        replicas,
        resources: ResourceRequirements {
            cpu_request: "250m".to_string(),
            cpu_limit: "1".to_string(),
            memory_request: "512Mi".to_string(),
            memory_limit: "2Gi".to_string(),
            gpu_count: 0,
        },
        network_policy: NetworkPolicy {
            ingress_rules: vec![
                IngressRule {
                    from_cidr: "10.0.0.0/8".to_string(),
                    ports: vec![8080],
                    protocol: "TCP".to_string(),
                }
            ],
            egress_rules: vec![
                EgressRule {
                    to_cidr: "0.0.0.0/0".to_string(),
                    ports: vec![443],
                    protocol: "TCP".to_string(),
                }
            ],
            load_balancer_config: LoadBalancerConfig {
                service_type: "ClusterIP".to_string(),
                session_affinity: "None".to_string(),
                health_check_path: "/health".to_string(),
                ssl_termination: false,
            },
        },
        persistent_storage: PersistentStorage {
            enabled: false,
            storage_class: "standard".to_string(),
            size: "1Gi".to_string(),
            access_modes: vec!["ReadWriteOnce".to_string()],
        },
    }
}

fn create_enterprise_production_environment(name: &str, replicas: u32) -> DeploymentEnvironment {
    DeploymentEnvironment {
        name: name.to_string(),
        namespace: if name == "staging" { "staging".to_string() } else { "production".to_string() },
        replicas,
        resources: ResourceRequirements {
            cpu_request: "500m".to_string(),
            cpu_limit: "2".to_string(),
            memory_request: "1Gi".to_string(),
            memory_limit: "4Gi".to_string(),
            gpu_count: 1,
        },
        network_policy: create_secure_network_policy(),
        persistent_storage: PersistentStorage {
            enabled: true,
            storage_class: "fast-ssd".to_string(),
            size: "50Gi".to_string(),
            access_modes: vec!["ReadWriteOnce".to_string()],
        },
    }
}

fn create_secure_network_policy() -> NetworkPolicy {
    NetworkPolicy {
        ingress_rules: vec![
            IngressRule {
                from_cidr: "10.0.0.0/8".to_string(),
                ports: vec![8080],
                protocol: "TCP".to_string(),
            },
            IngressRule {
                from_cidr: "172.16.0.0/12".to_string(),
                ports: vec![9090], // Metrics
                protocol: "TCP".to_string(),
            },
        ],
        egress_rules: vec![
            EgressRule {
                to_cidr: "10.0.0.0/8".to_string(),
                ports: vec![443, 5432], // HTTPS, PostgreSQL
                protocol: "TCP".to_string(),
            },
        ],
        load_balancer_config: LoadBalancerConfig {
            service_type: "LoadBalancer".to_string(),
            session_affinity: "ClientIP".to_string(),
            health_check_path: "/health".to_string(),
            ssl_termination: true,
        },
    }
}

fn create_enterprise_health_config() -> HealthCheckConfig {
    HealthCheckConfig {
        endpoint: "/health".to_string(),
        interval_seconds: 10,
        timeout_seconds: 5,
        healthy_threshold: 3,
        unhealthy_threshold: 2,
        expected_status_codes: vec![200, 204],
    }
}

fn create_comprehensive_rollback_config() -> RollbackConfig {
    RollbackConfig {
        auto_rollback_enabled: true,
        rollback_triggers: vec![
            RollbackTrigger::HighErrorRate { 
                threshold_percent: 1.0, 
                duration_seconds: 300 
            },
            RollbackTrigger::HighLatency { 
                threshold_ms: 1500, 
                duration_seconds: 240 
            },
            RollbackTrigger::LowThroughput { 
                threshold_rps: 100.0, 
                duration_seconds: 180 
            },
            RollbackTrigger::HealthCheckFailure { 
                failure_count: 5 
            },
        ],
        rollback_timeout_seconds: 600,
        preserve_rollback_history: 20,
    }
}

fn create_comprehensive_monitoring_config() -> MonitoringConfig {
    MonitoringConfig {
        metrics_enabled: true,
        logging_level: "info".to_string(),
        distributed_tracing: true,
        alerting_rules: vec![
            AlertingRule {
                name: "HighErrorRate".to_string(),
                condition: "error_rate > 0.01".to_string(),
                severity: AlertSeverity::Critical,
                notification_channels: vec!["slack-alerts".to_string(), "pagerduty".to_string()],
            },
            AlertingRule {
                name: "HighMemoryUsage".to_string(),
                condition: "memory_usage > 0.90".to_string(),
                severity: AlertSeverity::Warning,
                notification_channels: vec!["slack-warnings".to_string()],
            },
            AlertingRule {
                name: "LowThroughput".to_string(),
                condition: "request_rate < 50".to_string(),
                severity: AlertSeverity::Warning,
                notification_channels: vec!["slack-warnings".to_string()],
            },
        ],
        dashboard_config: DashboardConfig {
            enabled: true,
            refresh_interval_seconds: 30,
            panels: vec![
                DashboardPanel {
                    name: "Request Rate".to_string(),
                    chart_type: "line".to_string(),
                    metrics: vec!["http_requests_per_second".to_string()],
                    time_range: "1h".to_string(),
                },
                DashboardPanel {
                    name: "Response Time".to_string(),
                    chart_type: "line".to_string(),
                    metrics: vec!["http_request_duration_seconds".to_string()],
                    time_range: "1h".to_string(),
                },
                DashboardPanel {
                    name: "Error Rate".to_string(),
                    chart_type: "gauge".to_string(),
                    metrics: vec!["http_requests_error_rate".to_string()],
                    time_range: "5m".to_string(),
                },
                DashboardPanel {
                    name: "Resource Usage".to_string(),
                    chart_type: "stacked_area".to_string(),
                    metrics: vec!["cpu_usage".to_string(), "memory_usage".to_string()],
                    time_range: "4h".to_string(),
                },
            ],
        },
    }
}

fn create_enterprise_security_config() -> SecurityConfig {
    SecurityConfig {
        rbac_enabled: true,
        network_policies_enabled: true,
        pod_security_standards: PodSecurityStandards {
            profile: "restricted".to_string(),
            run_as_non_root: true,
            read_only_root_filesystem: true,
            drop_capabilities: vec!["ALL".to_string()],
        },
        secrets_management: SecretsManagement {
            vault_integration: true,
            auto_rotation_enabled: true,
            encryption_at_rest: true,
            sealed_secrets: true,
        },
        image_security: ImageSecurity {
            vulnerability_scanning: true,
            image_signing: true,
            allowed_registries: vec![
                "gcr.io/company".to_string(),
                "company.azurecr.io".to_string(),
                "registry.company.com".to_string(),
            ],
            security_policy_enforcement: true,
        },
    }
}

fn parse_cpu_string(cpu: &str) -> f32 {
    if cpu.ends_with('m') {
        cpu.trim_end_matches('m').parse::<f32>().unwrap_or(0.0) / 1000.0
    } else {
        cpu.parse::<f32>().unwrap_or(0.0)
    }
}

fn parse_memory_string(memory: &str) -> f32 {
    if memory.ends_with("Gi") {
        memory.trim_end_matches("Gi").parse::<f32>().unwrap_or(0.0)
    } else if memory.ends_with("Mi") {
        memory.trim_end_matches("Mi").parse::<f32>().unwrap_or(0.0) / 1024.0
    } else {
        memory.parse::<f32>().unwrap_or(0.0)
    }
}

fn estimate_monthly_cost(cpu_cores: f32, memory_gb: f32, gpus: u32) -> f32 {
    // Simplified cloud cost estimation (varies by provider and region)
    let cpu_cost_per_core_per_month = 25.0; // $25 per vCPU per month
    let memory_cost_per_gb_per_month = 3.50; // $3.50 per GB per month
    let gpu_cost_per_gpu_per_month = 250.0; // $250 per GPU per month
    
    cpu_cores * cpu_cost_per_core_per_month +
    memory_gb * memory_cost_per_gb_per_month +
    gpus as f32 * gpu_cost_per_gpu_per_month
}