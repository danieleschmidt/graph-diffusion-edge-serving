//! DGDM Server Binary - Ultra-Fast Graph Diffusion on Edge TPU

use graph_diffusion_edge::{
    serving::start_server_with_config,
    core::{GraphConfig, ProcessingConfig},
    Result,
};

#[cfg(feature = "tpu")]
use graph_diffusion_edge::tpu::{EdgeTPU, TpuConfig};

use clap::{Parser, Subcommand};
use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tracing::{info, error, warn, Level};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[derive(Parser)]
#[command(name = "dgdm-server")]
#[command(about = "Ultra-Fast Dynamic Graph Diffusion Models on Edge TPU v5")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(author = "Terragon Labs <dev@terragonlabs.com>")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Server bind address
    #[arg(short, long)]
    bind: Option<String>,

    /// Log level
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Enable TPU acceleration
    #[arg(long)]
    #[cfg(feature = "tpu")]
    enable_tpu: bool,

    /// TPU device path
    #[arg(long, default_value = "/dev/apex_0")]
    #[cfg(feature = "tpu")]
    tpu_device: String,

    /// TPU model path
    #[arg(long, default_value = "models/dgdm_quantized.tflite")]
    #[cfg(feature = "tpu")]
    tpu_model: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the DGDM server
    Serve {
        /// Port to bind to
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
    /// Validate configuration
    ValidateConfig,
    /// Show server information
    Info,
    /// Benchmark server performance
    Benchmark {
        /// Number of requests to send
        #[arg(short, long, default_value = "1000")]
        requests: usize,
        /// Concurrent connections
        #[arg(short, long, default_value = "10")]
        concurrency: usize,
    },
    #[cfg(feature = "tpu")]
    /// TPU-related commands
    Tpu {
        #[command(subcommand)]
        tpu_command: TpuCommands,
    },
}

#[cfg(feature = "tpu")]
#[derive(Subcommand)]
enum TpuCommands {
    /// List available TPU devices
    List,
    /// Show TPU device information
    Info {
        /// Device path
        #[arg(short, long, default_value = "/dev/apex_0")]
        device: String,
    },
    /// Test TPU functionality
    Test {
        /// Device path
        #[arg(short, long, default_value = "/dev/apex_0")]
        device: String,
        /// Model path
        #[arg(short, long, default_value = "models/dgdm_quantized.tflite")]
        model: String,
    },
}

#[derive(Debug, Deserialize, Serialize)]
struct ServerConfig {
    server: ServerSettings,
    graph: GraphConfig,
    processing: ProcessingConfig,
    #[cfg(feature = "tpu")]
    tpu: Option<TpuConfig>,
    logging: LoggingConfig,
}

#[derive(Debug, Deserialize, Serialize)]
struct ServerSettings {
    bind_address: String,
    enable_cors: bool,
    enable_compression: bool,
    max_request_size_mb: usize,
    request_timeout_seconds: u64,
}

#[derive(Debug, Deserialize, Serialize)]
struct LoggingConfig {
    level: String,
    format: String,
    enable_json: bool,
    enable_tracing: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            server: ServerSettings {
                bind_address: "0.0.0.0:8080".to_string(),
                enable_cors: true,
                enable_compression: true,
                max_request_size_mb: 100,
                request_timeout_seconds: 30,
            },
            graph: GraphConfig::default(),
            processing: ProcessingConfig::default(),
            #[cfg(feature = "tpu")]
            tpu: None,
            logging: LoggingConfig {
                level: "info".to_string(),
                format: "compact".to_string(),
                enable_json: false,
                enable_tracing: false,
            },
        }
    }
}

impl ServerConfig {
    fn load(config_path: &str) -> Result<Self> {
        let settings = Config::builder()
            .add_source(File::with_name(config_path).required(false))
            .add_source(Environment::with_prefix("DGDM").separator("__"))
            .build()
            .map_err(|e| graph_diffusion_edge::error::Error::configuration(e.to_string(), "config_build"))?;

        settings
            .try_deserialize()
            .map_err(|e| graph_diffusion_edge::error::Error::configuration(e.to_string(), "config_deserialize"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Load configuration
    let config = match ServerConfig::load(&cli.config) {
        Ok(config) => {
            info!("Configuration loaded from {}", cli.config);
            config
        }
        Err(e) => {
            warn!("Failed to load config from {}: {}. Using defaults.", cli.config, e);
            ServerConfig::default()
        }
    };

    // Initialize logging
    init_logging(&config.logging, &cli.log_level)?;

    info!("🚀 DGDM Server v{}", env!("CARGO_PKG_VERSION"));
    info!("💻 Platform: {}", std::env::consts::OS);
    info!("🏗️  Build: {} {}", 
          std::env::consts::ARCH, 
          if cfg!(debug_assertions) { "debug" } else { "release" }
    );

    match cli.command {
        Some(Commands::Serve { port }) => {
            let bind_address = cli.bind.unwrap_or_else(|| {
                format!("0.0.0.0:{}", port)
            });
            
            info!("🌐 Starting server on {}", bind_address);
            serve_command(bind_address, config).await
        }
        Some(Commands::ValidateConfig) => {
            validate_config_command(config)
        }
        Some(Commands::Info) => {
            info_command()
        }
        Some(Commands::Benchmark { requests, concurrency }) => {
            benchmark_command(requests, concurrency, config).await
        }
        #[cfg(feature = "tpu")]
        Some(Commands::Tpu { tpu_command }) => {
            tpu_command_handler(tpu_command).await
        }
        None => {
            // Default: start server
            let bind_address = cli.bind.unwrap_or_else(|| config.server.bind_address.clone());
            info!("🌐 Starting server on {}", bind_address);
            serve_command(bind_address, config).await
        }
    }
}

fn init_logging(logging_config: &LoggingConfig, cli_level: &str) -> Result<()> {
    let level = cli_level.parse::<Level>()
        .or_else(|_| logging_config.level.parse::<Level>())
        .unwrap_or(Level::INFO);

    let env_filter = EnvFilter::from_default_env()
        .add_directive(format!("graph_diffusion_edge={}", level).parse().unwrap())
        .add_directive(format!("dgdm_server={}", level).parse().unwrap());

    if logging_config.enable_json {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().compact())
            .init();
    }

    info!("📊 Logging initialized at {} level", level);
    Ok(())
}

async fn serve_command(bind_address: String, config: ServerConfig) -> Result<()> {
    info!("⚙️  Configuration:");
    info!("   • Max nodes: {}", config.graph.max_nodes);
    info!("   • Max edges: {}", config.graph.max_edges);
    info!("   • Diffusion steps: {}", config.graph.diffusion_steps);
    info!("   • Batch size: {}", config.graph.batch_size);
    info!("   • Quantization: {}", config.graph.quantization);

    #[cfg(feature = "tpu")]
    if let Some(ref tpu_config) = config.tpu {
        info!("🔥 TPU Configuration:");
        info!("   • Device: {}", tpu_config.device_path);
        info!("   • Model: {}", tpu_config.model_path);
        info!("   • Power limit: {}W", tpu_config.power_limit_watts);
        info!("   • Quantization: {}", tpu_config.quantization);
    }

    let addr: SocketAddr = bind_address
        .parse()
        .map_err(|e| graph_diffusion_edge::error::Error::configuration(
            format!("Invalid bind address: {}", e),
            "bind_address"
        ))?;

    start_server_with_config(
        &bind_address,
        config.graph,
        config.processing,
    ).await
}

fn validate_config_command(config: ServerConfig) -> Result<()> {
    info!("✅ Configuration validation");
    
    // Validate server settings
    if config.server.max_request_size_mb > 1000 {
        warn!("⚠️  Max request size is very large: {}MB", config.server.max_request_size_mb);
    }

    // Validate graph settings
    if config.graph.max_nodes > 10_000_000 {
        warn!("⚠️  Max nodes is very large: {}", config.graph.max_nodes);
    }

    if config.graph.diffusion_steps > 100 {
        warn!("⚠️  Many diffusion steps may impact performance: {}", config.graph.diffusion_steps);
    }

    // Validate processing settings
    if config.processing.temperature <= 0.0 {
        error!("❌ Temperature must be positive: {}", config.processing.temperature);
        return Err(graph_diffusion_edge::error::Error::configuration(
            "Invalid temperature value",
            "processing.temperature"
        ));
    }

    #[cfg(feature = "tpu")]
    if let Some(ref tpu_config) = config.tpu {
        if tpu_config.power_limit_watts > 6.0 {
            error!("❌ TPU power limit exceeds maximum: {}W", tpu_config.power_limit_watts);
            return Err(graph_diffusion_edge::error::Error::configuration(
                "Invalid TPU power limit",
                "tpu.power_limit_watts"
            ));
        }

        if !std::path::Path::new(&tpu_config.device_path).exists() {
            warn!("⚠️  TPU device not found: {}", tpu_config.device_path);
        }

        if !std::path::Path::new(&tpu_config.model_path).exists() {
            warn!("⚠️  TPU model not found: {}", tpu_config.model_path);
        }
    }

    info!("✅ Configuration is valid");
    Ok(())
}

fn info_command() -> Result<()> {
    println!("🔍 DGDM Server Information");
    println!();
    println!("📦 Package: {}", env!("CARGO_PKG_NAME"));
    println!("🏷️  Version: {}", env!("CARGO_PKG_VERSION"));
    println!("📝 Description: {}", env!("CARGO_PKG_DESCRIPTION"));
    println!("👥 Authors: {}", env!("CARGO_PKG_AUTHORS"));
    println!("🔗 Repository: {}", env!("CARGO_PKG_REPOSITORY"));
    println!();
    println!("🏗️  Build Information:");
    println!("   • Target: {}", std::env::consts::ARCH);
    println!("   • OS: {}", std::env::consts::OS);
    println!("   • Profile: {}", if cfg!(debug_assertions) { "debug" } else { "release" });
    println!();
    println!("🎯 Features:");
    println!("   • TPU Support: {}", cfg!(feature = "tpu"));
    println!("   • WASM Support: {}", cfg!(target_arch = "wasm32"));
    println!("   • Benchmarking: {}", cfg!(feature = "benchmark"));
    println!();
    println!("📊 Runtime Information:");
    println!("   • Available CPU cores: {}", num_cpus::get());
    
    if let Ok(memory) = sys_info::mem_info() {
        println!("   • Total memory: {:.1} GB", memory.total as f64 / 1024.0 / 1024.0);
        println!("   • Available memory: {:.1} GB", memory.free as f64 / 1024.0 / 1024.0);
    }

    #[cfg(feature = "tpu")]
    {
        println!();
        println!("🔥 TPU Information:");
        match graph_diffusion_edge::tpu::list_available_devices() {
            Ok(devices) => {
                if devices.is_empty() {
                    println!("   • No TPU devices found");
                } else {
                    for device in devices {
                        println!("   • Device: {} ({})", device.device_path, 
                               if device.is_available { "available" } else { "unavailable" });
                    }
                }
            }
            Err(e) => {
                println!("   • Error listing TPU devices: {}", e);
            }
        }
    }

    Ok(())
}

async fn benchmark_command(requests: usize, concurrency: usize, _config: ServerConfig) -> Result<()> {
    info!("🏃 Starting benchmark: {} requests with {} concurrent connections", requests, concurrency);
    
    // This is a placeholder for actual benchmarking implementation
    warn!("⚠️  Benchmark command not yet implemented");
    warn!("   This would:");
    warn!("   • Start a test server");
    warn!("   • Generate {} synthetic graph requests", requests);
    warn!("   • Send them with {} concurrent connections", concurrency);
    warn!("   • Report latency, throughput, and error rates");
    
    Ok(())
}

#[cfg(feature = "tpu")]
async fn tpu_command_handler(tpu_command: TpuCommands) -> Result<()> {
    match tpu_command {
        TpuCommands::List => {
            info!("🔍 Listing available TPU devices...");
            
            match graph_diffusion_edge::tpu::list_available_devices() {
                Ok(devices) => {
                    if devices.is_empty() {
                        println!("❌ No TPU devices found");
                        println!("💡 Make sure Edge TPU drivers are installed and devices are connected");
                    } else {
                        println!("✅ Found {} TPU device(s):", devices.len());
                        for device in devices {
                            println!("   📱 {}", device.device_path);
                            println!("      • ID: {}", device.device_id);
                            println!("      • Available: {}", device.is_available);
                            println!("      • Firmware: {}", device.firmware_version);
                            println!("      • Max Power: {}W", device.max_power_watts);
                            println!();
                        }
                    }
                }
                Err(e) => {
                    error!("❌ Failed to list TPU devices: {}", e);
                    return Err(e);
                }
            }
        }
        
        TpuCommands::Info { device } => {
            info!("🔍 Getting TPU device information: {}", device);
            
            match EdgeTPU::new().device_path(&device).build() {
                Ok(tpu) => {
                    match tpu.get_stats() {
                        Ok(stats) => {
                            println!("✅ TPU Device: {}", device);
                            println!("   🌡️  Temperature: {:.1}°C", stats.temperature_celsius);
                            println!("   ⚡ Power: {:.1}W", stats.power_consumption_watts);
                            println!("   📊 Utilization: {:.1}%", stats.utilization_percent);
                            println!("   💾 Memory: {:.1}MB", stats.memory_usage_mb);
                            println!("   🚀 Inferences: {}", stats.inference_count);
                            println!("   ⏱️  Avg Latency: {:.2}ms", stats.avg_latency_ms);
                        }
                        Err(e) => {
                            error!("❌ Failed to get TPU stats: {}", e);
                        }
                    }
                }
                Err(e) => {
                    error!("❌ Failed to initialize TPU: {}", e);
                    return Err(e);
                }
            }
        }
        
        TpuCommands::Test { device, model } => {
            info!("🧪 Testing TPU functionality: {} with model {}", device, model);
            
            match EdgeTPU::new()
                .device_path(&device)
                .model_path(&model)
                .build() 
            {
                Ok(tpu) => {
                    info!("✅ TPU initialized successfully");
                    
                    // Test inference with dummy data
                    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                    
                    match tpu.inference(&test_data) {
                        Ok(result) => {
                            info!("✅ Test inference successful");
                            info!("   📥 Input: {:?}", test_data);
                            info!("   📤 Output: {:?}", result);
                            println!("🎉 TPU test passed!");
                        }
                        Err(e) => {
                            error!("❌ Test inference failed: {}", e);
                            return Err(e);
                        }
                    }
                }
                Err(e) => {
                    error!("❌ TPU initialization failed: {}", e);
                    return Err(e);
                }
            }
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.server.bind_address, "0.0.0.0:8080");
        assert!(config.server.enable_cors);
        assert_eq!(config.graph.max_nodes, 100_000);
    }

    #[test]
    fn test_cli_parsing() {
        let cli = Cli::try_parse_from(&["dgdm-server", "--log-level", "debug"]).unwrap();
        assert_eq!(cli.log_level, "debug");
        assert_eq!(cli.config, "config.toml");
    }
}