# API Reference Guide

## Core Graph Processing API

### Graph Structures

#### `Graph`
Primary graph data structure for building and manipulating graphs.

```rust
impl Graph {
    pub fn new() -> Self
    pub fn add_node(&mut self, node: Node)
    pub fn add_edge(&mut self, edge: Edge) 
    pub fn to_compact(&self) -> Result<CompactGraph, GraphError>
}
```

#### `Node`
Represents a graph vertex with features and metadata.

```rust
pub struct Node {
    pub id: u32,
    pub features: Vec<f32>,
    pub label: Option<String>,
}
```

#### `Edge`
Represents a graph edge with weight and type information.

```rust
pub struct Edge {
    pub source: u32,
    pub target: u32,
    pub weight: f32,
    pub edge_type: Option<String>,
}
```

### DGDM Processor

#### `DGDMProcessor::new(graph_config: GraphConfig, processing_config: ProcessingConfig)`
Creates a new DGDM processor instance.

**Parameters:**
- `graph_config`: Configuration for graph handling
- `processing_config`: Processing parameters and optimization settings

**Returns:** `DGDMProcessor` instance

#### `DGDMProcessor::process(&self, graph: &CompactGraph) -> Result<ProcessingResult, ProcessingError>`
Processes a graph using dynamic graph diffusion models.

**Parameters:**
- `graph`: Input graph in compact format

**Returns:** 
- `Ok(ProcessingResult)`: Successful processing result with diffusion scores
- `Err(ProcessingError)`: Processing failure with error details

---

## Quantum Enhancement API

### Quantum Graph Transformer

#### `QuantumGraphTransformer::new(config: QuantumConfig) -> Self`
Initializes quantum enhancement system.

**Configuration:**
```rust
pub struct QuantumConfig {
    pub qubits_per_node: u32,        // Default: 4
    pub entanglement_strength: f32,   // Range: 0.0-1.0
    pub coherence_time: u32,         // Microseconds
    pub measurement_threshold: f32,   // Range: 0.0-1.0
    pub error_correction: bool,      // Enable quantum error correction
    pub quantum_dimensions: u32,     // Hilbert space dimensions
}
```

#### `quantum_enhance_diffusion(&mut self, graph: &CompactGraph, classical_result: ProcessingResult) -> Result<QuantumResult, QuantumError>`
Applies quantum enhancement to classical diffusion results.

**Parameters:**
- `graph`: Input graph structure
- `classical_result`: Output from classical DGDM processing

**Returns:**
```rust
pub struct QuantumResult {
    pub classical_result: ProcessingResult,
    pub quantum_coherence_score: f32,    // 0.0-1.0
    pub entanglement_entropy: f32,       // Quantum correlation measure
    pub superposition_count: u32,        // Active quantum states
    pub measurement_outcomes: Vec<f32>,   // Collapsed state values
}
```

---

## Adaptive Neural Architecture Search API

### AdaptiveNeuralArchitectureSearch

#### `AdaptiveNeuralArchitectureSearch::new(config: NASConfig) -> Self`
Creates evolutionary architecture search system.

**Configuration:**
```rust
pub struct NASConfig {
    pub population_size: u32,              // Evolution population size
    pub max_generations: u32,              // Maximum evolution cycles
    pub mutation_rate: f32,                // Range: 0.0-1.0
    pub crossover_rate: f32,               // Range: 0.0-1.0
    pub elite_ratio: f32,                  // Percentage of top performers
    pub max_layers: u32,                   // Architecture depth limit
    pub min_layers: u32,                   // Minimum architecture depth
    pub performance_weight_accuracy: f32,   // Multi-objective weight
    pub performance_weight_speed: f32,     // Multi-objective weight
    pub performance_weight_memory: f32,    // Multi-objective weight
    pub adaptive_mutation: bool,           // Dynamic mutation rate
}
```

#### `initialize_population(&mut self) -> Result<(), NASError>`
Initializes random architecture population for evolution.

#### `evolve_architecture(&mut self, training_graphs: &[CompactGraph], validation_graphs: &[CompactGraph]) -> Result<ArchitectureGene, NASError>`
Evolves neural architectures using evolutionary algorithms.

**Parameters:**
- `training_graphs`: Graphs for architecture training
- `validation_graphs`: Graphs for architecture validation

**Returns:**
```rust
pub struct ArchitectureGene {
    pub layers: Vec<LayerConfig>,
    pub skip_connections: Vec<(u32, u32)>,
    pub attention_heads: Vec<u32>,
    pub quantum_config: Option<QuantumConfig>,
}
```

#### `get_best_performance(&self) -> Option<ArchitecturePerformance>`
Retrieves performance metrics of the best evolved architecture.

**Returns:**
```rust
pub struct ArchitecturePerformance {
    pub fitness_score: f32,        // Overall fitness (0.0-1.0)
    pub accuracy: f32,             // Model accuracy (0.0-1.0)
    pub latency_ms: f32,           // Inference latency
    pub memory_usage: u64,         // Memory consumption in bytes
    pub quantum_coherence: f32,    // Quantum enhancement factor
}
```

---

## Robust Error Recovery API

### RobustErrorRecoverySystem

#### `RobustErrorRecoverySystem::new(config: RecoveryConfig) -> Self`
Initializes comprehensive error recovery system.

**Configuration:**
```rust
pub struct RecoveryConfig {
    pub max_retries: u32,                  // Maximum retry attempts
    pub base_delay_ms: u64,               // Base backoff delay
    pub max_delay_ms: u64,                // Maximum backoff delay
    pub circuit_breaker_threshold: u32,    // Failure threshold
    pub circuit_breaker_timeout_ms: u64,  // Circuit breaker timeout
    pub health_check_interval_ms: u64,    // Health check frequency
}
```

#### `execute_with_recovery<T, F, Fut>(&self, operation_name: &str, operation: F) -> Result<T, RecoveryError>`
Executes operations with comprehensive error recovery.

**Parameters:**
- `operation_name`: Identifier for monitoring
- `operation`: Async operation to execute with recovery

**Returns:**
- `Ok(T)`: Successful operation result
- `Err(RecoveryError)`: Unrecoverable error after all retry attempts

### Circuit Breaker States
- **Closed**: Normal operation, requests pass through
- **Open**: Failure threshold exceeded, requests fail fast
- **Half-Open**: Test request allowed to check service recovery

---

## Health Monitoring API

### IntelligentHealthMonitoring

#### `IntelligentHealthMonitoring::new(config: MonitoringConfig) -> Self`
Creates advanced health monitoring system with anomaly detection.

#### `start_monitoring(&mut self) -> Result<(), MonitoringError>`
Begins continuous health monitoring and anomaly detection.

#### `get_system_health(&self) -> SystemHealthReport`
Retrieves comprehensive system health status.

**Returns:**
```rust
pub struct SystemHealthReport {
    pub overall_health: HealthStatus,
    pub component_health: HashMap<String, HealthStatus>,
    pub anomalies_detected: Vec<AnomalyReport>,
    pub performance_metrics: PerformanceMetrics,
    pub predictions: Vec<HealthPrediction>,
}

pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}
```

---

## Distributed Processing API

### HighPerformanceDistributedEngine

#### `HighPerformanceDistributedEngine::new(config: DistributedConfig) -> Self`
Initializes distributed processing engine with worker pools.

**Configuration:**
```rust
pub struct DistributedConfig {
    pub max_workers: u32,
    pub task_queue_size: u32,
    pub worker_timeout_ms: u64,
    pub result_cache_size: u32,
    pub load_balancing_strategy: LoadBalancingStrategy,
}

pub enum LoadBalancingStrategy {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    ResourceBased,
}
```

#### `distribute_processing(&self, graphs: Vec<CompactGraph>) -> Result<Vec<ProcessingResult>, DistributedError>`
Distributes graph processing across worker pool.

#### `get_worker_health(&self) -> Vec<WorkerHealthStatus>`
Retrieves health status of all workers in the pool.

---

## Auto-Scaling API

### IntelligentAutoScalingSystem

#### `IntelligentAutoScalingSystem::new(config: AutoScalingConfig) -> Self`
Creates ML-powered auto-scaling system.

**Configuration:**
```rust
pub struct AutoScalingConfig {
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub target_cpu_utilization: f32,
    pub target_memory_utilization: f32,
    pub scale_up_cooldown_seconds: u64,
    pub scale_down_cooldown_seconds: u64,
    pub prediction_window_minutes: u32,
}
```

#### `start_auto_scaling(&mut self) -> Result<(), AutoScalingError>`
Begins automatic scaling based on metrics and ML predictions.

#### `get_scaling_recommendations(&self) -> ScalingRecommendations`
Provides scaling recommendations based on current analysis.

---

## Quality Gates API

### ComprehensiveQualityGates

#### `ComprehensiveQualityGates::new(config: QualityConfig) -> Self`
Initializes comprehensive quality assurance system.

#### `run_all_quality_gates(&self, project_path: &str) -> Result<QualityGateReport, QualityError>`
Executes all quality gates and returns comprehensive report.

**Returns:**
```rust
pub struct QualityGateReport {
    pub overall_status: QualityStatus,
    pub test_results: TestResults,
    pub security_scan: SecurityScanResults,
    pub performance_validation: PerformanceResults,
    pub code_quality: CodeQualityResults,
    pub recommendations: Vec<QualityRecommendation>,
}
```

---

## Production Deployment API

### ProductionDeploymentOrchestrator

#### `ProductionDeploymentOrchestrator::new(config: DeploymentConfig) -> Self`
Creates production deployment orchestration system.

**Configuration:**
```rust
pub struct DeploymentConfig {
    pub strategy: DeploymentStrategy,
    pub cluster_config: ClusterConfig,
    pub security_config: SecurityConfig,
    pub monitoring_config: MonitoringConfig,
}

pub enum DeploymentStrategy {
    BlueGreen,
    CanaryRelease { traffic_percentage: f32 },
    RollingUpdate { batch_size: u32 },
}
```

#### `deploy(&self, application_spec: ApplicationSpec) -> Result<DeploymentResult, DeploymentError>`
Executes production deployment with specified strategy.

#### `rollback(&self, deployment_id: String) -> Result<RollbackResult, DeploymentError>`
Performs automatic rollback to previous stable version.

#### `get_deployment_status(&self, deployment_id: String) -> DeploymentStatus`
Retrieves current status of ongoing or completed deployment.

---

## Error Types

### Common Error Patterns
All APIs use consistent error handling patterns with detailed error information:

```rust
pub enum GraphError {
    InvalidNode(String),
    InvalidEdge(String),
    ConversionError(String),
}

pub enum ProcessingError {
    ComputationFailed(String),
    ResourceExhausted(String),
    ConfigurationError(String),
}

pub enum QuantumError {
    DecoherenceError(String),
    MeasurementFailed(String),
    StatePreparationFailed(String),
}
```

Each error type implements `std::error::Error` and provides detailed context for debugging and monitoring.

---

## Performance Guidelines

### Optimal Graph Sizes
- **Small graphs** (≤100 nodes): Use single-threaded processing
- **Medium graphs** (100-1000 nodes): Enable quantum enhancement
- **Large graphs** (≥1000 nodes): Use distributed processing

### Memory Management
- Use `CompactGraph` format for processing to minimize memory usage
- Enable result caching for repeated computations
- Configure appropriate worker pool sizes based on available memory

### Quantum Enhancement
- Most effective for graphs with high connectivity (density > 0.3)
- Requires minimum coherence time of 10 microseconds
- Best results with 4-8 qubits per node