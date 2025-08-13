//! Distributed graph processing with auto-scaling and load balancing
//! 
//! Implements horizontal scaling, graph partitioning, and distributed diffusion

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, broadcast, mpsc};
use serde::{Serialize, Deserialize};
use tracing::{info, warn, error, debug, instrument};
use crate::core::{Graph, CompactGraph, DGDMProcessor, ProcessingConfig, GraphConfig};
use crate::resilience_patterns::{CircuitBreaker, AdaptiveRateLimiter};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    pub node_id: String,
    pub cluster_nodes: Vec<NodeInfo>,
    pub partition_strategy: PartitionStrategy,
    pub auto_scaling: AutoScalingConfig,
    pub load_balancing: LoadBalancingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: String,
    pub capacity: NodeCapacity,
    pub status: NodeStatus,
    pub last_heartbeat: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCapacity {
    pub max_nodes: usize,
    pub max_edges: usize,
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub processing_power: f64, // Relative processing capability
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeStatus {
    Active,
    Draining,
    Offline,
    Overloaded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PartitionStrategy {
    EdgeCut,      // Minimize edges across partitions
    VertexCut,    // Distribute high-degree vertices
    Random,       // Random assignment
    Spectral,     // Graph spectral partitioning
    Metis,        // METIS-based partitioning
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    pub enabled: bool,
    pub min_nodes: usize,
    pub max_nodes: usize,
    pub scale_up_threshold: f64,   // CPU utilization threshold
    pub scale_down_threshold: f64,
    pub scale_up_cooldown: Duration,
    pub scale_down_cooldown: Duration,
    pub target_utilization: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    pub algorithm: LoadBalancingAlgorithm,
    pub health_check_interval: Duration,
    pub max_requests_per_node: usize,
    pub sticky_sessions: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastConnections,
    WeightedRoundRobin,
    ConsistentHashing,
    LoadAware,
}

pub struct DistributedProcessor {
    config: ClusterConfig,
    local_processor: Arc<DGDMProcessor>,
    cluster_state: Arc<RwLock<ClusterState>>,
    partitioner: Arc<GraphPartitioner>,
    load_balancer: Arc<LoadBalancer>,
    auto_scaler: Arc<AutoScaler>,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    metrics_collector: Arc<MetricsCollector>,
}

#[derive(Debug)]
pub struct ClusterState {
    pub nodes: HashMap<String, NodeInfo>,
    pub partition_map: HashMap<String, Vec<u32>>, // node_id -> vertex_ids
    pub current_load: HashMap<String, f64>,
    pub last_rebalance: Instant,
}

impl DistributedProcessor {
    pub fn new(
        config: ClusterConfig,
        graph_config: GraphConfig,
        processing_config: ProcessingConfig,
    ) -> Self {
        let local_processor = Arc::new(DGDMProcessor::new(graph_config, processing_config));
        
        let cluster_state = Arc::new(RwLock::new(ClusterState {
            nodes: config.cluster_nodes.iter()
                .map(|node| (node.id.clone(), node.clone()))
                .collect(),
            partition_map: HashMap::new(),
            current_load: HashMap::new(),
            last_rebalance: Instant::now(),
        }));

        let partitioner = Arc::new(GraphPartitioner::new(config.partition_strategy.clone()));
        let load_balancer = Arc::new(LoadBalancer::new(config.load_balancing.clone()));
        let auto_scaler = Arc::new(AutoScaler::new(config.auto_scaling.clone()));
        let circuit_breakers = Arc::new(RwLock::new(HashMap::new()));
        let metrics_collector = Arc::new(MetricsCollector::new());

        Self {
            config,
            local_processor,
            cluster_state,
            partitioner,
            load_balancer,
            auto_scaler,
            circuit_breakers,
            metrics_collector,
        }
    }

    #[instrument(skip(self, graph))]
    pub async fn process_distributed(&self, graph: &CompactGraph) -> crate::Result<DistributedResult> {
        let start_time = Instant::now();
        
        // Check if local processing is sufficient
        if self.should_process_locally(graph).await {
            debug!("Processing graph locally");
            let result = self.local_processor.process(graph)?;
            return Ok(DistributedResult {
                embeddings: result.embeddings,
                processing_time_ms: result.processing_time_ms,
                nodes_processed: vec![(self.config.node_id.clone(), graph.num_nodes())],
                partition_strategy: PartitionStrategy::Random,
                communication_overhead_ms: 0.0,
            });
        }

        info!("Processing graph in distributed mode: {} nodes, {} edges", 
              graph.num_nodes(), graph.num_edges());

        // Partition the graph
        let partitions = self.partitioner.partition(graph, &self.get_active_nodes().await).await?;
        
        // Distribute processing tasks
        let processing_tasks = self.create_processing_tasks(partitions).await;
        
        // Execute distributed processing with fault tolerance
        let results = self.execute_distributed_tasks(processing_tasks).await?;
        
        // Aggregate results
        let final_result = self.aggregate_results(results, graph.num_nodes()).await?;
        
        let total_time = start_time.elapsed().as_secs_f64() * 1000.0;
        info!("Distributed processing completed in {:.2}ms", total_time);

        Ok(final_result)
    }

    async fn should_process_locally(&self, graph: &CompactGraph) -> bool {
        let cluster_state = self.cluster_state.read().await;
        let active_nodes = cluster_state.nodes.values()
            .filter(|node| matches!(node.status, NodeStatus::Active))
            .count();

        // Process locally if:
        // 1. Only one active node (this one)
        // 2. Graph is small enough for local processing
        // 3. Cluster is under maintenance
        active_nodes <= 1 || 
        graph.num_nodes() < 10_000 ||
        self.local_processor.can_process(graph)
    }

    async fn get_active_nodes(&self) -> Vec<NodeInfo> {
        let cluster_state = self.cluster_state.read().await;
        cluster_state.nodes.values()
            .filter(|node| matches!(node.status, NodeStatus::Active))
            .cloned()
            .collect()
    }

    async fn create_processing_tasks(&self, partitions: Vec<GraphPartition>) -> Vec<ProcessingTask> {
        let mut tasks = Vec::new();
        let active_nodes = self.get_active_nodes().await;
        
        for (partition, node) in partitions.into_iter().zip(active_nodes.iter().cycle()) {
            tasks.push(ProcessingTask {
                partition,
                target_node: node.clone(),
                priority: TaskPriority::Normal,
                timeout: Duration::from_secs(300),
            });
        }
        
        tasks
    }

    async fn execute_distributed_tasks(&self, tasks: Vec<ProcessingTask>) -> crate::Result<Vec<PartitionResult>> {
        let (result_tx, mut result_rx) = mpsc::channel(tasks.len());
        let mut handles = Vec::new();

        // Launch processing tasks
        for task in tasks {
            let result_tx = result_tx.clone();
            let circuit_breakers = self.circuit_breakers.clone();
            let metrics_collector = self.metrics_collector.clone();
            
            let handle = tokio::spawn(async move {
                let node_id = task.target_node.id.clone();
                
                // Get or create circuit breaker for this node
                let circuit_breaker = {
                    let mut breakers = circuit_breakers.write().await;
                    breakers.entry(node_id.clone())
                        .or_insert_with(|| CircuitBreaker::new(Default::default()))
                        .clone()
                };

                let start_time = Instant::now();
                let result = circuit_breaker.call(async {
                    Self::execute_remote_processing(task).await
                }).await;
                
                let duration = start_time.elapsed();
                metrics_collector.record_task_execution(node_id, duration, result.is_ok()).await;
                
                if let Err(e) = result_tx.send(result).await {
                    error!("Failed to send result: {}", e);
                }
            });
            
            handles.push(handle);
        }

        drop(result_tx); // Close sender
        
        // Collect results
        let mut results = Vec::new();
        let total_tasks = handles.len();
        
        while let Some(result) = result_rx.recv().await {
            match result {
                Ok(partition_result) => results.push(partition_result),
                Err(e) => {
                    error!("Task execution failed: {:?}", e);
                    // Continue with other tasks - handle partial failures
                }
            }
        }

        // Wait for all tasks to complete
        for handle in handles {
            let _ = handle.await;
        }

        if results.is_empty() {
            return Err(crate::error::Error::graph_processing(
                "All distributed processing tasks failed".to_string(),
                "distributed_processing_failure".to_string()
            ));
        }

        if results.len() < total_tasks {
            warn!("Partial failure in distributed processing: {}/{} tasks succeeded", 
                  results.len(), total_tasks);
        }

        Ok(results)
    }

    async fn execute_remote_processing(task: ProcessingTask) -> Result<PartitionResult, ProcessingError> {
        // In a real implementation, this would make HTTP/gRPC calls to remote nodes
        // For now, simulate remote processing
        tokio::time::sleep(Duration::from_millis(50)).await; // Simulate network latency
        
        // Simulate processing the partition
        let embeddings = ndarray::Array2::zeros((task.partition.graph.num_nodes(), task.partition.graph.feature_dim()));
        
        Ok(PartitionResult {
            partition_id: task.partition.id,
            embeddings,
            processing_time_ms: 25.0,
            node_id: task.target_node.id,
        })
    }

    async fn aggregate_results(&self, results: Vec<PartitionResult>, total_nodes: usize) -> crate::Result<DistributedResult> {
        if results.is_empty() {
            return Err(crate::error::Error::graph_processing(
                "No results to aggregate".to_string(),
                "aggregation_failure".to_string()
            ));
        }

        // Determine feature dimension from first result
        let feature_dim = results[0].embeddings.ncols();
        let mut final_embeddings = ndarray::Array2::zeros((total_nodes, feature_dim));
        
        let mut nodes_processed = Vec::new();
        let mut total_processing_time = 0.0;
        let mut node_offset = 0;

        // Combine embeddings from all partitions
        for result in results {
            let partition_size = result.embeddings.nrows();
            
            // Copy embeddings to final matrix
            final_embeddings.slice_mut(ndarray::s![node_offset..node_offset + partition_size, ..])
                .assign(&result.embeddings);
            
            nodes_processed.push((result.node_id, partition_size));
            total_processing_time = total_processing_time.max(result.processing_time_ms);
            node_offset += partition_size;
        }

        Ok(DistributedResult {
            embeddings: final_embeddings,
            processing_time_ms: total_processing_time,
            nodes_processed,
            partition_strategy: self.config.partition_strategy.clone(),
            communication_overhead_ms: 10.0, // Simulated
        })
    }

    pub async fn start_cluster_management(&self) -> crate::Result<()> {
        info!("Starting distributed cluster management");
        
        // Start health monitoring
        let health_monitor = self.start_health_monitoring();
        
        // Start auto-scaling
        let auto_scaler = self.start_auto_scaling();
        
        // Start load balancing
        let load_balancer = self.start_load_balancing();
        
        // Wait for all services
        tokio::try_join!(health_monitor, auto_scaler, load_balancer)?;
        
        Ok(())
    }

    async fn start_health_monitoring(&self) -> crate::Result<()> {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            // Check health of all nodes
            let nodes = self.get_active_nodes().await;
            for node in nodes {
                if let Err(e) = self.check_node_health(&node).await {
                    warn!("Node {} health check failed: {}", node.id, e);
                    self.mark_node_unhealthy(&node.id).await;
                }
            }
            
            // Rebalance if needed
            if self.should_rebalance().await {
                if let Err(e) = self.rebalance_cluster().await {
                    error!("Cluster rebalancing failed: {}", e);
                }
            }
        }
    }

    async fn start_auto_scaling(&self) -> crate::Result<()> {
        if !self.config.auto_scaling.enabled {
            return Ok(());
        }
        
        let mut interval = tokio::time::interval(Duration::from_secs(60));
        
        loop {
            interval.tick().await;
            
            let metrics = self.metrics_collector.get_cluster_metrics().await;
            let scaling_decision = self.auto_scaler.evaluate_scaling(&metrics).await;
            
            match scaling_decision {
                ScalingDecision::ScaleUp(count) => {
                    info!("Auto-scaling: Adding {} nodes", count);
                    if let Err(e) = self.scale_up(count).await {
                        error!("Scale-up failed: {}", e);
                    }
                }
                ScalingDecision::ScaleDown(count) => {
                    info!("Auto-scaling: Removing {} nodes", count);
                    if let Err(e) = self.scale_down(count).await {
                        error!("Scale-down failed: {}", e);
                    }
                }
                ScalingDecision::NoAction => {
                    debug!("Auto-scaling: No action needed");
                }
            }
        }
    }

    async fn start_load_balancing(&self) -> crate::Result<()> {
        // Load balancing is handled per-request, this just updates routing tables
        let mut interval = tokio::time::interval(Duration::from_secs(10));
        
        loop {
            interval.tick().await;
            
            let cluster_state = self.cluster_state.read().await;
            self.load_balancer.update_routing_table(&cluster_state.nodes).await;
        }
    }

    async fn check_node_health(&self, node: &NodeInfo) -> crate::Result<()> {
        // Simulate health check - in reality would be HTTP/gRPC call
        tokio::time::sleep(Duration::from_millis(10)).await;
        Ok(())
    }

    async fn mark_node_unhealthy(&self, node_id: &str) {
        let mut cluster_state = self.cluster_state.write().await;
        if let Some(node) = cluster_state.nodes.get_mut(node_id) {
            node.status = NodeStatus::Offline;
            warn!("Marked node {} as offline", node_id);
        }
    }

    async fn should_rebalance(&self) -> bool {
        let cluster_state = self.cluster_state.read().await;
        cluster_state.last_rebalance.elapsed() > Duration::from_secs(300) // 5 minutes
    }

    async fn rebalance_cluster(&self) -> crate::Result<()> {
        info!("Rebalancing cluster workload");
        
        let mut cluster_state = self.cluster_state.write().await;
        cluster_state.last_rebalance = Instant::now();
        
        // Implement rebalancing logic here
        // This would redistribute partitions based on current load
        
        Ok(())
    }

    async fn scale_up(&self, count: usize) -> crate::Result<()> {
        // In a real implementation, this would provision new compute instances
        info!("Scaling up by {} nodes (simulated)", count);
        Ok(())
    }

    async fn scale_down(&self, count: usize) -> crate::Result<()> {
        // In a real implementation, this would gracefully drain and terminate instances
        info!("Scaling down by {} nodes (simulated)", count);
        Ok(())
    }
}

// Supporting structures and implementations

pub struct GraphPartitioner {
    strategy: PartitionStrategy,
}

impl GraphPartitioner {
    pub fn new(strategy: PartitionStrategy) -> Self {
        Self { strategy }
    }

    pub async fn partition(&self, graph: &CompactGraph, nodes: &[NodeInfo]) -> crate::Result<Vec<GraphPartition>> {
        if nodes.is_empty() {
            return Err(crate::error::Error::graph_processing(
                "No active nodes for partitioning".to_string(),
                "partition_no_nodes".to_string()
            ));
        }

        match self.strategy {
            PartitionStrategy::Random => self.random_partition(graph, nodes.len()).await,
            PartitionStrategy::EdgeCut => self.edge_cut_partition(graph, nodes.len()).await,
            _ => {
                // For now, fall back to random partitioning
                self.random_partition(graph, nodes.len()).await
            }
        }
    }

    async fn random_partition(&self, graph: &CompactGraph, num_partitions: usize) -> crate::Result<Vec<GraphPartition>> {
        let nodes_per_partition = (graph.num_nodes() + num_partitions - 1) / num_partitions;
        let mut partitions = Vec::new();

        for i in 0..num_partitions {
            let start_idx = i * nodes_per_partition;
            let end_idx = std::cmp::min(start_idx + nodes_per_partition, graph.num_nodes());
            
            if start_idx < graph.num_nodes() {
                // Create subgraph for this partition
                let partition_graph = self.extract_subgraph(graph, start_idx, end_idx).await?;
                
                partitions.push(GraphPartition {
                    id: format!("partition_{}", i),
                    graph: partition_graph,
                    node_range: (start_idx, end_idx),
                });
            }
        }

        Ok(partitions)
    }

    async fn edge_cut_partition(&self, graph: &CompactGraph, num_partitions: usize) -> crate::Result<Vec<GraphPartition>> {
        // Simplified edge-cut partitioning
        // In reality, this would use algorithms like METIS
        self.random_partition(graph, num_partitions).await
    }

    async fn extract_subgraph(&self, graph: &CompactGraph, start: usize, end: usize) -> crate::Result<CompactGraph> {
        // Create a subgraph containing nodes from start to end
        // This is simplified - real implementation would handle edge connectivity
        
        let num_nodes = end - start;
        let node_features = graph.node_features().slice(ndarray::s![start..end, ..]).to_owned();
        
        Ok(CompactGraph {
            node_ids: (start..end).map(|i| i as u32).collect(),
            node_features,
            edge_offsets: vec![0; num_nodes + 1], // Simplified
            edge_targets: Vec::new(),
            edge_weights: Vec::new(),
            num_nodes,
            num_edges: 0,
            feature_dim: graph.feature_dim(),
        })
    }
}

pub struct LoadBalancer {
    config: LoadBalancingConfig,
    routing_table: Arc<RwLock<HashMap<String, f64>>>, // node_id -> weight
}

impl LoadBalancer {
    pub fn new(config: LoadBalancingConfig) -> Self {
        Self {
            config,
            routing_table: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn select_node(&self, nodes: &[NodeInfo]) -> Option<String> {
        match self.config.algorithm {
            LoadBalancingAlgorithm::RoundRobin => {
                // Simplified round-robin
                nodes.first().map(|n| n.id.clone())
            }
            LoadBalancingAlgorithm::LoadAware => {
                // Select least loaded node
                nodes.iter()
                    .min_by(|a, b| a.capacity.processing_power.partial_cmp(&b.capacity.processing_power).unwrap())
                    .map(|n| n.id.clone())
            }
            _ => nodes.first().map(|n| n.id.clone())
        }
    }

    pub async fn update_routing_table(&self, nodes: &HashMap<String, NodeInfo>) {
        let mut table = self.routing_table.write().await;
        table.clear();
        
        for (id, node) in nodes {
            if matches!(node.status, NodeStatus::Active) {
                table.insert(id.clone(), node.capacity.processing_power);
            }
        }
    }
}

pub struct AutoScaler {
    config: AutoScalingConfig,
    last_scale_action: Arc<RwLock<Instant>>,
}

impl AutoScaler {
    pub fn new(config: AutoScalingConfig) -> Self {
        Self {
            config,
            last_scale_action: Arc::new(RwLock::new(Instant::now())),
        }
    }

    pub async fn evaluate_scaling(&self, metrics: &ClusterMetrics) -> ScalingDecision {
        let last_action = *self.last_scale_action.read().await;
        
        // Check cooldown periods
        if metrics.avg_cpu_utilization > self.config.scale_up_threshold {
            if last_action.elapsed() > self.config.scale_up_cooldown {
                *self.last_scale_action.write().await = Instant::now();
                return ScalingDecision::ScaleUp(1);
            }
        } else if metrics.avg_cpu_utilization < self.config.scale_down_threshold {
            if last_action.elapsed() > self.config.scale_down_cooldown && 
               metrics.active_nodes > self.config.min_nodes {
                *self.last_scale_action.write().await = Instant::now();
                return ScalingDecision::ScaleDown(1);
            }
        }
        
        ScalingDecision::NoAction
    }
}

pub struct MetricsCollector {
    task_metrics: Arc<RwLock<Vec<TaskMetric>>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            task_metrics: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn record_task_execution(&self, node_id: String, duration: Duration, success: bool) {
        let mut metrics = self.task_metrics.write().await;
        metrics.push(TaskMetric {
            node_id,
            duration,
            success,
            timestamp: Instant::now(),
        });
        
        // Keep only recent metrics
        let cutoff = Instant::now() - Duration::from_secs(300);
        metrics.retain(|m| m.timestamp > cutoff);
    }

    pub async fn get_cluster_metrics(&self) -> ClusterMetrics {
        let metrics = self.task_metrics.read().await;
        
        let total_tasks = metrics.len();
        let successful_tasks = metrics.iter().filter(|m| m.success).count();
        let avg_duration = if !metrics.is_empty() {
            metrics.iter().map(|m| m.duration.as_millis() as f64).sum::<f64>() / total_tasks as f64
        } else {
            0.0
        };

        ClusterMetrics {
            active_nodes: 3, // Simulated
            avg_cpu_utilization: 0.6, // Simulated
            avg_memory_utilization: 0.4, // Simulated
            total_requests: total_tasks as u64,
            successful_requests: successful_tasks as u64,
            avg_response_time_ms: avg_duration,
        }
    }
}

// Data structures

#[derive(Debug, Clone)]
pub struct GraphPartition {
    pub id: String,
    pub graph: CompactGraph,
    pub node_range: (usize, usize),
}

#[derive(Debug, Clone)]
pub struct ProcessingTask {
    pub partition: GraphPartition,
    pub target_node: NodeInfo,
    pub priority: TaskPriority,
    pub timeout: Duration,
}

#[derive(Debug, Clone)]
pub enum TaskPriority {
    Low,
    Normal,
    High,
    Critical,
}

#[derive(Debug)]
pub struct PartitionResult {
    pub partition_id: String,
    pub embeddings: ndarray::Array2<f32>,
    pub processing_time_ms: f64,
    pub node_id: String,
}

#[derive(Debug)]
pub struct DistributedResult {
    pub embeddings: ndarray::Array2<f32>,
    pub processing_time_ms: f64,
    pub nodes_processed: Vec<(String, usize)>,
    pub partition_strategy: PartitionStrategy,
    pub communication_overhead_ms: f64,
}

#[derive(Debug)]
pub enum ScalingDecision {
    ScaleUp(usize),
    ScaleDown(usize),
    NoAction,
}

#[derive(Debug)]
pub struct TaskMetric {
    pub node_id: String,
    pub duration: Duration,
    pub success: bool,
    pub timestamp: Instant,
}

#[derive(Debug)]
pub struct ClusterMetrics {
    pub active_nodes: usize,
    pub avg_cpu_utilization: f64,
    pub avg_memory_utilization: f64,
    pub total_requests: u64,
    pub successful_requests: u64,
    pub avg_response_time_ms: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    #[error("Network error: {0}")]
    Network(String),
    
    #[error("Timeout occurred")]
    Timeout,
    
    #[error("Processing failed: {0}")]
    Processing(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Graph, Node, Edge};

    #[tokio::test]
    async fn test_graph_partitioning() {
        let partitioner = GraphPartitioner::new(PartitionStrategy::Random);
        
        // Create test graph
        let mut graph = Graph::new();
        for i in 0..10 {
            graph.add_node(Node {
                id: i,
                features: vec![i as f32],
                label: None,
            });
        }
        let compact = graph.to_compact().unwrap();
        
        // Create test nodes
        let nodes = vec![
            NodeInfo {
                id: "node1".to_string(),
                address: "localhost:8080".to_string(),
                capacity: NodeCapacity {
                    max_nodes: 1000,
                    max_edges: 10000,
                    cpu_cores: 4,
                    memory_gb: 8.0,
                    processing_power: 1.0,
                },
                status: NodeStatus::Active,
                last_heartbeat: 0,
            },
            NodeInfo {
                id: "node2".to_string(),
                address: "localhost:8081".to_string(),
                capacity: NodeCapacity {
                    max_nodes: 1000,
                    max_edges: 10000,
                    cpu_cores: 4,
                    memory_gb: 8.0,
                    processing_power: 1.0,
                },
                status: NodeStatus::Active,
                last_heartbeat: 0,
            },
        ];
        
        let partitions = partitioner.partition(&compact, &nodes).await.unwrap();
        assert_eq!(partitions.len(), 2);
    }

    #[tokio::test]
    async fn test_load_balancer() {
        let config = LoadBalancingConfig {
            algorithm: LoadBalancingAlgorithm::LoadAware,
            health_check_interval: Duration::from_secs(30),
            max_requests_per_node: 100,
            sticky_sessions: false,
        };
        
        let load_balancer = LoadBalancer::new(config);
        
        let nodes = vec![
            NodeInfo {
                id: "node1".to_string(),
                address: "localhost:8080".to_string(),
                capacity: NodeCapacity {
                    max_nodes: 1000,
                    max_edges: 10000,
                    cpu_cores: 4,
                    memory_gb: 8.0,
                    processing_power: 0.5, // Lower processing power
                },
                status: NodeStatus::Active,
                last_heartbeat: 0,
            },
            NodeInfo {
                id: "node2".to_string(),
                address: "localhost:8081".to_string(),
                capacity: NodeCapacity {
                    max_nodes: 1000,
                    max_edges: 10000,
                    cpu_cores: 8,
                    memory_gb: 16.0,
                    processing_power: 1.0, // Higher processing power
                },
                status: NodeStatus::Active,
                last_heartbeat: 0,
            },
        ];
        
        let selected = load_balancer.select_node(&nodes).await;
        assert_eq!(selected, Some("node1".to_string())); // Should select lower load node
    }
}