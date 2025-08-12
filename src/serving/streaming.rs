//! Real-time graph streaming and WebSocket handling

use crate::core::{Graph, Node, Edge, DGDMProcessor};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, RwLock, Mutex};
use tokio_stream::{Stream, StreamExt};
use tracing::{info, warn, error, instrument, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GraphUpdate {
    AddNode { 
        id: u32, 
        features: Vec<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    UpdateNode { 
        id: u32, 
        features: Vec<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },
    RemoveNode { 
        id: u32 
    },
    AddEdge { 
        source: u32, 
        target: u32, 
        weight: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        edge_type: Option<String>,
    },
    UpdateEdge { 
        source: u32, 
        target: u32, 
        weight: f32,
        #[serde(skip_serializing_if = "Option::is_none")]
        edge_type: Option<String>,
    },
    RemoveEdge { 
        source: u32, 
        target: u32 
    },
    Clear,
    BatchUpdate { 
        updates: Vec<GraphUpdate>
    },
    RequestDiffusion {
        steps: usize,
        temperature: f32,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamingResult {
    pub embeddings: Vec<Vec<f32>>,
    pub update_id: u64,
    pub processing_time_ms: f64,
    pub graph_stats: StreamingGraphStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamingGraphStats {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub update_count: u64,
    pub last_update_timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub auto_diffuse: bool,
    pub batch_size: usize,
    pub max_buffer_size: usize,
    pub diffusion_interval_ms: u64,
    pub enable_incremental: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            auto_diffuse: true,
            batch_size: 32,
            max_buffer_size: 10000,
            diffusion_interval_ms: 100,
            enable_incremental: true,
        }
    }
}

pub struct StreamingDGDM {
    processor: Arc<DGDMProcessor>,
    graph: Arc<RwLock<Graph>>,
    update_sender: broadcast::Sender<GraphUpdate>,
    result_sender: broadcast::Sender<StreamingResult>,
    config: StreamingConfig,
    update_counter: Arc<RwLock<u64>>,
    pending_updates: Arc<RwLock<Vec<GraphUpdate>>>,
    last_diffusion: Arc<Mutex<Instant>>,
    processing_queue: Arc<Mutex<Vec<GraphUpdate>>>,
}

impl StreamingDGDM {
    pub fn new(processor: Arc<DGDMProcessor>, config: StreamingConfig) -> Self {
        let (update_sender, _) = broadcast::channel(1000);
        let (result_sender, _) = broadcast::channel(1000);

        let streaming = Self {
            processor,
            graph: Arc::new(RwLock::new(Graph::new())),
            update_sender,
            result_sender,
            config,
            update_counter: Arc::new(RwLock::new(0)),
            pending_updates: Arc::new(RwLock::new(Vec::new())),
            last_diffusion: Arc::new(Mutex::new(Instant::now())),
            processing_queue: Arc::new(Mutex::new(Vec::new())),
        };
        
        streaming
    }

    pub fn subscribe_updates(&self) -> broadcast::Receiver<GraphUpdate> {
        self.update_sender.subscribe()
    }

    pub fn subscribe_results(&self) -> broadcast::Receiver<StreamingResult> {
        self.result_sender.subscribe()
    }

    #[instrument(skip(self))]
    pub async fn apply_update(&self, update: GraphUpdate) -> crate::Result<()> {
        info!("Applying graph update: {:?}", update);

        {
            let mut graph = self.graph.write().await;
            self.apply_update_to_graph(&mut graph, &update).await?;
        }

        let mut counter = self.update_counter.write().await;
        *counter += 1;

        if let Err(e) = self.update_sender.send(update.clone()) {
            warn!("Failed to broadcast update: {}", e);
        }

        if self.config.auto_diffuse {
            let mut queue = self.processing_queue.lock().await;
            queue.push(update.clone());
            
            // Trigger immediate processing if queue is full or enough time has passed
            let should_process = queue.len() >= self.config.batch_size || {
                let last_diffusion = self.last_diffusion.lock().await;
                last_diffusion.elapsed() >= Duration::from_millis(self.config.diffusion_interval_ms)
            };
            
            if should_process {
                let updates = std::mem::take(&mut *queue);
                drop(queue); // Release lock before processing
                self.trigger_diffusion(updates).await?;
            }
        }

        Ok(())
    }

    #[instrument(skip(self, updates))]
    pub async fn apply_batch_updates(&self, updates: Vec<GraphUpdate>) -> crate::Result<()> {
        info!("Applying batch of {} updates", updates.len());

        if updates.len() > self.config.max_buffer_size {
            return Err(crate::error::Error::validation(
                format!("Batch size {} exceeds maximum {}", updates.len(), self.config.max_buffer_size),
                format!("batch_size: {}", updates.len()),
                format!("max_batch_size: {}", self.config.max_buffer_size)
            ));
        }

        {
            let mut graph = self.graph.write().await;
            for update in &updates {
                self.apply_update_to_graph(&mut graph, update).await?;
            }
        }

        let mut counter = self.update_counter.write().await;
        *counter += updates.len() as u64;

        for update in &updates {
            if let Err(e) = self.update_sender.send(update.clone()) {
                warn!("Failed to broadcast update: {}", e);
            }
        }

        if self.config.auto_diffuse {
            self.trigger_diffusion(updates).await?;
        }

        Ok(())
    }

    async fn apply_update_to_graph(&self, graph: &mut Graph, update: &GraphUpdate) -> crate::Result<()> {
        match update {
            GraphUpdate::AddNode { id, features, label } => {
                let node = Node {
                    id: *id,
                    features: features.clone(),
                    label: label.clone(),
                };
                graph.add_node(node);
            }
            
            GraphUpdate::UpdateNode { id, features, label } => {
                if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == *id) {
                    node.features = features.clone();
                    node.label = label.clone();
                } else {
                    warn!("Attempted to update non-existent node: {}", id);
                }
            }
            
            GraphUpdate::RemoveNode { id } => {
                graph.nodes.retain(|n| n.id != *id);
                graph.edges.retain(|e| e.source != *id && e.target != *id);
            }
            
            GraphUpdate::AddEdge { source, target, weight, edge_type } => {
                let edge = Edge {
                    source: *source,
                    target: *target,
                    weight: *weight,
                    edge_type: edge_type.clone(),
                };
                graph.add_edge(edge);
            }
            
            GraphUpdate::UpdateEdge { source, target, weight, edge_type } => {
                if let Some(edge) = graph.edges.iter_mut()
                    .find(|e| e.source == *source && e.target == *target) {
                    edge.weight = *weight;
                    edge.edge_type = edge_type.clone();
                } else {
                    warn!("Attempted to update non-existent edge: {} -> {}", source, target);
                }
            }
            
            GraphUpdate::RemoveEdge { source, target } => {
                graph.edges.retain(|e| !(e.source == *source && e.target == *target));
            }
            
            GraphUpdate::Clear => {
                graph.nodes.clear();
                graph.edges.clear();
                graph.metadata.clear();
            }
            
            GraphUpdate::BatchUpdate { updates } => {
                for nested_update in updates {
                    Box::pin(self.apply_update_to_graph(graph, nested_update)).await?;
                }
            }
            
            GraphUpdate::RequestDiffusion { .. } => {
                // This is handled separately in the trigger_diffusion flow
                debug!("RequestDiffusion update received in apply_update_to_graph");
            }
        }

        Ok(())
    }

    #[instrument(skip(self, _updates))]
    async fn trigger_diffusion(&self, _updates: Vec<GraphUpdate>) -> crate::Result<()> {
        // Update last diffusion timestamp
        {
            let mut last_diffusion = self.last_diffusion.lock().await;
            *last_diffusion = Instant::now();
        }
        let start_time = std::time::Instant::now();
        
        let graph = self.graph.read().await;
        
        if graph.nodes.is_empty() {
            info!("Skipping diffusion on empty graph");
            return Ok(());
        }

        let compact_graph = graph.to_compact()?;
        drop(graph);

        if !self.processor.can_process(&compact_graph) {
            warn!("Graph exceeds processing limits, skipping diffusion");
            return Ok(());
        }

        let result = self.processor.process(&compact_graph)?;
        
        let embeddings: Vec<Vec<f32>> = result.embeddings
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        let update_id = *self.update_counter.read().await;
        let processing_time = start_time.elapsed();

        let streaming_result = StreamingResult {
            embeddings,
            update_id,
            processing_time_ms: processing_time.as_secs_f64() * 1000.0,
            graph_stats: StreamingGraphStats {
                num_nodes: compact_graph.num_nodes(),
                num_edges: compact_graph.num_edges(),
                update_count: update_id,
                last_update_timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
        };

        if let Err(e) = self.result_sender.send(streaming_result) {
            warn!("Failed to broadcast diffusion result: {}", e);
        }

        info!(
            "Diffusion completed in {:.2}ms for graph with {} nodes",
            processing_time.as_secs_f64() * 1000.0,
            compact_graph.num_nodes()
        );

        Ok(())
    }

    pub async fn manual_diffusion(&self) -> crate::Result<StreamingResult> {
        let start_time = std::time::Instant::now();
        
        let graph = self.graph.read().await;
        
        if graph.nodes.is_empty() {
            return Err(crate::error::Error::graph_processing(
                "Cannot perform diffusion on empty graph",
                "graph has no nodes"
            ));
        }

        let compact_graph = graph.to_compact()?;
        drop(graph);

        let result = self.processor.process(&compact_graph)?;
        
        let embeddings: Vec<Vec<f32>> = result.embeddings
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        let update_id = *self.update_counter.read().await;
        let processing_time = start_time.elapsed();

        let streaming_result = StreamingResult {
            embeddings,
            update_id,
            processing_time_ms: processing_time.as_secs_f64() * 1000.0,
            graph_stats: StreamingGraphStats {
                num_nodes: compact_graph.num_nodes(),
                num_edges: compact_graph.num_edges(),
                update_count: update_id,
                last_update_timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            },
        };

        Ok(streaming_result)
    }

    pub async fn get_graph_snapshot(&self) -> crate::Result<Graph> {
        let graph = self.graph.read().await;
        Ok(graph.clone())
    }

    pub async fn get_stats(&self) -> StreamingGraphStats {
        let graph = self.graph.read().await;
        let update_count = *self.update_counter.read().await;
        
        StreamingGraphStats {
            num_nodes: graph.num_nodes(),
            num_edges: graph.num_edges(),
            update_count,
            last_update_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    pub async fn start_periodic_diffusion(&self) -> crate::Result<()> {
        let interval = std::time::Duration::from_millis(self.config.diffusion_interval_ms);
        let mut interval_timer = tokio::time::interval(interval);

        loop {
            interval_timer.tick().await;
            
            let pending_updates = {
                let mut pending = self.pending_updates.write().await;
                if pending.is_empty() {
                    continue;
                }
                std::mem::take(&mut *pending)
            };

            if let Err(e) = self.trigger_diffusion(pending_updates).await {
                error!("Periodic diffusion failed: {}", e);
            }
        }
    }
}

pub fn create_update_stream(
    streaming_dgdm: Arc<StreamingDGDM>,
) -> impl Stream<Item = crate::Result<GraphUpdate>> {
    let receiver = streaming_dgdm.subscribe_updates();
    tokio_stream::wrappers::BroadcastStream::new(receiver)
        .map(|result| result.map_err(|e| crate::error::Error::network(e.to_string(), "websocket_stream")))
}

pub fn create_result_stream(
    streaming_dgdm: Arc<StreamingDGDM>,
) -> impl Stream<Item = crate::Result<StreamingResult>> {
    let receiver = streaming_dgdm.subscribe_results();
    tokio_stream::wrappers::BroadcastStream::new(receiver)
        .map(|result| result.map_err(|e| crate::error::Error::network(e.to_string(), "websocket_stream")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{GraphConfig, ProcessingConfig, DGDMProcessor};

    async fn create_test_streaming() -> StreamingDGDM {
        let config = GraphConfig::default();
        let processing_config = ProcessingConfig::default();
        let processor = Arc::new(DGDMProcessor::new(config, processing_config));
        
        StreamingDGDM::new(processor, StreamingConfig::default())
    }

    #[tokio::test]
    async fn test_add_node() {
        let streaming = create_test_streaming().await;
        
        let update = GraphUpdate::AddNode {
            id: 1,
            features: vec![1.0, 2.0],
            label: Some("test".to_string()),
        };

        streaming.apply_update(update).await.unwrap();
        
        let graph = streaming.get_graph_snapshot().await.unwrap();
        assert_eq!(graph.num_nodes(), 1);
        assert_eq!(graph.nodes[0].id, 1);
    }

    #[tokio::test]
    async fn test_add_edge() {
        let streaming = create_test_streaming().await;
        
        // Add nodes first
        streaming.apply_update(GraphUpdate::AddNode {
            id: 1,
            features: vec![1.0],
            label: None,
        }).await.unwrap();
        
        streaming.apply_update(GraphUpdate::AddNode {
            id: 2,
            features: vec![2.0],
            label: None,
        }).await.unwrap();

        // Add edge
        streaming.apply_update(GraphUpdate::AddEdge {
            source: 1,
            target: 2,
            weight: 0.5,
            edge_type: None,
        }).await.unwrap();
        
        let graph = streaming.get_graph_snapshot().await.unwrap();
        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.edges[0].source, 1);
        assert_eq!(graph.edges[0].target, 2);
    }

    #[tokio::test]
    async fn test_batch_updates() {
        let streaming = create_test_streaming().await;
        
        let updates = vec![
            GraphUpdate::AddNode { id: 1, features: vec![1.0], label: None },
            GraphUpdate::AddNode { id: 2, features: vec![2.0], label: None },
            GraphUpdate::AddEdge { source: 1, target: 2, weight: 1.0, edge_type: None },
        ];

        streaming.apply_batch_updates(updates).await.unwrap();
        
        let graph = streaming.get_graph_snapshot().await.unwrap();
        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);
    }

    #[tokio::test]
    async fn test_manual_diffusion() {
        let streaming = create_test_streaming().await;
        
        // Add some graph data
        streaming.apply_update(GraphUpdate::AddNode {
            id: 1,
            features: vec![1.0, 0.0],
            label: None,
        }).await.unwrap();
        
        streaming.apply_update(GraphUpdate::AddNode {
            id: 2,
            features: vec![0.0, 1.0],
            label: None,
        }).await.unwrap();

        streaming.apply_update(GraphUpdate::AddEdge {
            source: 1,
            target: 2,
            weight: 1.0,
            edge_type: None,
        }).await.unwrap();

        let result = streaming.manual_diffusion().await.unwrap();
        
        assert_eq!(result.embeddings.len(), 2);
        assert!(result.processing_time_ms > 0.0);
        assert_eq!(result.graph_stats.num_nodes, 2);
        assert_eq!(result.graph_stats.num_edges, 1);
    }
}