//! Adaptive Neural Architecture Search for Graph Diffusion Models
//!
//! Automatically discovers and evolves optimal neural architectures for specific graph types
//! using reinforcement learning and evolutionary strategies.

use crate::core::graph::CompactGraph;
use crate::core::dgdm::{DGDMProcessor, ProcessingConfig, DiffusionResult};
use crate::quantum_graph_transformer::{QuantumGraphTransformer, QuantumConfig};
use ndarray::{Array1, Array2};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use tracing::{info, debug, warn};
use rand::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureGene {
    /// Layer configuration
    pub layers: Vec<LayerConfig>,
    /// Activation functions per layer
    pub activations: Vec<ActivationType>,
    /// Skip connection patterns
    pub skip_connections: Vec<(usize, usize)>,
    /// Attention mechanisms
    pub attention_heads: Vec<usize>,
    /// Quantum enhancement settings
    pub quantum_config: Option<QuantumConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub layer_type: LayerType,
    pub hidden_dim: usize,
    pub dropout_rate: f32,
    pub normalization: NormalizationType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Linear,
    GraphConvolutional,
    GraphAttention,
    GraphTransformer,
    ResidualBlock,
    QuantumLayer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    GELU,
    Swish,
    LeakyReLU,
    Tanh,
    Sigmoid,
    Quantum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    GraphNorm,
    QuantumNorm,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturePerformance {
    pub accuracy: f32,
    pub latency_ms: f64,
    pub memory_usage: usize,
    pub convergence_speed: f32,
    pub quantum_coherence: f32,
    pub fitness_score: f32,
}

#[derive(Debug, Clone)]
pub struct NASConfig {
    pub population_size: usize,
    pub max_generations: usize,
    pub mutation_rate: f32,
    pub crossover_rate: f32,
    pub elite_ratio: f32,
    pub max_layers: usize,
    pub min_layers: usize,
    pub performance_weight_accuracy: f32,
    pub performance_weight_speed: f32,
    pub performance_weight_memory: f32,
    pub adaptive_mutation: bool,
}

impl Default for NASConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            mutation_rate: 0.15,
            crossover_rate: 0.8,
            elite_ratio: 0.1,
            max_layers: 10,
            min_layers: 2,
            performance_weight_accuracy: 0.5,
            performance_weight_speed: 0.3,
            performance_weight_memory: 0.2,
            adaptive_mutation: true,
        }
    }
}

pub struct AdaptiveNeuralArchitectureSearch {
    config: NASConfig,
    population: Vec<ArchitectureGene>,
    performance_history: HashMap<String, ArchitecturePerformance>,
    generation: usize,
    best_architecture: Option<ArchitectureGene>,
    best_performance: Option<ArchitecturePerformance>,
}

impl AdaptiveNeuralArchitectureSearch {
    pub fn new(config: NASConfig) -> Self {
        Self {
            config,
            population: Vec::new(),
            performance_history: HashMap::new(),
            generation: 0,
            best_architecture: None,
            best_performance: None,
        }
    }

    /// Initialize random population of architectures
    pub fn initialize_population(&mut self) -> crate::Result<()> {
        info!("Initializing NAS population with {} architectures", self.config.population_size);
        
        self.population.clear();
        let mut rng = thread_rng();

        for _ in 0..self.config.population_size {
            let architecture = self.generate_random_architecture(&mut rng)?;
            self.population.push(architecture);
        }

        Ok(())
    }

    /// Evolve architectures for optimal performance on given graph dataset
    pub async fn evolve_architecture(
        &mut self, 
        training_graphs: &[CompactGraph],
        validation_graphs: &[CompactGraph]
    ) -> crate::Result<ArchitectureGene> {
        
        info!("Starting NAS evolution for {} generations", self.config.max_generations);
        
        for generation in 0..self.config.max_generations {
            self.generation = generation;
            
            // Evaluate current population
            let performances = self.evaluate_population(training_graphs).await?;
            
            // Validate on separate dataset
            let best_idx = self.find_best_architecture(&performances);
            let validation_performance = self.evaluate_architecture(
                &self.population[best_idx], 
                validation_graphs
            ).await?;
            
            info!("Generation {}: Best fitness {:.4}, validation accuracy {:.4}", 
                  generation, performances[best_idx].fitness_score, validation_performance.accuracy);

            // Update best architecture
            if self.best_performance.is_none() || 
               validation_performance.fitness_score > self.best_performance.as_ref().unwrap().fitness_score {
                self.best_architecture = Some(self.population[best_idx].clone());
                self.best_performance = Some(validation_performance);
                info!("New best architecture found! Fitness: {:.4}", validation_performance.fitness_score);
            }

            // Evolution operations
            if generation < self.config.max_generations - 1 {
                self.evolve_population(&performances)?;
            }
        }

        self.best_architecture.clone()
            .ok_or_else(|| crate::error::Error::graph_processing(
                "No best architecture found during evolution",
                "NAS evolution failed"
            ))
    }

    async fn evaluate_population(
        &self,
        graphs: &[CompactGraph]
    ) -> crate::Result<Vec<ArchitecturePerformance>> {
        let mut performances = Vec::new();

        for (idx, architecture) in self.population.iter().enumerate() {
            debug!("Evaluating architecture {} of {}", idx + 1, self.population.len());
            
            let performance = self.evaluate_architecture(architecture, graphs).await?;
            performances.push(performance);
        }

        Ok(performances)
    }

    async fn evaluate_architecture(
        &self,
        architecture: &ArchitectureGene,
        graphs: &[CompactGraph]
    ) -> crate::Result<ArchitecturePerformance> {
        
        let architecture_hash = self.compute_architecture_hash(architecture);
        
        // Check cache
        if let Some(cached_performance) = self.performance_history.get(&architecture_hash) {
            return Ok(cached_performance.clone());
        }

        let mut total_accuracy = 0.0;
        let mut total_latency = 0.0;
        let mut total_memory = 0;
        let mut total_convergence = 0.0;
        let mut total_quantum_coherence = 0.0;

        for graph in graphs.iter() {
            let (accuracy, latency, memory, convergence, quantum_coherence) = 
                self.evaluate_single_graph(architecture, graph).await?;
            
            total_accuracy += accuracy;
            total_latency += latency;
            total_memory += memory;
            total_convergence += convergence;
            total_quantum_coherence += quantum_coherence;
        }

        let num_graphs = graphs.len() as f32;
        let avg_accuracy = total_accuracy / num_graphs;
        let avg_latency = total_latency / num_graphs as f64;
        let avg_memory = total_memory / graphs.len();
        let avg_convergence = total_convergence / num_graphs;
        let avg_quantum_coherence = total_quantum_coherence / num_graphs;

        // Compute fitness score
        let normalized_latency = 1.0 / (1.0 + avg_latency as f32 / 1000.0); // Normalize to 0-1, prefer lower latency
        let normalized_memory = 1.0 / (1.0 + avg_memory as f32 / 1_000_000.0); // Normalize memory usage
        
        let fitness_score = 
            avg_accuracy * self.config.performance_weight_accuracy +
            normalized_latency * self.config.performance_weight_speed +
            normalized_memory * self.config.performance_weight_memory;

        let performance = ArchitecturePerformance {
            accuracy: avg_accuracy,
            latency_ms: avg_latency,
            memory_usage: avg_memory,
            convergence_speed: avg_convergence,
            quantum_coherence: avg_quantum_coherence,
            fitness_score,
        };

        Ok(performance)
    }

    async fn evaluate_single_graph(
        &self,
        architecture: &ArchitectureGene,
        graph: &CompactGraph
    ) -> crate::Result<(f32, f64, usize, f32, f32)> {
        
        // Convert architecture to processing config
        let processing_config = self.architecture_to_config(architecture);
        
        // Create processor
        let processor = DGDMProcessor::new(
            crate::core::GraphConfig::default(),
            processing_config
        );

        // Measure performance
        let start_time = std::time::Instant::now();
        let memory_before = self.get_memory_usage();
        
        let result = processor.process(graph)?;
        
        let latency = start_time.elapsed().as_secs_f64() * 1000.0;
        let memory_after = self.get_memory_usage();
        let memory_usage = memory_after.saturating_sub(memory_before);

        // Compute metrics
        let accuracy = self.compute_accuracy_metric(&result);
        let convergence_speed = 1.0 / (result.convergence_score + 1e-6);
        
        // Apply quantum enhancement if configured
        let quantum_coherence = if let Some(ref quantum_config) = architecture.quantum_config {
            let mut quantum_transformer = QuantumGraphTransformer::new(quantum_config.clone());
            match quantum_transformer.quantum_enhance_diffusion(graph, result).await {
                Ok(quantum_result) => quantum_result.quantum_coherence_score,
                Err(e) => {
                    warn!("Quantum enhancement failed: {}", e);
                    0.0
                }
            }
        } else {
            0.0
        };

        Ok((accuracy, latency, memory_usage, convergence_speed, quantum_coherence))
    }

    fn evolve_population(&mut self, performances: &[ArchitecturePerformance]) -> crate::Result<()> {
        let mut new_population = Vec::new();
        let mut rng = thread_rng();

        // Elitism: keep best architectures
        let elite_count = (self.config.population_size as f32 * self.config.elite_ratio) as usize;
        let mut sorted_indices: Vec<usize> = (0..performances.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            performances[b].fitness_score.partial_cmp(&performances[a].fitness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for i in 0..elite_count {
            new_population.push(self.population[sorted_indices[i]].clone());
        }

        // Crossover and mutation
        while new_population.len() < self.config.population_size {
            // Selection
            let parent1_idx = self.tournament_selection(performances, &mut rng);
            let parent2_idx = self.tournament_selection(performances, &mut rng);

            let mut child = if rng.gen::<f32>() < self.config.crossover_rate {
                self.crossover(&self.population[parent1_idx], &self.population[parent2_idx], &mut rng)?
            } else {
                self.population[parent1_idx].clone()
            };

            // Adaptive mutation rate
            let mutation_rate = if self.config.adaptive_mutation {
                self.compute_adaptive_mutation_rate(performances)
            } else {
                self.config.mutation_rate
            };

            if rng.gen::<f32>() < mutation_rate {
                self.mutate_architecture(&mut child, &mut rng)?;
            }

            new_population.push(child);
        }

        self.population = new_population;
        Ok(())
    }

    fn tournament_selection(
        &self,
        performances: &[ArchitecturePerformance],
        rng: &mut ThreadRng
    ) -> usize {
        let tournament_size = 3;
        let mut best_idx = rng.gen_range(0..performances.len());
        let mut best_fitness = performances[best_idx].fitness_score;

        for _ in 1..tournament_size {
            let candidate_idx = rng.gen_range(0..performances.len());
            let candidate_fitness = performances[candidate_idx].fitness_score;
            
            if candidate_fitness > best_fitness {
                best_idx = candidate_idx;
                best_fitness = candidate_fitness;
            }
        }

        best_idx
    }

    fn crossover(
        &self,
        parent1: &ArchitectureGene,
        parent2: &ArchitectureGene,
        rng: &mut ThreadRng
    ) -> crate::Result<ArchitectureGene> {
        
        // Layer crossover
        let crossover_point = rng.gen_range(1..parent1.layers.len().min(parent2.layers.len()));
        let mut layers = parent1.layers[..crossover_point].to_vec();
        layers.extend_from_slice(&parent2.layers[crossover_point..]);

        // Activation crossover
        let mut activations = Vec::new();
        for (i, _) in layers.iter().enumerate() {
            if i < parent1.activations.len() && i < parent2.activations.len() {
                activations.push(if rng.gen_bool(0.5) {
                    parent1.activations[i].clone()
                } else {
                    parent2.activations[i].clone()
                });
            }
        }

        // Skip connections crossover
        let skip_connections = if rng.gen_bool(0.5) {
            parent1.skip_connections.clone()
        } else {
            parent2.skip_connections.clone()
        };

        // Attention heads crossover
        let attention_heads = if rng.gen_bool(0.5) {
            parent1.attention_heads.clone()
        } else {
            parent2.attention_heads.clone()
        };

        // Quantum config crossover
        let quantum_config = match (&parent1.quantum_config, &parent2.quantum_config) {
            (Some(q1), Some(q2)) => {
                Some(if rng.gen_bool(0.5) { q1.clone() } else { q2.clone() })
            }
            (Some(q), None) | (None, Some(q)) => {
                if rng.gen_bool(0.3) { Some(q.clone()) } else { None }
            }
            (None, None) => None,
        };

        Ok(ArchitectureGene {
            layers,
            activations,
            skip_connections,
            attention_heads,
            quantum_config,
        })
    }

    fn mutate_architecture(
        &self,
        architecture: &mut ArchitectureGene,
        rng: &mut ThreadRng
    ) -> crate::Result<()> {
        
        // Layer mutations
        if rng.gen_bool(0.3) && architecture.layers.len() < self.config.max_layers {
            // Add layer
            let new_layer = self.generate_random_layer(rng);
            let insert_pos = rng.gen_range(0..=architecture.layers.len());
            architecture.layers.insert(insert_pos, new_layer);
        }

        if rng.gen_bool(0.2) && architecture.layers.len() > self.config.min_layers {
            // Remove layer
            let remove_idx = rng.gen_range(0..architecture.layers.len());
            architecture.layers.remove(remove_idx);
        }

        // Mutate existing layers
        for layer in &mut architecture.layers {
            if rng.gen_bool(0.2) {
                layer.hidden_dim = self.mutate_dimension(layer.hidden_dim, rng);
            }
            if rng.gen_bool(0.2) {
                layer.dropout_rate = (layer.dropout_rate + rng.gen_range(-0.1..0.1)).clamp(0.0, 0.9);
            }
            if rng.gen_bool(0.1) {
                layer.layer_type = self.generate_random_layer_type(rng);
            }
        }

        // Activation mutations
        for activation in &mut architecture.activations {
            if rng.gen_bool(0.15) {
                *activation = self.generate_random_activation(rng);
            }
        }

        // Quantum config mutation
        if rng.gen_bool(0.1) {
            if architecture.quantum_config.is_some() {
                if rng.gen_bool(0.3) {
                    architecture.quantum_config = None; // Remove quantum
                } else {
                    // Mutate quantum config
                    if let Some(ref mut qconfig) = architecture.quantum_config {
                        if rng.gen_bool(0.5) {
                            qconfig.qubits_per_node = rng.gen_range(2..=8);
                        }
                        if rng.gen_bool(0.5) {
                            qconfig.entanglement_strength = rng.gen_range(0.1..1.0);
                        }
                    }
                }
            } else if rng.gen_bool(0.2) {
                // Add quantum enhancement
                architecture.quantum_config = Some(QuantumConfig::default());
            }
        }

        Ok(())
    }

    fn generate_random_architecture(&self, rng: &mut ThreadRng) -> crate::Result<ArchitectureGene> {
        let num_layers = rng.gen_range(self.config.min_layers..=self.config.max_layers);
        
        let mut layers = Vec::new();
        let mut activations = Vec::new();
        
        for _ in 0..num_layers {
            layers.push(self.generate_random_layer(rng));
            activations.push(self.generate_random_activation(rng));
        }

        // Random skip connections
        let mut skip_connections = Vec::new();
        for i in 0..num_layers {
            for j in (i + 2)..num_layers {
                if rng.gen_bool(0.2) {
                    skip_connections.push((i, j));
                }
            }
        }

        // Random attention heads
        let attention_heads = (0..num_layers)
            .map(|_| if rng.gen_bool(0.3) { rng.gen_range(1..=8) } else { 1 })
            .collect();

        // Optional quantum enhancement
        let quantum_config = if rng.gen_bool(0.3) {
            Some(QuantumConfig::default())
        } else {
            None
        };

        Ok(ArchitectureGene {
            layers,
            activations,
            skip_connections,
            attention_heads,
            quantum_config,
        })
    }

    fn generate_random_layer(&self, rng: &mut ThreadRng) -> LayerConfig {
        LayerConfig {
            layer_type: self.generate_random_layer_type(rng),
            hidden_dim: self.generate_random_dimension(rng),
            dropout_rate: rng.gen_range(0.0..0.5),
            normalization: self.generate_random_normalization(rng),
        }
    }

    fn generate_random_layer_type(&self, rng: &mut ThreadRng) -> LayerType {
        match rng.gen_range(0..6) {
            0 => LayerType::Linear,
            1 => LayerType::GraphConvolutional,
            2 => LayerType::GraphAttention,
            3 => LayerType::GraphTransformer,
            4 => LayerType::ResidualBlock,
            _ => LayerType::QuantumLayer,
        }
    }

    fn generate_random_activation(&self, rng: &mut ThreadRng) -> ActivationType {
        match rng.gen_range(0..7) {
            0 => ActivationType::ReLU,
            1 => ActivationType::GELU,
            2 => ActivationType::Swish,
            3 => ActivationType::LeakyReLU,
            4 => ActivationType::Tanh,
            5 => ActivationType::Sigmoid,
            _ => ActivationType::Quantum,
        }
    }

    fn generate_random_normalization(&self, rng: &mut ThreadRng) -> NormalizationType {
        match rng.gen_range(0..5) {
            0 => NormalizationType::BatchNorm,
            1 => NormalizationType::LayerNorm,
            2 => NormalizationType::GraphNorm,
            3 => NormalizationType::QuantumNorm,
            _ => NormalizationType::None,
        }
    }

    fn generate_random_dimension(&self, rng: &mut ThreadRng) -> usize {
        let dimensions = [32, 64, 128, 256, 512, 1024];
        dimensions[rng.gen_range(0..dimensions.len())]
    }

    fn mutate_dimension(&self, current: usize, rng: &mut ThreadRng) -> usize {
        let directions = [-1, 0, 1];
        let direction = directions[rng.gen_range(0..3)];
        
        match direction {
            -1 => (current / 2).max(32),
            0 => current,
            _ => (current * 2).min(1024),
        }
    }

    fn compute_adaptive_mutation_rate(&self, performances: &[ArchitecturePerformance]) -> f32 {
        if performances.is_empty() {
            return self.config.mutation_rate;
        }

        // Increase mutation rate if population diversity is low
        let fitness_values: Vec<f32> = performances.iter().map(|p| p.fitness_score).collect();
        let mean_fitness: f32 = fitness_values.iter().sum::<f32>() / fitness_values.len() as f32;
        let variance: f32 = fitness_values.iter()
            .map(|&x| (x - mean_fitness).powi(2))
            .sum::<f32>() / fitness_values.len() as f32;
        
        let diversity_factor = 1.0 - (variance / (mean_fitness + 1e-6));
        self.config.mutation_rate * (1.0 + diversity_factor)
    }

    fn find_best_architecture(&self, performances: &[ArchitecturePerformance]) -> usize {
        performances.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.fitness_score.partial_cmp(&b.fitness_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    fn compute_architecture_hash(&self, architecture: &ArchitectureGene) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        architecture.layers.len().hash(&mut hasher);
        for layer in &architecture.layers {
            layer.hidden_dim.hash(&mut hasher);
        }
        hasher.finish().to_string()
    }

    fn architecture_to_config(&self, architecture: &ArchitectureGene) -> ProcessingConfig {
        // Convert NAS architecture to processing configuration
        ProcessingConfig {
            temperature: 0.1,
            dropout_rate: architecture.layers.first()
                .map(|l| l.dropout_rate)
                .unwrap_or(0.1),
            use_attention: architecture.attention_heads.iter().any(|&h| h > 1),
            normalize_features: architecture.layers.iter()
                .any(|l| !matches!(l.normalization, NormalizationType::None)),
            mixed_precision: false,
        }
    }

    fn compute_accuracy_metric(&self, result: &DiffusionResult) -> f32 {
        // Simple accuracy metric based on convergence
        (1.0 - result.convergence_score).max(0.0).min(1.0)
    }

    fn get_memory_usage(&self) -> usize {
        // Simplified memory usage estimation
        std::mem::size_of_val(&self.population) + 
        std::mem::size_of_val(&self.performance_history) * self.performance_history.len()
    }

    pub fn get_best_architecture(&self) -> Option<&ArchitectureGene> {
        self.best_architecture.as_ref()
    }

    pub fn get_best_performance(&self) -> Option<&ArchitecturePerformance> {
        self.best_performance.as_ref()
    }

    pub fn get_generation(&self) -> usize {
        self.generation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::graph::{Graph, Node, Edge};

    fn create_test_graphs() -> Vec<CompactGraph> {
        let mut graphs = Vec::new();
        
        for i in 0..3 {
            let mut graph = Graph::new();
            graph.add_node(Node { id: i + 1, features: vec![1.0, 0.0], label: None });
            graph.add_node(Node { id: i + 2, features: vec![0.0, 1.0], label: None });
            graph.add_edge(Edge { source: i + 1, target: i + 2, weight: 1.0, edge_type: None });
            graphs.push(graph.to_compact().unwrap());
        }
        
        graphs
    }

    #[test]
    fn test_nas_initialization() {
        let config = NASConfig::default();
        let mut nas = AdaptiveNeuralArchitectureSearch::new(config);
        
        let result = nas.initialize_population();
        assert!(result.is_ok());
        assert_eq!(nas.population.len(), 50);
    }

    #[test]
    fn test_architecture_generation() {
        let config = NASConfig::default();
        let nas = AdaptiveNeuralArchitectureSearch::new(config);
        let mut rng = thread_rng();
        
        let architecture = nas.generate_random_architecture(&mut rng).unwrap();
        
        assert!(!architecture.layers.is_empty());
        assert_eq!(architecture.layers.len(), architecture.activations.len());
    }

    #[tokio::test]
    async fn test_architecture_evaluation() {
        let config = NASConfig {
            population_size: 5,
            max_generations: 2,
            ..NASConfig::default()
        };
        let mut nas = AdaptiveNeuralArchitectureSearch::new(config);
        nas.initialize_population().unwrap();
        
        let graphs = create_test_graphs();
        let performance = nas.evaluate_architecture(&nas.population[0], &graphs).await;
        
        assert!(performance.is_ok());
        let perf = performance.unwrap();
        assert!(perf.fitness_score >= 0.0);
    }
}