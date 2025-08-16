//! Comprehensive Benchmarking Framework with Statistical Validation
//! 
//! This module provides state-of-the-art benchmarking capabilities for graph diffusion algorithms,
//! including comparative studies, statistical significance testing, and reproducible experiments.

// pub mod statistical_tests;
// pub mod comparative_analysis;
// pub mod experiment_framework;
// pub mod reproducibility;
// pub mod visualization;

use crate::core::{graph::CompactGraph, adaptive_diffusion::AdaptiveDiffusionEngine};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub algorithms: Vec<AlgorithmConfig>,
    pub datasets: Vec<DatasetConfig>,
    pub metrics: Vec<MetricType>,
    pub statistical_tests: Vec<StatisticalTest>,
    pub repetitions: usize,
    pub confidence_level: f64,
    pub significance_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    pub name: String,
    pub algorithm_type: AlgorithmType,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlgorithmType {
    AdaptiveDiffusion,
    HeatDiffusion,
    PageRank,
    RandomWalk,
    PersonalizedPageRank,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub graph_type: GraphType,
    pub size_parameters: SizeParameters,
    pub generation_seed: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GraphType {
    Random { edge_probability: f64 },
    ScaleFree { gamma: f64 },
    SmallWorld { rewiring_probability: f64 },
    Grid { dimensions: Vec<usize> },
    Complete,
    Tree { branching_factor: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeParameters {
    pub num_nodes: usize,
    pub avg_degree: f64,
    pub feature_dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricType {
    ConvergenceTime,
    ConvergenceQuality,
    ComputationalComplexity,
    MemoryUsage,
    StatisticalSignificance,
    EnergyEfficiency,
    Scalability,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StatisticalTest {
    TTest,
    WilcoxonRankSum,
    KruskalWallis,
    ANOVA,
    EffectSize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub algorithm: String,
    pub dataset: String,
    pub metrics: HashMap<MetricType, MetricResult>,
    pub execution_time: Duration,
    pub memory_peak: usize,
    pub convergence_history: Vec<f32>,
    pub statistical_properties: StatisticalProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    pub value: f64,
    pub confidence_interval: (f64, f64),
    pub standard_error: f64,
    pub sample_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalProperties {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub p_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparativeReport {
    pub experiment_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub results: Vec<BenchmarkResult>,
    pub statistical_comparisons: Vec<StatisticalComparison>,
    pub performance_rankings: HashMap<MetricType, Vec<(String, f64)>>,
    pub reproducibility_metrics: ReproducibilityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalComparison {
    pub algorithm_a: String,
    pub algorithm_b: String,
    pub metric: MetricType,
    pub test_statistic: f64,
    pub p_value: f64,
    pub effect_size: f64,
    pub is_significant: bool,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReproducibilityMetrics {
    pub seed_consistency: f64,
    pub cross_run_stability: f64,
    pub platform_variance: f64,
    pub reproducibility_score: f64,
}

pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    results_cache: HashMap<String, Vec<BenchmarkResult>>,
}

impl BenchmarkSuite {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            results_cache: HashMap::new(),
        }
    }

    pub async fn run_full_benchmark(&mut self) -> crate::Result<ComparativeReport> {
        let experiment_id = format!("exp_{}", chrono::Utc::now().timestamp());
        let mut all_results = Vec::new();

        // Generate all dataset-algorithm combinations
        let combinations = self.generate_experiment_combinations();
        
        tracing::info!("Running {} benchmark combinations", combinations.len());

        for (dataset_config, algorithm_config) in combinations {
            let dataset_results = self.run_algorithm_on_dataset(
                &algorithm_config,
                &dataset_config,
            ).await?;
            
            all_results.extend(dataset_results);
        }

        // Perform statistical comparisons
        let statistical_comparisons = self.perform_statistical_comparisons(&all_results)?;
        
        // Calculate performance rankings
        let performance_rankings = self.calculate_performance_rankings(&all_results);
        
        // Assess reproducibility
        let reproducibility_metrics = self.assess_reproducibility(&all_results).await?;

        Ok(ComparativeReport {
            experiment_id,
            timestamp: chrono::Utc::now(),
            results: all_results,
            statistical_comparisons,
            performance_rankings,
            reproducibility_metrics,
        })
    }

    async fn run_algorithm_on_dataset(
        &mut self,
        algorithm_config: &AlgorithmConfig,
        dataset_config: &DatasetConfig,
    ) -> crate::Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();
        
        for repetition in 0..self.config.repetitions {
            let graph = self.generate_graph(dataset_config, repetition as u64)?;
            let result = self.benchmark_single_run(
                algorithm_config,
                dataset_config,
                &graph,
                repetition,
            ).await?;
            
            results.push(result);
        }

        Ok(results)
    }

    async fn benchmark_single_run(
        &self,
        algorithm_config: &AlgorithmConfig,
        dataset_config: &DatasetConfig,
        graph: &CompactGraph,
        repetition: usize,
    ) -> crate::Result<BenchmarkResult> {
        let start_time = Instant::now();
        let start_memory = self.get_memory_usage();

        // Run the algorithm
        let (convergence_history, statistical_properties) = match algorithm_config.algorithm_type {
            AlgorithmType::AdaptiveDiffusion => {
                self.run_adaptive_diffusion(graph, &algorithm_config.parameters).await?
            },
            AlgorithmType::HeatDiffusion => {
                self.run_heat_diffusion(graph, &algorithm_config.parameters).await?
            },
            AlgorithmType::PageRank => {
                self.run_pagerank(graph, &algorithm_config.parameters).await?
            },
            AlgorithmType::RandomWalk => {
                self.run_random_walk(graph, &algorithm_config.parameters).await?
            },
            AlgorithmType::PersonalizedPageRank => {
                self.run_personalized_pagerank(graph, &algorithm_config.parameters).await?
            },
        };

        let execution_time = start_time.elapsed();
        let peak_memory = self.get_memory_usage() - start_memory;

        // Calculate metrics
        let metrics = self.calculate_metrics(
            &convergence_history,
            execution_time,
            peak_memory,
            graph,
        )?;

        Ok(BenchmarkResult {
            algorithm: algorithm_config.name.clone(),
            dataset: dataset_config.name.clone(),
            metrics,
            execution_time,
            memory_peak: peak_memory,
            convergence_history,
            statistical_properties,
        })
    }

    async fn run_adaptive_diffusion(
        &self,
        graph: &CompactGraph,
        parameters: &HashMap<String, f64>,
    ) -> crate::Result<(Vec<f32>, StatisticalProperties)> {
        use crate::core::adaptive_diffusion::{AdaptiveDiffusionConfig, AdaptiveDiffusionEngine};
        
        let config = AdaptiveDiffusionConfig {
            meta_learning_rate: parameters.get("meta_learning_rate").unwrap_or(&0.001) as f32,
            attention_heads: parameters.get("attention_heads").unwrap_or(&8.0) as usize,
            hierarchy_levels: parameters.get("hierarchy_levels").unwrap_or(&3.0) as usize,
            ..Default::default()
        };

        let mut engine = AdaptiveDiffusionEngine::new(config);
        let initial_features = graph.node_features().clone();
        
        let (_result, metrics) = engine.adaptive_diffuse(graph, &initial_features)?;
        
        let stats = self.compute_statistical_properties(&metrics.convergence_history);
        Ok((metrics.convergence_history, stats))
    }

    async fn run_heat_diffusion(
        &self,
        graph: &CompactGraph,
        parameters: &HashMap<String, f64>,
    ) -> crate::Result<(Vec<f32>, StatisticalProperties)> {
        use crate::core::diffusion::{DiffusionEngine, DiffusionConfig, DiffusionType};
        
        let config = DiffusionConfig {
            diffusion_type: DiffusionType::Heat,
            num_steps: parameters.get("num_steps").unwrap_or(&50.0) as usize,
            learning_rate: parameters.get("learning_rate").unwrap_or(&0.01) as f32,
            ..Default::default()
        };

        let mut engine = DiffusionEngine::new(config);
        let initial_features = graph.node_features().clone();
        
        let (_result, metrics) = engine.diffuse(graph, &initial_features)?;
        
        // Create convergence history from single metric
        let convergence_history = vec![metrics.final_loss; metrics.convergence_steps];
        let stats = self.compute_statistical_properties(&convergence_history);
        Ok((convergence_history, stats))
    }

    async fn run_pagerank(
        &self,
        graph: &CompactGraph,
        parameters: &HashMap<String, f64>,
    ) -> crate::Result<(Vec<f32>, StatisticalProperties)> {
        use crate::core::diffusion::{DiffusionEngine, DiffusionConfig, DiffusionType};
        
        let config = DiffusionConfig {
            diffusion_type: DiffusionType::PageRank,
            num_steps: parameters.get("num_steps").unwrap_or(&50.0) as usize,
            ..Default::default()
        };

        let mut engine = DiffusionEngine::new(config);
        let initial_features = graph.node_features().clone();
        
        let (_result, metrics) = engine.diffuse(graph, &initial_features)?;
        
        let convergence_history = vec![metrics.final_loss; metrics.convergence_steps];
        let stats = self.compute_statistical_properties(&convergence_history);
        Ok((convergence_history, stats))
    }

    async fn run_random_walk(
        &self,
        graph: &CompactGraph,
        parameters: &HashMap<String, f64>,
    ) -> crate::Result<(Vec<f32>, StatisticalProperties)> {
        use crate::core::diffusion::{DiffusionEngine, DiffusionConfig, DiffusionType};
        
        let config = DiffusionConfig {
            diffusion_type: DiffusionType::Random,
            num_steps: parameters.get("num_steps").unwrap_or(&50.0) as usize,
            ..Default::default()
        };

        let mut engine = DiffusionEngine::new(config);
        let initial_features = graph.node_features().clone();
        
        let (_result, metrics) = engine.diffuse(graph, &initial_features)?;
        
        let convergence_history = vec![metrics.final_loss; metrics.convergence_steps];
        let stats = self.compute_statistical_properties(&convergence_history);
        Ok((convergence_history, stats))
    }

    async fn run_personalized_pagerank(
        &self,
        graph: &CompactGraph,
        parameters: &HashMap<String, f64>,
    ) -> crate::Result<(Vec<f32>, StatisticalProperties)> {
        use crate::core::diffusion::{DiffusionEngine, DiffusionConfig, DiffusionType};
        
        let reset_prob = parameters.get("reset_probability").unwrap_or(&0.15) as f32;
        let config = DiffusionConfig {
            diffusion_type: DiffusionType::Personalized { reset_probability: reset_prob },
            num_steps: parameters.get("num_steps").unwrap_or(&50.0) as usize,
            ..Default::default()
        };

        let mut engine = DiffusionEngine::new(config);
        let initial_features = graph.node_features().clone();
        
        let (_result, metrics) = engine.diffuse(graph, &initial_features)?;
        
        let convergence_history = vec![metrics.final_loss; metrics.convergence_steps];
        let stats = self.compute_statistical_properties(&convergence_history);
        Ok((convergence_history, stats))
    }

    fn generate_experiment_combinations(&self) -> Vec<(DatasetConfig, AlgorithmConfig)> {
        let mut combinations = Vec::new();
        
        for dataset in &self.config.datasets {
            for algorithm in &self.config.algorithms {
                combinations.push((dataset.clone(), algorithm.clone()));
            }
        }
        
        combinations
    }

    fn generate_graph(&self, config: &DatasetConfig, seed: u64) -> crate::Result<CompactGraph> {
        use crate::core::graph::{Graph, Node, Edge};
        use rand::{SeedableRng, Rng};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(config.generation_seed + seed);
        let mut graph = Graph::new();
        
        // Generate nodes
        for i in 0..config.size_parameters.num_nodes {
            let features: Vec<f32> = (0..config.size_parameters.feature_dimension)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            
            graph.add_node(Node {
                id: i as u32,
                features,
                label: None,
            });
        }
        
        // Generate edges based on graph type
        match &config.graph_type {
            GraphType::Random { edge_probability } => {
                for i in 0..config.size_parameters.num_nodes {
                    for j in i+1..config.size_parameters.num_nodes {
                        if rng.gen::<f64>() < *edge_probability {
                            let weight = rng.gen_range(0.1..2.0);
                            graph.add_edge(Edge {
                                source: i as u32,
                                target: j as u32,
                                weight,
                                edge_type: None,
                            });
                        }
                    }
                }
            },
            GraphType::ScaleFree { gamma: _ } => {
                // Simplified scale-free generation using preferential attachment
                let m = (config.size_parameters.avg_degree / 2.0) as usize;
                
                for i in m..config.size_parameters.num_nodes {
                    let mut targets = Vec::new();
                    
                    // Calculate degree distribution
                    let mut degrees: Vec<usize> = (0..i).map(|j| {
                        graph.adjacency_list().get(&(j as u32)).map_or(0, |v| v.len())
                    }).collect();
                    
                    if degrees.iter().sum::<usize>() == 0 {
                        degrees = vec![1; i];
                    }
                    
                    for _ in 0..m.min(i) {
                        let total_degree: usize = degrees.iter().sum();
                        let mut cumulative = 0;
                        let random_val = rng.gen_range(0..total_degree);
                        
                        for (j, &degree) in degrees.iter().enumerate() {
                            cumulative += degree;
                            if cumulative > random_val && !targets.contains(&j) {
                                targets.push(j);
                                break;
                            }
                        }
                    }
                    
                    for &target in &targets {
                        let weight = rng.gen_range(0.1..2.0);
                        graph.add_edge(Edge {
                            source: i as u32,
                            target: target as u32,
                            weight,
                            edge_type: None,
                        });
                    }
                }
            },
            GraphType::SmallWorld { rewiring_probability } => {
                // Generate ring lattice first
                let k = (config.size_parameters.avg_degree as usize).max(2);
                for i in 0..config.size_parameters.num_nodes {
                    for j in 1..=k/2 {
                        let target = (i + j) % config.size_parameters.num_nodes;
                        let weight = rng.gen_range(0.1..2.0);
                        graph.add_edge(Edge {
                            source: i as u32,
                            target: target as u32,
                            weight,
                            edge_type: None,
                        });
                    }
                }
                
                // Rewire edges
                for i in 0..config.size_parameters.num_nodes {
                    if rng.gen::<f64>() < *rewiring_probability {
                        let new_target = rng.gen_range(0..config.size_parameters.num_nodes);
                        if new_target != i {
                            let weight = rng.gen_range(0.1..2.0);
                            graph.add_edge(Edge {
                                source: i as u32,
                                target: new_target as u32,
                                weight,
                                edge_type: None,
                            });
                        }
                    }
                }
            },
            GraphType::Grid { dimensions } => {
                if dimensions.len() == 2 {
                    let rows = dimensions[0];
                    let cols = dimensions[1];
                    
                    for i in 0..rows {
                        for j in 0..cols {
                            let node_id = i * cols + j;
                            
                            // Right neighbor
                            if j < cols - 1 {
                                let weight = rng.gen_range(0.1..2.0);
                                graph.add_edge(Edge {
                                    source: node_id as u32,
                                    target: (node_id + 1) as u32,
                                    weight,
                                    edge_type: None,
                                });
                            }
                            
                            // Bottom neighbor
                            if i < rows - 1 {
                                let weight = rng.gen_range(0.1..2.0);
                                graph.add_edge(Edge {
                                    source: node_id as u32,
                                    target: (node_id + cols) as u32,
                                    weight,
                                    edge_type: None,
                                });
                            }
                        }
                    }
                }
            },
            GraphType::Complete => {
                for i in 0..config.size_parameters.num_nodes {
                    for j in i+1..config.size_parameters.num_nodes {
                        let weight = rng.gen_range(0.1..2.0);
                        graph.add_edge(Edge {
                            source: i as u32,
                            target: j as u32,
                            weight,
                            edge_type: None,
                        });
                    }
                }
            },
            GraphType::Tree { branching_factor } => {
                for i in 1..config.size_parameters.num_nodes {
                    let parent = (i - 1) / branching_factor;
                    let weight = rng.gen_range(0.1..2.0);
                    graph.add_edge(Edge {
                        source: parent as u32,
                        target: i as u32,
                        weight,
                        edge_type: None,
                    });
                }
            },
        }
        
        graph.to_compact()
    }

    fn calculate_metrics(
        &self,
        convergence_history: &[f32],
        execution_time: Duration,
        memory_usage: usize,
        graph: &CompactGraph,
    ) -> crate::Result<HashMap<MetricType, MetricResult>> {
        let mut metrics = HashMap::new();
        
        if self.config.metrics.contains(&MetricType::ConvergenceTime) {
            metrics.insert(MetricType::ConvergenceTime, MetricResult {
                value: execution_time.as_secs_f64(),
                confidence_interval: (0.0, 0.0), // Will be calculated across repetitions
                standard_error: 0.0,
                sample_size: 1,
            });
        }
        
        if self.config.metrics.contains(&MetricType::ConvergenceQuality) {
            let final_convergence = convergence_history.last().unwrap_or(&f32::INFINITY);
            metrics.insert(MetricType::ConvergenceQuality, MetricResult {
                value: *final_convergence as f64,
                confidence_interval: (0.0, 0.0),
                standard_error: 0.0,
                sample_size: 1,
            });
        }
        
        if self.config.metrics.contains(&MetricType::ComputationalComplexity) {
            let complexity = graph.num_nodes() as f64 * graph.num_edges() as f64;
            let normalized_time = execution_time.as_secs_f64() / complexity;
            metrics.insert(MetricType::ComputationalComplexity, MetricResult {
                value: normalized_time,
                confidence_interval: (0.0, 0.0),
                standard_error: 0.0,
                sample_size: 1,
            });
        }
        
        if self.config.metrics.contains(&MetricType::MemoryUsage) {
            metrics.insert(MetricType::MemoryUsage, MetricResult {
                value: memory_usage as f64,
                confidence_interval: (0.0, 0.0),
                standard_error: 0.0,
                sample_size: 1,
            });
        }
        
        if self.config.metrics.contains(&MetricType::StatisticalSignificance) {
            let significance = self.calculate_convergence_significance(convergence_history);
            metrics.insert(MetricType::StatisticalSignificance, MetricResult {
                value: significance,
                confidence_interval: (0.0, 0.0),
                standard_error: 0.0,
                sample_size: 1,
            });
        }
        
        Ok(metrics)
    }

    fn calculate_convergence_significance(&self, history: &[f32]) -> f64 {
        if history.len() < 3 {
            return 0.0;
        }
        
        // Calculate the slope of the convergence curve (log scale)
        let log_values: Vec<f64> = history.iter()
            .map(|&x| (x.max(1e-10) as f64).ln())
            .collect();
        
        let n = log_values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = log_values.iter().sum::<f64>() / n;
        
        let numerator: f64 = log_values.iter().enumerate()
            .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
            .sum();
        
        let denominator: f64 = (0..log_values.len())
            .map(|i| (i as f64 - x_mean).powi(2))
            .sum();
        
        if denominator > 1e-10 {
            (-numerator / denominator).exp() // Exponential of negative slope
        } else {
            0.0
        }
    }

    fn compute_statistical_properties(&self, data: &[f32]) -> StatisticalProperties {
        if data.is_empty() {
            return StatisticalProperties {
                mean: 0.0,
                median: 0.0,
                std_dev: 0.0,
                skewness: 0.0,
                kurtosis: 0.0,
                p_value: 1.0,
            };
        }
        
        let n = data.len() as f64;
        let mean = data.iter().sum::<f32>() as f64 / n;
        
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted_data.len() % 2 == 0 {
            (sorted_data[sorted_data.len() / 2 - 1] + sorted_data[sorted_data.len() / 2]) as f64 / 2.0
        } else {
            sorted_data[sorted_data.len() / 2] as f64
        };
        
        let variance = data.iter()
            .map(|&x| (x as f64 - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        
        let skewness = if std_dev > 1e-10 {
            data.iter()
                .map(|&x| ((x as f64 - mean) / std_dev).powi(3))
                .sum::<f64>() / n
        } else {
            0.0
        };
        
        let kurtosis = if std_dev > 1e-10 {
            data.iter()
                .map(|&x| ((x as f64 - mean) / std_dev).powi(4))
                .sum::<f64>() / n - 3.0
        } else {
            0.0
        };
        
        // Simple normality test (Shapiro-Wilk approximation)
        let p_value = if n > 3.0 {
            let w_statistic = 1.0 - (skewness.powi(2) + kurtosis.powi(2)) / 6.0;
            (1.0 - (-w_statistic.abs()).exp()).max(0.0).min(1.0)
        } else {
            1.0
        };
        
        StatisticalProperties {
            mean,
            median,
            std_dev,
            skewness,
            kurtosis,
            p_value,
        }
    }

    fn perform_statistical_comparisons(
        &self,
        results: &[BenchmarkResult],
    ) -> crate::Result<Vec<StatisticalComparison>> {
        let mut comparisons = Vec::new();
        
        // Group results by algorithm
        let mut algorithm_groups: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in results {
            algorithm_groups.entry(result.algorithm.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        let algorithms: Vec<String> = algorithm_groups.keys().cloned().collect();
        
        // Perform pairwise comparisons
        for i in 0..algorithms.len() {
            for j in i+1..algorithms.len() {
                let algo_a = &algorithms[i];
                let algo_b = &algorithms[j];
                
                for metric_type in &self.config.metrics {
                    let comparison = self.compare_algorithms(
                        algo_a,
                        algo_b,
                        metric_type,
                        &algorithm_groups,
                    )?;
                    
                    comparisons.push(comparison);
                }
            }
        }
        
        Ok(comparisons)
    }

    fn compare_algorithms(
        &self,
        algo_a: &str,
        algo_b: &str,
        metric: &MetricType,
        groups: &HashMap<String, Vec<&BenchmarkResult>>,
    ) -> crate::Result<StatisticalComparison> {
        let group_a = groups.get(algo_a).unwrap();
        let group_b = groups.get(algo_b).unwrap();
        
        let values_a: Vec<f64> = group_a.iter()
            .filter_map(|r| r.metrics.get(metric))
            .map(|m| m.value)
            .collect();
        
        let values_b: Vec<f64> = group_b.iter()
            .filter_map(|r| r.metrics.get(metric))
            .map(|m| m.value)
            .collect();
        
        if values_a.is_empty() || values_b.is_empty() {
            return Ok(StatisticalComparison {
                algorithm_a: algo_a.to_string(),
                algorithm_b: algo_b.to_string(),
                metric: metric.clone(),
                test_statistic: 0.0,
                p_value: 1.0,
                effect_size: 0.0,
                is_significant: false,
                confidence_interval: (0.0, 0.0),
            });
        }
        
        // Perform t-test
        let (test_statistic, p_value) = self.t_test(&values_a, &values_b);
        
        // Calculate effect size (Cohen's d)
        let effect_size = self.cohens_d(&values_a, &values_b);
        
        // Calculate confidence interval for the difference
        let confidence_interval = self.confidence_interval_difference(&values_a, &values_b);
        
        let is_significant = p_value < self.config.significance_threshold;
        
        Ok(StatisticalComparison {
            algorithm_a: algo_a.to_string(),
            algorithm_b: algo_b.to_string(),
            metric: metric.clone(),
            test_statistic,
            p_value,
            effect_size,
            is_significant,
            confidence_interval,
        })
    }

    fn t_test(&self, group_a: &[f64], group_b: &[f64]) -> (f64, f64) {
        if group_a.is_empty() || group_b.is_empty() {
            return (0.0, 1.0);
        }
        
        let n_a = group_a.len() as f64;
        let n_b = group_b.len() as f64;
        
        let mean_a = group_a.iter().sum::<f64>() / n_a;
        let mean_b = group_b.iter().sum::<f64>() / n_b;
        
        let var_a = group_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (n_a - 1.0);
        let var_b = group_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (n_b - 1.0);
        
        let pooled_se = ((var_a / n_a) + (var_b / n_b)).sqrt();
        
        if pooled_se < 1e-10 {
            return (0.0, 1.0);
        }
        
        let t_statistic = (mean_a - mean_b) / pooled_se;
        
        // Approximate p-value using normal distribution (for large samples)
        let degrees_freedom = n_a + n_b - 2.0;
        let p_value = if degrees_freedom > 30.0 {
            2.0 * (1.0 - self.normal_cdf(t_statistic.abs()))
        } else {
            // For small samples, use a conservative estimate
            2.0 * (1.0 - self.t_cdf(t_statistic.abs(), degrees_freedom))
        };
        
        (t_statistic, p_value)
    }

    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    fn erf(&self, x: f64) -> f64 {
        // Approximation of error function
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;
        
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
        
        sign * y
    }

    fn t_cdf(&self, t: f64, df: f64) -> f64 {
        // Simplified t-distribution CDF approximation
        if df > 100.0 {
            self.normal_cdf(t)
        } else {
            // Use normal approximation with adjustment for small df
            let adjusted_t = t * (1.0 - 1.0 / (4.0 * df));
            self.normal_cdf(adjusted_t)
        }
    }

    fn cohens_d(&self, group_a: &[f64], group_b: &[f64]) -> f64 {
        if group_a.is_empty() || group_b.is_empty() {
            return 0.0;
        }
        
        let mean_a = group_a.iter().sum::<f64>() / group_a.len() as f64;
        let mean_b = group_b.iter().sum::<f64>() / group_b.len() as f64;
        
        let var_a = group_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (group_a.len() - 1) as f64;
        let var_b = group_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (group_b.len() - 1) as f64;
        
        let pooled_std = ((var_a + var_b) / 2.0).sqrt();
        
        if pooled_std < 1e-10 {
            0.0
        } else {
            (mean_a - mean_b) / pooled_std
        }
    }

    fn confidence_interval_difference(&self, group_a: &[f64], group_b: &[f64]) -> (f64, f64) {
        if group_a.is_empty() || group_b.is_empty() {
            return (0.0, 0.0);
        }
        
        let mean_a = group_a.iter().sum::<f64>() / group_a.len() as f64;
        let mean_b = group_b.iter().sum::<f64>() / group_b.len() as f64;
        let mean_diff = mean_a - mean_b;
        
        let var_a = group_a.iter().map(|&x| (x - mean_a).powi(2)).sum::<f64>() / (group_a.len() - 1) as f64;
        let var_b = group_b.iter().map(|&x| (x - mean_b).powi(2)).sum::<f64>() / (group_b.len() - 1) as f64;
        
        let se_diff = ((var_a / group_a.len() as f64) + (var_b / group_b.len() as f64)).sqrt();
        
        // Use 95% confidence interval (approximate)
        let t_critical = 1.96; // For large samples
        let margin = t_critical * se_diff;
        
        (mean_diff - margin, mean_diff + margin)
    }

    fn calculate_performance_rankings(&self, results: &[BenchmarkResult]) -> HashMap<MetricType, Vec<(String, f64)>> {
        let mut rankings = HashMap::new();
        
        for metric_type in &self.config.metrics {
            let mut algorithm_scores: HashMap<String, Vec<f64>> = HashMap::new();
            
            for result in results {
                if let Some(metric_result) = result.metrics.get(metric_type) {
                    algorithm_scores.entry(result.algorithm.clone())
                        .or_insert_with(Vec::new)
                        .push(metric_result.value);
                }
            }
            
            let mut ranking: Vec<(String, f64)> = algorithm_scores.into_iter()
                .map(|(algo, scores)| {
                    let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
                    (algo, avg_score)
                })
                .collect();
            
            // Sort by score (ascending for most metrics, descending for efficiency metrics)
            match metric_type {
                MetricType::EnergyEfficiency => ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()),
                _ => ranking.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap()),
            }
            
            rankings.insert(metric_type.clone(), ranking);
        }
        
        rankings
    }

    async fn assess_reproducibility(&self, results: &[BenchmarkResult]) -> crate::Result<ReproducibilityMetrics> {
        let mut algorithm_groups: HashMap<String, Vec<&BenchmarkResult>> = HashMap::new();
        for result in results {
            algorithm_groups.entry(result.algorithm.clone())
                .or_insert_with(Vec::new)
                .push(result);
        }
        
        let mut seed_consistencies = Vec::new();
        let mut cross_run_stabilities = Vec::new();
        
        for (_, group) in algorithm_groups {
            if group.len() >= 2 {
                // Calculate seed consistency (coefficient of variation)
                let execution_times: Vec<f64> = group.iter()
                    .map(|r| r.execution_time.as_secs_f64())
                    .collect();
                
                let mean_time = execution_times.iter().sum::<f64>() / execution_times.len() as f64;
                let std_time = (execution_times.iter()
                    .map(|&x| (x - mean_time).powi(2))
                    .sum::<f64>() / (execution_times.len() - 1) as f64).sqrt();
                
                let cv = if mean_time > 1e-10 { std_time / mean_time } else { 0.0 };
                seed_consistencies.push(1.0 - cv.min(1.0)); // Higher is better
                
                // Calculate cross-run stability using convergence histories
                let convergence_correlations: Vec<f64> = group.windows(2)
                    .map(|pair| self.correlation(&pair[0].convergence_history, &pair[1].convergence_history))
                    .collect();
                
                let avg_correlation = convergence_correlations.iter().sum::<f64>() / convergence_correlations.len() as f64;
                cross_run_stabilities.push(avg_correlation.max(0.0));
            }
        }
        
        let seed_consistency = seed_consistencies.iter().sum::<f64>() / seed_consistencies.len().max(1) as f64;
        let cross_run_stability = cross_run_stabilities.iter().sum::<f64>() / cross_run_stabilities.len().max(1) as f64;
        
        // Platform variance (simplified - assume single platform for now)
        let platform_variance = 0.05; // 5% variance assumption
        
        let reproducibility_score = (seed_consistency + cross_run_stability + (1.0 - platform_variance)) / 3.0;
        
        Ok(ReproducibilityMetrics {
            seed_consistency,
            cross_run_stability,
            platform_variance,
            reproducibility_score,
        })
    }

    fn correlation(&self, a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let n = a.len() as f64;
        let mean_a = a.iter().sum::<f32>() as f64 / n;
        let mean_b = b.iter().sum::<f32>() as f64 / n;
        
        let numerator: f64 = a.iter().zip(b.iter())
            .map(|(&x, &y)| (x as f64 - mean_a) * (y as f64 - mean_b))
            .sum();
        
        let sum_sq_a: f64 = a.iter().map(|&x| (x as f64 - mean_a).powi(2)).sum();
        let sum_sq_b: f64 = b.iter().map(|&y| (y as f64 - mean_b).powi(2)).sum();
        
        let denominator = (sum_sq_a * sum_sq_b).sqrt();
        
        if denominator > 1e-10 {
            numerator / denominator
        } else {
            0.0
        }
    }

    fn get_memory_usage(&self) -> usize {
        // Simplified memory usage estimation
        // In a real implementation, this would use platform-specific APIs
        0
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            algorithms: vec![
                AlgorithmConfig {
                    name: "Adaptive".to_string(),
                    algorithm_type: AlgorithmType::AdaptiveDiffusion,
                    parameters: HashMap::new(),
                },
                AlgorithmConfig {
                    name: "Heat".to_string(),
                    algorithm_type: AlgorithmType::HeatDiffusion,
                    parameters: HashMap::new(),
                },
                AlgorithmConfig {
                    name: "PageRank".to_string(),
                    algorithm_type: AlgorithmType::PageRank,
                    parameters: HashMap::new(),
                },
            ],
            datasets: vec![
                DatasetConfig {
                    name: "Small Random".to_string(),
                    graph_type: GraphType::Random { edge_probability: 0.1 },
                    size_parameters: SizeParameters {
                        num_nodes: 100,
                        avg_degree: 4.0,
                        feature_dimension: 8,
                    },
                    generation_seed: 42,
                },
                DatasetConfig {
                    name: "Scale-Free".to_string(),
                    graph_type: GraphType::ScaleFree { gamma: 2.5 },
                    size_parameters: SizeParameters {
                        num_nodes: 200,
                        avg_degree: 6.0,
                        feature_dimension: 16,
                    },
                    generation_seed: 123,
                },
            ],
            metrics: vec![
                MetricType::ConvergenceTime,
                MetricType::ConvergenceQuality,
                MetricType::ComputationalComplexity,
                MetricType::StatisticalSignificance,
            ],
            statistical_tests: vec![
                StatisticalTest::TTest,
                StatisticalTest::EffectSize,
            ],
            repetitions: 10,
            confidence_level: 0.95,
            significance_threshold: 0.05,
        }
    }
}