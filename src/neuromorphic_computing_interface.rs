//! Neuromorphic Computing Interface
//!
//! Revolutionary brain-inspired computing system that mimics neural spike patterns
//! for ultra-efficient graph processing with biological-level energy consumption.

use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use parking_lot::{RwLock, Mutex};
use tokio::sync::{mpsc, broadcast, watch, Semaphore};
use tokio::time::{interval, sleep_until, timeout};
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, error, instrument, span, Level};
use uuid::Uuid;
use rand::{thread_rng, Rng, distributions::Distribution};
use ndarray::{Array1, Array2, Array3, Axis};

use crate::core::{Graph, Node, Edge};
use crate::error::{Error, Result};

/// Neuromorphic Computing Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicConfig {
    /// Number of neuromorphic processing units
    pub num_processing_units: usize,
    /// Neurons per processing unit
    pub neurons_per_unit: usize,
    /// Synaptic connections per neuron (average)
    pub synapses_per_neuron: usize,
    /// Membrane potential threshold for spiking
    pub spike_threshold: f32,
    /// Resting potential of neurons
    pub resting_potential: f32,
    /// Refractory period duration (microseconds)
    pub refractory_period_us: u64,
    /// Synaptic delay range (microseconds)
    pub synaptic_delay_range: (u64, u64),
    /// Learning rate for synaptic plasticity
    pub plasticity_learning_rate: f32,
    /// Time constant for membrane potential decay
    pub membrane_time_constant_ms: f32,
    /// Spike-timing-dependent plasticity (STDP) parameters
    pub stdp_config: STDPConfig,
    /// Event-driven simulation configuration
    pub event_driven_config: EventDrivenConfig,
}

impl Default for NeuromorphicConfig {
    fn default() -> Self {
        Self {
            num_processing_units: 8,
            neurons_per_unit: 1024,
            synapses_per_neuron: 64,
            spike_threshold: -55.0, // mV
            resting_potential: -70.0, // mV
            refractory_period_us: 2000, // 2ms
            synaptic_delay_range: (100, 2000), // 0.1-2ms
            plasticity_learning_rate: 0.01,
            membrane_time_constant_ms: 20.0,
            stdp_config: STDPConfig::default(),
            event_driven_config: EventDrivenConfig::default(),
        }
    }
}

/// Spike-Timing-Dependent Plasticity Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPConfig {
    /// Time window for STDP (milliseconds)
    pub stdp_window_ms: f32,
    /// Potentiation amplitude
    pub potentiation_amplitude: f32,
    /// Depression amplitude
    pub depression_amplitude: f32,
    /// Time constant for potentiation
    pub tau_potentiation_ms: f32,
    /// Time constant for depression
    pub tau_depression_ms: f32,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            stdp_window_ms: 100.0,
            potentiation_amplitude: 0.1,
            depression_amplitude: 0.08,
            tau_potentiation_ms: 20.0,
            tau_depression_ms: 20.0,
        }
    }
}

/// Event-Driven Simulation Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventDrivenConfig {
    /// Simulation time step (microseconds)
    pub time_step_us: u64,
    /// Maximum events per processing cycle
    pub max_events_per_cycle: usize,
    /// Event buffer size
    pub event_buffer_size: usize,
    /// Background noise rate (spikes/second)
    pub background_noise_rate: f32,
}

impl Default for EventDrivenConfig {
    fn default() -> Self {
        Self {
            time_step_us: 10, // 10 microseconds
            max_events_per_cycle: 10000,
            event_buffer_size: 1000000,
            background_noise_rate: 1.0, // 1 Hz per neuron
        }
    }
}

/// Neuromorphic spike event
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SpikeEvent {
    pub timestamp_us: u64,
    pub neuron_id: NeuronId,
    pub spike_amplitude: u16, // Quantized amplitude
    pub event_type: SpikeEventType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SpikeEventType {
    ExcitatorySpike,
    InhibitorySpike,
    PlasticityUpdate,
    NoiseSpike,
}

/// Neuron identifier (processing unit, local neuron index)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NeuronId {
    pub unit_id: usize,
    pub local_id: usize,
}

impl NeuronId {
    pub fn new(unit_id: usize, local_id: usize) -> Self {
        Self { unit_id, local_id }
    }
    
    pub fn global_id(&self, neurons_per_unit: usize) -> usize {
        self.unit_id * neurons_per_unit + self.local_id
    }
}

/// Spiking neuron model with membrane dynamics
#[derive(Debug, Clone)]
pub struct SpikingNeuron {
    pub id: NeuronId,
    pub membrane_potential: f32, // mV
    pub last_spike_time: Option<u64>, // microseconds
    pub refractory_end_time: Option<u64>,
    pub input_synapses: Vec<Synapse>,
    pub output_connections: Vec<NeuronId>,
    pub neuron_type: NeuronType,
    pub adaptation_state: AdaptationState,
    pub energy_consumption: f64, // picojoules
}

#[derive(Debug, Clone)]
pub enum NeuronType {
    Excitatory,
    Inhibitory,
    Modulatory,
    InputNeuron,
    OutputNeuron,
}

#[derive(Debug, Clone)]
pub struct AdaptationState {
    pub adaptation_current: f32,
    pub threshold_adaptation: f32,
    pub last_adaptation_time: u64,
}

/// Synaptic connection with plasticity
#[derive(Debug, Clone)]
pub struct Synapse {
    pub pre_neuron_id: NeuronId,
    pub post_neuron_id: NeuronId,
    pub weight: f32,
    pub delay_us: u64,
    pub last_pre_spike: Option<u64>,
    pub last_post_spike: Option<u64>,
    pub plasticity_trace: f32,
    pub synapse_type: SynapseType,
}

#[derive(Debug, Clone)]
pub enum SynapseType {
    Excitatory,
    Inhibitory,
    Modulatory,
    Plastic,
    Fixed,
}

/// Event-driven neuromorphic processing unit
#[derive(Debug)]
pub struct NeuromorphicProcessingUnit {
    pub unit_id: usize,
    pub neurons: Vec<SpikingNeuron>,
    pub event_queue: BTreeMap<u64, Vec<SpikeEvent>>, // Timestamp -> Events
    pub current_time: u64, // microseconds since start
    pub total_spikes: u64,
    pub energy_consumed: f64, // picojoules
    pub spike_history: VecDeque<(u64, usize)>, // (timestamp, spike_count)
    pub plasticity_engine: SynapticPlasticityEngine,
}

impl NeuromorphicProcessingUnit {
    pub fn new(unit_id: usize, config: &NeuromorphicConfig) -> Result<Self> {
        let mut neurons = Vec::with_capacity(config.neurons_per_unit);
        let mut rng = thread_rng();
        
        // Create neurons with diverse properties
        for local_id in 0..config.neurons_per_unit {
            let neuron_id = NeuronId::new(unit_id, local_id);
            
            let neuron_type = match local_id {
                i if i < config.neurons_per_unit * 4 / 5 => NeuronType::Excitatory,
                i if i < config.neurons_per_unit * 9 / 10 => NeuronType::Inhibitory,
                _ => NeuronType::Modulatory,
            };
            
            let neuron = SpikingNeuron {
                id: neuron_id,
                membrane_potential: config.resting_potential + rng.gen_range(-5.0..5.0),
                last_spike_time: None,
                refractory_end_time: None,
                input_synapses: Vec::new(),
                output_connections: Vec::new(),
                neuron_type,
                adaptation_state: AdaptationState {
                    adaptation_current: 0.0,
                    threshold_adaptation: 0.0,
                    last_adaptation_time: 0,
                },
                energy_consumption: 0.0,
            };
            
            neurons.push(neuron);
        }
        
        // Create synaptic connections
        Self::create_synaptic_connections(&mut neurons, config)?;
        
        Ok(Self {
            unit_id,
            neurons,
            event_queue: BTreeMap::new(),
            current_time: 0,
            total_spikes: 0,
            energy_consumed: 0.0,
            spike_history: VecDeque::with_capacity(1000),
            plasticity_engine: SynapticPlasticityEngine::new(config),
        })
    }

    fn create_synaptic_connections(neurons: &mut [SpikingNeuron], config: &NeuromorphicConfig) -> Result<()> {
        let mut rng = thread_rng();
        
        for post_idx in 0..neurons.len() {
            let post_id = neurons[post_idx].id;
            let target_synapses = config.synapses_per_neuron + rng.gen_range(-10..10).max(1) as usize;
            
            for _ in 0..target_synapses {
                let pre_idx = rng.gen_range(0..neurons.len());
                if pre_idx == post_idx { continue; } // No self-connections
                
                let pre_id = neurons[pre_idx].id;
                let weight = match neurons[pre_idx].neuron_type {
                    NeuronType::Excitatory => rng.gen_range(0.1..1.0),
                    NeuronType::Inhibitory => rng.gen_range(-1.0..-0.1),
                    NeuronType::Modulatory => rng.gen_range(-0.5..0.5),
                    _ => rng.gen_range(0.1..0.5),
                };
                
                let delay_us = rng.gen_range(config.synaptic_delay_range.0..=config.synaptic_delay_range.1);
                
                let synapse = Synapse {
                    pre_neuron_id: pre_id,
                    post_neuron_id: post_id,
                    weight,
                    delay_us,
                    last_pre_spike: None,
                    last_post_spike: None,
                    plasticity_trace: 0.0,
                    synapse_type: if matches!(neurons[pre_idx].neuron_type, NeuronType::Excitatory) {
                        SynapseType::Plastic
                    } else {
                        SynapseType::Fixed
                    },
                };
                
                neurons[post_idx].input_synapses.push(synapse);
                neurons[pre_idx].output_connections.push(post_id);
            }
        }
        
        Ok(())
    }

    /// Process events in the neuromorphic processing unit
    #[instrument(skip(self))]
    pub fn process_events(&mut self, config: &NeuromorphicConfig) -> Result<ProcessingResults> {
        let start_time = self.current_time;
        let end_time = self.current_time + config.event_driven_config.time_step_us;
        
        let mut spikes_fired = 0;
        let mut events_processed = 0;
        let mut plastic_updates = 0;
        
        // Process all events in current time window
        while let Some((&event_time, _)) = self.event_queue.range(start_time..end_time).next() {
            if let Some(events) = self.event_queue.remove(&event_time) {
                for event in events {
                    if events_processed >= config.event_driven_config.max_events_per_cycle {
                        // Return unused events to queue
                        self.event_queue.entry(event_time).or_insert_with(Vec::new);
                        break;
                    }
                    
                    self.process_single_event(event, config)?;
                    events_processed += 1;
                    
                    if event.event_type == SpikeEventType::ExcitatorySpike || 
                       event.event_type == SpikeEventType::InhibitorySpike {
                        spikes_fired += 1;
                    } else if event.event_type == SpikeEventType::PlasticityUpdate {
                        plastic_updates += 1;
                    }
                }
            }
        }
        
        // Update neuron membrane potentials
        for neuron in &mut self.neurons {
            self.update_neuron_dynamics(neuron, config)?;
            
            // Check for spike generation
            if self.should_neuron_spike(neuron, config) {
                self.generate_spike(neuron, config)?;
                spikes_fired += 1;
            }
        }
        
        // Update time
        self.current_time = end_time;
        self.total_spikes += spikes_fired as u64;
        
        // Record spike history
        self.spike_history.push_back((self.current_time, spikes_fired));
        if self.spike_history.len() > 1000 {
            self.spike_history.pop_front();
        }
        
        Ok(ProcessingResults {
            events_processed,
            spikes_fired,
            plastic_updates,
            energy_consumed: self.calculate_energy_consumption(spikes_fired),
            processing_time_us: end_time - start_time,
        })
    }

    fn process_single_event(&mut self, event: SpikeEvent, config: &NeuromorphicConfig) -> Result<()> {
        let neuron_idx = self.find_neuron_index(event.neuron_id)?;
        let neuron = &mut self.neurons[neuron_idx];
        
        match event.event_type {
            SpikeEventType::ExcitatorySpike | SpikeEventType::InhibitorySpike => {
                // Apply synaptic input
                self.apply_synaptic_input(neuron, event, config)?;
            }
            SpikeEventType::PlasticityUpdate => {
                // Update synaptic plasticity
                self.plasticity_engine.update_plasticity(neuron, event, config)?;
            }
            SpikeEventType::NoiseSpike => {
                // Apply background noise
                neuron.membrane_potential += thread_rng().gen_range(-1.0..1.0);
            }
        }
        
        Ok(())
    }

    fn update_neuron_dynamics(&mut self, neuron: &mut SpikingNeuron, config: &NeuromorphicConfig) -> Result<()> {
        let dt_ms = config.event_driven_config.time_step_us as f32 / 1000.0;
        
        // Check if neuron is in refractory period
        if let Some(refractory_end) = neuron.refractory_end_time {
            if self.current_time < refractory_end {
                return Ok(()); // Skip dynamics update during refractory period
            } else {
                neuron.refractory_end_time = None;
            }
        }
        
        // Membrane potential decay toward resting potential
        let decay_factor = (-dt_ms / config.membrane_time_constant_ms).exp();
        neuron.membrane_potential = config.resting_potential + 
            (neuron.membrane_potential - config.resting_potential) * decay_factor;
        
        // Apply adaptation current
        neuron.membrane_potential -= neuron.adaptation_state.adaptation_current * dt_ms;
        
        // Decay adaptation current
        neuron.adaptation_state.adaptation_current *= 0.99;
        
        // Threshold adaptation
        if let Some(last_spike) = neuron.last_spike_time {
            let time_since_spike = self.current_time.saturating_sub(last_spike) as f32 / 1000.0; // ms
            neuron.adaptation_state.threshold_adaptation = 
                5.0 * (-time_since_spike / 100.0).exp(); // Threshold increases after spike
        }
        
        Ok(())
    }

    fn should_neuron_spike(&self, neuron: &SpikingNeuron, config: &NeuromorphicConfig) -> bool {
        let effective_threshold = config.spike_threshold + neuron.adaptation_state.threshold_adaptation;
        neuron.membrane_potential >= effective_threshold && neuron.refractory_end_time.is_none()
    }

    fn generate_spike(&mut self, neuron: &mut SpikingNeuron, config: &NeuromorphicConfig) -> Result<()> {
        let spike_time = self.current_time;
        
        // Reset membrane potential and set refractory period
        neuron.membrane_potential = config.resting_potential;
        neuron.last_spike_time = Some(spike_time);
        neuron.refractory_end_time = Some(spike_time + config.refractory_period_us);
        
        // Add adaptation current
        neuron.adaptation_state.adaptation_current += 2.0;
        
        // Energy consumption for spike
        neuron.energy_consumption += 1.0; // 1 picojoule per spike (approximate)
        self.energy_consumed += neuron.energy_consumption;
        
        // Propagate spike to connected neurons
        for &target_id in &neuron.output_connections {
            // Find appropriate synapse for this connection
            for synapse in &neuron.input_synapses {
                if synapse.pre_neuron_id == neuron.id && synapse.post_neuron_id == target_id {
                    let arrival_time = spike_time + synapse.delay_us;
                    
                    let event_type = match synapse.synapse_type {
                        SynapseType::Excitatory | SynapseType::Plastic => SpikeEventType::ExcitatorySpike,
                        SynapseType::Inhibitory => SpikeEventType::InhibitorySpike,
                        _ => SpikeEventType::ExcitatorySpike,
                    };
                    
                    let event = SpikeEvent {
                        timestamp_us: arrival_time,
                        neuron_id: target_id,
                        spike_amplitude: (synapse.weight.abs() * 1000.0) as u16, // Convert to quantized amplitude
                        event_type,
                    };
                    
                    self.event_queue.entry(arrival_time).or_insert_with(Vec::new).push(event);
                    break;
                }
            }
        }
        
        Ok(())
    }

    fn apply_synaptic_input(&mut self, neuron: &mut SpikingNeuron, event: SpikeEvent, _config: &NeuromorphicConfig) -> Result<()> {
        let amplitude = event.spike_amplitude as f32 / 1000.0; // Convert back from quantized
        
        match event.event_type {
            SpikeEventType::ExcitatorySpike => {
                neuron.membrane_potential += amplitude;
            }
            SpikeEventType::InhibitorySpike => {
                neuron.membrane_potential -= amplitude;
            }
            _ => {}
        }
        
        // Update plasticity traces
        for synapse in &mut neuron.input_synapses {
            if synapse.pre_neuron_id.global_id(1024) == event.neuron_id.global_id(1024) {
                synapse.last_pre_spike = Some(event.timestamp_us);
                synapse.plasticity_trace += 0.1;
            }
        }
        
        Ok(())
    }

    fn find_neuron_index(&self, neuron_id: NeuronId) -> Result<usize> {
        self.neurons
            .iter()
            .position(|n| n.id == neuron_id)
            .ok_or_else(|| Error::graph_processing("Neuron not found", &format!("NeuronId: {:?}", neuron_id)))
    }

    fn calculate_energy_consumption(&self, spikes_fired: usize) -> f64 {
        // Energy model: base consumption + spike-dependent + synaptic transmission
        let base_energy = 0.1; // picojoules per time step
        let spike_energy = spikes_fired as f64 * 1.0; // 1 pJ per spike
        let synaptic_energy = spikes_fired as f64 * 0.5 * self.neurons.len() as f64 / 100.0; // Transmission cost
        
        base_energy + spike_energy + synaptic_energy
    }
}

#[derive(Debug)]
pub struct ProcessingResults {
    pub events_processed: usize,
    pub spikes_fired: usize,
    pub plastic_updates: usize,
    pub energy_consumed: f64,
    pub processing_time_us: u64,
}

/// Synaptic Plasticity Engine implementing STDP
#[derive(Debug)]
pub struct SynapticPlasticityEngine {
    stdp_config: STDPConfig,
    plasticity_updates: u64,
}

impl SynapticPlasticityEngine {
    pub fn new(config: &NeuromorphicConfig) -> Self {
        Self {
            stdp_config: config.stdp_config.clone(),
            plasticity_updates: 0,
        }
    }

    pub fn update_plasticity(&mut self, neuron: &mut SpikingNeuron, event: SpikeEvent, _config: &NeuromorphicConfig) -> Result<()> {
        for synapse in &mut neuron.input_synapses {
            if let Some(pre_spike_time) = synapse.last_pre_spike {
                let dt = event.timestamp_us as f32 - pre_spike_time as f32; // microseconds
                let dt_ms = dt / 1000.0;
                
                if dt_ms.abs() <= self.stdp_config.stdp_window_ms {
                    let weight_change = if dt_ms > 0.0 {
                        // Post-before-pre: depression
                        -self.stdp_config.depression_amplitude * 
                        (-dt_ms / self.stdp_config.tau_depression_ms).exp()
                    } else {
                        // Pre-before-post: potentiation
                        self.stdp_config.potentiation_amplitude * 
                        (dt_ms / self.stdp_config.tau_potentiation_ms).exp()
                    };
                    
                    synapse.weight += weight_change;
                    synapse.weight = synapse.weight.clamp(-2.0, 2.0); // Bound weights
                    
                    self.plasticity_updates += 1;
                }
            }
        }
        
        Ok(())
    }
}

/// Graph-to-Spike Encoder
#[derive(Debug)]
pub struct GraphSpikeEncoder {
    config: NeuromorphicConfig,
    encoding_strategy: EncodingStrategy,
    node_to_neuron_map: HashMap<usize, Vec<NeuronId>>,
    edge_to_synapse_map: HashMap<(usize, usize), Vec<(NeuronId, NeuronId)>>,
}

#[derive(Debug, Clone)]
pub enum EncodingStrategy {
    RateEncoding,
    TemporalEncoding,
    PopulationEncoding,
    SpatialEncoding,
    HybridEncoding,
}

impl GraphSpikeEncoder {
    pub fn new(config: NeuromorphicConfig) -> Self {
        Self {
            config,
            encoding_strategy: EncodingStrategy::HybridEncoding,
            node_to_neuron_map: HashMap::new(),
            edge_to_synapse_map: HashMap::new(),
        }
    }

    /// Encode graph structure into spike patterns
    #[instrument(skip(self, graph))]
    pub fn encode_graph_to_spikes(&mut self, graph: &Graph) -> Result<Vec<SpikeEvent>> {
        info!("Encoding graph with {} nodes and {} edges to neuromorphic spikes", 
              graph.node_count(), graph.edge_count());
        
        let mut spike_events = Vec::new();
        let mut current_time = 0u64;
        
        // Map graph nodes to neuron populations
        self.map_nodes_to_neurons(graph)?;
        
        // Map graph edges to synaptic connections
        self.map_edges_to_synapses(graph)?;
        
        // Generate spike patterns based on encoding strategy
        match self.encoding_strategy {
            EncodingStrategy::RateEncoding => {
                spike_events.extend(self.rate_encode_graph(graph, &mut current_time)?);
            }
            EncodingStrategy::TemporalEncoding => {
                spike_events.extend(self.temporal_encode_graph(graph, &mut current_time)?);
            }
            EncodingStrategy::PopulationEncoding => {
                spike_events.extend(self.population_encode_graph(graph, &mut current_time)?);
            }
            EncodingStrategy::HybridEncoding => {
                // Combine multiple encoding strategies
                spike_events.extend(self.rate_encode_graph(graph, &mut current_time)?);
                spike_events.extend(self.temporal_encode_graph(graph, &mut current_time)?);
                spike_events.extend(self.population_encode_graph(graph, &mut current_time)?);
            }
            _ => {
                return Err(Error::graph_processing("Unsupported encoding strategy", 
                                                 "GraphSpikeEncoder"));
            }
        }
        
        // Sort events by timestamp
        spike_events.sort_by_key(|e| e.timestamp_us);
        
        info!("Generated {} spike events for graph processing", spike_events.len());
        Ok(spike_events)
    }

    fn map_nodes_to_neurons(&mut self, graph: &Graph) -> Result<()> {
        let neurons_per_node = 8; // Use multiple neurons per graph node for population encoding
        let mut neuron_counter = 0;
        
        for node in graph.nodes() {
            let mut neuron_population = Vec::new();
            
            for _ in 0..neurons_per_node {
                let unit_id = neuron_counter / self.config.neurons_per_unit;
                let local_id = neuron_counter % self.config.neurons_per_unit;
                
                if unit_id >= self.config.num_processing_units {
                    return Err(Error::graph_processing("Not enough neuromorphic processing units", 
                                                     "GraphSpikeEncoder"));
                }
                
                neuron_population.push(NeuronId::new(unit_id, local_id));
                neuron_counter += 1;
            }
            
            self.node_to_neuron_map.insert(node.id, neuron_population);
        }
        
        Ok(())
    }

    fn map_edges_to_synapses(&mut self, graph: &Graph) -> Result<()> {
        for edge in graph.edges() {
            let source_neurons = self.node_to_neuron_map.get(&edge.source)
                .ok_or_else(|| Error::graph_processing("Source node not mapped", "GraphSpikeEncoder"))?;
            let target_neurons = self.node_to_neuron_map.get(&edge.target)
                .ok_or_else(|| Error::graph_processing("Target node not mapped", "GraphSpikeEncoder"))?;
            
            let mut synapse_connections = Vec::new();
            
            // Create all-to-all connections between populations (with probability)
            let mut rng = thread_rng();
            for &source_neuron in source_neurons {
                for &target_neuron in target_neurons {
                    if rng.gen::<f32>() < 0.3 { // 30% connection probability
                        synapse_connections.push((source_neuron, target_neuron));
                    }
                }
            }
            
            self.edge_to_synapse_map.insert((edge.source, edge.target), synapse_connections);
        }
        
        Ok(())
    }

    fn rate_encode_graph(&self, graph: &Graph, current_time: &mut u64) -> Result<Vec<SpikeEvent>> {
        let mut events = Vec::new();
        let encoding_duration_us = 50000; // 50ms encoding window
        
        for node in graph.nodes() {
            if let Some(neuron_population) = self.node_to_neuron_map.get(&node.id) {
                // Calculate firing rate based on node features
                let feature_magnitude = node.features.iter().map(|&f| f.abs()).sum::<f32>();
                let base_rate = (feature_magnitude * 100.0).min(1000.0); // Max 1kHz
                
                // Generate Poisson spike train
                let mut rng = thread_rng();
                for &neuron_id in neuron_population {
                    let num_spikes = Poisson::new(base_rate as f64 * encoding_duration_us as f64 / 1e6)
                        .map_err(|_| Error::graph_processing("Invalid Poisson parameter", "RateEncoding"))?
                        .sample(&mut rng) as usize;
                    
                    for _ in 0..num_spikes {
                        let spike_time = *current_time + rng.gen_range(0..encoding_duration_us);
                        events.push(SpikeEvent {
                            timestamp_us: spike_time,
                            neuron_id,
                            spike_amplitude: 1000, // Standard amplitude
                            event_type: SpikeEventType::ExcitatorySpike,
                        });
                    }
                }
            }
        }
        
        *current_time += encoding_duration_us;
        Ok(events)
    }

    fn temporal_encode_graph(&self, graph: &Graph, current_time: &mut u64) -> Result<Vec<SpikeEvent>> {
        let mut events = Vec::new();
        let time_per_node = 5000; // 5ms per node
        
        for (node_index, node) in graph.nodes().enumerate() {
            if let Some(neuron_population) = self.node_to_neuron_map.get(&node.id) {
                let node_time_offset = node_index as u64 * time_per_node;
                
                // Encode node features as spike timing
                for (feature_index, &feature_value) in node.features.iter().enumerate() {
                    let neuron_index = feature_index % neuron_population.len();
                    let neuron_id = neuron_population[neuron_index];
                    
                    // Convert feature value to spike timing
                    let timing_offset = ((feature_value + 1.0) * 0.5 * time_per_node as f32) as u64;
                    let spike_time = *current_time + node_time_offset + timing_offset;
                    
                    events.push(SpikeEvent {
                        timestamp_us: spike_time,
                        neuron_id,
                        spike_amplitude: (feature_value.abs() * 1000.0) as u16,
                        event_type: SpikeEventType::ExcitatorySpike,
                    });
                }
            }
        }
        
        *current_time += graph.node_count() as u64 * time_per_node;
        Ok(events)
    }

    fn population_encode_graph(&self, graph: &Graph, current_time: &mut u64) -> Result<Vec<SpikeEvent>> {
        let mut events = Vec::new();
        let encoding_window = 20000; // 20ms window
        
        for node in graph.nodes() {
            if let Some(neuron_population) = self.node_to_neuron_map.get(&node.id) {
                // Use population vector encoding
                for (i, &neuron_id) in neuron_population.iter().enumerate() {
                    let preferred_value = (i as f32) / (neuron_population.len() as f32 - 1.0) * 2.0 - 1.0; // -1 to 1
                    
                    // Calculate response based on distance from preferred value
                    let mut response_strength = 0.0;
                    for &feature_value in &node.features {
                        let distance = (feature_value - preferred_value).abs();
                        response_strength += (1.0 - distance / 2.0).max(0.0);
                    }
                    response_strength /= node.features.len() as f32;
                    
                    // Generate spikes based on response strength
                    let spike_rate = response_strength * 500.0; // Max 500Hz
                    let mut rng = thread_rng();
                    let num_spikes = Poisson::new(spike_rate as f64 * encoding_window as f64 / 1e6)
                        .map_err(|_| Error::graph_processing("Invalid Poisson parameter", "PopulationEncoding"))?
                        .sample(&mut rng) as usize;
                    
                    for _ in 0..num_spikes {
                        let spike_time = *current_time + rng.gen_range(0..encoding_window);
                        events.push(SpikeEvent {
                            timestamp_us: spike_time,
                            neuron_id,
                            spike_amplitude: (response_strength * 1000.0) as u16,
                            event_type: SpikeEventType::ExcitatorySpike,
                        });
                    }
                }
            }
        }
        
        *current_time += encoding_window;
        Ok(events)
    }
}

/// Spike-to-Graph Decoder
#[derive(Debug)]
pub struct SpikeGraphDecoder {
    config: NeuromorphicConfig,
    decoding_strategy: DecodingStrategy,
    spike_buffer: VecDeque<SpikeEvent>,
    output_buffer: Array2<f32>,
}

#[derive(Debug, Clone)]
pub enum DecodingStrategy {
    RateDecoding,
    PopulationVector,
    TemporalPattern,
    BayesianDecoding,
}

impl SpikeGraphDecoder {
    pub fn new(config: NeuromorphicConfig, output_size: (usize, usize)) -> Self {
        Self {
            config,
            decoding_strategy: DecodingStrategy::PopulationVector,
            spike_buffer: VecDeque::with_capacity(10000),
            output_buffer: Array2::zeros(output_size),
        }
    }

    /// Decode spike patterns back to graph embeddings
    #[instrument(skip(self, spike_events))]
    pub fn decode_spikes_to_embeddings(&mut self, spike_events: &[SpikeEvent]) -> Result<Array2<f32>> {
        info!("Decoding {} spike events to graph embeddings", spike_events.len());
        
        // Add new spikes to buffer
        for &event in spike_events {
            self.spike_buffer.push_back(event);
        }
        
        // Keep only recent spikes (last 100ms)
        let cutoff_time = spike_events.last()
            .map(|e| e.timestamp_us.saturating_sub(100000))
            .unwrap_or(0);
        
        while self.spike_buffer.front()
            .map(|e| e.timestamp_us < cutoff_time)
            .unwrap_or(false) {
            self.spike_buffer.pop_front();
        }
        
        // Decode based on strategy
        match self.decoding_strategy {
            DecodingStrategy::RateDecoding => self.rate_decode()?,
            DecodingStrategy::PopulationVector => self.population_vector_decode()?,
            DecodingStrategy::TemporalPattern => self.temporal_pattern_decode()?,
            DecodingStrategy::BayesianDecoding => self.bayesian_decode()?,
        }
        
        Ok(self.output_buffer.clone())
    }

    fn rate_decode(&mut self) -> Result<()> {
        // Count spikes per neuron and convert to firing rates
        let mut spike_counts: HashMap<NeuronId, usize> = HashMap::new();
        
        for event in &self.spike_buffer {
            *spike_counts.entry(event.neuron_id).or_insert(0) += 1;
        }
        
        // Convert to output embeddings
        for (neuron_id, count) in spike_counts {
            let output_row = neuron_id.unit_id % self.output_buffer.nrows();
            let output_col = neuron_id.local_id % self.output_buffer.ncols();
            
            let rate = count as f32 / 100.0; // Normalize by time window (100ms)
            self.output_buffer[[output_row, output_col]] = rate;
        }
        
        Ok(())
    }

    fn population_vector_decode(&mut self) -> Result<()> {
        // Group spikes by processing unit (representing graph nodes)
        let mut unit_spikes: HashMap<usize, Vec<&SpikeEvent>> = HashMap::new();
        
        for event in &self.spike_buffer {
            unit_spikes.entry(event.neuron_id.unit_id)
                .or_insert_with(Vec::new)
                .push(event);
        }
        
        // Decode each unit's population activity
        for (unit_id, spikes) in unit_spikes {
            if unit_id >= self.output_buffer.nrows() { continue; }
            
            // Calculate population vector
            let neurons_per_unit = self.config.neurons_per_unit;
            let mut activity_vector = vec![0.0; neurons_per_unit];
            
            for spike in spikes {
                if spike.neuron_id.local_id < neurons_per_unit {
                    activity_vector[spike.neuron_id.local_id] += spike.spike_amplitude as f32 / 1000.0;
                }
            }
            
            // Compute population vector direction
            let mut weighted_sum = 0.0;
            let mut total_activity = 0.0;
            
            for (i, &activity) in activity_vector.iter().enumerate() {
                let preferred_direction = (i as f32) / (neurons_per_unit as f32 - 1.0) * 2.0 - 1.0;
                weighted_sum += activity * preferred_direction;
                total_activity += activity;
            }
            
            let decoded_value = if total_activity > 0.0 {
                weighted_sum / total_activity
            } else {
                0.0
            };
            
            // Store in output buffer
            for col in 0..self.output_buffer.ncols().min(16) {
                self.output_buffer[[unit_id, col]] = decoded_value + (col as f32 - 8.0) * 0.1;
            }
        }
        
        Ok(())
    }

    fn temporal_pattern_decode(&mut self) -> Result<()> {
        // Analyze spike timing patterns
        let mut timing_patterns: HashMap<usize, Vec<u64>> = HashMap::new();
        
        for event in &self.spike_buffer {
            timing_patterns.entry(event.neuron_id.unit_id)
                .or_insert_with(Vec::new)
                .push(event.timestamp_us);
        }
        
        // Decode temporal patterns to embeddings
        for (unit_id, mut timestamps) in timing_patterns {
            if unit_id >= self.output_buffer.nrows() { continue; }
            
            timestamps.sort_unstable();
            
            // Calculate inter-spike intervals
            let intervals: Vec<f32> = timestamps.windows(2)
                .map(|w| (w[1] - w[0]) as f32 / 1000.0) // Convert to milliseconds
                .collect();
            
            // Use statistical measures of intervals as features
            if !intervals.is_empty() {
                let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
                let variance = intervals.iter()
                    .map(|&x| (x - mean_interval).powi(2))
                    .sum::<f32>() / intervals.len() as f32;
                
                let cv = if mean_interval > 0.0 { variance.sqrt() / mean_interval } else { 0.0 };
                
                // Store temporal features
                if self.output_buffer.ncols() >= 3 {
                    self.output_buffer[[unit_id, 0]] = mean_interval / 100.0; // Normalize
                    self.output_buffer[[unit_id, 1]] = variance / 1000.0;
                    self.output_buffer[[unit_id, 2]] = cv;
                }
            }
        }
        
        Ok(())
    }

    fn bayesian_decode(&mut self) -> Result<()> {
        // Simplified Bayesian decoding using maximum likelihood
        // In practice, would use proper Bayesian inference
        
        // Count spike patterns
        let mut pattern_counts: HashMap<(usize, u16), f32> = HashMap::new();
        
        for event in &self.spike_buffer {
            let pattern_key = (event.neuron_id.unit_id, event.spike_amplitude / 100); // Quantize amplitude
            *pattern_counts.entry(pattern_key).or_insert(0.0) += 1.0;
        }
        
        // Convert to posterior probabilities (simplified)
        for ((unit_id, amplitude_bin), count) in pattern_counts {
            if unit_id >= self.output_buffer.nrows() { continue; }
            
            let col = (amplitude_bin as usize) % self.output_buffer.ncols();
            
            // Simple likelihood model
            let likelihood = count / self.spike_buffer.len() as f32;
            self.output_buffer[[unit_id, col]] = likelihood;
        }
        
        Ok(())
    }
}

/// Main Neuromorphic Computing Interface
pub struct NeuromorphicComputingInterface {
    config: NeuromorphicConfig,
    processing_units: Vec<NeuromorphicProcessingUnit>,
    spike_encoder: GraphSpikeEncoder,
    spike_decoder: SpikeGraphDecoder,
    event_router: EventRouter,
    performance_monitor: NeuromorphicPerformanceMonitor,
    power_manager: PowerManager,
}

impl NeuromorphicComputingInterface {
    pub fn new(config: NeuromorphicConfig) -> Result<Self> {
        let mut processing_units = Vec::with_capacity(config.num_processing_units);
        
        for unit_id in 0..config.num_processing_units {
            let unit = NeuromorphicProcessingUnit::new(unit_id, &config)?;
            processing_units.push(unit);
        }
        
        let spike_encoder = GraphSpikeEncoder::new(config.clone());
        let spike_decoder = SpikeGraphDecoder::new(config.clone(), (config.num_processing_units, 128));
        
        Ok(Self {
            config: config.clone(),
            processing_units,
            spike_encoder,
            spike_decoder,
            event_router: EventRouter::new(&config),
            performance_monitor: NeuromorphicPerformanceMonitor::new(),
            power_manager: PowerManager::new(&config),
        })
    }

    /// Process graph using neuromorphic computing
    #[instrument(skip(self, graph))]
    pub async fn process_graph_neuromorphic(&mut self, graph: &Graph) -> Result<NeuromorphicProcessingResult> {
        let start_time = Instant::now();
        info!("Starting neuromorphic graph processing for {} nodes, {} edges", 
              graph.node_count(), graph.edge_count());

        // Encode graph to spike events
        let spike_events = self.spike_encoder.encode_graph_to_spikes(graph)?;
        debug!("Encoded graph to {} spike events", spike_events.len());

        // Route events to processing units
        let routed_events = self.event_router.route_events(spike_events).await?;
        
        // Process events in parallel across neuromorphic units
        let mut processing_tasks = Vec::new();
        
        for (unit_id, unit_events) in routed_events {
            if let Some(unit) = self.processing_units.get_mut(unit_id) {
                // Add events to unit's event queue
                for event in unit_events {
                    unit.event_queue.entry(event.timestamp_us)
                        .or_insert_with(Vec::new)
                        .push(event);
                }
                
                // Process events
                let processing_result = unit.process_events(&self.config)?;
                processing_tasks.push((unit_id, processing_result));
            }
        }
        
        // Collect all spike outputs
        let mut all_output_spikes = Vec::new();
        let mut total_energy = 0.0;
        let mut total_spikes = 0;
        
        for (unit_id, result) in processing_tasks {
            total_energy += result.energy_consumed;
            total_spikes += result.spikes_fired;
            debug!("Unit {} processed {} events, fired {} spikes, consumed {:.2} pJ", 
                   unit_id, result.events_processed, result.spikes_fired, result.energy_consumed);
            
            // Extract output spikes from this unit
            if let Some(unit) = self.processing_units.get(unit_id) {
                for neuron in &unit.neurons {
                    if let Some(spike_time) = neuron.last_spike_time {
                        all_output_spikes.push(SpikeEvent {
                            timestamp_us: spike_time,
                            neuron_id: neuron.id,
                            spike_amplitude: (neuron.membrane_potential.abs() * 10.0) as u16,
                            event_type: match neuron.neuron_type {
                                NeuronType::Excitatory => SpikeEventType::ExcitatorySpike,
                                NeuronType::Inhibitory => SpikeEventType::InhibitorySpike,
                                _ => SpikeEventType::ExcitatorySpike,
                            },
                        });
                    }
                }
            }
        }
        
        // Decode spike outputs to graph embeddings
        let embeddings = self.spike_decoder.decode_spikes_to_embeddings(&all_output_spikes)?;
        
        let processing_time = start_time.elapsed();
        
        // Update performance monitoring
        self.performance_monitor.record_processing_cycle(
            graph.node_count(),
            graph.edge_count(),
            total_spikes,
            total_energy,
            processing_time,
        );
        
        // Power management
        self.power_manager.update_power_state(total_energy, processing_time).await?;
        
        let result = NeuromorphicProcessingResult {
            embeddings,
            spike_statistics: SpikeStatistics {
                total_input_events: spike_events.len(),
                total_output_spikes: all_output_spikes.len(),
                spikes_per_neuron: total_spikes as f64 / (self.config.num_processing_units * self.config.neurons_per_unit) as f64,
                average_spike_rate: total_spikes as f64 / processing_time.as_secs_f64(),
            },
            energy_consumption: total_energy,
            processing_time,
            neuromorphic_efficiency: total_spikes as f64 / total_energy.max(1e-9), // spikes per picojoule
        };
        
        info!("Neuromorphic processing completed: {:.2} pJ energy, {} output spikes, {:.2} spikes/pJ efficiency",
              total_energy, all_output_spikes.len(), result.neuromorphic_efficiency);
        
        Ok(result)
    }

    /// Get current neuromorphic system statistics
    pub fn get_system_statistics(&self) -> NeuromorphicSystemStats {
        let mut total_neurons = 0;
        let mut total_synapses = 0;
        let mut total_energy = 0.0;
        
        for unit in &self.processing_units {
            total_neurons += unit.neurons.len();
            total_energy += unit.energy_consumed;
            
            for neuron in &unit.neurons {
                total_synapses += neuron.input_synapses.len();
            }
        }
        
        NeuromorphicSystemStats {
            total_processing_units: self.processing_units.len(),
            total_neurons,
            total_synapses,
            total_energy_consumed: total_energy,
            average_spike_rate: self.performance_monitor.get_average_spike_rate(),
            system_efficiency: self.performance_monitor.get_system_efficiency(),
        }
    }
}

// Supporting structures and implementations

#[derive(Debug)]
pub struct NeuromorphicProcessingResult {
    pub embeddings: Array2<f32>,
    pub spike_statistics: SpikeStatistics,
    pub energy_consumption: f64,
    pub processing_time: Duration,
    pub neuromorphic_efficiency: f64,
}

#[derive(Debug)]
pub struct SpikeStatistics {
    pub total_input_events: usize,
    pub total_output_spikes: usize,
    pub spikes_per_neuron: f64,
    pub average_spike_rate: f64,
}

#[derive(Debug)]
pub struct NeuromorphicSystemStats {
    pub total_processing_units: usize,
    pub total_neurons: usize,
    pub total_synapses: usize,
    pub total_energy_consumed: f64,
    pub average_spike_rate: f64,
    pub system_efficiency: f64,
}

#[derive(Debug)]
pub struct EventRouter {
    routing_table: HashMap<usize, Vec<usize>>, // Graph node -> Processing units
    load_balancer: LoadBalancer,
}

impl EventRouter {
    pub fn new(config: &NeuromorphicConfig) -> Self {
        Self {
            routing_table: HashMap::new(),
            load_balancer: LoadBalancer::new(config.num_processing_units),
        }
    }

    pub async fn route_events(&mut self, events: Vec<SpikeEvent>) -> Result<HashMap<usize, Vec<SpikeEvent>>> {
        let mut routed_events: HashMap<usize, Vec<SpikeEvent>> = HashMap::new();
        
        for event in events {
            let target_unit = self.load_balancer.select_unit_for_event(&event).await;
            routed_events.entry(target_unit).or_insert_with(Vec::new).push(event);
        }
        
        Ok(routed_events)
    }
}

#[derive(Debug)]
pub struct LoadBalancer {
    unit_loads: Vec<f64>,
    round_robin_counter: usize,
}

impl LoadBalancer {
    pub fn new(num_units: usize) -> Self {
        Self {
            unit_loads: vec![0.0; num_units],
            round_robin_counter: 0,
        }
    }

    pub async fn select_unit_for_event(&mut self, _event: &SpikeEvent) -> usize {
        // Simple round-robin load balancing
        let selected_unit = self.round_robin_counter % self.unit_loads.len();
        self.round_robin_counter += 1;
        
        self.unit_loads[selected_unit] += 1.0;
        selected_unit
    }
}

#[derive(Debug)]
pub struct NeuromorphicPerformanceMonitor {
    processing_history: VecDeque<ProcessingRecord>,
    total_graphs_processed: usize,
}

#[derive(Debug)]
pub struct ProcessingRecord {
    timestamp: Instant,
    nodes: usize,
    edges: usize,
    spikes: usize,
    energy: f64,
    duration: Duration,
}

impl NeuromorphicPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            processing_history: VecDeque::with_capacity(1000),
            total_graphs_processed: 0,
        }
    }

    pub fn record_processing_cycle(&mut self, nodes: usize, edges: usize, spikes: usize, energy: f64, duration: Duration) {
        let record = ProcessingRecord {
            timestamp: Instant::now(),
            nodes,
            edges,
            spikes,
            energy,
            duration,
        };
        
        self.processing_history.push_back(record);
        if self.processing_history.len() > 1000 {
            self.processing_history.pop_front();
        }
        
        self.total_graphs_processed += 1;
    }

    pub fn get_average_spike_rate(&self) -> f64 {
        if self.processing_history.is_empty() {
            return 0.0;
        }
        
        let total_spikes: usize = self.processing_history.iter().map(|r| r.spikes).sum();
        let total_time: f64 = self.processing_history.iter()
            .map(|r| r.duration.as_secs_f64())
            .sum();
        
        if total_time > 0.0 {
            total_spikes as f64 / total_time
        } else {
            0.0
        }
    }

    pub fn get_system_efficiency(&self) -> f64 {
        if self.processing_history.is_empty() {
            return 0.0;
        }
        
        let total_spikes: usize = self.processing_history.iter().map(|r| r.spikes).sum();
        let total_energy: f64 = self.processing_history.iter().map(|r| r.energy).sum();
        
        if total_energy > 0.0 {
            total_spikes as f64 / total_energy
        } else {
            0.0
        }
    }
}

#[derive(Debug)]
pub struct PowerManager {
    config: NeuromorphicConfig,
    power_budget: f64, // picojoules per second
    current_power_draw: f64,
    power_scaling_factor: f64,
}

impl PowerManager {
    pub fn new(config: &NeuromorphicConfig) -> Self {
        let total_neurons = config.num_processing_units * config.neurons_per_unit;
        let power_budget = total_neurons as f64 * 0.1; // 0.1 pJ/s per neuron
        
        Self {
            config: config.clone(),
            power_budget,
            current_power_draw: 0.0,
            power_scaling_factor: 1.0,
        }
    }

    pub async fn update_power_state(&mut self, energy_consumed: f64, duration: Duration) -> Result<()> {
        self.current_power_draw = energy_consumed / duration.as_secs_f64();
        
        // Adjust power scaling if exceeding budget
        if self.current_power_draw > self.power_budget {
            self.power_scaling_factor *= 0.9; // Reduce by 10%
            warn!("Power consumption exceeded budget, scaling down by 10%");
        } else if self.current_power_draw < self.power_budget * 0.5 {
            self.power_scaling_factor *= 1.05; // Increase by 5%
            self.power_scaling_factor = self.power_scaling_factor.min(1.0);
        }
        
        Ok(())
    }
}

// Poisson distribution for spike generation
#[derive(Debug)]
struct Poisson {
    lambda: f64,
}

impl Poisson {
    fn new(lambda: f64) -> Result<Self, &'static str> {
        if lambda <= 0.0 {
            Err("Lambda must be positive")
        } else {
            Ok(Self { lambda })
        }
    }
}

impl Distribution<u32> for Poisson {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
        // Knuth's algorithm for Poisson sampling
        let l = (-self.lambda).exp();
        let mut k = 0;
        let mut p = 1.0;
        
        loop {
            k += 1;
            p *= rng.gen::<f64>();
            if p <= l {
                break;
            }
        }
        
        k - 1
    }
}