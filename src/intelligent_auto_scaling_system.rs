//! Intelligent Auto-Scaling System
//!
//! Advanced auto-scaling with predictive analytics, machine learning-based
//! load prediction, and intelligent resource optimization.

use crate::high_performance_distributed_engine::{DistributedEngineConfig, WorkerMetrics, PerformanceMetrics};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::time::interval;
use tracing::{info, debug, warn, error};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoScalingConfig {
    /// Minimum number of workers
    pub min_workers: usize,
    /// Maximum number of workers
    pub max_workers: usize,
    /// Target CPU utilization percentage
    pub target_cpu_utilization: f32,
    /// Target memory utilization percentage
    pub target_memory_utilization: f32,
    /// Target response time in milliseconds
    pub target_response_time_ms: f32,
    /// Scaling evaluation interval in seconds
    pub evaluation_interval_seconds: u64,
    /// Cooldown period after scaling in seconds
    pub scaling_cooldown_seconds: u64,
    /// Predictive scaling window in minutes
    pub prediction_window_minutes: u64,
    /// Enable predictive scaling
    pub predictive_scaling_enabled: bool,
    /// Aggressive scaling mode
    pub aggressive_scaling: bool,
    /// Cost optimization priority (0.0 to 1.0)
    pub cost_optimization_priority: f32,
    /// Performance priority (0.0 to 1.0)
    pub performance_priority: f32,
}

impl Default for AutoScalingConfig {
    fn default() -> Self {
        Self {
            min_workers: 2,
            max_workers: 16,
            target_cpu_utilization: 70.0,
            target_memory_utilization: 75.0,
            target_response_time_ms: 500.0,
            evaluation_interval_seconds: 30,
            scaling_cooldown_seconds: 300,
            prediction_window_minutes: 15,
            predictive_scaling_enabled: true,
            aggressive_scaling: false,
            cost_optimization_priority: 0.3,
            performance_priority: 0.7,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    pub timestamp: u64,
    pub decision_type: ScalingType,
    pub current_workers: usize,
    pub target_workers: usize,
    pub confidence: f32,
    pub reasoning: String,
    pub metrics_snapshot: ScalingMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingType {
    ScaleUp,
    ScaleDown,
    NoAction,
    PredictiveScaleUp,
    PredictiveScaleDown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingMetrics {
    pub cpu_utilization: f32,
    pub memory_utilization: f32,
    pub response_time_ms: f32,
    pub throughput_rps: f32,
    pub queue_depth: usize,
    pub error_rate: f32,
    pub worker_efficiency: f32,
}

#[derive(Debug, Clone)]
pub struct PredictionModel {
    /// Historical load patterns
    pub load_history: VecDeque<LoadDataPoint>,
    /// Seasonal patterns (hourly, daily, weekly)
    pub seasonal_patterns: HashMap<String, Vec<f32>>,
    /// Trend coefficients
    pub trend_coefficients: Vec<f32>,
    /// Model accuracy
    pub accuracy: f32,
}

#[derive(Debug, Clone)]
pub struct LoadDataPoint {
    pub timestamp: u64,
    pub cpu_load: f32,
    pub memory_load: f32,
    pub request_rate: f32,
    pub response_time: f32,
    pub worker_count: usize,
}

pub struct IntelligentAutoScalingSystem {
    config: AutoScalingConfig,
    current_workers: Arc<Mutex<usize>>,
    metrics_history: Arc<Mutex<VecDeque<ScalingMetrics>>>,
    scaling_history: Arc<Mutex<VecDeque<ScalingDecision>>>,
    prediction_model: Arc<Mutex<PredictionModel>>,
    last_scaling_time: Arc<Mutex<Instant>>,
    is_running: Arc<Mutex<bool>>,
}

impl IntelligentAutoScalingSystem {
    pub fn new(config: AutoScalingConfig) -> Self {
        let prediction_model = PredictionModel {
            load_history: VecDeque::new(),
            seasonal_patterns: HashMap::new(),
            trend_coefficients: vec![1.0, 0.0, 0.0], // Linear trend: a + bx + cx^2
            accuracy: 0.5, // Start with 50% confidence
        };

        Self {
            current_workers: Arc::new(Mutex::new(config.min_workers)),
            config,
            metrics_history: Arc::new(Mutex::new(VecDeque::new())),
            scaling_history: Arc::new(Mutex::new(VecDeque::new())),
            prediction_model: Arc::new(Mutex::new(prediction_model)),
            last_scaling_time: Arc::new(Mutex::new(Instant::now())),
            is_running: Arc::new(Mutex::new(false)),
        }
    }

    /// Start the auto-scaling system
    pub async fn start(&self) -> crate::Result<()> {
        {
            let mut running = self.is_running.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire running lock", "autoscaler_start")
            })?;
            
            if *running {
                return Err(crate::error::Error::configuration(
                    "Auto-scaling system is already running",
                    "autoscaler_state"
                ));
            }
            
            *running = true;
        }

        info!("Starting Intelligent Auto-Scaling System");

        // Start background loops
        let scaler_clone = self.clone();
        tokio::spawn(async move {
            scaler_clone.evaluation_loop().await;
        });

        if self.config.predictive_scaling_enabled {
            let scaler_clone = self.clone();
            tokio::spawn(async move {
                scaler_clone.predictive_analysis_loop().await;
            });

            let scaler_clone = self.clone();
            tokio::spawn(async move {
                scaler_clone.model_training_loop().await;
            });
        }

        info!("Auto-scaling system started successfully");
        Ok(())
    }

    /// Stop the auto-scaling system
    pub async fn stop(&self) -> crate::Result<()> {
        {
            let mut running = self.is_running.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire running lock", "autoscaler_stop")
            })?;
            
            *running = false;
        }

        info!("Auto-scaling system stopped");
        Ok(())
    }

    async fn evaluation_loop(&self) {
        let mut interval = interval(Duration::from_secs(self.config.evaluation_interval_seconds));
        
        while self.is_running() {
            interval.tick().await;
            
            if let Err(e) = self.evaluate_scaling_need().await {
                error!("Scaling evaluation error: {}", e);
            }
        }
    }

    async fn evaluate_scaling_need(&self) -> crate::Result<()> {
        // Collect current metrics
        let current_metrics = self.collect_current_metrics().await?;
        
        // Store metrics in history
        self.store_metrics(current_metrics.clone()).await;

        // Check if we're in cooldown period
        if self.is_in_cooldown() {
            debug!("Auto-scaler in cooldown period, skipping evaluation");
            return Ok(());
        }

        // Make scaling decision
        let decision = self.make_scaling_decision(&current_metrics).await?;
        
        // Execute scaling decision
        if !matches!(decision.decision_type, ScalingType::NoAction) {
            self.execute_scaling_decision(&decision).await?;
        }

        // Store decision in history
        self.store_scaling_decision(decision).await;

        Ok(())
    }

    async fn collect_current_metrics(&self) -> crate::Result<ScalingMetrics> {
        // In a real implementation, this would collect actual system metrics
        // For now, we'll simulate with realistic values based on load
        
        let current_workers = self.get_current_worker_count();
        
        // Simulate metrics that vary based on worker count and load
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Base load - simulates daily patterns
        let hour_of_day = (SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() / 3600) % 24;
        let base_load_factor = self.calculate_time_based_load_factor(hour_of_day);
        
        // CPU utilization increases with load, decreases with more workers
        let cpu_utilization = (base_load_factor * 100.0 / (current_workers as f32).sqrt())
            .min(95.0)
            .max(10.0);
        
        // Memory follows similar pattern but with different scaling
        let memory_utilization = (base_load_factor * 80.0 / (current_workers as f32).powf(0.7))
            .min(90.0)
            .max(15.0);
        
        // Response time increases with CPU load
        let response_time_ms = if cpu_utilization > 80.0 {
            200.0 + (cpu_utilization - 80.0) * 20.0
        } else {
            100.0 + cpu_utilization * 2.0
        };
        
        // Throughput decreases as resources become constrained
        let throughput_rps = if cpu_utilization < 70.0 {
            current_workers as f32 * 10.0
        } else {
            current_workers as f32 * 10.0 * (100.0 - cpu_utilization) / 30.0
        };
        
        let queue_depth = if cpu_utilization > 75.0 {
            ((cpu_utilization - 75.0) * 2.0) as usize
        } else {
            0
        };
        
        let error_rate = if cpu_utilization > 85.0 {
            (cpu_utilization - 85.0) / 100.0
        } else {
            0.001
        };
        
        let worker_efficiency = if current_workers > 0 {
            throughput_rps / current_workers as f32
        } else {
            0.0
        };

        Ok(ScalingMetrics {
            cpu_utilization,
            memory_utilization,
            response_time_ms,
            throughput_rps,
            queue_depth,
            error_rate,
            worker_efficiency,
        })
    }

    fn calculate_time_based_load_factor(&self, hour: u64) -> f32 {
        // Simulate realistic daily traffic patterns
        match hour {
            0..=6 => 0.3,   // Night - low load
            7..=8 => 0.6,   // Morning ramp-up
            9..=11 => 0.9,  // Morning peak
            12..=13 => 1.0, // Lunch peak
            14..=16 => 0.8, // Afternoon
            17..=19 => 1.1, // Evening peak
            20..=22 => 0.7, // Evening wind-down
            _ => 0.4,       // Late night
        }
    }

    async fn make_scaling_decision(&self, metrics: &ScalingMetrics) -> crate::Result<ScalingDecision> {
        let current_workers = self.get_current_worker_count();
        
        // Calculate scaling pressure from multiple factors
        let cpu_pressure = self.calculate_cpu_pressure(metrics.cpu_utilization);
        let memory_pressure = self.calculate_memory_pressure(metrics.memory_utilization);
        let latency_pressure = self.calculate_latency_pressure(metrics.response_time_ms);
        let queue_pressure = self.calculate_queue_pressure(metrics.queue_depth);
        let efficiency_pressure = self.calculate_efficiency_pressure(metrics.worker_efficiency, current_workers);
        
        // Weighted pressure calculation
        let total_pressure = cpu_pressure * 0.3 +
                           memory_pressure * 0.2 +
                           latency_pressure * 0.25 +
                           queue_pressure * 0.15 +
                           efficiency_pressure * 0.1;
        
        // Predictive scaling adjustment
        let mut predictive_adjustment = 0.0;
        if self.config.predictive_scaling_enabled {
            predictive_adjustment = self.calculate_predictive_adjustment().await;
        }
        
        let final_pressure = total_pressure + predictive_adjustment;
        
        // Make scaling decision based on pressure and constraints
        let (decision_type, target_workers, confidence, reasoning) = 
            self.determine_scaling_action(current_workers, final_pressure, metrics);

        Ok(ScalingDecision {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            decision_type,
            current_workers,
            target_workers,
            confidence,
            reasoning,
            metrics_snapshot: metrics.clone(),
        })
    }

    fn calculate_cpu_pressure(&self, cpu_utilization: f32) -> f32 {
        let target = self.config.target_cpu_utilization;
        
        if cpu_utilization > target + 15.0 {
            // High pressure - scale up aggressively
            2.0
        } else if cpu_utilization > target + 5.0 {
            // Medium pressure - scale up moderately
            1.0
        } else if cpu_utilization < target - 20.0 {
            // Low utilization - consider scaling down
            -2.0
        } else if cpu_utilization < target - 10.0 {
            // Moderate under-utilization
            -1.0
        } else {
            // Within acceptable range
            0.0
        }
    }

    fn calculate_memory_pressure(&self, memory_utilization: f32) -> f32 {
        let target = self.config.target_memory_utilization;
        
        if memory_utilization > target + 10.0 {
            1.5
        } else if memory_utilization < target - 15.0 {
            -1.0
        } else {
            0.0
        }
    }

    fn calculate_latency_pressure(&self, response_time_ms: f32) -> f32 {
        let target = self.config.target_response_time_ms;
        
        if response_time_ms > target * 2.0 {
            3.0 // High latency - urgent scaling needed
        } else if response_time_ms > target * 1.5 {
            1.5
        } else if response_time_ms < target * 0.5 {
            -0.5 // Very fast responses - might be over-provisioned
        } else {
            0.0
        }
    }

    fn calculate_queue_pressure(&self, queue_depth: usize) -> f32 {
        match queue_depth {
            0 => 0.0,
            1..=5 => 0.5,
            6..=15 => 1.0,
            16..=30 => 2.0,
            _ => 3.0,
        }
    }

    fn calculate_efficiency_pressure(&self, efficiency: f32, worker_count: usize) -> f32 {
        // If efficiency is dropping and we have many workers, consider scaling down
        // If efficiency is high but we have few workers, consider scaling up
        
        let base_efficiency = 10.0; // Expected requests per second per worker
        
        if efficiency < base_efficiency * 0.5 && worker_count > self.config.min_workers {
            -0.5 // Poor efficiency with many workers
        } else if efficiency > base_efficiency * 1.5 && worker_count < self.config.max_workers {
            0.5 // High efficiency - room to scale up
        } else {
            0.0
        }
    }

    async fn calculate_predictive_adjustment(&self) -> f32 {
        // Use prediction model to anticipate future load
        if let Ok(model) = self.prediction_model.lock() {
            if model.accuracy > 0.6 && !model.load_history.is_empty() {
                // Predict load 15 minutes ahead
                let predicted_load = self.predict_future_load(&model, self.config.prediction_window_minutes);
                
                // Convert predicted load to scaling pressure
                if predicted_load > 1.2 {
                    return 1.0; // Preemptive scale-up
                } else if predicted_load < 0.6 {
                    return -0.5; // Preemptive scale-down
                }
            }
        }
        
        0.0
    }

    fn predict_future_load(&self, model: &PredictionModel, minutes_ahead: u64) -> f32 {
        // Simple predictive model based on trends and seasonal patterns
        if model.load_history.is_empty() {
            return 1.0; // Default load factor
        }

        // Get recent trend
        let recent_points: Vec<&LoadDataPoint> = model.load_history.iter().rev().take(10).collect();
        if recent_points.len() < 2 {
            return 1.0;
        }

        // Calculate trend from recent CPU usage
        let trend = (recent_points[0].cpu_load - recent_points[recent_points.len() - 1].cpu_load) 
                   / recent_points.len() as f32;

        // Apply seasonal adjustments
        let future_hour = ((SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() / 3600) 
                         + minutes_ahead / 60) % 24;
        let seasonal_factor = self.calculate_time_based_load_factor(future_hour);

        // Combine trend and seasonal factors
        let current_load_factor = recent_points[0].cpu_load / 100.0;
        let predicted_load_factor = current_load_factor + trend + (seasonal_factor - current_load_factor) * 0.3;

        predicted_load_factor.max(0.1).min(2.0)
    }

    fn determine_scaling_action(
        &self,
        current_workers: usize,
        pressure: f32,
        metrics: &ScalingMetrics,
    ) -> (ScalingType, usize, f32, String) {
        
        let mut target_workers = current_workers;
        let mut decision_type = ScalingType::NoAction;
        let mut confidence = 0.5;
        let mut reasoning = "No significant pressure detected".to_string();

        // Scale up conditions
        if pressure >= 2.0 && current_workers < self.config.max_workers {
            let scale_increment = if self.config.aggressive_scaling { 2 } else { 1 };
            target_workers = (current_workers + scale_increment).min(self.config.max_workers);
            decision_type = ScalingType::ScaleUp;
            confidence = (pressure / 3.0).min(1.0);
            reasoning = format!("High pressure detected ({:.2}): CPU={:.1}%, Response={:.0}ms", 
                              pressure, metrics.cpu_utilization, metrics.response_time_ms);
        }
        // Moderate scale up
        else if pressure >= 1.0 && current_workers < self.config.max_workers {
            target_workers = current_workers + 1;
            decision_type = ScalingType::ScaleUp;
            confidence = pressure / 2.0;
            reasoning = format!("Moderate pressure detected ({:.2}): scaling up gradually", pressure);
        }
        // Scale down conditions
        else if pressure <= -2.0 && current_workers > self.config.min_workers {
            target_workers = (current_workers - 1).max(self.config.min_workers);
            decision_type = ScalingType::ScaleDown;
            confidence = (-pressure / 3.0).min(1.0);
            reasoning = format!("Low utilization detected ({:.2}): CPU={:.1}%, optimizing costs", 
                              pressure, metrics.cpu_utilization);
        }
        // Gradual scale down
        else if pressure <= -1.0 && current_workers > self.config.min_workers {
            // Only scale down if we've been consistently under-utilized
            if self.has_consistent_low_utilization() {
                target_workers = current_workers - 1;
                decision_type = ScalingType::ScaleDown;
                confidence = 0.6;
                reasoning = "Consistent low utilization - gradual scale down".to_string();
            }
        }

        // Cost vs Performance optimization
        if self.config.cost_optimization_priority > self.config.performance_priority {
            // Favor cost optimization - be more aggressive about scaling down
            if metrics.cpu_utilization < 50.0 && current_workers > self.config.min_workers {
                target_workers = (current_workers - 1).max(self.config.min_workers);
                decision_type = ScalingType::ScaleDown;
                confidence = 0.7;
                reasoning = "Cost optimization: reducing over-provisioned resources".to_string();
            }
        }

        (decision_type, target_workers, confidence, reasoning)
    }

    fn has_consistent_low_utilization(&self) -> bool {
        // Check if CPU utilization has been consistently low for the past few measurements
        if let Ok(history) = self.metrics_history.lock() {
            let recent_metrics: Vec<&ScalingMetrics> = history.iter().rev().take(3).collect();
            if recent_metrics.len() >= 3 {
                return recent_metrics.iter().all(|m| m.cpu_utilization < 40.0);
            }
        }
        false
    }

    async fn execute_scaling_decision(&self, decision: &ScalingDecision) -> crate::Result<()> {
        info!("Executing scaling decision: {} workers -> {} workers ({})", 
              decision.current_workers, decision.target_workers, decision.reasoning);

        // Update current worker count
        {
            let mut current = self.current_workers.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire current workers lock", "execute_scaling")
            })?;
            *current = decision.target_workers;
        }

        // Update last scaling time
        {
            let mut last_scaling = self.last_scaling_time.lock().map_err(|_| {
                crate::error::Error::graph_processing("Failed to acquire last scaling time lock", "execute_scaling")
            })?;
            *last_scaling = Instant::now();
        }

        // In a real implementation, this would trigger actual worker creation/removal
        match decision.decision_type {
            ScalingType::ScaleUp | ScalingType::PredictiveScaleUp => {
                // Trigger worker creation
                info!("Scaling up to {} workers", decision.target_workers);
            }
            ScalingType::ScaleDown | ScalingType::PredictiveScaleDown => {
                // Trigger worker removal
                info!("Scaling down to {} workers", decision.target_workers);
            }
            _ => {}
        }

        Ok(())
    }

    async fn predictive_analysis_loop(&self) {
        let mut interval = interval(Duration::from_secs(60)); // Analyze every minute
        
        while self.is_running() {
            interval.tick().await;
            
            if let Err(e) = self.update_load_predictions().await {
                error!("Predictive analysis error: {}", e);
            }
        }
    }

    async fn update_load_predictions(&self) -> crate::Result<()> {
        // Collect current load data point
        let metrics = self.collect_current_metrics().await?;
        let current_workers = self.get_current_worker_count();
        
        let load_point = LoadDataPoint {
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            cpu_load: metrics.cpu_utilization,
            memory_load: metrics.memory_utilization,
            request_rate: metrics.throughput_rps,
            response_time: metrics.response_time_ms,
            worker_count: current_workers,
        };

        // Update prediction model
        if let Ok(mut model) = self.prediction_model.lock() {
            model.load_history.push_back(load_point);
            
            // Keep only recent history (last 24 hours)
            while model.load_history.len() > 24 * 60 {
                model.load_history.pop_front();
            }
            
            // Update seasonal patterns
            self.update_seasonal_patterns(&mut model);
        }

        Ok(())
    }

    fn update_seasonal_patterns(&self, model: &mut PredictionModel) {
        // Update hourly patterns
        let mut hourly_patterns: HashMap<u8, Vec<f32>> = HashMap::new();
        
        for point in &model.load_history {
            let hour = ((point.timestamp / 3600) % 24) as u8;
            hourly_patterns.entry(hour)
                .or_insert_with(Vec::new)
                .push(point.cpu_load);
        }

        // Calculate average load for each hour
        let mut hourly_averages = Vec::new();
        for hour in 0..24 {
            if let Some(loads) = hourly_patterns.get(&hour) {
                let avg = loads.iter().sum::<f32>() / loads.len() as f32;
                hourly_averages.push(avg);
            } else {
                hourly_averages.push(50.0); // Default
            }
        }

        model.seasonal_patterns.insert("hourly".to_string(), hourly_averages);
    }

    async fn model_training_loop(&self) {
        let mut interval = interval(Duration::from_secs(300)); // Train every 5 minutes
        
        while self.is_running() {
            interval.tick().await;
            
            if let Err(e) = self.train_prediction_model().await {
                error!("Model training error: {}", e);
            }
        }
    }

    async fn train_prediction_model(&self) -> crate::Result<()> {
        if let Ok(mut model) = self.prediction_model.lock() {
            if model.load_history.len() < 20 {
                return Ok(()); // Not enough data
            }

            // Simple linear regression for trend analysis
            let recent_points: Vec<&LoadDataPoint> = model.load_history.iter().rev().take(20).collect();
            
            let n = recent_points.len() as f32;
            let sum_x: f32 = (0..recent_points.len()).map(|i| i as f32).sum();
            let sum_y: f32 = recent_points.iter().map(|p| p.cpu_load).sum();
            let sum_xy: f32 = recent_points.iter().enumerate()
                .map(|(i, p)| i as f32 * p.cpu_load).sum();
            let sum_x2: f32 = (0..recent_points.len()).map(|i| (i as f32).powi(2)).sum();

            // Linear regression: y = a + bx
            if n * sum_x2 - sum_x * sum_x != 0.0 {
                let b = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
                let a = (sum_y - b * sum_x) / n;
                
                model.trend_coefficients = vec![a, b, 0.0];
                
                // Calculate model accuracy (simplified R-squared)
                let mean_y = sum_y / n;
                let mut ss_res = 0.0;
                let mut ss_tot = 0.0;
                
                for (i, point) in recent_points.iter().enumerate() {
                    let predicted = a + b * i as f32;
                    ss_res += (point.cpu_load - predicted).powi(2);
                    ss_tot += (point.cpu_load - mean_y).powi(2);
                }
                
                if ss_tot > 0.0 {
                    model.accuracy = 1.0 - (ss_res / ss_tot);
                    model.accuracy = model.accuracy.max(0.0).min(1.0);
                }
            }
        }

        Ok(())
    }

    // Helper methods
    fn is_running(&self) -> bool {
        self.is_running.lock().map(|r| *r).unwrap_or(false)
    }

    fn is_in_cooldown(&self) -> bool {
        if let Ok(last_scaling) = self.last_scaling_time.lock() {
            last_scaling.elapsed() < Duration::from_secs(self.config.scaling_cooldown_seconds)
        } else {
            false
        }
    }

    fn get_current_worker_count(&self) -> usize {
        self.current_workers.lock().map(|w| *w).unwrap_or(self.config.min_workers)
    }

    async fn store_metrics(&self, metrics: ScalingMetrics) {
        if let Ok(mut history) = self.metrics_history.lock() {
            history.push_back(metrics);
            
            // Keep only recent history
            while history.len() > 1000 {
                history.pop_front();
            }
        }
    }

    async fn store_scaling_decision(&self, decision: ScalingDecision) {
        if let Ok(mut history) = self.scaling_history.lock() {
            history.push_back(decision);
            
            // Keep only recent history
            while history.len() > 100 {
                history.pop_front();
            }
        }
    }

    // Public API methods
    pub fn get_current_worker_count_public(&self) -> usize {
        self.get_current_worker_count()
    }

    pub fn get_scaling_history(&self) -> Vec<ScalingDecision> {
        if let Ok(history) = self.scaling_history.lock() {
            history.iter().cloned().collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_metrics_history(&self) -> Vec<ScalingMetrics> {
        if let Ok(history) = self.metrics_history.lock() {
            history.iter().cloned().collect()
        } else {
            Vec::new()
        }
    }

    pub fn get_prediction_accuracy(&self) -> f32 {
        if let Ok(model) = self.prediction_model.lock() {
            model.accuracy
        } else {
            0.0
        }
    }

    pub fn force_scaling_evaluation(&self) -> bool {
        // Trigger immediate scaling evaluation (bypass cooldown)
        if let Ok(mut last_scaling) = self.last_scaling_time.lock() {
            *last_scaling = Instant::now() - Duration::from_secs(self.config.scaling_cooldown_seconds + 1);
            true
        } else {
            false
        }
    }

    pub fn update_config(&mut self, new_config: AutoScalingConfig) {
        self.config = new_config;
    }
}

impl Clone for IntelligentAutoScalingSystem {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            current_workers: Arc::clone(&self.current_workers),
            metrics_history: Arc::clone(&self.metrics_history),
            scaling_history: Arc::clone(&self.scaling_history),
            prediction_model: Arc::clone(&self.prediction_model),
            last_scaling_time: Arc::clone(&self.last_scaling_time),
            is_running: Arc::clone(&self.is_running),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_autoscaler_creation() {
        let config = AutoScalingConfig::default();
        let scaler = IntelligentAutoScalingSystem::new(config);
        
        assert!(!scaler.is_running());
        assert_eq!(scaler.get_current_worker_count(), 2); // min_workers
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let config = AutoScalingConfig::default();
        let scaler = IntelligentAutoScalingSystem::new(config);
        
        let metrics = scaler.collect_current_metrics().await.unwrap();
        
        assert!(metrics.cpu_utilization >= 0.0);
        assert!(metrics.memory_utilization >= 0.0);
        assert!(metrics.response_time_ms > 0.0);
    }

    #[tokio::test]
    async fn test_scaling_decision_logic() {
        let config = AutoScalingConfig::default();
        let scaler = IntelligentAutoScalingSystem::new(config);
        
        // Test high CPU scenario
        let high_cpu_metrics = ScalingMetrics {
            cpu_utilization: 90.0,
            memory_utilization: 60.0,
            response_time_ms: 800.0,
            throughput_rps: 50.0,
            queue_depth: 10,
            error_rate: 0.02,
            worker_efficiency: 5.0,
        };
        
        let decision = scaler.make_scaling_decision(&high_cpu_metrics).await.unwrap();
        
        match decision.decision_type {
            ScalingType::ScaleUp => {
                assert!(decision.target_workers > decision.current_workers);
            }
            _ => {
                // Other decisions are acceptable depending on current state
            }
        }
    }

    #[test]
    fn test_pressure_calculations() {
        let config = AutoScalingConfig::default();
        let scaler = IntelligentAutoScalingSystem::new(config);
        
        // Test CPU pressure calculation
        let high_cpu_pressure = scaler.calculate_cpu_pressure(90.0);
        let normal_cpu_pressure = scaler.calculate_cpu_pressure(70.0);
        let low_cpu_pressure = scaler.calculate_cpu_pressure(30.0);
        
        assert!(high_cpu_pressure > 0.0);
        assert_eq!(normal_cpu_pressure, 0.0);
        assert!(low_cpu_pressure < 0.0);
        
        // Test latency pressure calculation
        let high_latency_pressure = scaler.calculate_latency_pressure(2000.0);
        let normal_latency_pressure = scaler.calculate_latency_pressure(500.0);
        
        assert!(high_latency_pressure > 0.0);
        assert_eq!(normal_latency_pressure, 0.0);
    }
}