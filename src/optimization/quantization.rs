//! Adaptive quantization for optimizing model performance and memory usage

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    pub target_precision: QuantizationPrecision,
    pub calibration_samples: usize,
    pub accuracy_threshold: f32,
    pub enable_mixed_precision: bool,
    pub per_layer_optimization: bool,
    pub dynamic_quantization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizationPrecision {
    FP32,
    FP16,
    INT8,
    INT4,
    MIXED,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            target_precision: QuantizationPrecision::INT8,
            calibration_samples: 1000,
            accuracy_threshold: 0.95,
            enable_mixed_precision: true,
            per_layer_optimization: true,
            dynamic_quantization: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizationStats {
    pub scale: f32,
    pub zero_point: i32,
    pub min_val: f32,
    pub max_val: f32,
    pub precision: QuantizationPrecision,
}

#[derive(Debug, Clone)]
pub struct LayerQuantizationInfo {
    pub layer_name: String,
    pub input_stats: QuantizationStats,
    pub weight_stats: QuantizationStats,
    pub output_stats: QuantizationStats,
    pub accuracy_impact: f32,
}

pub struct AdaptiveQuantizer {
    config: QuantizationConfig,
    calibration_data: HashMap<String, Vec<Array2<f32>>>,
    layer_stats: HashMap<String, LayerQuantizationInfo>,
}

impl AdaptiveQuantizer {
    pub fn new(config: QuantizationConfig) -> Self {
        info!("Initializing adaptive quantizer with precision: {:?}", config.target_precision);
        
        Self {
            config,
            calibration_data: HashMap::new(),
            layer_stats: HashMap::new(),
        }
    }

    pub fn add_calibration_data(&mut self, layer_name: String, data: Array2<f32>) {
        self.calibration_data
            .entry(layer_name)
            .or_insert_with(Vec::new)
            .push(data);
    }

    pub fn calibrate(&mut self) -> crate::Result<()> {
        info!("Starting quantization calibration with {} samples per layer", 
              self.config.calibration_samples);

        for (layer_name, data_samples) in &self.calibration_data {
            if data_samples.is_empty() {
                warn!("No calibration data for layer: {}", layer_name);
                continue;
            }

            // Aggregate statistics across all samples
            let (min_val, max_val) = self.compute_value_range(data_samples);
            let (scale, zero_point) = self.compute_quantization_params(
                min_val, 
                max_val, 
                &self.config.target_precision
            );

            let stats = QuantizationStats {
                scale,
                zero_point,
                min_val,
                max_val,
                precision: self.config.target_precision.clone(),
            };

            // For now, use the same stats for input, weight, and output
            // In a real implementation, these would be computed separately
            let layer_info = LayerQuantizationInfo {
                layer_name: layer_name.clone(),
                input_stats: stats.clone(),
                weight_stats: stats.clone(),
                output_stats: stats,
                accuracy_impact: self.estimate_accuracy_impact(data_samples, min_val, max_val),
            };

            self.layer_stats.insert(layer_name.clone(), layer_info);
        }

        // Optimize mixed precision if enabled
        if self.config.enable_mixed_precision {
            self.optimize_mixed_precision()?;
        }

        info!("Calibration completed for {} layers", self.layer_stats.len());
        Ok(())
    }

    pub fn quantize_array(&self, data: &Array2<f32>, layer_name: &str) -> crate::Result<Array2<i8>> {
        let stats = self.layer_stats.get(layer_name)
            .ok_or_else(|| crate::error::Error::graph_processing(
                format!("No quantization stats for layer: {}", layer_name),
                "quantization_stats_lookup"
            ))?;

        let quantized = data.map(|&value| {
            let quantized_value = ((value - stats.input_stats.min_val) / stats.input_stats.scale + 
                                   stats.input_stats.zero_point as f32).round();
            quantized_value.clamp(i8::MIN as f32, i8::MAX as f32) as i8
        });

        debug!("Quantized array for layer: {} ({:?})", layer_name, stats.input_stats.precision);
        Ok(quantized)
    }

    pub fn dequantize_array(&self, data: &Array2<i8>, layer_name: &str) -> crate::Result<Array2<f32>> {
        let stats = self.layer_stats.get(layer_name)
            .ok_or_else(|| crate::error::Error::graph_processing(
                format!("No quantization stats for layer: {}", layer_name),
                "quantization_stats_lookup"
            ))?;

        let dequantized = data.map(|&value| {
            (value as f32 - stats.input_stats.zero_point as f32) * stats.input_stats.scale + 
            stats.input_stats.min_val
        });

        Ok(dequantized)
    }

    pub fn quantize_weights(&self, weights: &Array2<f32>) -> Array2<i8> {
        // Simple symmetric quantization for weights
        let abs_max = weights.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let scale = abs_max / 127.0;

        weights.map(|&w| {
            let quantized = (w / scale).round();
            quantized.clamp(-128.0, 127.0) as i8
        })
    }

    pub fn get_layer_stats(&self, layer_name: &str) -> Option<&LayerQuantizationInfo> {
        self.layer_stats.get(layer_name)
    }

    pub fn get_compression_ratio(&self) -> f32 {
        match self.config.target_precision {
            QuantizationPrecision::FP32 => 1.0,
            QuantizationPrecision::FP16 => 2.0,
            QuantizationPrecision::INT8 => 4.0,
            QuantizationPrecision::INT4 => 8.0,
            QuantizationPrecision::MIXED => 2.5, // Average estimate
        }
    }

    pub fn export_quantization_info(&self) -> HashMap<String, serde_json::Value> {
        let mut info = HashMap::new();

        for (layer_name, layer_info) in &self.layer_stats {
            let layer_json = serde_json::json!({
                "precision": format!("{:?}", layer_info.input_stats.precision),
                "scale": layer_info.input_stats.scale,
                "zero_point": layer_info.input_stats.zero_point,
                "value_range": {
                    "min": layer_info.input_stats.min_val,
                    "max": layer_info.input_stats.max_val
                },
                "accuracy_impact": layer_info.accuracy_impact
            });
            info.insert(layer_name.clone(), layer_json);
        }

        info.insert("config".to_string(), serde_json::json!({
            "target_precision": format!("{:?}", self.config.target_precision),
            "compression_ratio": self.get_compression_ratio(),
            "mixed_precision": self.config.enable_mixed_precision
        }));

        info
    }

    fn compute_value_range(&self, data_samples: &[Array2<f32>]) -> (f32, f32) {
        let mut global_min = f32::INFINITY;
        let mut global_max = f32::NEG_INFINITY;

        for sample in data_samples {
            for &value in sample.iter() {
                if value < global_min {
                    global_min = value;
                }
                if value > global_max {
                    global_max = value;
                }
            }
        }

        // Handle edge cases
        if global_min == global_max {
            global_max = global_min + 1e-8;
        }

        (global_min, global_max)
    }

    fn compute_quantization_params(
        &self, 
        min_val: f32, 
        max_val: f32, 
        precision: &QuantizationPrecision
    ) -> (f32, i32) {
        match precision {
            QuantizationPrecision::INT8 => {
                let scale = (max_val - min_val) / 255.0;
                let zero_point = (-min_val / scale).round() as i32 - 128;
                (scale, zero_point)
            }
            QuantizationPrecision::INT4 => {
                let scale = (max_val - min_val) / 15.0;
                let zero_point = (-min_val / scale).round() as i32 - 8;
                (scale, zero_point)
            }
            _ => {
                // For floating point, return identity mapping
                (1.0, 0)
            }
        }
    }

    fn estimate_accuracy_impact(&self, data_samples: &[Array2<f32>], min_val: f32, max_val: f32) -> f32 {
        if data_samples.is_empty() {
            return 0.0;
        }

        let (scale, zero_point) = self.compute_quantization_params(
            min_val, 
            max_val, 
            &self.config.target_precision
        );

        let mut total_error = 0.0;
        let mut total_elements = 0;

        for sample in data_samples.iter().take(10) { // Sample a few for estimation
            for &original_value in sample.iter() {
                // Simulate quantization round-trip
                let quantized = ((original_value - min_val) / scale + zero_point as f32).round();
                let clamped = quantized.clamp(i8::MIN as f32, i8::MAX as f32);
                let dequantized = (clamped - zero_point as f32) * scale + min_val;
                
                let error = (original_value - dequantized).abs();
                total_error += error;
                total_elements += 1;
            }
        }

        if total_elements > 0 {
            let mean_error = total_error / total_elements as f32;
            let max_possible_value = max_val - min_val;
            
            if max_possible_value > 0.0 {
                1.0 - (mean_error / max_possible_value)
            } else {
                1.0
            }
        } else {
            1.0
        }
    }

    fn optimize_mixed_precision(&mut self) -> crate::Result<()> {
        info!("Optimizing mixed precision quantization");

        // Sort layers by accuracy impact
        let mut layer_impacts: Vec<_> = self.layer_stats.iter()
            .map(|(name, info)| (name.clone(), info.accuracy_impact))
            .collect();
        
        layer_impacts.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep most sensitive layers in higher precision
        let high_precision_count = (layer_impacts.len() as f32 * 0.3).ceil() as usize;
        
        for (i, (layer_name, _)) in layer_impacts.iter().enumerate() {
            if let Some(layer_info) = self.layer_stats.get_mut(layer_name) {
                if i < high_precision_count {
                    // Use FP16 for most sensitive layers
                    layer_info.input_stats.precision = QuantizationPrecision::FP16;
                    layer_info.weight_stats.precision = QuantizationPrecision::FP16;
                    layer_info.output_stats.precision = QuantizationPrecision::FP16;
                } else {
                    // Use INT8 for less sensitive layers
                    layer_info.input_stats.precision = QuantizationPrecision::INT8;
                    layer_info.weight_stats.precision = QuantizationPrecision::INT8;
                    layer_info.output_stats.precision = QuantizationPrecision::INT8;
                }
            }
        }

        info!("Mixed precision optimization completed: {} layers in FP16, {} layers in INT8",
              high_precision_count, layer_impacts.len() - high_precision_count);

        Ok(())
    }
}

pub fn quantize_activations_dynamic(activations: &Array2<f32>) -> (Array2<i8>, f32, i32) {
    let min_val = activations.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = activations.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    
    let scale = (max_val - min_val) / 255.0;
    let zero_point = (-min_val / scale).round() as i32 - 128;
    
    let quantized = activations.map(|&x| {
        let q = ((x - min_val) / scale + zero_point as f32).round();
        q.clamp(i8::MIN as f32, i8::MAX as f32) as i8
    });
    
    (quantized, scale, zero_point)
}

pub fn benchmark_quantization_performance(
    original: &Array2<f32>,
    quantized: &Array2<i8>,
    scale: f32,
    zero_point: i32,
    min_val: f32
) -> QuantizationBenchmark {
    let dequantized = quantized.map(|&q| {
        (q as f32 - zero_point as f32) * scale + min_val
    });
    
    let mse = (&dequantized - original).mapv(|x| x * x).sum() / original.len() as f32;
    let mae = (&dequantized - original).mapv(|x| x.abs()).sum() / original.len() as f32;
    
    let max_error = (&dequantized - original).mapv(|x| x.abs()).iter().cloned()
        .fold(0.0f32, f32::max);
    
    let snr = if mse > 0.0 {
        10.0 * (original.mapv(|x| x * x).sum() / (original.len() as f32 * mse)).log10()
    } else {
        f32::INFINITY
    };
    
    QuantizationBenchmark {
        mse,
        mae,
        max_error,
        snr_db: snr,
        compression_ratio: std::mem::size_of::<f32>() as f32 / std::mem::size_of::<i8>() as f32,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QuantizationBenchmark {
    pub mse: f32,           // Mean Squared Error
    pub mae: f32,           // Mean Absolute Error  
    pub max_error: f32,     // Maximum absolute error
    pub snr_db: f32,        // Signal-to-Noise Ratio in dB
    pub compression_ratio: f32, // Memory compression ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::default();
        assert!(matches!(config.target_precision, QuantizationPrecision::INT8));
        assert_eq!(config.calibration_samples, 1000);
        assert!(config.enable_mixed_precision);
    }

    #[test]
    fn test_adaptive_quantizer_creation() {
        let config = QuantizationConfig::default();
        let quantizer = AdaptiveQuantizer::new(config);
        assert_eq!(quantizer.layer_stats.len(), 0);
    }

    #[test]
    fn test_quantization_params_int8() {
        let config = QuantizationConfig::default();
        let quantizer = AdaptiveQuantizer::new(config);
        
        let min_val = -10.0;
        let max_val = 10.0;
        let (scale, zero_point) = quantizer.compute_quantization_params(
            min_val, 
            max_val, 
            &QuantizationPrecision::INT8
        );
        
        assert!((scale - (20.0 / 255.0)).abs() < 1e-6);
        assert!(zero_point >= i8::MIN as i32 && zero_point <= i8::MAX as i32);
    }

    #[test]
    fn test_dynamic_quantization() {
        let data = Array::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (quantized, scale, zero_point) = quantize_activations_dynamic(&data);
        
        assert_eq!(quantized.shape(), data.shape());
        assert!(scale > 0.0);
        assert!(zero_point >= i8::MIN as i32 && zero_point <= i8::MAX as i32);
    }

    #[test]
    fn test_quantization_round_trip() {
        let original = Array::from_shape_vec((2, 2), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();
        let (quantized, scale, zero_point) = quantize_activations_dynamic(&original);
        
        let dequantized = quantized.map(|&q| {
            (q as f32 - zero_point as f32) * scale + (-1.0) // min_val = -1.0
        });
        
        // Check that round-trip error is reasonable
        let max_error = (&dequantized - &original).mapv(|x| x.abs()).iter().cloned()
            .fold(0.0f32, f32::max);
        
        assert!(max_error < 0.5); // Allow some quantization error
    }

    #[test]
    fn test_compression_ratio() {
        let config = QuantizationConfig {
            target_precision: QuantizationPrecision::INT8,
            ..Default::default()
        };
        let quantizer = AdaptiveQuantizer::new(config);
        
        assert_eq!(quantizer.get_compression_ratio(), 4.0);
    }

    #[test]
    fn test_benchmark_quantization() {
        let original = Array::from_shape_vec((3, 3), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        let (quantized, scale, zero_point) = quantize_activations_dynamic(&original);
        
        let benchmark = benchmark_quantization_performance(
            &original, &quantized, scale, zero_point, 1.0
        );
        
        assert!(benchmark.compression_ratio > 1.0);
        assert!(benchmark.mse >= 0.0);
        assert!(benchmark.mae >= 0.0);
        assert!(benchmark.max_error >= 0.0);
    }
}