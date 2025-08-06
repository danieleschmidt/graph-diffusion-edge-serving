//! Mock TPU runtime for testing without hardware dependencies

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn, error, instrument};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpuConfig {
    pub device_path: String,
    pub model_path: String,
    pub power_limit_watts: f32,
    pub temperature_limit_celsius: f32,
    pub enable_profiling: bool,
    pub batch_size: usize,
    pub quantization: String,
}

impl Default for TpuConfig {
    fn default() -> Self {
        Self {
            device_path: "/dev/apex_0".to_string(),
            model_path: "models/dgdm_quantized.tflite".to_string(),
            power_limit_watts: 4.0,
            temperature_limit_celsius: 85.0,
            enable_profiling: false,
            batch_size: 32,
            quantization: "int8".to_string(),
        }
    }
}

#[derive(Debug)]
pub struct EdgeTPU {
    config: TpuConfig,
    device_id: usize,
    model_loaded: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TpuStats {
    pub temperature_celsius: f32,
    pub power_consumption_watts: f32,
    pub utilization_percent: f32,
    pub memory_usage_mb: f32,
    pub inference_count: u64,
    pub avg_latency_ms: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct TpuDevice {
    pub device_path: String,
    pub device_id: usize,
    pub is_available: bool,
    pub firmware_version: String,
    pub max_power_watts: f32,
}

impl EdgeTPU {
    pub fn new() -> EdgeTPUBuilder {
        EdgeTPUBuilder::new()
    }

    #[instrument(skip(self))]
    pub fn initialize(&mut self) -> crate::Result<()> {
        info!("Initializing mock TPU at {}", self.config.device_path);

        // Mock device availability check
        info!("Mock TPU device found at {}", self.config.device_path);
        
        self.configure_power_limits()?;
        self.load_model()?;

        info!("Mock TPU initialized successfully");
        Ok(())
    }

    fn configure_power_limits(&self) -> crate::Result<()> {
        info!(
            "Configuring mock TPU power limit to {} watts", 
            self.config.power_limit_watts
        );
        
        if self.config.power_limit_watts > 6.0 {
            warn!("Power limit exceeds recommended maximum of 6W for Edge TPU");
        }

        Ok(())
    }

    fn load_model(&mut self) -> crate::Result<()> {
        let model_path = PathBuf::from(&self.config.model_path);
        
        info!("Loading mock model from {}", self.config.model_path);

        // Mock model loading - always succeeds
        self.model_loaded = true;
        info!("Mock model loaded successfully");
        
        Ok(())
    }

    #[instrument(skip(self, input_data))]
    pub fn inference(&self, input_data: &[f32]) -> crate::Result<Vec<f32>> {
        if !self.model_loaded {
            return Err(crate::error::Error::TpuRuntime(
                "Mock model not loaded".to_string()
            ));
        }

        let start_time = std::time::Instant::now();

        // Mock inference - simple transformation
        let output_size = input_data.len();
        let mut output = vec![0.0; output_size];
        
        for (i, &input) in input_data.iter().enumerate() {
            output[i] = input * 0.5 + 0.1; // Simple mock transformation
        }

        let inference_time = start_time.elapsed();
        info!(
            "Mock TPU inference completed in {:.2}ms", 
            inference_time.as_secs_f64() * 1000.0
        );

        Ok(output)
    }

    pub fn batch_inference(&self, batch_data: &[&[f32]]) -> crate::Result<Vec<Vec<f32>>> {
        if batch_data.len() > self.config.batch_size {
            return Err(crate::error::Error::TpuRuntime(
                format!(
                    "Batch size {} exceeds limit {}",
                    batch_data.len(),
                    self.config.batch_size
                )
            ));
        }

        let mut results = Vec::with_capacity(batch_data.len());
        
        for input in batch_data {
            let result = self.inference(input)?;
            results.push(result);
        }

        Ok(results)
    }

    pub fn get_stats(&self) -> crate::Result<TpuStats> {
        Ok(TpuStats {
            temperature_celsius: self.get_temperature()?,
            power_consumption_watts: self.get_power_consumption()?,
            utilization_percent: self.get_utilization()?,
            memory_usage_mb: self.get_memory_usage()?,
            inference_count: 42, // Mock value
            avg_latency_ms: 2.5, // Mock value
        })
    }

    fn get_temperature(&self) -> crate::Result<f32> {
        // Mock temperature - safe range
        let temp = 45.0 + (rand::random::<f32>() * 10.0);
        
        if temp > self.config.temperature_limit_celsius {
            warn!("Mock TPU temperature ({:.1}°C) exceeds limit ({:.1}°C)", 
                 temp, self.config.temperature_limit_celsius);
        }
        
        Ok(temp)
    }

    fn get_power_consumption(&self) -> crate::Result<f32> {
        // Mock power consumption
        let base_power = 1.5;
        let dynamic_power = rand::random::<f32>() * 2.5;
        Ok(base_power + dynamic_power)
    }

    fn get_utilization(&self) -> crate::Result<f32> {
        // Mock utilization
        Ok(rand::random::<f32>() * 100.0)
    }

    fn get_memory_usage(&self) -> crate::Result<f32> {
        // Mock memory usage in MB
        Ok(128.0 + rand::random::<f32>() * 64.0)
    }

    pub fn reset(&mut self) -> crate::Result<()> {
        info!("Resetting mock TPU");
        self.model_loaded = false;
        
        // Mock reset delay
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        self.initialize()
    }

    pub fn set_power_limit(&mut self, watts: f32) -> crate::Result<()> {
        if watts > 6.0 {
            return Err(crate::error::Error::TpuRuntime(
                "Power limit cannot exceed 6W for Edge TPU".to_string()
            ));
        }

        info!("Setting mock TPU power limit to {}W", watts);
        self.config.power_limit_watts = watts;
        Ok(())
    }
}

pub struct EdgeTPUBuilder {
    config: TpuConfig,
}

impl EdgeTPUBuilder {
    pub fn new() -> Self {
        Self {
            config: TpuConfig::default(),
        }
    }

    pub fn device_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.device_path = path.into();
        self
    }

    pub fn model_path<S: Into<String>>(mut self, path: S) -> Self {
        self.config.model_path = path.into();
        self
    }

    pub fn power_limit(mut self, watts: f32) -> Self {
        self.config.power_limit_watts = watts;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn quantization<S: Into<String>>(mut self, quant: S) -> Self {
        self.config.quantization = quant.into();
        self
    }

    pub fn build(self) -> crate::Result<EdgeTPU> {
        let mut tpu = EdgeTPU {
            config: self.config,
            device_id: 0,
            model_loaded: false,
        };

        tpu.initialize()?;
        Ok(tpu)
    }
}

impl Default for EdgeTPUBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn list_available_devices() -> crate::Result<Vec<TpuDevice>> {
    info!("Listing mock TPU devices");
    
    // Mock device list
    let devices = vec![
        TpuDevice {
            device_path: "/dev/apex_0".to_string(),
            device_id: 0,
            is_available: true,
            firmware_version: "1.0.0-mock".to_string(),
            max_power_watts: 6.0,
        }
    ];

    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpu_config_default() {
        let config = TpuConfig::default();
        assert_eq!(config.device_path, "/dev/apex_0");
        assert_eq!(config.power_limit_watts, 4.0);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_tpu_builder() {
        let builder = EdgeTPU::new()
            .device_path("/dev/apex_1")
            .power_limit(2.0)
            .batch_size(64);

        assert_eq!(builder.config.device_path, "/dev/apex_1");
        assert_eq!(builder.config.power_limit_watts, 2.0);
        assert_eq!(builder.config.batch_size, 64);
    }

    #[test]
    fn test_mock_inference() {
        let tpu = EdgeTPU::new().build().unwrap();
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let result = tpu.inference(&input_data).unwrap();
        
        assert_eq!(result.len(), input_data.len());
        // Check mock transformation: input * 0.5 + 0.1
        assert_eq!(result[0], 1.0 * 0.5 + 0.1);
    }

    #[test]
    fn test_power_limit_validation() {
        let mut tpu = EdgeTPU {
            config: TpuConfig::default(),
            device_id: 0,
            model_loaded: false,
        };

        // Valid power limit
        assert!(tpu.set_power_limit(4.0).is_ok());
        
        // Invalid power limit
        assert!(tpu.set_power_limit(10.0).is_err());
    }

    #[test]
    fn test_batch_inference() {
        let tpu = EdgeTPU::new().build().unwrap();
        
        let batch_data = vec![
            vec![1.0, 2.0].as_slice(),
            vec![3.0, 4.0].as_slice(),
        ];

        let results = tpu.batch_inference(&batch_data).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 2);
        assert_eq!(results[1].len(), 2);
    }
}