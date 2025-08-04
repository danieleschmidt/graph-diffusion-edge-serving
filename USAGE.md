# DGDM Server Usage Guide

## Quick Start

### 1. Build and Run Locally

```bash
# Build the server (CPU-only)
./scripts/build.sh

# Run with default configuration
./target/release/dgdm-server

# Run with custom configuration
./target/release/dgdm-server --config config.toml --log-level debug
```

### 2. Docker Deployment

```bash
# Build and deploy with Docker Compose
./scripts/deploy.sh

# With TPU support
./scripts/deploy.sh --tpu

# With monitoring stack
./scripts/deploy.sh --monitoring
```

### 3. WASM Development

```bash
# Build WASM module
./scripts/build.sh --wasm

# Serve demo locally
python3 -m http.server 8000
# Open http://localhost:8000/examples/wasm_demo.html
```

## API Usage

### REST API Examples

#### Health Check
```bash
curl http://localhost:8080/health
```

#### Process Single Graph
```bash
curl -X POST http://localhost:8080/diffuse \
  -H "Content-Type: application/json" \
  -d '{
    "graph": {
      "nodes": [
        {"id": 1, "features": [0.1, 0.2, 0.3]},
        {"id": 2, "features": [0.4, 0.5, 0.6]}
      ],
      "edges": [
        {"source": 1, "target": 2, "weight": 0.8}
      ]
    },
    "return_metrics": true
  }'
```

#### Batch Processing
```bash
curl -X POST http://localhost:8080/batch \
  -H "Content-Type: application/json" \
  -d '[
    {
      "graph": {
        "nodes": [{"id": 1, "features": [1.0, 0.0]}],
        "edges": []
      }
    },
    {
      "graph": {
        "nodes": [{"id": 1, "features": [0.0, 1.0]}],
        "edges": []
      }
    }
  ]'
```

### JavaScript/WASM Usage

```javascript
import init, { DGDMClient } from './pkg/graph_diffusion_edge.js';

async function main() {
    await init();
    
    const client = new DGDMClient();
    
    // Create a simple graph
    const graph = {
        nodes: [
            { id: 1, features: [0.1, 0.2] },
            { id: 2, features: [0.3, 0.4] }
        ],
        edges: [
            { source: 1, target: 2, weight: 1.0 }
        ]
    };
    
    // Process the graph
    const result = client.process_graph(graph);
    console.log('Embeddings:', result.embeddings);
    console.log('Processing time:', result.processing_time_ms);
}
```

### Python Client Example

```python
import requests
import json

class DGDMClient:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
    
    def process_graph(self, nodes, edges, return_metrics=False):
        payload = {
            "graph": {
                "nodes": nodes,
                "edges": edges
            },
            "return_metrics": return_metrics
        }
        
        response = requests.post(
            f"{self.base_url}/diffuse",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status()
        return response.json()

# Usage
client = DGDMClient()

nodes = [
    {"id": 1, "features": [0.1, 0.2, 0.3]},
    {"id": 2, "features": [0.4, 0.5, 0.6]},
    {"id": 3, "features": [0.7, 0.8, 0.9]}
]

edges = [
    {"source": 1, "target": 2, "weight": 1.0},
    {"source": 2, "target": 3, "weight": 0.5}
]

result = client.process_graph(nodes, edges, return_metrics=True)
print(f"Processed {len(result['embeddings'])} nodes in {result['processing_time_ms']:.2f}ms")
```

## Configuration

### Environment Variables

```bash
# Server configuration
export DGDM__SERVER__BIND_ADDRESS="0.0.0.0:8080"
export DGDM__SERVER__MAX_REQUEST_SIZE_MB=100

# Graph processing
export DGDM__GRAPH__MAX_NODES=1000000
export DGDM__GRAPH__DIFFUSION_STEPS=5

# TPU configuration (if available)
export DGDM__TPU__DEVICE_PATH="/dev/apex_0"
export DGDM__TPU__POWER_LIMIT_WATTS=4.0

# Logging
export RUST_LOG=info
export DGDM__LOGGING__LEVEL=debug
```

### Configuration File (config.toml)

```toml
[server]
bind_address = "0.0.0.0:8080"
max_request_size_mb = 100

[graph]
max_nodes = 1000000
max_edges = 10000000
diffusion_steps = 5
batch_size = 32

[processing]
temperature = 0.1
normalize_features = true

[logging]
level = "info"
enable_json = false
```

## Performance Optimization

### TPU Acceleration

```bash
# Check TPU availability
./target/release/dgdm-server tpu list

# Test TPU functionality
./target/release/dgdm-server tpu test --device /dev/apex_0

# Run with TPU support
./target/release/dgdm-server --enable-tpu --tpu-device /dev/apex_0
```

### Batch Processing

For better throughput, use batch processing:

```bash
# Process multiple graphs in a single request
curl -X POST http://localhost:8080/batch \
  -H "Content-Type: application/json" \
  -d '[graph1, graph2, graph3, ...]'
```

### Memory Management

Monitor memory usage:

```bash
# Get server metrics
curl http://localhost:8080/metrics

# Monitor with Prometheus/Grafana
./scripts/deploy.sh --monitoring
```

## Development

### Running Tests

```bash
# Run all tests
cargo test --no-default-features

# Run specific test
cargo test --no-default-features test_graph_creation

# Run benchmarks
cargo bench --features benchmark
```

### Building for Different Targets

```bash
# Build for Linux (default)
./scripts/build.sh --release

# Build with TPU support
./scripts/build.sh --tpu

# Build WASM module
./scripts/build.sh --wasm

# Cross-compile for ARM64
./scripts/build.sh --target aarch64-unknown-linux-gnu
```

### Debugging

```bash
# Run with debug logging
RUST_LOG=debug ./target/debug/dgdm-server

# Enable tracing
RUST_LOG=graph_diffusion_edge=trace ./target/debug/dgdm-server

# Memory debugging with Valgrind (Linux)
valgrind --tool=memcheck ./target/debug/dgdm-server
```

## Monitoring and Observability

### Metrics

The server exposes Prometheus metrics at `/metrics`:

- `dgdm_requests_total` - Total number of requests
- `dgdm_processing_duration_seconds` - Processing time histogram
- `dgdm_graphs_processed_total` - Total graphs processed
- `dgdm_processing_latency_seconds` - Latency distribution

### Health Checks

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed health information
curl http://localhost:8080/health | jq .
```

### Logging

Structured logging is available in JSON format:

```bash
# Enable JSON logging
DGDM__LOGGING__ENABLE_JSON=true ./target/release/dgdm-server
```

## Troubleshooting

### Common Issues

1. **TPU Not Detected**
   ```bash
   # Check TPU devices
   ls /dev/apex_*
   
   # Reset TPU
   sudo modprobe -r apex && sudo modprobe apex
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size
   export DGDM__GRAPH__BATCH_SIZE=16
   
   # Limit graph size
   export DGDM__GRAPH__MAX_NODES=100000
   ```

3. **Performance Issues**
   ```bash
   # Enable profiling
   DGDM__TPU__ENABLE_PROFILING=true ./target/release/dgdm-server
   
   # Check temperature
   curl http://localhost:8080/health | jq .tpu_temperature
   ```

### Getting Help

- Check logs: `docker-compose logs dgdm-server`
- View metrics: `curl http://localhost:8080/metrics`
- Server info: `curl http://localhost:8080/info`
- GitHub Issues: [Create an issue](https://github.com/danieleschmidt/graph-diffusion-edge-serving/issues)