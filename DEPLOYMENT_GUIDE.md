# 🚀 DGDM Server Deployment Guide

## Overview

This guide covers production deployment of the Dynamic Graph Diffusion Models (DGDM) server, a high-performance graph processing system optimized for Edge TPU v5.

## 📋 Prerequisites

### System Requirements

**Minimum:**
- CPU: 4 cores, 2.4GHz
- RAM: 8GB
- Storage: 50GB SSD
- Network: 1Gbps

**Recommended:**
- CPU: 8+ cores, 3.0GHz+ (Intel Xeon or AMD EPYC)
- RAM: 32GB+
- Storage: 200GB+ NVMe SSD
- Network: 10Gbps
- GPU: Edge TPU v5 (optional, for maximum performance)

### Software Requirements

- Docker 24.0+
- Docker Compose 2.20+
- Linux kernel 5.4+ (for optimal performance)
- SSL certificates (for HTTPS)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   DGDM Server   │    │   Monitoring    │
│   (Nginx)       │────│   (Rust/Axum)   │────│   (Prometheus)  │
│   Port 80/443   │    │   Port 8080     │    │   Port 9090     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Edge TPU v5   │
                       │   (Optional)    │
                       └─────────────────┘
```

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/danieleschmidt/quantum-inspired-task-planner.git
cd quantum-inspired-task-planner
```

### 2. Configuration

Copy and customize the production configuration:

```bash
cp config.prod.toml config.toml
# Edit config.toml for your environment
```

Key settings to review:
- `server.bind_address`: Server binding address
- `graph.max_nodes`: Maximum graph size limits
- `processing.temperature`: Diffusion temperature
- `security.*`: Security configurations

### 3. SSL Certificates

Place SSL certificates in `nginx/ssl/`:

```bash
mkdir -p nginx/ssl
# Copy your certificates:
# - nginx/ssl/cert.pem
# - nginx/ssl/key.pem
```

### 4. Deploy with Docker Compose

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f dgdm-server
```

## 🔧 Configuration Details

### Graph Processing Limits

Configure based on your hardware:

```toml
[graph]
max_nodes = 1_000_000      # Adjust based on memory
max_edges = 10_000_000     # Adjust based on memory
feature_dim = 512          # Feature vector dimension
diffusion_steps = 10       # Processing iterations
batch_size = 32           # Concurrent graph batch size
```

**Memory estimation:** `nodes × features × 4 bytes × 2 (working memory)`

### Performance Tuning

```toml
[optimization]
enable_prefetching = true     # CPU cache optimization
enable_vectorization = true   # SIMD instructions
thread_pool_size = 8         # Set to CPU core count
chunk_size = 2048           # Processing chunk size
```

### Security Configuration

```toml
[security]
enable_rate_limiting = true
requests_per_minute = 1000
allowed_origins = ["https://yourdomain.com"]
max_concurrent_connections = 1000
```

## 📊 Monitoring

### Metrics Endpoints

- **Health Check:** `GET /health`
- **Metrics:** `GET /metrics` (Prometheus format)
- **Server Info:** `GET /info`

### Grafana Dashboard

Access Grafana at `http://localhost:3000` (admin/admin123!)

Key metrics to monitor:
- Request latency (95th percentile < 100ms)
- Throughput (requests/second)
- Memory usage (< 80% of available)
- CPU utilization (< 70% average)
- Error rate (< 0.1%)

### Prometheus Queries

```promql
# Request rate
rate(dgdm_requests_total[5m])

# Average processing time
rate(dgdm_processing_duration_seconds_sum[5m]) / rate(dgdm_processing_duration_seconds_count[5m])

# Memory usage
process_resident_memory_bytes / (1024 * 1024)

# Error rate
rate(dgdm_requests_total{status!="200"}[5m]) / rate(dgdm_requests_total[5m])
```

## 🔍 Troubleshooting

### Common Issues

**1. High Memory Usage**
```bash
# Check current memory usage
docker stats dgdm-server

# Reduce graph limits in config
[graph]
max_nodes = 500_000
max_edges = 5_000_000
```

**2. Slow Processing**
```bash
# Check CPU usage
htop

# Increase thread pool
[optimization]
thread_pool_size = 16
```

**3. Connection Timeouts**
```bash
# Increase timeouts
[server]
request_timeout_seconds = 600
```

### Health Check Failures

```bash
# Manual health check
curl -f http://localhost:8080/health

# Check server logs
docker logs dgdm-server

# Restart service
docker-compose -f docker-compose.prod.yml restart dgdm-server
```

### Performance Debugging

```bash
# Enable debug logging
RUST_LOG=debug docker-compose up

# Profile with flamegraph
cargo flamegraph --bin dgdm-server

# Memory profiling
valgrind --tool=massif ./dgdm-server
```

## 🌐 Load Balancing

### Multiple Instances

```yaml
# docker-compose.prod.yml
services:
  dgdm-server-1:
    # ... configuration
  dgdm-server-2:
    # ... configuration
  
  nginx:
    volumes:
      - ./nginx/nginx-lb.conf:/etc/nginx/nginx.conf
```

### Nginx Configuration

```nginx
upstream dgdm_backend {
    least_conn;
    server dgdm-server-1:8080 weight=1 max_fails=3 fail_timeout=30s;
    server dgdm-server-2:8080 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    location / {
        proxy_pass http://dgdm_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🔒 Security Best Practices

### 1. Network Security

```bash
# Configure firewall
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8080/tcp  # Block direct access
ufw enable
```

### 2. Container Security

```yaml
# docker-compose.prod.yml
services:
  dgdm-server:
    security_opt:
      - no-new-privileges:true
    user: "1001:1001"
    read_only: true
```

### 3. API Security

```toml
[security]
enable_api_keys = true
allowed_origins = ["https://trusted-domain.com"]
enable_rate_limiting = true
```

## 📈 Scaling

### Horizontal Scaling

1. **Stateless Design**: DGDM server is stateless and scales horizontally
2. **Load Balancer**: Use nginx or cloud load balancers
3. **Database**: Use external cache/database for shared state

### Vertical Scaling

1. **CPU**: Increase thread pool size
2. **Memory**: Increase graph processing limits
3. **TPU**: Add multiple Edge TPU devices

### Auto-scaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: dgdm-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: dgdm-server
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## 🚨 Disaster Recovery

### Backup Strategy

1. **Configuration**: Version-controlled configs
2. **Models**: Backup trained models to cloud storage
3. **Monitoring**: Export Prometheus data regularly

### Recovery Procedures

```bash
# Quick recovery
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d

# Full rebuild
docker-compose -f docker-compose.prod.yml build --no-cache
docker-compose -f docker-compose.prod.yml up -d
```

## 📞 Support

For production support:
- 📧 Email: support@terragonlabs.com
- 🐛 Issues: [GitHub Issues](https://github.com/danieleschmidt/quantum-inspired-task-planner/issues)
- 📚 Documentation: [Complete API Docs](https://docs.terragonlabs.com/dgdm)

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.