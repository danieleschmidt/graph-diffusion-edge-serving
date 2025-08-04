#!/bin/bash
# Deployment script for DGDM Server

set -e

# Default values
ENVIRONMENT="production"
COMPOSE_FILE="docker-compose.yml"
TPU_MODE=false
MONITORING=false
BUILD_IMAGES=true

print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENVIRONMENT="$2"
            shift 2 ;;
        --tpu)
            TPU_MODE=true
            shift ;;
        --monitoring)
            MONITORING=true
            shift ;;
        --no-build)
            BUILD_IMAGES=false
            shift ;;
        --help)
            echo "DGDM Server Deployment Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env ENV          Environment (development|production) [default: production]"
            echo "  --tpu              Enable TPU support"
            echo "  --monitoring       Enable monitoring stack (Prometheus + Grafana)"
            echo "  --no-build         Skip building Docker images"
            echo "  --help             Show this help message"
            exit 0 ;;
        *)
            print_error "Unknown argument: $1"
            exit 1 ;;
    esac
done

print_status "🚀 Deploying DGDM Server..."
print_status "Environment: $ENVIRONMENT"
print_status "TPU Support: $TPU_MODE"
print_status "Monitoring: $MONITORING"

# Check prerequisites
print_status "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "docker-compose is not installed"
    exit 1
fi

# Check if running as root for TPU mode
if [ "$TPU_MODE" = true ] && [ "$EUID" -ne 0 ]; then
    print_warning "TPU mode may require root privileges for device access"
fi

# Build profiles
PROFILES=""
if [ "$TPU_MODE" = true ]; then
    PROFILES="$PROFILES,tpu"
fi
if [ "$MONITORING" = true ]; then
    PROFILES="$PROFILES,monitoring"
fi

# Remove leading comma
PROFILES="${PROFILES#,}"

# Build Docker images if requested
if [ "$BUILD_IMAGES" = true ]; then
    print_status "Building Docker images..."
    
    # Build main image
    docker-compose build dgdm-server
    
    # Build TPU image if needed
    if [ "$TPU_MODE" = true ]; then
        docker-compose build dgdm-server-tpu
    fi
    
    print_success "Docker images built successfully"
fi

# Create necessary directories
print_status "Creating directories..."
mkdir -p logs models monitoring/grafana/{dashboards,datasources}

# Set environment variables
export ENVIRONMENT="$ENVIRONMENT"
export COMPOSE_PROFILES="$PROFILES"

# Deploy based on environment
print_status "Deploying services..."

if [ "$ENVIRONMENT" = "development" ]; then
    # Development deployment
    print_status "Starting development environment..."
    
    if [ -n "$PROFILES" ]; then
        docker-compose --profile "$PROFILES" up -d
    else
        docker-compose up -d dgdm-server
    fi
    
elif [ "$ENVIRONMENT" = "production" ]; then
    # Production deployment
    print_status "Starting production environment..."
    
    # Check if SSL certificates exist
    if [ ! -d "nginx/ssl" ]; then
        print_warning "SSL certificates not found. HTTPS will not be available."
    fi
    
    # Add production profile
    if [ -n "$PROFILES" ]; then
        PROFILES="$PROFILES,production"
    else
        PROFILES="production"
    fi
    
    docker-compose --profile "$PROFILES" up -d
    
else
    print_error "Unknown environment: $ENVIRONMENT"
    exit 1
fi

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 10

# Health checks
print_status "Performing health checks..."

# Check main server
if curl -f http://localhost:8080/health &> /dev/null; then
    print_success "Main server is healthy"
else
    print_error "Main server health check failed"
fi

# Check TPU server if enabled
if [ "$TPU_MODE" = true ]; then
    if curl -f http://localhost:8081/health &> /dev/null; then
        print_success "TPU server is healthy"
    else
        print_warning "TPU server health check failed"
    fi
fi

# Check monitoring if enabled
if [ "$MONITORING" = true ]; then
    if curl -f http://localhost:9090/-/healthy &> /dev/null; then
        print_success "Prometheus is healthy"
    else
        print_warning "Prometheus health check failed"
    fi
    
    if curl -f http://localhost:3000/api/health &> /dev/null; then
        print_success "Grafana is healthy"
    else
        print_warning "Grafana health check failed"
    fi
fi

# Display service information
print_status "📋 Service Information:"
echo ""
echo "🌐 DGDM Server:"
echo "  • Main API: http://localhost:8080"
echo "  • Health: http://localhost:8080/health"
echo "  • Metrics: http://localhost:8080/metrics"
echo "  • Info: http://localhost:8080/info"

if [ "$TPU_MODE" = true ]; then
    echo ""
    echo "🔥 DGDM Server (TPU):"
    echo "  • TPU API: http://localhost:8081"
    echo "  • Health: http://localhost:8081/health"
fi

if [ "$MONITORING" = true ]; then
    echo ""
    echo "📊 Monitoring:"
    echo "  • Prometheus: http://localhost:9090"
    echo "  • Grafana: http://localhost:3000 (admin/admin)"
fi

echo ""
echo "🔧 Management Commands:"
echo "  • View logs: docker-compose logs -f"
echo "  • Stop services: docker-compose down"
echo "  • Restart: docker-compose restart"
echo "  • Update: docker-compose pull && docker-compose up -d"

print_success "🎉 Deployment completed successfully!"

# Show running containers
echo ""
print_status "Running containers:"
docker-compose ps