#!/bin/bash
# Build script for DGDM Server

set -e

echo "🏗️  Building DGDM Server..."

# Function to print colored output
print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Parse command line arguments
BUILD_TYPE="release"
FEATURES=""
TARGET=""
WASM_BUILD=false
DOCKER_BUILD=false
TPU_SUPPORT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="debug"
            shift ;;
        --tpu)
            TPU_SUPPORT=true
            shift ;;
        --wasm)
            WASM_BUILD=true
            shift ;;
        --docker)
            DOCKER_BUILD=true
            shift ;;
        --target)
            TARGET="--target $2"
            shift 2 ;;
        *)
            print_error "Unknown argument: $1"
            exit 1 ;;
    esac
done

# Set features based on options
if [ "$TPU_SUPPORT" = true ]; then
    FEATURES="--features tpu"
    print_status "Building with TPU support"
else
    FEATURES="--no-default-features"
    print_status "Building without TPU support (CPU only)"
fi

# Build native binary
print_status "Building native binary..."
if [ "$BUILD_TYPE" = "release" ]; then
    cargo build --release $FEATURES $TARGET --bin dgdm-server
    BINARY_PATH="target/release/dgdm-server"
else
    cargo build $FEATURES $TARGET --bin dgdm-server
    BINARY_PATH="target/debug/dgdm-server"
fi

if [ -f "$BINARY_PATH" ]; then
    print_success "Native binary built successfully: $BINARY_PATH"
    ls -lh "$BINARY_PATH"
else
    print_error "Binary build failed"
    exit 1
fi

# Build WASM if requested
if [ "$WASM_BUILD" = true ]; then
    print_status "Building WASM module..."
    
    # Check if wasm-pack is installed
    if ! command -v wasm-pack &> /dev/null; then
        print_error "wasm-pack is not installed. Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
        exit 1
    fi
    
    wasm-pack build --target web --release --no-default-features
    
    if [ -d "pkg" ]; then
        print_success "WASM module built successfully: pkg/"
        ls -la pkg/
    else
        print_error "WASM build failed"
        exit 1
    fi
fi

# Build Docker image if requested
if [ "$DOCKER_BUILD" = true ]; then
    print_status "Building Docker image..."
    
    if [ "$TPU_SUPPORT" = true ]; then
        DOCKERFILE="Dockerfile.tpu"
        IMAGE_TAG="dgdm-server:tpu"
    else
        DOCKERFILE="Dockerfile"
        IMAGE_TAG="dgdm-server:latest"
    fi
    
    docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" .
    
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully: $IMAGE_TAG"
        docker images | grep dgdm-server
    else
        print_error "Docker build failed"
        exit 1
    fi
fi

print_success "Build completed successfully! 🎉"

# Display next steps
echo ""
echo "📋 Next steps:"
echo "  • Run server: $BINARY_PATH"
echo "  • Run with config: $BINARY_PATH --config config.toml"
echo "  • Run tests: cargo test $FEATURES"
if [ "$WASM_BUILD" = true ]; then
    echo "  • Serve WASM demo: python3 -m http.server 8000 (then open examples/wasm_demo.html)"
fi
if [ "$DOCKER_BUILD" = true ]; then
    echo "  • Run Docker: docker run -p 8080:8080 $IMAGE_TAG"
    echo "  • Or use docker-compose: docker-compose up"
fi