# Multi-stage Docker build for DGDM Server
# Optimized for production deployment with minimal image size

# Build stage
FROM rust:1.75-slim-bullseye AS builder

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency files first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release --no-default-features && rm -rf src

# Copy source code
COPY src ./src
COPY benches ./benches

# Build the actual application
RUN cargo build --release --no-default-features --bin dgdm-server

# Runtime stage
FROM debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libgomp1 \
    libssl1.1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r dgdm && useradd -r -g dgdm dgdm

# Create app directory and set ownership
WORKDIR /app
RUN chown dgdm:dgdm /app

# Copy binary from builder stage
COPY --from=builder /app/target/release/dgdm-server /usr/local/bin/dgdm-server
RUN chmod +x /usr/local/bin/dgdm-server

# Copy configuration files if they exist
COPY --chown=dgdm:dgdm config.toml* ./

# Create directory for models
RUN mkdir -p models && chown dgdm:dgdm models

# Switch to non-root user
USER dgdm

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["dgdm-server", "--bind", "0.0.0.0:8080"]