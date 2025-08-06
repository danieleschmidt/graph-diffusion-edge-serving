//! Custom middleware for rate limiting, authentication, and security

use axum::{
    extract::{Request, State},
    http::{StatusCode, HeaderMap},
    middleware::Next,
    response::Response,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, warn, error};

// Rate limiting middleware
#[derive(Clone)]
pub struct RateLimiter {
    windows: Arc<RwLock<HashMap<String, RateWindow>>>,
    requests_per_minute: u32,
    window_size: Duration,
}

#[derive(Debug)]
struct RateWindow {
    count: u32,
    window_start: Instant,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            windows: Arc::new(RwLock::new(HashMap::new())),
            requests_per_minute,
            window_size: Duration::from_secs(60),
        }
    }

    pub async fn check_rate_limit(&self, client_id: &str) -> Result<(), StatusCode> {
        let mut windows = self.windows.write().await;
        let now = Instant::now();
        
        let window = windows.entry(client_id.to_string()).or_insert(RateWindow {
            count: 0,
            window_start: now,
        });

        // Reset window if expired
        if now.duration_since(window.window_start) >= self.window_size {
            window.count = 0;
            window.window_start = now;
        }

        if window.count >= self.requests_per_minute {
            warn!("Rate limit exceeded for client: {}", client_id);
            return Err(StatusCode::TOO_MANY_REQUESTS);
        }

        window.count += 1;
        debug!("Rate limit check passed for {}: {}/{}", client_id, window.count, self.requests_per_minute);
        Ok(())
    }

    // Cleanup expired windows
    pub async fn cleanup_expired(&self) {
        let mut windows = self.windows.write().await;
        let now = Instant::now();
        
        windows.retain(|_, window| {
            now.duration_since(window.window_start) < self.window_size * 2
        });
    }
}

pub async fn rate_limiting_middleware(
    State(rate_limiter): State<Arc<RateLimiter>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract client identifier (could be IP, API key, etc.)
    let client_id = extract_client_id(&request);
    
    // Check rate limit
    rate_limiter.check_rate_limit(&client_id).await?;
    
    let response = next.run(request).await;
    Ok(response)
}

fn extract_client_id(request: &Request) -> String {
    // Try to get client IP from headers (for reverse proxy setups)
    if let Some(forwarded_for) = request.headers().get("x-forwarded-for") {
        if let Ok(ip_str) = forwarded_for.to_str() {
            return ip_str.split(',').next().unwrap_or("unknown").trim().to_string();
        }
    }
    
    if let Some(real_ip) = request.headers().get("x-real-ip") {
        if let Ok(ip_str) = real_ip.to_str() {
            return ip_str.to_string();
        }
    }
    
    // Fallback to connection info (would need to be passed in real implementation)
    "unknown".to_string()
}

// Security headers middleware
pub async fn security_headers_middleware(
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    
    let headers = response.headers_mut();
    
    // Security headers
    headers.insert("x-content-type-options", "nosniff".parse().unwrap());
    headers.insert("x-frame-options", "DENY".parse().unwrap());
    headers.insert("x-xss-protection", "1; mode=block".parse().unwrap());
    headers.insert(
        "strict-transport-security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );
    headers.insert(
        "content-security-policy",
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:".parse().unwrap(),
    );
    headers.insert("referrer-policy", "strict-origin-when-cross-origin".parse().unwrap());
    headers.insert(
        "permissions-policy",
        "camera=(), microphone=(), geolocation=()".parse().unwrap(),
    );
    
    response
}

// Request size validation middleware
pub async fn request_size_middleware(
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    const MAX_NODES: usize = 1_000_000;
    const MAX_EDGES: usize = 10_000_000;
    
    // Check content length
    if let Some(content_length) = request.headers().get("content-length") {
        if let Ok(length_str) = content_length.to_str() {
            if let Ok(length) = length_str.parse::<usize>() {
                if length > 100 * 1024 * 1024 { // 100MB
                    warn!("Request exceeds size limit: {} bytes", length);
                    return Err(StatusCode::PAYLOAD_TOO_LARGE);
                }
            }
        }
    }
    
    let response = next.run(request).await;
    Ok(response)
}

// API key authentication middleware (placeholder)
pub async fn auth_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Check for API key in headers
    if let Some(api_key) = headers.get("x-api-key") {
        if let Ok(key_str) = api_key.to_str() {
            // In a real implementation, validate against a database/store
            if validate_api_key(key_str).await {
                let response = next.run(request).await;
                return Ok(response);
            }
        }
    }
    
    // For now, allow all requests (authentication disabled)
    let response = next.run(request).await;
    Ok(response)
}

async fn validate_api_key(_key: &str) -> bool {
    // Placeholder implementation
    // In production, this would validate against a secure store
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(5);
        
        // Should allow first 5 requests
        for i in 1..=5 {
            assert!(limiter.check_rate_limit("test_client").await.is_ok(), "Request {} should succeed", i);
        }
        
        // Should block 6th request
        assert!(limiter.check_rate_limit("test_client").await.is_err(), "Request 6 should be rate limited");
    }

    #[tokio::test]
    async fn test_rate_limiter_cleanup() {
        let limiter = RateLimiter::new(1);
        
        // Add a client
        assert!(limiter.check_rate_limit("test_client").await.is_ok());
        
        // Cleanup should remove expired entries
        limiter.cleanup_expired().await;
        
        // Check that windows map is managed properly
        let windows = limiter.windows.read().await;
        assert!(!windows.is_empty()); // Should still contain recent entry
    }
}