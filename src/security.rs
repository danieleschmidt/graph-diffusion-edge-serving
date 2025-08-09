//! Security middleware and authentication for the DGDM server

use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::Response,
};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use std::collections::HashMap;
use tracing::{debug, error, warn};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use rand::rngs::OsRng;
use regex::Regex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,       // Subject (user ID)
    pub exp: usize,        // Expiration time
    pub iat: usize,        // Issued at
    pub role: String,      // User role
    pub permissions: Vec<String>, // Fine-grained permissions
}

#[derive(Debug, Clone)]
pub struct AuthConfig {
    pub jwt_secret: String,
    pub token_expiration_hours: usize,
    pub require_https: bool,
    pub allowed_origins: Vec<String>,
    pub rate_limit_per_minute: usize,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            jwt_secret: "your-super-secret-jwt-key-change-in-production".to_string(),
            token_expiration_hours: 24,
            require_https: true,
            allowed_origins: vec!["https://localhost:3000".to_string()],
            rate_limit_per_minute: 100,
        }
    }
}

#[derive(Clone)]
pub struct SecurityMiddleware {
    config: AuthConfig,
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
}

impl std::fmt::Debug for SecurityMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecurityMiddleware")
         .field("config", &self.config)
         .field("encoding_key", &"[REDACTED]")
         .field("decoding_key", &"[REDACTED]")
         .finish()
    }
}

impl SecurityMiddleware {
    pub fn new(config: AuthConfig) -> Self {
        let encoding_key = EncodingKey::from_secret(config.jwt_secret.as_ref());
        let decoding_key = DecodingKey::from_secret(config.jwt_secret.as_ref());
        
        Self {
            config,
            encoding_key,
            decoding_key,
        }
    }

    pub fn generate_token(&self, user_id: &str, role: &str, permissions: Vec<String>) -> Result<String, jsonwebtoken::errors::Error> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() as usize;
        
        let claims = Claims {
            sub: user_id.to_string(),
            exp: now + (self.config.token_expiration_hours * 3600),
            iat: now,
            role: role.to_string(),
            permissions,
        };

        encode(&Header::default(), &claims, &self.encoding_key)
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
        let validation = Validation::new(Algorithm::HS256);
        let token_data = decode::<Claims>(token, &self.decoding_key, &validation)?;
        Ok(token_data.claims)
    }
}

// JWT Authentication middleware
pub async fn jwt_auth_middleware(
    State(security): State<Arc<SecurityMiddleware>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Skip auth for health endpoint
    if request.uri().path() == "/health" {
        return Ok(next.run(request).await);
    }

    let auth_header = headers
        .get("authorization")
        .and_then(|header| header.to_str().ok());

    let token = match auth_header {
        Some(header) if header.starts_with("Bearer ") => &header[7..],
        _ => {
            warn!("Missing or invalid authorization header");
            return Err(StatusCode::UNAUTHORIZED);
        }
    };

    match security.validate_token(token) {
        Ok(claims) => {
            debug!("Authenticated user: {} with role: {}", claims.sub, claims.role);
            // Add claims to request extensions for use in handlers
            let mut request = request;
            request.extensions_mut().insert(claims);
            Ok(next.run(request).await)
        }
        Err(e) => {
            warn!("Token validation failed: {}", e);
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

// Permission-based authorization middleware
pub async fn require_permission_middleware(
    required_permission: String,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    if let Some(claims) = request.extensions().get::<Claims>() {
        if claims.permissions.contains(&required_permission) || 
           claims.role == "admin" {
            Ok(next.run(request).await)
        } else {
            warn!("User {} lacks required permission: {}", claims.sub, required_permission);
            Err(StatusCode::FORBIDDEN)
        }
    } else {
        error!("Claims not found in request extensions");
        Err(StatusCode::UNAUTHORIZED)
    }
}

// CORS security headers middleware
pub async fn security_headers_middleware(
    request: Request,
    next: Next,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    // Security headers
    headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
    headers.insert("X-Frame-Options", "DENY".parse().unwrap());
    headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
    headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
    headers.insert(
        "Strict-Transport-Security",
        "max-age=31536000; includeSubDomains".parse().unwrap(),
    );
    headers.insert(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self' 'unsafe-inline'".parse().unwrap(),
    );

    response
}

// Request size limiting middleware
pub async fn request_size_limit_middleware(
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    const MAX_REQUEST_SIZE: u64 = 500 * 1024 * 1024; // 500 MB

    if let Some(content_length) = headers.get("content-length") {
        if let Ok(length_str) = content_length.to_str() {
            if let Ok(length) = length_str.parse::<u64>() {
                if length > MAX_REQUEST_SIZE {
                    warn!("Request size {} exceeds maximum {}", length, MAX_REQUEST_SIZE);
                    return Err(StatusCode::PAYLOAD_TOO_LARGE);
                }
            }
        }
    }

    Ok(next.run(request).await)
}

// IP-based rate limiting (simple in-memory implementation)

pub struct RateLimiter {
    requests: Arc<Mutex<HashMap<String, Vec<Instant>>>>,
    max_requests: usize,
    window_duration: std::time::Duration,
}

impl RateLimiter {
    pub fn new(max_requests: usize, window_seconds: u64) -> Self {
        Self {
            requests: Arc::new(Mutex::new(HashMap::new())),
            max_requests,
            window_duration: std::time::Duration::from_secs(window_seconds),
        }
    }

    pub fn check_rate_limit(&self, client_ip: &str) -> bool {
        let now = Instant::now();
        let mut requests = self.requests.lock().unwrap();
        
        let client_requests = requests.entry(client_ip.to_string()).or_insert_with(Vec::new);
        
        // Remove old requests outside the window
        client_requests.retain(|&request_time| now.duration_since(request_time) < self.window_duration);
        
        if client_requests.len() >= self.max_requests {
            false
        } else {
            client_requests.push(now);
            true
        }
    }
}

pub async fn rate_limit_middleware(
    State(rate_limiter): State<Arc<RateLimiter>>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract client IP (simplified - in production use forwarded headers carefully)
    let client_ip = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown");

    if !rate_limiter.check_rate_limit(client_ip) {
        warn!("Rate limit exceeded for IP: {}", client_ip);
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    Ok(next.run(request).await)
}

// Password hashing utilities
pub struct PasswordManager;

impl PasswordManager {
    pub fn hash_password(password: &str) -> Result<String, argon2::password_hash::Error> {
        let salt = argon2::password_hash::SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        let password_hash = argon2.hash_password(password.as_bytes(), &salt)?;
        Ok(password_hash.to_string())
    }

    pub fn verify_password(password: &str, hash: &str) -> Result<bool, argon2::password_hash::Error> {
        let parsed_hash = PasswordHash::new(hash)?;
        let argon2 = Argon2::default();
        match argon2.verify_password(password.as_bytes(), &parsed_hash) {
            Ok(()) => Ok(true),
            Err(argon2::password_hash::Error::Password) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

// Authentication endpoints
#[derive(Debug, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize)]
pub struct LoginResponse {
    pub token: String,
    pub expires_in: usize,
}

#[derive(Debug, Serialize)]
pub struct AuthError {
    pub error: String,
    pub message: String,
}

// Secure session management
pub struct SessionManager {
    security: Arc<SecurityMiddleware>,
}

impl SessionManager {
    pub fn new(security: Arc<SecurityMiddleware>) -> Self {
        Self { security }
    }

    pub fn create_session(&self, user_id: &str, role: &str) -> Result<String, jsonwebtoken::errors::Error> {
        let permissions = match role {
            "admin" => vec!["read".to_string(), "write".to_string(), "admin".to_string()],
            "user" => vec!["read".to_string(), "write".to_string()],
            "readonly" => vec!["read".to_string()],
            _ => vec![], // No permissions for unknown roles
        };

        self.security.generate_token(user_id, role, permissions)
    }
}

// Input sanitization for security

pub struct InputSanitizer {
    xss_patterns: Vec<Regex>,
    sql_patterns: Vec<Regex>,
}

impl Default for InputSanitizer {
    fn default() -> Self {
        Self::new()
    }
}

impl InputSanitizer {
    pub fn new() -> Self {
        let xss_patterns = vec![
            Regex::new(r"(?i)<script[^>]*>.*?</script>").unwrap(),
            Regex::new(r"(?i)javascript:").unwrap(),
            Regex::new(r"(?i)vbscript:").unwrap(),
            Regex::new(r"(?i)data:text/html").unwrap(),
        ];

        let sql_patterns = vec![
            Regex::new(r"(?i)(union|select|insert|update|delete|drop|create|alter)\s").unwrap(),
            Regex::new(r"(?i)(\sor\s|\sand\s).*[=<>]").unwrap(),
            Regex::new(r"[';]").unwrap(),
        ];

        Self {
            xss_patterns,
            sql_patterns,
        }
    }

    pub fn sanitize_string(&self, input: &str) -> String {
        let mut sanitized = input.to_string();

        // Remove potential XSS
        for pattern in &self.xss_patterns {
            sanitized = pattern.replace_all(&sanitized, "").to_string();
        }

        // Remove potential SQL injection
        for pattern in &self.sql_patterns {
            sanitized = pattern.replace_all(&sanitized, "").to_string();
        }

        // HTML entity encoding for remaining < > & "
        sanitized = sanitized
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#x27;");

        sanitized
    }

    pub fn is_safe(&self, input: &str) -> bool {
        !self.xss_patterns.iter().any(|p| p.is_match(input)) &&
        !self.sql_patterns.iter().any(|p| p.is_match(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_password_hashing() {
        let password = "test_password123";
        let hash = PasswordManager::hash_password(password).unwrap();
        
        assert!(PasswordManager::verify_password(password, &hash).unwrap());
        assert!(!PasswordManager::verify_password("wrong_password", &hash).unwrap());
    }

    #[test]
    fn test_input_sanitizer() {
        let sanitizer = InputSanitizer::new();
        
        let malicious_input = "<script>alert('xss')</script>";
        let sanitized = sanitizer.sanitize_string(malicious_input);
        assert!(!sanitized.contains("<script>"));
        
        assert!(!sanitizer.is_safe(malicious_input));
        assert!(sanitizer.is_safe("normal text"));
    }

    #[test]
    fn test_jwt_token() {
        let config = AuthConfig::default();
        let security = SecurityMiddleware::new(config);
        
        let permissions = vec!["read".to_string(), "write".to_string()];
        let token = security.generate_token("test_user", "user", permissions).unwrap();
        
        let claims = security.validate_token(&token).unwrap();
        assert_eq!(claims.sub, "test_user");
        assert_eq!(claims.role, "user");
    }

    #[test]
    fn test_rate_limiter() {
        let rate_limiter = RateLimiter::new(2, 60);
        
        assert!(rate_limiter.check_rate_limit("192.168.1.1"));
        assert!(rate_limiter.check_rate_limit("192.168.1.1"));
        assert!(!rate_limiter.check_rate_limit("192.168.1.1")); // Third request should fail
    }
}