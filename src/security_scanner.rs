//! Runtime security scanning and threat detection

use crate::{Result, error::Error};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{warn, error, info};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityAlert {
    pub threat_level: ThreatLevel,
    pub threat_type: ThreatType,
    pub description: String,
    pub source_ip: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThreatType {
    InjectionAttempt,
    SuspiciousPattern,
    RateLimitViolation,
    AuthenticationBypass,
    DataExfiltration,
    DenialOfService,
    AnomalousTraffic,
}

pub struct SecurityScanner {
    alerts: Arc<RwLock<Vec<SecurityAlert>>>,
    threat_patterns: Vec<regex::Regex>,
    ip_reputation_cache: Arc<RwLock<HashMap<String, ReputationScore>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    pub score: u8, // 0-100, higher is more suspicious
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub reasons: Vec<String>,
}

impl SecurityScanner {
    pub fn new() -> Self {
        let threat_patterns = vec![
            // SQL injection patterns
            regex::Regex::new(r"(?i)(union\s+select|insert\s+into|drop\s+table|delete\s+from)").unwrap(),
            // XSS patterns
            regex::Regex::new(r"(?i)(<script|javascript:|data:text/html|vbscript:)").unwrap(),
            // Command injection
            regex::Regex::new(r"(;|\|&|&&|\|\||\$\(|`|<\(|>\()").unwrap(),
            // Path traversal
            regex::Regex::new(r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e\\)").unwrap(),
            // LDAP injection
            regex::Regex::new(r"(\*|\(|\)|&|\||=|!|<|>).*(\*|\(|\)|&|\||=|!|<|>)").unwrap(),
        ];

        Self {
            alerts: Arc::new(RwLock::new(Vec::new())),
            threat_patterns,
            ip_reputation_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn scan_request(&self, 
                            content: &str, 
                            source_ip: Option<String>, 
                            headers: &HashMap<String, String>) -> Result<()> {
        let mut alerts = Vec::new();

        // Pattern-based detection
        for (i, pattern) in self.threat_patterns.iter().enumerate() {
            if pattern.is_match(content) {
                let threat_type = match i {
                    0 => ThreatType::InjectionAttempt,
                    1 => ThreatType::InjectionAttempt,
                    2 => ThreatType::InjectionAttempt,
                    3 => ThreatType::SuspiciousPattern,
                    _ => ThreatType::SuspiciousPattern,
                };

                let alert = SecurityAlert {
                    threat_level: ThreatLevel::High,
                    threat_type,
                    description: format!("Malicious pattern detected in request content"),
                    source_ip: source_ip.clone(),
                    timestamp: chrono::Utc::now(),
                    metadata: [("pattern_index".to_string(), i.to_string())].iter().cloned().collect(),
                };

                alerts.push(alert);
            }
        }

        // Header analysis
        for (header_name, header_value) in headers {
            if self.analyze_header(header_name, header_value).await? {
                alerts.push(SecurityAlert {
                    threat_level: ThreatLevel::Medium,
                    threat_type: ThreatType::SuspiciousPattern,
                    description: format!("Suspicious header detected: {}", header_name),
                    source_ip: source_ip.clone(),
                    timestamp: chrono::Utc::now(),
                    metadata: [(header_name.clone(), header_value.clone())].iter().cloned().collect(),
                });
            }
        }

        // IP reputation check
        if let Some(ref ip) = source_ip {
            let reputation = self.check_ip_reputation(ip).await?;
            if reputation.score > 70 {
                alerts.push(SecurityAlert {
                    threat_level: if reputation.score > 90 { ThreatLevel::Critical } else { ThreatLevel::High },
                    threat_type: ThreatType::AnomalousTraffic,
                    description: format!("Request from suspicious IP: {} (score: {})", ip, reputation.score),
                    source_ip: source_ip.clone(),
                    timestamp: chrono::Utc::now(),
                    metadata: [("reputation_score".to_string(), reputation.score.to_string())].iter().cloned().collect(),
                });
            }
        }

        // Store alerts and potentially block
        if !alerts.is_empty() {
            let mut stored_alerts = self.alerts.write().await;
            let critical_count = alerts.iter().filter(|a| matches!(a.threat_level, ThreatLevel::Critical)).count();
            let high_count = alerts.iter().filter(|a| matches!(a.threat_level, ThreatLevel::High)).count();

            for alert in &alerts {
                error!("Security alert: {:?}", alert);
                stored_alerts.push(alert.clone());
            }

            // Limit alert storage
            if stored_alerts.len() > 10000 {
                stored_alerts.drain(0..1000); // Remove oldest 1000 alerts
            }

            // Block critical threats or injection attempts
            if critical_count > 0 || high_count > 0 {
                return Err(Error::Security {
                    violation: format!("Multiple security violations detected: {} critical, {} high", critical_count, high_count),
                    context: format!("source_ip: {:?}", source_ip),
                });
            }
        }

        Ok(())
    }

    async fn analyze_header(&self, name: &str, value: &str) -> Result<bool> {
        let suspicious_headers = ["x-forwarded-host", "x-real-ip", "x-cluster-client-ip"];
        let suspicious_values = ["127.0.0.1", "localhost", "0.0.0.0"];

        if suspicious_headers.contains(&name.to_lowercase().as_str()) {
            if suspicious_values.iter().any(|&sv| value.contains(sv)) {
                return Ok(true);
            }
        }

        // Check for abnormally long headers
        if value.len() > 8192 {
            warn!("Abnormally long header value: {} = {} chars", name, value.len());
            return Ok(true);
        }

        // Check for binary content in headers
        if value.chars().any(|c| c.is_control() && c != '\t' && c != '\n' && c != '\r') {
            return Ok(true);
        }

        Ok(false)
    }

    pub async fn check_ip_reputation(&self, ip: &str) -> Result<ReputationScore> {
        let mut cache = self.ip_reputation_cache.write().await;
        
        // Check cache first
        if let Some(cached) = cache.get(ip) {
            if chrono::Utc::now().signed_duration_since(cached.last_updated).num_hours() < 24 {
                return Ok(cached.clone());
            }
        }

        // Simple reputation scoring based on patterns
        let mut score = 0u8;
        let mut reasons = Vec::new();

        // Private/internal IPs get lower scores
        if ip.starts_with("10.") || ip.starts_with("192.168.") || ip.starts_with("172.") {
            score += 10;
            reasons.push("Private IP range".to_string());
        }

        // Common attack source patterns
        if ip.contains("tor-exit") || ip.contains("proxy") {
            score += 50;
            reasons.push("Known proxy/anonymizer".to_string());
        }

        // Geographic risk (placeholder - would integrate with real GeoIP)
        if ip.starts_with("5.") || ip.starts_with("46.") {
            score += 20;
            reasons.push("High-risk geographic region".to_string());
        }

        let reputation = ReputationScore {
            score,
            last_updated: chrono::Utc::now(),
            reasons,
        };

        cache.insert(ip.to_string(), reputation.clone());

        // Limit cache size
        if cache.len() > 10000 {
            let to_remove: Vec<String> = cache.keys().take(1000).cloned().collect();
            for key in to_remove {
                cache.remove(&key);
            }
        }

        Ok(reputation)
    }

    pub async fn get_recent_alerts(&self, limit: usize) -> Vec<SecurityAlert> {
        let alerts = self.alerts.read().await;
        let start_idx = if alerts.len() > limit { alerts.len() - limit } else { 0 };
        alerts[start_idx..].to_vec()
    }

    pub async fn get_alert_summary(&self) -> HashMap<String, usize> {
        let alerts = self.alerts.read().await;
        let mut summary = HashMap::new();
        
        let now = chrono::Utc::now();
        let one_hour_ago = now - chrono::Duration::hours(1);
        
        for alert in alerts.iter() {
            if alert.timestamp > one_hour_ago {
                let key = format!("{:?}-{:?}", alert.threat_level, alert.threat_type);
                *summary.entry(key).or_insert(0) += 1;
            }
        }
        
        summary
    }

    pub async fn clear_old_alerts(&self, older_than_hours: i64) {
        let mut alerts = self.alerts.write().await;
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(older_than_hours);
        alerts.retain(|alert| alert.timestamp > cutoff);
        info!("Cleared old security alerts, {} remaining", alerts.len());
    }
}

impl Default for SecurityScanner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sql_injection_detection() {
        let scanner = SecurityScanner::new();
        let headers = HashMap::new();
        
        // Should detect SQL injection
        let result = scanner.scan_request(
            "'; DROP TABLE users; --",
            Some("192.168.1.1".to_string()),
            &headers
        ).await;
        
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_safe_content() {
        let scanner = SecurityScanner::new();
        let headers = HashMap::new();
        
        // Should pass clean content
        let result = scanner.scan_request(
            "This is safe content",
            Some("192.168.1.1".to_string()),
            &headers
        ).await;
        
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_reputation_scoring() {
        let scanner = SecurityScanner::new();
        
        let rep1 = scanner.check_ip_reputation("192.168.1.1").await.unwrap();
        assert!(rep1.score > 0); // Private IP gets some score
        
        let rep2 = scanner.check_ip_reputation("8.8.8.8").await.unwrap();
        assert_eq!(rep2.score, 0); // Public IP gets no score initially
    }
}