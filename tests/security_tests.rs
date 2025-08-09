//! Security-focused tests for DGDM server

use graph_diffusion_edge::{
    security_scanner::{SecurityScanner, ThreatLevel},
    validation::{SecurityValidator, GraphValidator},
    core::{Graph, Node, Edge},
};
use std::collections::HashMap;
use tokio::time::{timeout, Duration};

#[tokio::test]
async fn test_security_scanner_sql_injection() {
    let scanner = SecurityScanner::new();
    let headers = HashMap::new();
    
    let malicious_inputs = vec![
        "'; DROP TABLE users; --",
        "1 UNION SELECT * FROM passwords",
        "admin'; DELETE FROM accounts WHERE '1'='1",
        "1' OR '1'='1' --",
    ];
    
    for input in malicious_inputs {
        let result = scanner.scan_request(
            input,
            Some("192.168.1.100".to_string()),
            &headers
        ).await;
        
        assert!(result.is_err(), "Should detect SQL injection: {}", input);
    }
}

#[tokio::test]
async fn test_security_scanner_xss_detection() {
    let scanner = SecurityScanner::new();
    let headers = HashMap::new();
    
    let xss_attempts = vec![
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "<img src=x onerror=alert(1)>",
        "data:text/html,<script>alert(1)</script>",
    ];
    
    for attempt in xss_attempts {
        let result = scanner.scan_request(
            attempt,
            Some("10.0.0.1".to_string()),
            &headers
        ).await;
        
        assert!(result.is_err(), "Should detect XSS: {}", attempt);
    }
}

#[tokio::test]
async fn test_security_scanner_path_traversal() {
    let scanner = SecurityScanner::new();
    let headers = HashMap::new();
    
    let path_traversal_attempts = vec![
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    ];
    
    for attempt in path_traversal_attempts {
        let result = scanner.scan_request(
            attempt,
            Some("172.16.0.1".to_string()),
            &headers
        ).await;
        
        assert!(result.is_err(), "Should detect path traversal: {}", attempt);
    }
}

#[tokio::test]
async fn test_security_scanner_safe_content() {
    let scanner = SecurityScanner::new();
    let headers = HashMap::new();
    
    let safe_inputs = vec![
        "Hello world",
        "This is a normal graph request",
        "node_1 connects to node_2 with weight 0.5",
        "Processing 1000 nodes and 5000 edges",
    ];
    
    for input in safe_inputs {
        let result = scanner.scan_request(
            input,
            Some("8.8.8.8".to_string()),
            &headers
        ).await;
        
        assert!(result.is_ok(), "Should allow safe content: {}", input);
    }
}

#[tokio::test]
async fn test_security_validator_comprehensive() {
    let validator = SecurityValidator::new();
    
    // Test safe strings
    assert!(validator.validate_string("safe_input", "test").is_ok());
    assert!(validator.validate_string("Hello World 123", "test").is_ok());
    
    // Test length limits
    let long_string = "a".repeat(2000);
    assert!(validator.validate_string(&long_string, "test").is_err());
    
    // Test malicious patterns
    let malicious_patterns = vec![
        "DROP TABLE users",
        "<script>alert(1)</script>",
        "../../../etc/passwd",
        "$(rm -rf /)",
        "&&whoami",
    ];
    
    for pattern in malicious_patterns {
        assert!(validator.validate_string(pattern, "test").is_err(), "Should reject: {}", pattern);
    }
}

#[tokio::test]
async fn test_graph_validator_security() {
    let validator = GraphValidator::new();
    
    // Test with malicious node labels
    let mut malicious_graph = Graph::new();
    malicious_graph.add_node(Node {
        id: 1,
        features: vec![1.0, 2.0],
        label: Some("<script>alert('xss')</script>".to_string()),
    });
    
    // Should detect and reject malicious content
    let result = validator.validate_graph(&malicious_graph);
    assert!(result.is_err(), "Should reject graph with malicious labels");
}

#[tokio::test]
async fn test_resource_exhaustion_protection() {
    let validator = GraphValidator::new();
    
    // Create extremely large graph
    let mut huge_graph = Graph::new();
    
    // Add many nodes to trigger size limits
    for i in 0..50_000 {
        huge_graph.add_node(Node {
            id: i,
            features: vec![1.0; 100], // Large feature vectors
            label: Some(format!("node_{}", i)),
        });
    }
    
    // Should reject due to size limits
    let result = validator.validate_graph(&huge_graph);
    assert!(result.is_err(), "Should reject oversized graphs");
}

#[tokio::test]
async fn test_security_scanner_alert_management() {
    let scanner = SecurityScanner::new();
    let headers = HashMap::new();
    
    // Generate some alerts
    let _ = scanner.scan_request(
        "'; DROP TABLE test; --",
        Some("192.168.1.1".to_string()),
        &headers
    ).await;
    
    // Check alerts were recorded
    let alerts = scanner.get_recent_alerts(10).await;
    assert!(!alerts.is_empty(), "Should have recorded security alerts");
    
    let summary = scanner.get_alert_summary().await;
    assert!(!summary.is_empty(), "Should have alert summary");
}

#[tokio::test]
async fn test_ip_reputation_system() {
    let scanner = SecurityScanner::new();
    
    // Test private IP scoring
    let private_reputation = scanner.check_ip_reputation("192.168.1.1").await.unwrap();
    assert!(private_reputation.score > 0, "Private IPs should have some score");
    
    // Test public IP scoring
    let public_reputation = scanner.check_ip_reputation("8.8.8.8").await.unwrap();
    assert_eq!(public_reputation.score, 0, "Clean public IPs should have low score");
    
    // Test caching
    let cached_reputation = scanner.check_ip_reputation("192.168.1.1").await.unwrap();
    assert_eq!(private_reputation.score, cached_reputation.score, "Should return cached result");
}

#[tokio::test]
async fn test_concurrent_security_scanning() {
    let scanner = std::sync::Arc::new(SecurityScanner::new());
    let headers = HashMap::new();
    
    let mut handles = vec![];
    
    // Launch multiple concurrent scans
    for i in 0..10 {
        let scanner_clone = scanner.clone();
        let headers_clone = headers.clone();
        
        let handle = tokio::spawn(async move {
            let content = if i % 2 == 0 {
                "safe content"
            } else {
                "'; DROP TABLE test; --"
            };
            
            let result = scanner_clone.scan_request(
                content,
                Some(format!("192.168.1.{}", i)),
                &headers_clone
            ).await;
            
            (i, result)
        });
        
        handles.push(handle);
    }
    
    // Wait for all scans to complete
    let mut safe_count = 0;
    let mut blocked_count = 0;
    
    for handle in handles {
        let (i, result) = handle.await.unwrap();
        if i % 2 == 0 {
            assert!(result.is_ok(), "Safe content should pass");
            safe_count += 1;
        } else {
            assert!(result.is_err(), "Malicious content should be blocked");
            blocked_count += 1;
        }
    }
    
    assert_eq!(safe_count, 5, "Should allow safe requests");
    assert_eq!(blocked_count, 5, "Should block malicious requests");
}

#[tokio::test]
async fn test_security_timeout_handling() {
    let scanner = SecurityScanner::new();
    let headers = HashMap::new();
    
    // Test that security scanning completes within reasonable time
    let result = timeout(
        Duration::from_secs(1),
        scanner.scan_request("test content", Some("1.2.3.4".to_string()), &headers)
    ).await;
    
    assert!(result.is_ok(), "Security scanning should complete quickly");
}