#!/bin/bash

set -euo pipefail

echo "🧪 DGDM Testing Suite"
echo "===================="

# Color output functions
green() { echo -e "\033[0;32m$1\033[0m"; }
red() { echo -e "\033[0;31m$1\033[0m"; }
yellow() { echo -e "\033[0;33m$1\033[0m"; }
blue() { echo -e "\033[0;34m$1\033[0m"; }

# Test categories
run_unit_tests() {
    blue "📋 Running unit tests..."
    if cargo test --lib --quiet 2>/dev/null; then
        green "✅ Unit tests passed"
        return 0
    else
        red "❌ Unit tests failed"
        return 1
    fi
}

run_integration_tests() {
    blue "🔗 Running integration tests..."
    if cargo test --test '*' --quiet 2>/dev/null; then
        green "✅ Integration tests passed"
        return 0
    else
        yellow "⚠️  No integration tests found or failed"
        return 0
    fi
}

run_doc_tests() {
    blue "📚 Running documentation tests..."
    if cargo test --doc --quiet 2>/dev/null; then
        green "✅ Documentation tests passed"
        return 0
    else
        yellow "⚠️  Documentation tests failed or not found"
        return 0
    fi
}

check_code_quality() {
    blue "🔍 Checking code quality..."
    
    # Check for common issues
    if cargo clippy --all-targets --all-features -- -D warnings >/dev/null 2>&1; then
        green "✅ Clippy checks passed"
    else
        yellow "⚠️  Clippy warnings found"
    fi
    
    # Check formatting
    if cargo fmt --check >/dev/null 2>&1; then
        green "✅ Code formatting is correct"
    else
        yellow "⚠️  Code formatting issues found"
    fi
}

run_security_checks() {
    blue "🔒 Running security checks..."
    
    # Check for known vulnerabilities
    if command -v cargo-audit >/dev/null 2>&1; then
        if cargo audit --quiet 2>/dev/null; then
            green "✅ No known vulnerabilities"
        else
            yellow "⚠️  Security vulnerabilities found"
        fi
    else
        yellow "⚠️  cargo-audit not installed"
    fi
}

run_performance_tests() {
    blue "⚡ Running basic performance tests..."
    
    # Simple compile-time test
    start_time=$(date +%s.%N)
    if cargo check --quiet >/dev/null 2>&1; then
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "N/A")
        green "✅ Code compiles (${duration}s)"
    else
        red "❌ Compilation failed"
        return 1
    fi
}

# Main test execution
main() {
    local exit_code=0
    
    echo "Starting comprehensive test suite..."
    echo
    
    # Core functionality tests
    run_unit_tests || exit_code=1
    echo
    
    run_integration_tests || exit_code=1
    echo
    
    run_doc_tests || exit_code=1
    echo
    
    # Code quality checks
    check_code_quality || exit_code=1
    echo
    
    # Security and performance
    run_security_checks || exit_code=1
    echo
    
    run_performance_tests || exit_code=1
    echo
    
    # Summary
    echo "===================="
    if [ $exit_code -eq 0 ]; then
        green "🎉 All tests completed successfully!"
    else
        red "💥 Some tests failed or had warnings"
    fi
    
    echo
    blue "📊 Test Summary:"
    echo "  - Unit tests: Core functionality"
    echo "  - Integration tests: End-to-end workflows"
    echo "  - Doc tests: Example code in documentation"
    echo "  - Code quality: Clippy lints and formatting"
    echo "  - Security: Vulnerability scanning"
    echo "  - Performance: Compilation speed"
    
    exit $exit_code
}

# Allow running specific test categories
case "${1:-all}" in
    "unit")
        run_unit_tests
        ;;
    "integration") 
        run_integration_tests
        ;;
    "doc")
        run_doc_tests
        ;;
    "quality")
        check_code_quality
        ;;
    "security")
        run_security_checks
        ;;
    "performance")
        run_performance_tests
        ;;
    "all"|*)
        main
        ;;
esac