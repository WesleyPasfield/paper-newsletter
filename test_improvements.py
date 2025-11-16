#!/usr/bin/env python3
"""
Test script for the three improvements:
1. Rate limiting with token bucket and circuit breaker
2. Duplicate detection with fuzzy matching
3. JSON parsing robustness
"""

import sys
import time
import json
from functions.paper_analyzer import TokenBucket, CircuitBreaker, RateLimitMetrics, CircuitState
from functions.newsletter_generator import (
    normalize_arxiv_link,
    similarity_ratio,
    DuplicateDetector,
    extract_json_from_text,
    repair_json,
    validate_and_fix_newsletter_data,
    process_newsletter_content
)

def test_token_bucket():
    """Test token bucket rate limiting"""
    print("\n=== Testing Token Bucket ===")

    # Create bucket with 2 tokens per second, capacity of 5
    bucket = TokenBucket(rate=2.0, capacity=5.0)

    # Test 1: Should be able to consume 5 tokens immediately
    assert bucket.consume(5.0), "Should consume 5 tokens from full bucket"
    print("✓ Consumed 5 tokens from full bucket")

    # Test 2: Should not be able to consume more without waiting
    assert not bucket.consume(1.0), "Should not have tokens available"
    print("✓ Cannot consume when bucket is empty")

    # Test 3: Wait and refill
    wait_time = bucket.wait_time(2.0)
    print(f"✓ Need to wait {wait_time:.2f}s for 2 tokens")
    assert wait_time > 0, "Should need to wait for tokens"

    # Test 4: Small wait should refill some tokens
    time.sleep(0.5)  # Should refill 1 token (2 per second * 0.5s)
    assert bucket.consume(1.0), "Should have refilled ~1 token"
    print("✓ Tokens refilled over time")

    print("✓ Token Bucket tests passed!")

def test_circuit_breaker():
    """Test circuit breaker pattern"""
    print("\n=== Testing Circuit Breaker ===")

    # Create circuit breaker with low threshold for testing
    breaker = CircuitBreaker(failure_threshold=3, timeout=1.0, half_open_attempts=1)

    # Test 1: Initially closed
    assert breaker.state == CircuitState.CLOSED, "Should start CLOSED"
    assert breaker.call_allowed(), "Should allow calls when CLOSED"
    print("✓ Circuit breaker starts in CLOSED state")

    # Test 2: Record failures to open circuit
    for i in range(3):
        breaker.record_failure(is_rate_limit=False)

    assert breaker.state == CircuitState.OPEN, "Should be OPEN after threshold failures"
    assert not breaker.call_allowed(), "Should not allow calls when OPEN"
    print("✓ Circuit breaker opens after threshold failures")

    # Test 3: Transition to HALF_OPEN after timeout
    time.sleep(1.1)  # Wait for timeout
    assert breaker.call_allowed(), "Should allow call after timeout (HALF_OPEN)"
    assert breaker.state == CircuitState.HALF_OPEN, "Should be HALF_OPEN"
    print("✓ Circuit breaker transitions to HALF_OPEN after timeout")

    # Test 4: Success in HALF_OPEN closes circuit
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED, "Should be CLOSED after success in HALF_OPEN"
    print("✓ Circuit breaker closes after successful HALF_OPEN attempt")

    print("✓ Circuit Breaker tests passed!")

def test_rate_limit_metrics():
    """Test metrics tracking"""
    print("\n=== Testing Rate Limit Metrics ===")

    metrics = RateLimitMetrics()

    # Test initial state
    assert metrics.success_rate() == 1.0, "Should have 100% success with no requests"
    print("✓ Initial success rate is 100%")

    # Record some requests
    metrics.total_requests = 10
    metrics.successful_requests = 8
    metrics.rate_limit_errors = 2

    assert metrics.success_rate() == 0.8, "Should calculate correct success rate"
    print(f"✓ Success rate calculated correctly: {metrics.success_rate():.1%}")

    print("✓ Rate Limit Metrics tests passed!")

def test_normalize_arxiv_link():
    """Test arXiv link normalization"""
    print("\n=== Testing arXiv Link Normalization ===")

    test_cases = [
        ("https://arxiv.org/abs/2301.12345v1", "https://arxiv.org/abs/2301.12345"),
        ("https://arxiv.org/abs/2301.12345v2", "https://arxiv.org/abs/2301.12345"),
        ("https://arxiv.org/abs/2301.12345", "https://arxiv.org/abs/2301.12345"),
        ("https://arxiv.org/pdf/2301.12345v1", "https://arxiv.org/abs/2301.12345"),
        ("https://arxiv.org/abs/2301.12345/", "https://arxiv.org/abs/2301.12345"),
    ]

    for input_link, expected in test_cases:
        result = normalize_arxiv_link(input_link)
        assert result == expected, f"Failed for {input_link}: got {result}, expected {expected}"
        print(f"✓ {input_link[:40]}... → {result[:40]}...")

    print("✓ arXiv Link Normalization tests passed!")

def test_similarity_ratio():
    """Test fuzzy string matching"""
    print("\n=== Testing Similarity Ratio ===")

    # Test identical strings
    assert similarity_ratio("test", "test") == 1.0, "Identical strings should have ratio 1.0"
    print("✓ Identical strings: ratio = 1.0")

    # Test similar strings
    ratio = similarity_ratio(
        "Large Language Models for Code Generation",
        "Large Language Model for Code Generation"
    )
    assert ratio > 0.9, f"Similar strings should have high ratio (got {ratio:.2f})"
    print(f"✓ Similar strings: ratio = {ratio:.2%}")

    # Test different strings
    ratio = similarity_ratio("Deep Learning", "Quantum Computing")
    assert ratio < 0.5, f"Different strings should have low ratio (got {ratio:.2f})"
    print(f"✓ Different strings: ratio = {ratio:.2%}")

    print("✓ Similarity Ratio tests passed!")

def test_duplicate_detector():
    """Test duplicate detection with fuzzy matching"""
    print("\n=== Testing Duplicate Detector ===")

    previously_included = {
        "attention is all you need",
        "https://arxiv.org/abs/1706.03762",
        "bert: pre-training of deep bidirectional transformers",
    }

    detector = DuplicateDetector(previously_included, similarity_threshold=0.9)

    # Test 1: Exact title match
    is_dup, reason = detector.is_duplicate(
        "Attention is All You Need",
        "https://arxiv.org/abs/9999.99999"
    )
    assert is_dup, "Should detect exact title match (case-insensitive)"
    print(f"✓ Exact title match detected: {reason}")

    # Test 2: Exact link match with normalization
    is_dup, reason = detector.is_duplicate(
        "Some Other Paper",
        "https://arxiv.org/abs/1706.03762v2"
    )
    assert is_dup, "Should detect normalized link match"
    print(f"✓ Normalized link match detected: {reason}")

    # Test 3: Fuzzy title match
    is_dup, reason = detector.is_duplicate(
        "Attention Is All You Need!",  # Slightly different
        "https://arxiv.org/abs/9999.99999"
    )
    assert is_dup, "Should detect fuzzy title match"
    print(f"✓ Fuzzy title match detected: {reason}")

    # Test 4: Not a duplicate
    is_dup, reason = detector.is_duplicate(
        "A Completely Different Paper Title",
        "https://arxiv.org/abs/9999.88888"
    )
    assert not is_dup, "Should not detect as duplicate"
    print("✓ Non-duplicate correctly identified")

    print("✓ Duplicate Detector tests passed!")

def test_extract_json_from_text():
    """Test JSON extraction from text"""
    print("\n=== Testing JSON Extraction ===")

    # Test 1: JSON with surrounding text
    text = 'Here is some text {"key": "value", "num": 123} and more text'
    result = extract_json_from_text(text)
    assert result == '{"key": "value", "num": 123}', "Should extract JSON object"
    print("✓ Extracted JSON from surrounding text")

    # Test 2: Nested JSON
    text = 'Before {"outer": {"inner": "value"}} After'
    result = extract_json_from_text(text)
    parsed = json.loads(result)
    assert parsed["outer"]["inner"] == "value", "Should handle nested JSON"
    print("✓ Extracted nested JSON")

    # Test 3: JSON array
    text = 'Start [1, 2, 3] End'
    result = extract_json_from_text(text)
    assert result == '[1, 2, 3]', "Should extract JSON array"
    print("✓ Extracted JSON array")

    print("✓ JSON Extraction tests passed!")

def test_repair_json():
    """Test JSON repair functionality"""
    print("\n=== Testing JSON Repair ===")

    # Test 1: Remove trailing commas
    broken = '{"key": "value",}'
    fixed = repair_json(broken)
    parsed = json.loads(fixed)
    assert parsed["key"] == "value", "Should remove trailing comma"
    print("✓ Removed trailing comma")

    # Test 2: Array with trailing comma
    broken = '{"items": [1, 2, 3,]}'
    fixed = repair_json(broken)
    parsed = json.loads(fixed)
    assert len(parsed["items"]) == 3, "Should handle array trailing comma"
    print("✓ Fixed array trailing comma")

    print("✓ JSON Repair tests passed!")

def test_validate_and_fix_newsletter_data():
    """Test newsletter data validation"""
    print("\n=== Testing Newsletter Data Validation ===")

    # Test 1: Missing fields
    incomplete_data = {}
    fixed = validate_and_fix_newsletter_data(incomplete_data)

    assert 'overview' in fixed, "Should add missing overview"
    assert 'featured_papers' in fixed, "Should add missing featured_papers"
    assert 'additional_papers' in fixed, "Should add missing additional_papers"
    assert 'metadata' in fixed, "Should add missing metadata"
    print("✓ Added missing fields")

    # Test 2: Invalid paper entries
    data_with_invalid = {
        'featured_papers': [
            {'title': 'Valid Paper', 'link': 'http://example.com', 'summary': 'Good'},
            {'title': 'Missing link'},  # Invalid
            'not a dict',  # Invalid
        ],
        'additional_papers': [
            {'title': 'Valid', 'link': 'http://example.com'},
        ]
    }
    fixed = validate_and_fix_newsletter_data(data_with_invalid)

    assert len(fixed['featured_papers']) == 1, "Should filter out invalid papers"
    print("✓ Filtered invalid paper entries")

    # Test 3: Metadata counts
    assert fixed['metadata']['featured_papers_count'] == 1, "Should count featured papers"
    assert fixed['metadata']['additional_papers_count'] == 1, "Should count additional papers"
    print("✓ Calculated correct metadata counts")

    print("✓ Newsletter Data Validation tests passed!")

def test_process_newsletter_content():
    """Test full newsletter content processing"""
    print("\n=== Testing Newsletter Content Processing ===")

    # Test 1: Valid JSON in markdown code block
    content = '''```json
{
  "overview": "Test overview",
  "featured_papers": [
    {"title": "Paper 1", "link": "http://example.com/1", "summary": "Summary 1"}
  ],
  "additional_papers": [],
  "metadata": {}
}
```'''

    result = process_newsletter_content(content)
    assert result['overview'] == "Test overview", "Should parse JSON from code block"
    assert len(result['featured_papers']) == 1, "Should have 1 featured paper"
    print("✓ Processed JSON from markdown code block")

    # Test 2: JSON with extra text
    content = '''Here is the newsletter:
{
  "overview": "Another test",
  "featured_papers": [],
  "additional_papers": [
    {"title": "Paper 2", "link": "http://example.com/2"}
  ],
  "metadata": {}
}
That's all!'''

    result = process_newsletter_content(content)
    assert result['overview'] == "Another test", "Should extract JSON from text"
    assert len(result['additional_papers']) == 1, "Should have 1 additional paper"
    print("✓ Extracted and processed JSON from surrounding text")

    # Test 3: Incomplete data that needs fixing
    content = '{"overview": "Test", "featured_papers": []}'

    result = process_newsletter_content(content)
    assert 'additional_papers' in result, "Should add missing fields"
    assert 'metadata' in result, "Should add metadata"
    print("✓ Fixed incomplete newsletter data")

    print("✓ Newsletter Content Processing tests passed!")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Newsletter System Improvements")
    print("=" * 60)

    try:
        # Rate limiting tests
        test_token_bucket()
        test_circuit_breaker()
        test_rate_limit_metrics()

        # Duplicate detection tests
        test_normalize_arxiv_link()
        test_similarity_ratio()
        test_duplicate_detector()

        # JSON parsing tests
        test_extract_json_from_text()
        test_repair_json()
        test_validate_and_fix_newsletter_data()
        test_process_newsletter_content()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
