# functions/paper_analyzer.py
import feedparser
import requests
import time
import random
import math
from anthropic import APIError, Anthropic
import json
from datetime import datetime
import logging
from bs4 import BeautifulSoup
from typing import Set, Dict, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, stop trying
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class RateLimitMetrics:
    """Track rate limiting metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    rate_limit_errors: int = 0
    other_errors: int = 0
    circuit_breaker_trips: int = 0

    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    def __init__(self, rate: float, capacity: float):
        """
        Args:
            rate: Tokens added per second
            capacity: Maximum tokens in bucket
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.time()

    def consume(self, tokens: float = 1.0) -> bool:
        """
        Try to consume tokens. Returns True if successful.
        """
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self):
        """Refill tokens based on time elapsed"""
        now = time.time()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    def wait_time(self, tokens: float = 1.0) -> float:
        """Calculate how long to wait for tokens to be available"""
        self._refill()
        if self.tokens >= tokens:
            return 0.0
        tokens_needed = tokens - self.tokens
        return tokens_needed / self.rate

class CircuitBreaker:
    """Circuit breaker pattern for API calls"""
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, half_open_attempts: int = 1):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before trying again (OPEN -> HALF_OPEN)
            half_open_attempts: Number of test requests in HALF_OPEN state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.half_open_attempts = half_open_attempts

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_success_count = 0

    def call_allowed(self) -> bool:
        """Check if calls are allowed in current state"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if time.time() - self.last_failure_time >= self.timeout:
                logger.info("Circuit breaker transitioning to HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
                self.half_open_success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self):
        """Record successful API call"""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_success_count += 1
            if self.half_open_success_count >= self.half_open_attempts:
                logger.info("Circuit breaker transitioning to CLOSED state")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            # Gradually decrease failure count on success
            self.failure_count = max(0, self.failure_count - 1)

    def record_failure(self, is_rate_limit: bool = False):
        """Record failed API call"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        # Rate limit errors are more serious - count them double
        if is_rate_limit:
            self.failure_count += 1

        if self.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker reopening due to failure in HALF_OPEN state")
            self.state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker opening after {self.failure_count} failures")
            self.state = CircuitState.OPEN

@dataclass
class Paper:
    title: str
    link: str
    abstract: str = ""
    full_text: str = ""
    interest_score: float = 0.0
    summary: str = ""

    def __hash__(self):
        return hash((self.title, self.link))
    
    def __eq__(self, other):
        if not isinstance(other, Paper):
            return False
        return self.title == other.title or self.link == other.link

class PaperAnalyzer:
    def __init__(
        self,
        anthropic_api_key: str,
        eval_prompt: str,
        newsletter_prompt: str,
        previously_included_papers: Set[str] = None,
        model_cheap: Optional[str] = None,
        model_expensive: Optional[str] = None,
        model_fallback: Optional[str] = None,
        model_emergency: Optional[str] = None
    ):
        """
        Initialize PaperAnalyzer with configurable Claude models.

        Args:
            anthropic_api_key: Anthropic API key
            eval_prompt: Prompt for evaluating papers
            newsletter_prompt: Prompt for generating newsletter
            previously_included_papers: Set of previously included paper titles/links
            model_cheap: Model for cheap operations (default: claude-haiku-4-5-20251001)
            model_expensive: Model for expensive operations (default: claude-sonnet-4-5-20250929)
            model_fallback: Fallback model when rate limited (default: claude-3-7-sonnet-20250219)
            model_emergency: Emergency fallback model (default: claude-3-haiku-20240307)
        """
        self.client = Anthropic(api_key=anthropic_api_key)

        # Set models with defaults if not provided
        self.claude_model_cheap = model_cheap or "claude-haiku-4-5-20251001"
        self.claude_model_pricey = model_expensive or "claude-sonnet-4-5-20250929"
        self.claude_model_fallback = model_fallback or "claude-3-7-sonnet-20250219"
        self.claude_model_emergency = model_emergency or "claude-3-haiku-20240307"

        logger.info(f"Initialized with models - Cheap: {self.claude_model_cheap}, "
                   f"Expensive: {self.claude_model_pricey}, "
                   f"Fallback: {self.claude_model_fallback}, "
                   f"Emergency: {self.claude_model_emergency}")

        # Track fallback level: 0=normal, 1=fallback, 2=emergency
        self.fallback_level = 0
        self.eval_prompt = eval_prompt
        self.newsletter_prompt = newsletter_prompt
        self.processed_papers: Set[Paper] = set()
        self.previously_included_papers = previously_included_papers or set()

        # Convert all previously included papers to lowercase for case-insensitive matching
        self.previously_included_papers = {p.lower() for p in self.previously_included_papers}

        # Enhanced rate limiting with token bucket
        # Start conservative: 5 requests per second, burst capacity of 10
        self.token_bucket = TokenBucket(rate=5.0, capacity=10.0)

        # Circuit breaker to stop trying if we keep failing
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60.0,
            half_open_attempts=1
        )

        # Metrics tracking
        self.metrics = RateLimitMetrics()

        # Legacy rate limiting state (kept for compatibility)
        self.last_api_call_time = 0.0
        self.min_api_interval = 0.1  # Minimum 100ms between API calls
        self.recent_rate_limit_failures = 0  # Track recent rate limit failures
        self.input_token_acceleration_limit = False  # Track if we hit input token acceleration limit

        logger.info(f"Initialized PaperAnalyzer with {len(self.previously_included_papers)} previously included papers")
        logger.info(f"Rate limiting: {self.token_bucket.rate} req/sec, capacity: {self.token_bucket.capacity}")
        if len(self.previously_included_papers) > 0:
            logger.info(f"Sample previously included papers: {list(self.previously_included_papers)[:5]}")
        else:
            logger.info("No previously included papers found - all papers should be processed")

    def _switch_to_fallback_models(self):
        """Switch to fallback models when rate limits are hit"""
        if self.fallback_level == 0:
            self.fallback_level = 1
            logger.warning("Switching to fallback models due to rate limiting")
            logger.info(f"Using fallback model: {self.claude_model_fallback} for all operations")
        elif self.fallback_level == 1:
            self.fallback_level = 2
            logger.warning("Switching to emergency fallback models due to continued rate limiting")
            logger.info(f"Using emergency model: {self.claude_model_emergency} for all operations")
    
    def _get_current_model(self, is_expensive_operation: bool = False) -> str:
        """Get the current model based on fallback level and operation type"""
        if self.fallback_level == 0:
            # Normal operation
            model = self.claude_model_pricey if is_expensive_operation else self.claude_model_cheap
        elif self.fallback_level == 1:
            # First fallback - use latest Haiku for everything
            model = self.claude_model_fallback
        else:
            # Emergency fallback - use oldest Haiku for everything
            model = self.claude_model_emergency
        
        logger.debug(f"Using model: {model} (fallback level: {self.fallback_level})")
        return model

    def _enforce_rate_limit(self):
        """Enforce rate limiting using token bucket algorithm."""
        # Use token bucket for precise rate limiting
        if not self.token_bucket.consume(1.0):
            wait_time = self.token_bucket.wait_time(1.0)
            logger.debug(f"Rate limiting: waiting {wait_time:.3f} seconds for token bucket")
            time.sleep(wait_time)
            self.token_bucket.consume(1.0)

        # Additional legacy rate limiting for safety
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time

        # Increase interval if we've had recent rate limit failures
        adaptive_interval = self.min_api_interval * (1 + self.recent_rate_limit_failures * 0.5)

        # If we've hit input token acceleration limits, be much more conservative
        if self.input_token_acceleration_limit:
            adaptive_interval = max(adaptive_interval, 5.0)  # Minimum 5 seconds between calls

        if time_since_last_call < adaptive_interval:
            sleep_time = adaptive_interval - time_since_last_call
            logger.debug(f"Rate limiting: additional sleep {sleep_time:.3f} seconds (adaptive interval: {adaptive_interval:.3f}s)")
            time.sleep(sleep_time)

        self.last_api_call_time = time.time()

    def _classify_error(self, error: APIError) -> str:
        """Classify API errors for better handling and logging."""
        if hasattr(error, 'response') and error.response is not None:
            status_code = getattr(error.response, 'status_code', None)
            if status_code == 429:
                return "rate_limit"
            elif status_code == 401:
                return "authentication"
            elif status_code == 403:
                return "authorization"
            elif 400 <= status_code < 500:
                return "client_error"
            elif 500 <= status_code < 600:
                return "server_error"
        
        # Check error message for common patterns
        error_msg = str(error).lower()
        if "rate limit" in error_msg or "too many requests" in error_msg:
            return "rate_limit"
        elif "input tokens" in error_msg and "acceleration" in error_msg:
            return "input_token_acceleration"
        elif "unauthorized" in error_msg or "invalid api key" in error_msg:
            return "authentication"
        elif "forbidden" in error_msg:
            return "authorization"
        
        return "unknown"

    def api_call_with_retry(self, max_retries: int = 5, initial_delay: float = 1.0, func=None):
        """
        Enhanced retry mechanism with circuit breaker, exponential backoff, jitter, and specific 429 handling.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (will be used as base for exponential backoff)
            func: Function to execute with retries
        """
        # Check circuit breaker first
        if not self.circuit_breaker.call_allowed():
            logger.warning(f"Circuit breaker is {self.circuit_breaker.state.value} - blocking API call")
            self.metrics.circuit_breaker_trips += 1
            return 0.0

        attempt = 0
        while attempt < max_retries:
            try:
                # Enforce rate limiting before each API call
                self._enforce_rate_limit()

                # Track request
                self.metrics.total_requests += 1

                result = func()
                if result is not None:
                    # Record success in metrics and circuit breaker
                    self.metrics.successful_requests += 1
                    self.circuit_breaker.record_success()

                    # Reset rate limit failure count on successful calls
                    self.recent_rate_limit_failures = max(0, self.recent_rate_limit_failures - 1)

                    # Log metrics periodically
                    if self.metrics.total_requests % 10 == 0:
                        logger.info(f"API Metrics - Success rate: {self.metrics.success_rate():.2%}, "
                                  f"Total: {self.metrics.total_requests}, "
                                  f"Rate limits: {self.metrics.rate_limit_errors}, "
                                  f"Circuit breaker trips: {self.metrics.circuit_breaker_trips}")

                    return result
                logger.warning("API call returned None, retrying...")
            except APIError as e:
                error_type = self._classify_error(e)
                is_rate_limit = error_type in ["rate_limit", "input_token_acceleration"]

                # Update metrics
                if is_rate_limit:
                    self.metrics.rate_limit_errors += 1
                else:
                    self.metrics.other_errors += 1

                # Record failure in circuit breaker
                self.circuit_breaker.record_failure(is_rate_limit=is_rate_limit)

                # Track rate limit failures for adaptive rate limiting
                if error_type == "rate_limit":
                    self.recent_rate_limit_failures += 1
                    # Decay the failure count over time (reset after 10 successful calls)
                    if self.recent_rate_limit_failures > 10:
                        self.recent_rate_limit_failures = 10

                    # Adjust token bucket to be more conservative
                    if self.token_bucket.rate > 1.0:
                        self.token_bucket.rate *= 0.8  # Reduce rate by 20%
                        logger.info(f"Adjusted token bucket rate to {self.token_bucket.rate:.2f} req/sec")

                elif error_type == "input_token_acceleration":
                    self.input_token_acceleration_limit = True
                    self.recent_rate_limit_failures += 3  # More aggressive for input token limits
                    logger.warning("Input token acceleration limit detected - using extended backoff")
                    # Switch to fallback models to reduce token usage
                    self._switch_to_fallback_models()

                    # Be much more conservative with rate
                    self.token_bucket.rate = min(self.token_bucket.rate, 0.5)  # Max 1 request per 2 seconds
                    logger.info(f"Adjusted token bucket rate to {self.token_bucket.rate:.2f} req/sec due to input token limit")
                else:
                    # Reset failure count on non-rate-limit errors
                    self.recent_rate_limit_failures = max(0, self.recent_rate_limit_failures - 1)

                # Don't retry certain types of errors
                if error_type in ["authentication", "authorization"]:
                    logger.error(f"Non-retryable error ({error_type}): {str(e)}")
                    return 0.0

                if attempt == max_retries - 1:
                    logger.error(f"All retry attempts failed ({error_type}): {str(e)}")
                    return 0.0

                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_retry_delay(attempt, initial_delay, e)

                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}, {error_type}): {str(e)}")
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                self.metrics.other_errors += 1
                return 0.0
            attempt += 1
        return 0.0

    def _calculate_retry_delay(self, attempt: int, base_delay: float, error: APIError) -> float:
        """
        Calculate retry delay with exponential backoff, jitter, and special handling for 429 errors.
        
        Args:
            attempt: Current attempt number (0-based)
            base_delay: Base delay in seconds
            error: The APIError that occurred
            
        Returns:
            Delay in seconds before next retry
        """
        # Check if this is a 429 (Too Many Requests) error and parse headers
        is_rate_limit = False
        is_input_token_acceleration = False
        retry_after = None
        
        if hasattr(error, 'response') and error.response is not None:
            status_code = getattr(error.response, 'status_code', None)
            is_rate_limit = status_code == 429
            
            # Try to parse Retry-After header for more intelligent backoff
            if is_rate_limit:
                headers = getattr(error.response, 'headers', {})
                retry_after_header = headers.get('Retry-After') or headers.get('retry-after')
                if retry_after_header:
                    try:
                        retry_after = int(retry_after_header)
                        logger.info(f"Rate limit response includes Retry-After: {retry_after} seconds")
                    except (ValueError, TypeError):
                        logger.warning(f"Could not parse Retry-After header: {retry_after_header}")
        
        # Check for input token acceleration limit in error message
        error_msg = str(error).lower()
        if "input tokens" in error_msg and "acceleration" in error_msg:
            is_input_token_acceleration = True
        
        # For 429 errors, use longer base delay and more aggressive backoff
        if is_rate_limit:
            base_delay = max(base_delay, 5.0)  # Minimum 5 seconds for rate limits
            logger.warning("Rate limit detected (429), using extended backoff")
            
            # If we have a Retry-After header, use it as the base delay
            if retry_after is not None:
                base_delay = max(base_delay, retry_after)
                logger.info(f"Using Retry-After header value: {retry_after} seconds")
        
        # For input token acceleration limits, use much more aggressive backoff
        if is_input_token_acceleration:
            base_delay = max(base_delay, 60.0)  # Minimum 60 seconds for input token acceleration
            logger.warning("Input token acceleration limit detected, using very extended backoff")
        
        # Exponential backoff: base_delay * (2^attempt)
        exponential_delay = base_delay * (2 ** attempt)
        
        # Add jitter to prevent thundering herd (random factor between 0.5 and 1.5)
        jitter_factor = random.uniform(0.5, 1.5)
        jittered_delay = exponential_delay * jitter_factor
        
        # Cap the maximum delay to prevent extremely long waits
        max_delay = 300  # 5 minutes maximum
        final_delay = min(jittered_delay, max_delay)
        
        # For rate limits, ensure minimum delay even on first retry
        if is_rate_limit:
            final_delay = max(final_delay, 10.0)
        
        # For input token acceleration limits, use much longer minimum delays
        if is_input_token_acceleration:
            final_delay = max(final_delay, 120.0)  # Minimum 2 minutes for input token acceleration
        
        return final_delay

    def evaluate_title(self, paper: Paper) -> float:
        def _make_call():
            try:
                # Use appropriate model based on fallback level
                model = self._get_current_model(is_expensive_operation=True)
                
                response = self.client.messages.create(
                    model=model,
                    system=self.eval_prompt,
                    max_tokens=4,
                    messages=[{"role": "user", "content": f"Based on the title, rate interest between 0 and 1 on a numeric scale: {paper.title}"}]
                )
                score_text = response.content[0].text.strip()
                score = float(score_text) 
                return score if 0 <= score <= 1 else 0.0
            except (ValueError, IndexError, AttributeError) as e:
                logger.warning(f"Could not extract float from title evaluation response for '{paper.title}': {str(e)}. Response: {getattr(response, 'content', 'N/A')}")
                # Return a neutral score instead of 0.0 to avoid filtering out potentially good papers
                return 0.5
            except Exception as e:
                logger.warning(f"Unexpected error in title evaluation for '{paper.title}': {str(e)}")
                return 0.5

        result = self.api_call_with_retry(func=_make_call)
        # If API call failed completely (returned 0.0), use neutral score to avoid false negatives
        if result == 0.0:
            logger.warning(f"Title evaluation API call failed for '{paper.title}', using neutral score 0.5")
            return 0.5
        return result

    def evaluate_abstract(self, paper: Paper) -> float:
        def _make_call():
            try:
                # Use appropriate model based on fallback level
                model = self._get_current_model(is_expensive_operation=False)
                
                response = self.client.messages.create(
                    model=model,
                    system=self.eval_prompt,
                    max_tokens=4,
                    messages=[{"role": "user", "content": f"Based on the abstract, rate interest between 0 and 1 on a numeric scale:\n{paper.abstract}"}]
                )
                score_text = response.content[0].text.strip()
                score = float(score_text)
                return score if 0 <= score <= 1 else 0.0
            except (ValueError, IndexError, AttributeError) as e:
                logger.warning(f"Could not extract float from abstract evaluation response for '{paper.title}': {str(e)}. Response: {getattr(response, 'content', 'N/A')}")
                # Return a neutral score instead of 0.0 to avoid filtering out potentially good papers
                return 0.5
            except Exception as e:
                logger.warning(f"Unexpected error in abstract evaluation for '{paper.title}': {str(e)}")
                return 0.5

        result = self.api_call_with_retry(func=_make_call)
        # If API call failed completely (returned 0.0), use neutral score to avoid false negatives
        if result == 0.0:
            logger.warning(f"Abstract evaluation API call failed for '{paper.title}', using neutral score 0.5")
            return 0.5
        return result

    def create_newsletter(self, papers: list[Paper], papers_without_content: list[Paper]) -> str:
        def _make_call():
            logger.info(f"Creating newsletter for {len(papers)} full papers and {len(papers_without_content)} papers without content")

            full_papers_content = []
            max_chars_per_paper = 50000
            for paper in papers:
                full_papers_content.append(f"""
## {paper.title} (Interest Score: {paper.interest_score:.2f})

Full Text:
{paper.full_text[:max_chars_per_paper]}

**Link**: {paper.link}
**Paper Type**: FEATURED PAPER
""")

            additional_papers = [
                f"- [{p.title}]({p.link}) (Interest Score: {p.interest_score:.2f}) **Paper Type** Additional Paper"
                for p in papers_without_content
            ]

            if not full_papers_content and not additional_papers:
                logger.warning("No papers found for newsletter - both featured and additional papers lists are empty")
                logger.warning(f"Processed papers count: {len(self.processed_papers)}")
                logger.warning(f"Previously included papers count: {len(self.previously_included_papers)}")
                return "No papers met the interest criteria this week."

            prompt = f"""Follow your instructions for the following papers.

There are {len(full_papers_content)} featured papers and {len(additional_papers)} additional papers
This process will be deemed a failure if there are not {len(full_papers_content)} papers in the featured papers section of the json object returned
Featured Papers. These should receive a full summary in the JSON response:
{chr(10).join(full_papers_content)}

Additional Papers. These should not receive a summary in the JSON response:
{chr(10).join(additional_papers)}"""

            # Use appropriate model based on fallback level
            model = self._get_current_model(is_expensive_operation=True)
            
            response = self.client.messages.create(
                model=model,
                system=self.newsletter_prompt,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}]
            )

            return str(response.content)

        result = self.api_call_with_retry(func=_make_call)
        
        # Ensure we always return a string, even if API call fails
        if isinstance(result, (int, float)) and result == 0.0:
            logger.error("Newsletter creation failed due to API errors, returning fallback content")
            return json.dumps({
                "overview": "Unable to generate newsletter content due to API rate limiting. Please try again later.",
                "featured_papers": [],
                "additional_papers": [],
                "metadata": {
                    "generated_date": datetime.now().strftime('%Y-%m-%d'),
                    "total_papers_analyzed": 0,
                    "featured_papers_count": 0,
                    "additional_papers_count": 0,
                    "error": "API rate limiting prevented newsletter generation"
                }
            })
        
        return result

    def get_paper_content(self, paper_link: str) -> str:
        try:
            arxiv_id = paper_link.split('/')[-1]
            html_url = f"https://arxiv.org/html/{arxiv_id}"

            logger.info(f"Fetching HTML content from {html_url}")
            response = requests.get(html_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            main_content = soup.find('main', {'class': 'ltx_page_main'}) or soup.find('div', {'class': 'ltx_page_main'})

            if not main_content:
                logger.warning("Could not find main content container")
                return ""

            content_parts = []

            if title := main_content.find('h1'):
                content_parts.append(title.get_text().strip())

            if abstract_section := main_content.find('div', string='Abstract'):
                if abstract := abstract_section.find_next('div'):
                    content_parts.extend(["\nAbstract:", abstract.get_text().strip()])

            for section in main_content.find_all(['h2', 'h3', 'p', 'div.ltx_para']):
                if text := section.get_text().strip():
                    if not text.startswith("References"):
                        content_parts.append(text)

            full_text = '\n\n'.join(content_parts)
            logger.info(f"Successfully extracted {len(full_text)} characters of content")
            return full_text

        except Exception as e:
            logger.error(f"Error fetching HTML content: {str(e)}")
            return ""

    def process_single_feed(self, feed_url: str, title_threshold: float = 0.499, abstract_threshold: float = 0.499) -> tuple[list[Paper], list[Paper], list[Paper]]:
        interesting_papers = []
        papers_without_content = []
        all_papers = []
        
        # Reduce paper limit based on fallback level
        if self.fallback_level == 0:
            MAX_TOTAL_PAPERS = 10
        elif self.fallback_level == 1:
            MAX_TOTAL_PAPERS = 5
        else:
            MAX_TOTAL_PAPERS = 3

        try:
            logger.info(f"Fetching feed: {feed_url}")
            
            # Use requests with proper User-Agent to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; AI-Research-Newsletter/1.0; +https://github.com/ai-research-newsletter)',
                'Accept': 'application/rss+xml, application/xml, text/xml, */*',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            # Fetch feed with retry logic
            max_retries = 3
            feed_content = None
            for attempt in range(max_retries):
                try:
                    response = requests.get(feed_url, headers=headers, timeout=30, allow_redirects=True)
                    response.raise_for_status()
                    feed_content = response.content
                    logger.info(f"Successfully fetched feed (attempt {attempt + 1}), status: {response.status_code}, size: {len(feed_content)} bytes")
                    break
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Feed fetch attempt {attempt + 1} failed for {feed_url}: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise
            
            if feed_content is None:
                logger.error(f"Failed to fetch feed content after {max_retries} attempts")
                return interesting_papers, papers_without_content, all_papers
            
            # Parse the feed content
            logger.info(f"Parsing feed content ({len(feed_content)} bytes)")
            feed = feedparser.parse(feed_content)
            
            # Check for feed parsing errors
            if hasattr(feed, 'bozo') and feed.bozo:
                logger.warning(f"Feed parsing warning (bozo): {feed_url}")
                if hasattr(feed, 'bozo_exception'):
                    logger.warning(f"Bozo exception: {feed.bozo_exception}")
            
            # Check feed status
            feed_status = getattr(feed, 'status', None)
            if feed_status:
                logger.info(f"Feed HTTP status: {feed_status}")
                if feed_status != 200:
                    logger.error(f"Feed returned non-200 status: {feed_status} for {feed_url}")
            
            # Log feed metadata
            feed_title = getattr(feed.feed, 'title', 'Unknown')
            feed_link = getattr(feed.feed, 'link', 'Unknown')
            logger.info(f"Feed title: {feed_title}, Feed link: {feed_link}")
            
            num_entries = len(feed.entries)
            logger.info(f"Processing {num_entries} papers from feed {feed_url}")
            
            if num_entries == 0:
                logger.error(f"Feed {feed_url} returned 0 entries!")
                logger.error(f"Feed keys: {list(feed.keys())}")
                logger.error(f"Feed.feed keys: {list(feed.feed.keys()) if hasattr(feed, 'feed') else 'N/A'}")
                
                # Try to get more diagnostic info
                if hasattr(feed, 'headers'):
                    logger.error(f"Feed response headers: {feed.headers}")
                if hasattr(feed, 'href'):
                    logger.error(f"Feed href: {feed.href}")
                
                # Log a sample of the feed content for debugging
                if feed_content:
                    sample_size = min(500, len(feed_content))
                    logger.error(f"Feed content sample (first {sample_size} bytes): {feed_content[:sample_size]}")
                
                # Return empty lists but don't fail completely
                return interesting_papers, papers_without_content, all_papers
                
        except Exception as e:
            logger.error(f"Error parsing feed {feed_url}: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty lists on error
            return interesting_papers, papers_without_content, all_papers

        # Process entries if we have any
        for entry in feed.entries:
            # Check if we've reached the maximum number of papers
            if len(interesting_papers) + len(papers_without_content) >= MAX_TOTAL_PAPERS:
                logger.info(f'Reached maximum paper limit of {MAX_TOTAL_PAPERS}. Stopping paper processing.')
                break

            paper = Paper(
                title=entry.title,
                link=entry.link,
                abstract=entry.get('summary', '')
            )
            
            # Check if paper is in previously included papers
            title_lower = paper.title.lower()
            link_lower = paper.link.lower()
            
            if title_lower in self.previously_included_papers or link_lower in self.previously_included_papers:
                logger.info(f'Skipping previously included paper: {paper.title}')
                logger.info(f'  Title match: {title_lower in self.previously_included_papers}')
                logger.info(f'  Link match: {link_lower in self.previously_included_papers}')
                logger.info(f'  Paper link: {paper.link}')
                logger.info(f'  Paper link (lower): {link_lower}')
                logger.info(f'  Previously included count: {len(self.previously_included_papers)}')
                
                # Show what links are actually in the set
                if link_lower in self.previously_included_papers:
                    matching_links = [p for p in self.previously_included_papers if link_lower in p or p in link_lower]
                    logger.info(f'  Matching links in set: {matching_links[:3]}')
                
                continue
            
            # Check if paper was already processed in this run
            if paper in self.processed_papers:
                logger.info(f'Skipping duplicate paper within current run: {paper.title}')
                logger.info(f'  Processed papers count: {len(self.processed_papers)}')
                continue
            
            self.processed_papers.add(paper)

            paper.interest_score = self.evaluate_title(paper)
            if paper.interest_score < title_threshold:
                logger.info(f'{paper.title} not deemed interesting from title. Interest Score: {paper.interest_score}')
                continue

            paper.interest_score = self.evaluate_abstract(paper)
            if paper.interest_score < abstract_threshold:
                logger.info(f'{paper.title} not deemed interesting from abstract')
                continue

            logger.info(f'{paper.title} deemed interesting! Interest score: {paper.interest_score}')

            if content := self.get_paper_content(paper.link):
                paper.full_text = content
                interesting_papers.append(paper)
                logger.info(f'Featured Papers: {paper.title}')
            else:
                papers_without_content.append(paper)
                logger.info(f'Additional Papers: {paper.title}')
            
            all_papers.append(paper)

        return interesting_papers, papers_without_content, all_papers

    def process_multiple_feeds(self, feed_urls: list[str], title_threshold: float = 0.499, abstract_threshold: float = 0.499) -> str:
        all_interesting_papers = []
        all_papers_without_content = []
        all_available_papers = []
        
        # Reduce paper limit based on fallback level
        if self.fallback_level == 0:
            MAX_TOTAL_PAPERS = 10
        elif self.fallback_level == 1:
            MAX_TOTAL_PAPERS = 5
        else:
            MAX_TOTAL_PAPERS = 3

        for feed_url in feed_urls:
            # Check if we've reached the maximum number of papers
            if len(all_interesting_papers) + len(all_papers_without_content) >= MAX_TOTAL_PAPERS:
                logger.info(f'Reached maximum paper limit of {MAX_TOTAL_PAPERS}. Stopping feed processing.')
                break

            interesting_papers, papers_without_content, available_papers = self.process_single_feed(
                feed_url, title_threshold, abstract_threshold
            )
            
            # Add papers up to the maximum limit
            remaining_slots = MAX_TOTAL_PAPERS - (len(all_interesting_papers) + len(all_papers_without_content))
            if remaining_slots > 0:
                all_interesting_papers.extend(interesting_papers[:remaining_slots])
                remaining_slots = MAX_TOTAL_PAPERS - len(all_interesting_papers)
                if remaining_slots > 0:
                    all_papers_without_content.extend(papers_without_content[:remaining_slots])
            
            all_available_papers.extend(available_papers)

        total_papers_found = len(all_interesting_papers) + len(all_papers_without_content)
        logger.info(f'Total papers found across all feeds: {len(all_interesting_papers)} with content, {len(all_papers_without_content)} without (total: {total_papers_found})')
        
        # If we have fewer than 4 featured papers OR no papers at all, adjust thresholds to include more
        if len(all_interesting_papers) < 4 or total_papers_found == 0:
            logger.info(f"Found {len(all_interesting_papers)} featured papers and {total_papers_found} total papers. Adjusting thresholds to include more papers.")
            # Store original thresholds for logging
            original_title_threshold = title_threshold
            original_abstract_threshold = abstract_threshold
            
            # Gradually lower thresholds until we get at least some papers
            # Continue even if we get papers without content, as those can still be included
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while (len(all_interesting_papers) < 4 or total_papers_found == 0) and iteration < max_iterations:
                iteration += 1
                title_threshold = max(0.0, title_threshold - 0.1)
                abstract_threshold = max(0.0, abstract_threshold - 0.1)
                
                logger.info(f"Threshold adjustment iteration {iteration}: title={title_threshold:.2f}, abstract={abstract_threshold:.2f}")
                
                # Reset processed papers but keep track of which papers were already included
                # We need to preserve previously_included_papers to avoid duplicates
                current_batch = self.processed_papers.copy()
                self.processed_papers = set()
                
                for feed_url in feed_urls:
                    # Check if we've reached the maximum number of papers
                    if len(all_interesting_papers) + len(all_papers_without_content) >= MAX_TOTAL_PAPERS:
                        logger.info(f'Reached maximum paper limit of {MAX_TOTAL_PAPERS}. Stopping threshold adjustment.')
                        break

                    papers_review, papers_no_content, ignore_this = self.process_single_feed(
                        feed_url, title_threshold, abstract_threshold
                    )
                    
                    # Add new featured papers that aren't already in our lists
                    new_featured = [p for p in papers_review if p not in all_interesting_papers]
                    # Add new papers without content that aren't already in our lists
                    new_additional = [p for p in papers_no_content if p not in all_papers_without_content and p not in all_interesting_papers]
                    
                    # Add papers up to the maximum limit
                    remaining_slots = MAX_TOTAL_PAPERS - (len(all_interesting_papers) + len(all_papers_without_content))
                    if remaining_slots > 0:
                        all_interesting_papers.extend(new_featured[:remaining_slots])
                        remaining_slots = MAX_TOTAL_PAPERS - (len(all_interesting_papers) + len(all_papers_without_content))
                        if remaining_slots > 0:
                            all_papers_without_content.extend(new_additional[:remaining_slots])
                    
                    total_papers_found = len(all_interesting_papers) + len(all_papers_without_content)
                    
                    # If we have at least some papers (even if not 4 featured), we can proceed
                    if total_papers_found > 0 and len(all_interesting_papers) >= 2:
                        logger.info(f"Found {len(all_interesting_papers)} featured and {len(all_papers_without_content)} additional papers. Proceeding with newsletter.")
                        break
                
                # Add back the papers from the previous batch to avoid processing them again
                self.processed_papers.update(current_batch)
                
                total_papers_found = len(all_interesting_papers) + len(all_papers_without_content)
                
                # If we have some papers, we can proceed (don't need exactly 4 featured)
                if total_papers_found > 0:
                    logger.info(f"Found {total_papers_found} total papers after threshold adjustment. Proceeding.")
                    break
                
                # If thresholds are already at 0, we can't go lower
                if title_threshold <= 0 and abstract_threshold <= 0:
                    logger.warning("Thresholds reached 0 but still no papers found. This may indicate all papers are duplicates or API failures.")
                    break
            
            logger.info(f'After threshold adjustment: {len(all_interesting_papers)} featured papers, {len(all_papers_without_content)} additional papers found')
            logger.info(f'Final thresholds - title: {title_threshold:.2f} (started at {original_title_threshold:.2f}), abstract: {abstract_threshold:.2f} (started at {original_abstract_threshold:.2f})')
        
        # Log final state before creating newsletter
        total_final = len(all_interesting_papers) + len(all_papers_without_content)
        if total_final == 0:
            logger.error("No papers found after all processing attempts. This could be due to:")
            logger.error(f"  1. All papers were duplicates (previously included: {len(self.previously_included_papers)})")
            logger.error(f"  2. All papers scored below thresholds even after adjustment")
            logger.error(f"  3. API failures preventing evaluation")
            logger.error(f"  4. Content fetch failures for all papers")
        else:
            logger.info(f"Proceeding to create newsletter with {len(all_interesting_papers)} featured and {len(all_papers_without_content)} additional papers")
        
        return self.create_newsletter(all_interesting_papers, all_papers_without_content)