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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    def __init__(self, anthropic_api_key: str, eval_prompt: str, newsletter_prompt: str, previously_included_papers: Set[str] = None):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.claude_model_cheap = "claude-3-haiku-20240307"
        self.claude_model_pricey = "claude-3-5-sonnet-latest"
        self.eval_prompt = eval_prompt
        self.newsletter_prompt = newsletter_prompt
        self.processed_papers: Set[Paper] = set()
        self.previously_included_papers = previously_included_papers or set()
        
        # Convert all previously included papers to lowercase for case-insensitive matching
        self.previously_included_papers = {p.lower() for p in self.previously_included_papers}
        
        # Rate limiting state
        self.last_api_call_time = 0.0
        self.min_api_interval = 0.1  # Minimum 100ms between API calls
        self.recent_rate_limit_failures = 0  # Track recent rate limit failures
        
        logger.info(f"Initialized PaperAnalyzer with {len(self.previously_included_papers)} previously included papers")

    def _enforce_rate_limit(self):
        """Enforce minimum interval between API calls to avoid hitting rate limits."""
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        
        # Increase interval if we've had recent rate limit failures
        adaptive_interval = self.min_api_interval * (1 + self.recent_rate_limit_failures * 0.5)
        
        if time_since_last_call < adaptive_interval:
            sleep_time = adaptive_interval - time_since_last_call
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f} seconds (adaptive interval: {adaptive_interval:.3f}s)")
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
        elif "unauthorized" in error_msg or "invalid api key" in error_msg:
            return "authentication"
        elif "forbidden" in error_msg:
            return "authorization"
        
        return "unknown"

    def api_call_with_retry(self, max_retries: int = 5, initial_delay: float = 1.0, func=None):
        """
        Enhanced retry mechanism with exponential backoff, jitter, and specific 429 handling.
        
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (will be used as base for exponential backoff)
            func: Function to execute with retries
        """
        attempt = 0
        while attempt < max_retries:
            try:
                # Enforce rate limiting before each API call
                self._enforce_rate_limit()
                
                result = func()
                if result is not None:
                    # Reset rate limit failure count on successful calls
                    self.recent_rate_limit_failures = max(0, self.recent_rate_limit_failures - 1)
                    return result
                logger.warning("API call returned None, retrying...")
            except APIError as e:
                error_type = self._classify_error(e)
                
                # Track rate limit failures for adaptive rate limiting
                if error_type == "rate_limit":
                    self.recent_rate_limit_failures += 1
                    # Decay the failure count over time (reset after 10 successful calls)
                    if self.recent_rate_limit_failures > 10:
                        self.recent_rate_limit_failures = 10
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
        
        # For 429 errors, use longer base delay and more aggressive backoff
        if is_rate_limit:
            base_delay = max(base_delay, 5.0)  # Minimum 5 seconds for rate limits
            logger.warning("Rate limit detected (429), using extended backoff")
            
            # If we have a Retry-After header, use it as the base delay
            if retry_after is not None:
                base_delay = max(base_delay, retry_after)
                logger.info(f"Using Retry-After header value: {retry_after} seconds")
        
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
        
        return final_delay

    def evaluate_title(self, paper: Paper) -> float:
        def _make_call():
            try:
                response = self.client.messages.create( # Issue here right now
                    model=self.claude_model_pricey,
                    system=self.eval_prompt,
                    max_tokens=4,
                    messages=[{"role": "user", "content": f"Based on the title, rate interest between 0 and 1 on a numeric scale: {paper.title}"}]
                )
                score = float(response.content[0].text) 
                return score if 0 <= score <= 1 else 0.0
            except Exception as e:
                logger.warning(f"Could not extract float from response: {str(e)}")
                return 0.0

        return self.api_call_with_retry(func=_make_call)

    def evaluate_abstract(self, paper: Paper) -> float:
        def _make_call():
            try:
                response = self.client.messages.create(
                    model=self.claude_model_cheap,
                    system=self.eval_prompt,
                    max_tokens=4,
                    messages=[{"role": "user", "content": f"Based on the abstract, rate interest between 0 and 1 on a numeric scale::\n{paper.abstract}"}]
                )
                score = float(response.content[0].text)
                return score if 0 <= score <= 1 else 0.0
            except Exception as e:
                logger.warning(f"Could not extract float from response: {str(e)}")
                return 0.0

        return self.api_call_with_retry(func=_make_call)

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
                return "No papers met the interest criteria this week."

            prompt = f"""Follow your instructions for the following papers.

There are {len(full_papers_content)} featured papers and {len(additional_papers)} additional papers
This process will be deemed a failure if there are not {len(full_papers_content)} papers in the featured papers section of the json object returned
Featured Papers. These should receive a full summary in the JSON response:
{chr(10).join(full_papers_content)}

Additional Papers. These should not receive a summary in the JSON response:
{chr(10).join(additional_papers)}"""

            response = self.client.messages.create(
                model=self.claude_model_pricey,
                system=self.newsletter_prompt,
                max_tokens=8000,
                messages=[{"role": "user", "content": prompt}]
            )

            return str(response.content)

        return self.api_call_with_retry(func=_make_call)

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
        feed = feedparser.parse(feed_url)
        interesting_papers = []
        papers_without_content = []
        all_papers = []
        MAX_TOTAL_PAPERS = 10  # Maximum total papers to include

        logger.info(f"Processing {len(feed.entries)} papers from feed {feed_url}")

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
            if paper.title.lower() in self.previously_included_papers or paper.link.lower() in self.previously_included_papers:
                logger.info(f'Skipping previously included paper: {paper.title}')
                continue
            
            # Check if paper was already processed in this run
            if paper in self.processed_papers:
                logger.info(f'Skipping duplicate paper within current run: {paper.title}')
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
        MAX_TOTAL_PAPERS = 10  # Maximum total papers to include

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

        logger.info(f'Total papers found across all feeds: {len(all_interesting_papers)} with content, {len(all_papers_without_content)} without')
        
        # If we have fewer than 4 featured papers, adjust thresholds to include more
        if len(all_interesting_papers) < 4:
            logger.info("Found fewer than 4 featured papers. Adjusting thresholds to include more papers.")
            # Gradually lower thresholds until we get 4 featured papers
            while len(all_interesting_papers) < 4 and (title_threshold > 0 or abstract_threshold > 0):
                title_threshold -= 0.1
                abstract_threshold -= 0.1
                
                # Reset processed papers but keep track of which papers were already included
                # We need to preserve previously_included_papers to avoid duplicates
                current_batch = self.processed_papers.copy()
                self.processed_papers = set()
                
                for feed_url in feed_urls:
                    # Check if we've reached the maximum number of papers
                    if len(all_interesting_papers) + len(all_papers_without_content) >= MAX_TOTAL_PAPERS:
                        logger.info(f'Reached maximum paper limit of {MAX_TOTAL_PAPERS}. Stopping threshold adjustment.')
                        break

                    papers_review, ignore_this_1, ignore_this_two = self.process_single_feed(
                        feed_url, title_threshold, abstract_threshold
                    )
                    # Only add papers that aren't already in our lists
                    new_papers = [p for p in papers_review if p not in all_interesting_papers]
                    
                    # Add papers up to the maximum limit
                    remaining_slots = MAX_TOTAL_PAPERS - (len(all_interesting_papers) + len(all_papers_without_content))
                    if remaining_slots > 0:
                        all_interesting_papers.extend(new_papers[:remaining_slots])
                    
                    if len(all_interesting_papers) >= 4:
                        break
                
                # Add back the papers from the previous batch to avoid processing them again
                self.processed_papers.update(current_batch)
                
                if title_threshold <= 0 and abstract_threshold <= 0:
                    break
            
            logger.info(f'After threshold adjustment: {len(all_interesting_papers)} featured papers found with final thresholds - title: {title_threshold:.2f}, abstract: {abstract_threshold:.2f}')
        
        return self.create_newsletter(all_interesting_papers, all_papers_without_content)