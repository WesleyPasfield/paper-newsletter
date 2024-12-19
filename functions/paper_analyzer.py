# functions/paper_analyzer.py
import feedparser
import requests
import time
from anthropic import APIError, Anthropic
import json
from datetime import datetime
import logging
from bs4 import BeautifulSoup
from typing import Set, Dict
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
    def __init__(self, anthropic_api_key: str, eval_prompt: str, newsletter_prompt: str):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.claude_model_cheap = "claude-3-haiku-20240307"
        self.claude_model_pricey = "claude-3-5-sonnet-latest"
        self.eval_prompt = eval_prompt
        self.newsletter_prompt = newsletter_prompt
        self.processed_papers: Set[Paper] = set()

    def api_call_with_retry(self, max_retries: int = 5, initial_delay: int = 2, func=None):
        attempt = 0
        while attempt < max_retries:
            try:
                result = func()
                if result is not None:
                    return result
                logger.warning("API call returned None, retrying...")
            except APIError as e:
                if attempt == max_retries - 1:
                    logger.error(f"All retry attempts failed: {str(e)}")
                    return 0.0

                delay = initial_delay * (2 ** attempt)
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                return 0.0
            attempt += 1
        return 0.0

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

    def process_single_feed(self, feed_url: str, title_threshold: float = 0.499, abstract_threshold: float = 0.499) -> tuple[list[Paper], list[Paper]]:
        feed = feedparser.parse(feed_url)
        interesting_papers = []
        papers_without_content = []
        all_papers = []

        logger.info(f"Processing {len(feed.entries)} papers from feed {feed_url}")

        for entry in feed.entries:
            paper = Paper(
                title=entry.title,
                link=entry.link,
                abstract=entry.get('summary', '')
            )
            
            if paper in self.processed_papers:
                logger.info(f'Skipping duplicate paper: {paper.title}')
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

        for feed_url in feed_urls:
            interesting_papers, papers_without_content, available_papers = self.process_single_feed(
                feed_url, title_threshold, abstract_threshold
            )
            all_interesting_papers.extend(interesting_papers)
            all_papers_without_content.extend(papers_without_content)
            all_available_papers.extend(available_papers)

        logger.info(f'Total papers found across all feeds: {len(all_interesting_papers)} with content, {len(all_papers_without_content)} without')
        
        # If we have fewer than 4 featured papers, adjust thresholds to include more
        if len(all_interesting_papers) < 4:
            logger.info("Found fewer than 4 featured papers. Adjusting thresholds to include more papers.")
            # Gradually lower thresholds until we get 4 featured papers
            while len(all_interesting_papers) < 4 and (title_threshold > 0 or abstract_threshold > 0):
                title_threshold -= 0.1
                abstract_threshold -= 0.1
                self.processed_papers: Set[Paper] = set() # Reset processed papers
                
                for feed_url in feed_urls:
                    papers_review, ignore_this_1, ignore_this_two  = self.process_single_feed(
                        feed_url, title_threshold, abstract_threshold
                    )
                    # Only add papers that aren't already in our lists
                    new_papers = [p for p in papers_review if p not in all_interesting_papers]
                    all_interesting_papers.extend(new_papers)
                    
                    if len(all_interesting_papers) >= 4:
                        all_interesting_papers = all_interesting_papers[:4]
                        break
                
                if title_threshold <= 0 and abstract_threshold <= 0:
                    break
            
            logger.info(f'After threshold adjustment: {len(all_interesting_papers)} featured papers found with final thresholds - title: {title_threshold:.2f}, abstract: {abstract_threshold:.2f}')
        
        return self.create_newsletter(all_interesting_papers, all_papers_without_content)