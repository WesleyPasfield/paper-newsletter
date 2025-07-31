# functions/paper_analyzer.py
import feedparser
import requests
import time
import json
from datetime import datetime
import logging
from bs4 import BeautifulSoup
from typing import Set, Dict, Optional
from dataclasses import dataclass

from .llm_providers import LLMManager, LLMProvider, LLMConfig, DEFAULT_CONFIGS
from .dspy_prompts import DSPyPromptManager, TRAINING_EXAMPLES

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
    evaluation_reasoning: str = ""
    provider_used: Optional[str] = None

    def __hash__(self):
        return hash((self.title, self.link))
    
    def __eq__(self, other):
        if not isinstance(other, Paper):
            return False
        return self.title == other.title or self.link == other.link

class PaperAnalyzer:
    def __init__(self, eval_prompt: str = None, newsletter_prompt: str = None, previously_included_papers: Set[str] = None, preferred_provider: Optional[LLMProvider] = None):
        # Initialize multi-LLM system
        self.llm_manager = LLMManager()
        self.preferred_provider = preferred_provider or LLMProvider.CLAUDE
        
        # Initialize DSPy prompt manager
        self.dspy_manager = DSPyPromptManager(self.llm_manager)
        
        # Optimize prompts with training examples
        try:
            self.dspy_manager.optimize_prompts(TRAINING_EXAMPLES)
        except Exception as e:
            logger.warning(f"Prompt optimization failed, using default prompts: {str(e)}")
        
        # Legacy prompt support (deprecated but maintained for compatibility)
        self.eval_prompt = eval_prompt or self.dspy_manager.evaluation_criteria
        self.newsletter_prompt = newsletter_prompt or self.dspy_manager.newsletter_guidelines
        
        self.processed_papers: Set[Paper] = set()
        self.previously_included_papers = previously_included_papers or set()
        
        # Convert all previously included papers to lowercase for case-insensitive matching
        self.previously_included_papers = {p.lower() for p in self.previously_included_papers}
        
        available_providers = self.llm_manager.get_available_providers()
        logger.info(f"Initialized PaperAnalyzer with {len(available_providers)} LLM providers: {available_providers}")
        logger.info(f"Previously included papers: {len(self.previously_included_papers)}")

    def _get_llm_configs(self) -> Dict[LLMProvider, LLMConfig]:
        """Get LLM configurations for fallback"""
        configs = {}
        
        # Add available providers with their configs
        for provider in self.llm_manager.get_available_providers():
            if provider in DEFAULT_CONFIGS:
                configs[provider] = DEFAULT_CONFIGS[provider]['cheap']
                
        return configs

    def evaluate_title(self, paper: Paper) -> float:
        """Evaluate paper based on title using DSPy and multi-LLM system"""
        try:
            # Use DSPy for evaluation
            score, reasoning = self.dspy_manager.evaluate_paper(paper.title, "")
            paper.evaluation_reasoning = reasoning
            
            logger.info(f"Title evaluation for '{paper.title}': {score:.3f} - {reasoning[:100]}...")
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating title for {paper.title}: {str(e)}")
            # Fallback to legacy method with multi-LLM
            return self._legacy_evaluate_title(paper)
    
    def _legacy_evaluate_title(self, paper: Paper) -> float:
        """Legacy title evaluation with multi-LLM fallback"""
        try:
            configs = self._get_llm_configs()
            if not configs:
                logger.error("No LLM providers available")
                return 0.0
            
            user_prompt = f"Based on the title, rate interest between 0 and 1 on a numeric scale: {paper.title}"
            
            response = self.llm_manager.generate_with_fallback(
                system_prompt=self.eval_prompt,
                user_prompt=user_prompt,
                configs=configs,
                preferred_provider=self.preferred_provider
            )
            
            paper.provider_used = response.provider.value
            score = float(response.content.strip())
            return score if 0 <= score <= 1 else 0.0
            
        except Exception as e:
            logger.warning(f"Could not extract float from response: {str(e)}")
            return 0.0

    def evaluate_abstract(self, paper: Paper) -> float:
        """Evaluate paper based on abstract using DSPy and multi-LLM system"""
        try:
            # Use DSPy for evaluation with both title and abstract
            score, reasoning = self.dspy_manager.evaluate_paper(paper.title, paper.abstract)
            paper.evaluation_reasoning = reasoning
            
            logger.info(f"Abstract evaluation for '{paper.title}': {score:.3f} - {reasoning[:100]}...")
            return score
            
        except Exception as e:
            logger.error(f"Error evaluating abstract for {paper.title}: {str(e)}")
            # Fallback to legacy method with multi-LLM
            return self._legacy_evaluate_abstract(paper)
    
    def _legacy_evaluate_abstract(self, paper: Paper) -> float:
        """Legacy abstract evaluation with multi-LLM fallback"""
        try:
            configs = self._get_llm_configs()
            if not configs:
                logger.error("No LLM providers available")
                return 0.0
            
            user_prompt = f"Based on the abstract, rate interest between 0 and 1 on a numeric scale:\n{paper.abstract}"
            
            response = self.llm_manager.generate_with_fallback(
                system_prompt=self.eval_prompt,
                user_prompt=user_prompt,
                configs=configs,
                preferred_provider=self.preferred_provider
            )
            
            paper.provider_used = response.provider.value
            score = float(response.content.strip())
            return score if 0 <= score <= 1 else 0.0
            
        except Exception as e:
            logger.warning(f"Could not extract float from response: {str(e)}")
            return 0.0

    def create_newsletter(self, papers: list[Paper], papers_without_content: list[Paper]) -> str:
        """Create newsletter using DSPy and multi-LLM system"""
        logger.info(f"Creating newsletter for {len(papers)} full papers and {len(papers_without_content)} papers without content")
        try:
            # Prepare content for DSPy
            featured_papers_content = self._format_featured_papers(papers)
            additional_papers_content = self._format_additional_papers(papers_without_content)
            if not featured_papers_content and not additional_papers_content:
                return "No papers met the interest criteria this week."
            
            # Use DSPy for newsletter generation
            newsletter_json = self.dspy_manager.generate_newsletter(
                featured_papers_content=featured_papers_content,
                additional_papers_content=additional_papers_content
            )
            logger.info("Successfully generated newsletter using DSPy")
            return newsletter_json
            
        except Exception as e:
            logger.error(f"Error generating newsletter with DSPy: {str(e)}")
            # Fallback to legacy method
            return self._legacy_create_newsletter(papers, papers_without_content)
    
    def _format_featured_papers(self, papers: list[Paper]) -> str:
        """Format featured papers for DSPy processing"""
        full_papers_content = []
        max_chars_per_paper = 50000
        
        for paper in papers:
            paper_content = f"""
## {paper.title} (Interest Score: {paper.interest_score:.2f})

**Evaluation Reasoning**: {paper.evaluation_reasoning or 'N/A'}
**Provider Used**: {paper.provider_used or 'N/A'}

Full Text:
{paper.full_text[:max_chars_per_paper]}

**Link**: {paper.link}
**Paper Type**: FEATURED PAPER
"""
            full_papers_content.append(paper_content)
        
        return chr(10).join(full_papers_content)
    
    def _format_additional_papers(self, papers: list[Paper]) -> str:
        """Format additional papers for DSPy processing"""
        additional_papers = [
            f"- [{p.title}]({p.link}) (Interest Score: {p.interest_score:.2f}, Provider: {p.provider_used or 'N/A'}) **Paper Type** Additional Paper"
            for p in papers
        ]
        return chr(10).join(additional_papers)
    
    def _legacy_create_newsletter(self, papers: list[Paper], papers_without_content: list[Paper]) -> str:
        """Legacy newsletter creation with multi-LLM fallback"""
        try:
            configs = {}
            for provider in self.llm_manager.get_available_providers():
                if provider in DEFAULT_CONFIGS:
                    configs[provider] = DEFAULT_CONFIGS[provider]['expensive']
            
            if not configs:
                logger.error("No LLM providers available for newsletter generation")
                return "Error: No LLM providers available"
            
            featured_papers_content = self._format_featured_papers(papers)
            additional_papers_content = self._format_additional_papers(papers_without_content)
            
            prompt = f"""Follow your instructions for the following papers.

There are {len(papers)} featured papers and {len(papers_without_content)} additional papers
This process will be deemed a failure if there are not {len(papers)} papers in the featured papers section of the json object returned
Featured Papers. These should receive a full summary in the JSON response:
{featured_papers_content}

Additional Papers. These should not receive a summary in the JSON response:
{additional_papers_content}"""
            
            response = self.llm_manager.generate_with_fallback(
                system_prompt=self.newsletter_prompt,
                user_prompt=prompt,
                configs=configs,
                preferred_provider=self.preferred_provider
            )
            
            logger.info(f"Newsletter generated using {response.provider.value}")
            return response.content
            
        except Exception as e:
            logger.error(f"Error in legacy newsletter creation: {str(e)}")
            return f"Error generating newsletter: {str(e)}"

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

    def process_single_feed(self, feed_url: str, title_threshold: float = 0.24, abstract_threshold: float = 0.24) -> tuple[list[Paper], list[Paper], list[Paper]]:
        feed = feedparser.parse(feed_url)
        interesting_papers = []
        papers_without_content = []
        all_papers = []
        MAX_TOTAL_PAPERS = 10  # Maximum total papers to include

        logger.info(f"Processing {len(feed.entries)} papers from feed {feed_url}")

        for i, entry in enumerate(feed.entries):
            logger.info(f"Processing paper {i+1}/{len(feed.entries)}: {entry.title[:50]}...")
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
            logger.info(f'{paper.title} title evaluation score: {paper.interest_score:.3f} (threshold: {title_threshold:.3f})')
            if paper.interest_score < title_threshold:
                logger.info(f'{paper.title} not deemed interesting from title. Interest Score: {paper.interest_score}')
                continue

            paper.interest_score = self.evaluate_abstract(paper)
            logger.info(f'{paper.title} abstract evaluation score: {paper.interest_score:.3f} (threshold: {abstract_threshold:.3f})')
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

    def process_multiple_feeds(self, feed_urls: list[str], title_threshold: float = 0.24, abstract_threshold: float = 0.24) -> str:
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
            while len(all_interesting_papers) < 7 and (title_threshold > 0 or abstract_threshold > 0):
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
                    
                    if len(all_interesting_papers) >= 7:
                        break
                
                # Add back the papers from the previous batch to avoid processing them again
                self.processed_papers.update(current_batch)
                
                if title_threshold <= 0 and abstract_threshold <= 0:
                    break
            
            logger.info(f'After threshold adjustment: {len(all_interesting_papers)} featured papers found with final thresholds - title: {title_threshold:.2f}, abstract: {abstract_threshold:.2f}')
        
        return self.create_newsletter(all_interesting_papers, all_papers_without_content)