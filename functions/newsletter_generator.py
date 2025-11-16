# functions/newsletter_generator.py
import json
import os
from .paper_analyzer import PaperAnalyzer
from .email_sender import send_newsletter
import logging
from datetime import datetime
import codecs
from typing import Dict, List, Set, Tuple
import boto3
from botocore.exceptions import ClientError
import re
from difflib import SequenceMatcher

logger = logging.getLogger()
logger.setLevel(logging.INFO)

EVAL_PROMPT = """You are evaluating academic papers for an expert specializing in:

1. Practical LLM/Generative AI regulation:
- Data-driven evaluation frameworks over compute based metrics
- Domain-specific testing methodologies
- Production deployment challenges
- User experience validation
- Certification processes for AI systems
- Impact of test or inference time optimizations on model performance and governance strategies

2. Societal and economic impacts of generative AI:
- Knowledge work transformation
- Education system adaptation
- Business model viability in an AI-first world
- Implementation challenges at scale
- Impact of autonomous agent-based systems on labor markets
- Human and AI collaboration frameworks

3. Technical focus areas:
- LLM evaluation and benchmarking
- Hybrid systems combining LLMs with deterministic approaches
- Data curation and quality assessment
- Converting research demos to production applications
- Real-world deployment architectures
- Agent based LLM applications
- Test or Inference time optimization

Here are some example paper titles that I have found interesting lately for additional context:

- On the Limitations of Compute Thresholds as a Governance Strategy for Large Language Models
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- Evaluating Synthetic Data for Tool-Using LLMs
- Towards Monosemanticity: Decomposing Language Models With Dictionary Learning
- Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges
- LalaEval: A Holistic Human Evaluation Framework for Domain-Specific Large Language Models
- Large Language Model Influence on Diagnostic Reasoning A Randomized Clinical Trial
- InsightBench: Evaluating Business Analytics Agents Through Multi-Step Insight Generation
- Best Practices and Lessons Learned on Synthetic Data for Language Models
- Evaluating and Improving the Effectiveness of Synthetic Chest X-Rays for Medical Image Analysis
- The Social Impact of Generative LLM-Based AI
- Agent-as-a-Judge: Evaluate Agents with Agents
- Constructing Domain-Specific Evaluation Sets for LLM-as-a-judge
- On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey
- A Survey on Knowledge Distillation of Large Language Models
- How Well Do LLMs Generate Code for Different Application Domains? Benchmark and Evaluation
- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters
- Language Models and a second opinion use case: The Pocket Professional
- Turn Every Application into an Agent: Towards Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents

Rate papers 0-1 based on on a numeric scale:
- Alignment with above focus areas
- Emphasis on practical implementation over theory
- Concrete frameworks or solutions
- Data-driven approaches
- Relevance to real-world applications
- Similarity to example papers provided. If paper titles are not similar it should not be a disqualifying factor, but if paper titles are similar it should be a very strong positive signal.

0.0 indicates absolutely no relevance
0.25 indicates some relevance in one of the areas of interest
0.5 indicates either significant relevance in one area of interest, or relevance across multiple area of interest
0.75 and above indicates significant alignment with areas of interest
The score can be within those ranges provided as well if unclear or ambiguous

Provide a response that ONLY contains the rated score and no other text:

A score above 0.5 indicates its worth passing to the next stage of paper review."""

NEWSLETTER_PROMPT = """You are a research assistant creating an AI/ML research newsletter. Your task is to summarize the papers provided. 
Your response ONLY should contain the following JSON format replacing the values with relevant details from the provided content in this request.
You do NOT define what is a featured vs. additional paper, that information is provided. All featured papers should receive the full summary section. All Additional Papers should not. 
    {
  "overview": "Brief overview of common themes and key findings across papers, focusing on methodological advancements, practical applications, and emerging trends.",
  "featured_papers": [
    {
      "title": "Full title of the paper",
      "summary": "2-3 paragraphs summarizing technical contributions, practical implications and key takeaways. Emphasize key figures and be opinionated",
      "link": "https://arxiv.org/abs/paper-id"
    }
  ],
  "additional_papers": [
    {
      "title": "Title of paper without full content",
      "link": "https://arxiv.org/abs/paper-id"
    }
  ],
  "metadata": {
    "generated_date": "2024-11-21",
    "total_papers_analyzed": 10,
    "featured_papers_count": 7,
    "additional_papers_count": 3
  }
}"""

def normalize_arxiv_link(link: str) -> str:
    """
    Normalize arXiv links to handle different versions (v1, v2, etc.)

    Examples:
        https://arxiv.org/abs/2301.12345v1 -> https://arxiv.org/abs/2301.12345
        https://arxiv.org/abs/2301.12345v2 -> https://arxiv.org/abs/2301.12345
        https://arxiv.org/abs/2301.12345 -> https://arxiv.org/abs/2301.12345
    """
    # Remove version suffix (v1, v2, etc.)
    normalized = re.sub(r'v\d+$', '', link)

    # Handle both /abs/ and /pdf/ links
    normalized = normalized.replace('/pdf/', '/abs/')

    # Ensure it ends without trailing slash
    normalized = normalized.rstrip('/')

    return normalized.lower()

def similarity_ratio(str1: str, str2: str) -> float:
    """
    Calculate similarity ratio between two strings using SequenceMatcher.
    Returns a value between 0 (completely different) and 1 (identical).
    """
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

def is_duplicate_paper(title: str, link: str, previously_included: Set[str],
                       similarity_threshold: float = 0.9) -> Tuple[bool, str]:
    """
    Check if a paper is a duplicate using exact matching and fuzzy matching.

    Args:
        title: Paper title to check
        link: Paper link to check
        previously_included: Set of previously included titles and links
        similarity_threshold: Threshold for fuzzy title matching (0-1)

    Returns:
        Tuple of (is_duplicate, reason)
    """
    title_lower = title.lower()
    link_normalized = normalize_arxiv_link(link)

    # Check for exact title match
    if title_lower in previously_included:
        return True, f"Exact title match"

    # Check for exact link match (with normalization)
    for prev_item in previously_included:
        if prev_item.startswith('http'):
            if normalize_arxiv_link(prev_item) == link_normalized:
                return True, f"Exact link match (normalized)"

    # Check for fuzzy title match (for similar but not identical titles)
    for prev_item in previously_included:
        if not prev_item.startswith('http'):  # Only compare with titles
            similarity = similarity_ratio(title_lower, prev_item)
            if similarity >= similarity_threshold:
                return True, f"Fuzzy title match (similarity: {similarity:.2%})"

    return False, ""

class DuplicateDetector:
    """
    Efficient duplicate detection with fuzzy matching and normalization.
    """
    def __init__(self, previously_included: Set[str], similarity_threshold: float = 0.9):
        """
        Args:
            previously_included: Set of previously included paper titles and links
            similarity_threshold: Threshold for fuzzy title matching (0-1)
        """
        self.similarity_threshold = similarity_threshold

        # Separate titles and links for more efficient checking
        self.previous_titles: Set[str] = set()
        self.previous_links: Set[str] = set()

        for item in previously_included:
            item_lower = item.lower()
            if item_lower.startswith('http'):
                self.previous_links.add(normalize_arxiv_link(item_lower))
            else:
                self.previous_titles.add(item_lower)

        logger.info(f"DuplicateDetector initialized with {len(self.previous_titles)} titles "
                   f"and {len(self.previous_links)} links")

    def is_duplicate(self, title: str, link: str) -> Tuple[bool, str]:
        """
        Check if a paper is a duplicate.

        Returns:
            Tuple of (is_duplicate, reason)
        """
        title_lower = title.lower()
        link_normalized = normalize_arxiv_link(link)

        # Fast exact match checks first
        if title_lower in self.previous_titles:
            return True, "Exact title match"

        if link_normalized in self.previous_links:
            return True, "Exact link match (normalized)"

        # Slower fuzzy matching for titles
        for prev_title in self.previous_titles:
            similarity = similarity_ratio(title_lower, prev_title)
            if similarity >= self.similarity_threshold:
                return True, f"Fuzzy title match (similarity: {similarity:.2%}, matched: '{prev_title[:50]}...')"

        return False, ""

    def add_paper(self, title: str, link: str):
        """Add a paper to the duplicate detector after processing"""
        self.previous_titles.add(title.lower())
        self.previous_links.add(normalize_arxiv_link(link))

def get_secret():
    try:
        secret_name = "anthropic/api_key"
        region_name = os.environ.get('AWS_REGION', 'us-west-2')
        
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        
        if 'SecretString' in get_secret_value_response:
            try:
                secret = json.loads(get_secret_value_response['SecretString'])
                if 'anthropic_key' not in secret:
                    raise ValueError("Secret missing anthropic_key")
                return secret['anthropic_key']
            except json.JSONDecodeError:
                raise ValueError("Secret is not valid JSON")
        else:
            raise ValueError("Secret not found")
    except Exception as e:
        logger.error(f"Error retrieving secret: {str(e)}")
        raise

def validate_environment_variables():
    """Validate all required environment variables are present"""
    required_vars = [
        'ANTHROPIC_API_KEY',
        'SUBSCRIBERS_TABLE',
        'SENDER_EMAIL',
        'NEWSLETTER_BUCKET',
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

def store_newsletter(json_content: Dict) -> str:
    """Store newsletter content in S3"""
    s3_client = boto3.client('s3')
    bucket_name = os.environ['NEWSLETTER_BUCKET']
    
    # Create ISO timestamp-based key
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    key = f'newsletters/{timestamp}.json'
    
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=json.dumps(json_content, indent=2),
            ContentType='application/json'
        )
        return f"s3://{bucket_name}/{key}"
    except Exception as e:
        logger.error(f"Error storing newsletter: {str(e)}")
        raise

def extract_json_from_text(content: str) -> str:
    """
    Extract JSON from text that may contain extra content before/after.
    Tries multiple strategies to find and extract valid JSON.
    """
    # Strategy 1: Find JSON object by braces
    # Look for outermost { } pair
    first_brace = content.find('{')
    if first_brace != -1:
        # Find the matching closing brace by counting
        brace_count = 0
        for i in range(first_brace, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    return content[first_brace:i+1]

    # Strategy 2: Find JSON array by brackets
    first_bracket = content.find('[')
    if first_bracket != -1:
        bracket_count = 0
        for i in range(first_bracket, len(content)):
            if content[i] == '[':
                bracket_count += 1
            elif content[i] == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    return content[first_bracket:i+1]

    return content

def repair_json(content: str) -> str:
    """
    Attempt to repair common JSON formatting issues.
    """
    # Remove trailing commas before closing braces/brackets
    content = re.sub(r',(\s*[}\]])', r'\1', content)

    # Fix single quotes (should be double quotes in JSON)
    # Be careful not to break strings that intentionally contain single quotes
    # This is a simple heuristic and may not work in all cases
    content = re.sub(r"'([^']*)':", r'"\1":', content)

    # Remove any trailing incomplete entries
    # If JSON ends with a comma and incomplete object, remove it
    content = re.sub(r',\s*$', '', content)

    return content

def validate_and_fix_newsletter_data(data: Dict) -> Dict:
    """
    Validate newsletter data structure and fix common issues.
    """
    # Ensure all required top-level keys exist
    if 'overview' not in data:
        data['overview'] = "No papers met the criteria for this week's newsletter."
        logger.warning("Missing 'overview' field, added default")

    if 'featured_papers' not in data:
        data['featured_papers'] = []
        logger.warning("Missing 'featured_papers' field, initialized as empty list")

    if 'additional_papers' not in data:
        data['additional_papers'] = []
        logger.warning("Missing 'additional_papers' field, initialized as empty list")

    if 'metadata' not in data:
        data['metadata'] = {}
        logger.warning("Missing 'metadata' field, initialized as empty dict")

    # Validate and fix featured papers
    valid_featured = []
    for i, paper in enumerate(data.get('featured_papers', [])):
        if not isinstance(paper, dict):
            logger.warning(f"Featured paper {i} is not a dict, skipping")
            continue

        if 'title' not in paper or 'link' not in paper:
            logger.warning(f"Featured paper {i} missing required fields, skipping: {paper}")
            continue

        # Featured papers should have summaries
        if 'summary' not in paper or not paper['summary']:
            logger.warning(f"Featured paper {i} missing summary: {paper.get('title', 'Unknown')}")
            paper['summary'] = "Summary not available."

        valid_featured.append(paper)

    data['featured_papers'] = valid_featured

    # Validate and fix additional papers
    valid_additional = []
    for i, paper in enumerate(data.get('additional_papers', [])):
        if not isinstance(paper, dict):
            logger.warning(f"Additional paper {i} is not a dict, skipping")
            continue

        if 'title' not in paper or 'link' not in paper:
            logger.warning(f"Additional paper {i} missing required fields, skipping: {paper}")
            continue

        valid_additional.append(paper)

    data['additional_papers'] = valid_additional

    # Fix metadata
    if 'generated_date' not in data['metadata']:
        data['metadata']['generated_date'] = datetime.now().strftime('%Y-%m-%d')

    data['metadata'].update({
        'featured_papers_count': len(data['featured_papers']),
        'additional_papers_count': len(data['additional_papers']),
        'total_papers_analyzed': len(data['featured_papers']) + len(data['additional_papers'])
    })

    return data

def process_newsletter_content(content: str) -> Dict:
    """
    Process newsletter content returned from API with robust error recovery.
    Tries multiple strategies to extract and parse JSON.
    """
    logger.info("Processing newsletter content")
    original_content = content

    try:
        # Step 1: Handle TextBlock formatting if present
        if "TextBlock(text='" in content:
            content = content.split("TextBlock(text='", 1)[1].rsplit("')", 1)[0]
            content = content.split("', type='text", 1)[0]
            content = codecs.escape_decode(content)[0].decode('utf-8')
            logger.info("Extracted content from TextBlock formatting")

        # Step 2: Handle markdown code blocks (```json ... ``` or ``` ... ```)
        if '```' in content:
            # Find all code blocks
            code_block_pattern = r'```(?:json)?\s*\n(.*?)\n```'
            matches = re.findall(code_block_pattern, content, re.DOTALL)

            if matches:
                # Use the largest code block (likely to be the JSON)
                content = max(matches, key=len)
                logger.info("Extracted JSON from markdown code block")
            else:
                # Try simple extraction
                if content.strip().startswith('```json'):
                    lines = content.strip().split('\n')
                    json_lines = lines[1:-1] if lines[-1].strip() == '```' else lines[1:]
                    content = '\n'.join(json_lines)
                    logger.info("Extracted JSON from markdown code block (simple method)")
                elif content.strip().startswith('```'):
                    lines = content.strip().split('\n')
                    json_lines = lines[1:-1] if lines[-1].strip() == '```' else lines[1:]
                    content = '\n'.join(json_lines)
                    logger.info("Extracted JSON from generic code block")

        # Step 3: Extract JSON from surrounding text
        content = extract_json_from_text(content)
        logger.info(f"Attempting to parse JSON content (first 200 chars): {content[:200]}...")

        # Step 4: Try to parse JSON
        try:
            newsletter_data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Initial JSON parse failed: {str(e)}, attempting repair")

            # Try to repair the JSON
            repaired_content = repair_json(content)

            try:
                newsletter_data = json.loads(repaired_content)
                logger.info("Successfully parsed JSON after repair")
            except json.JSONDecodeError as e2:
                logger.error(f"JSON repair failed: {str(e2)}")
                logger.error(f"Original content (first 500 chars): {original_content[:500]}...")
                logger.error(f"Processed content (first 500 chars): {content[:500]}...")
                raise ValueError(f"Unable to parse JSON even after repair attempts: {str(e2)}")

        # Step 5: Validate and fix the data structure
        newsletter_data = validate_and_fix_newsletter_data(newsletter_data)

        logger.info(f"Successfully processed newsletter with {len(newsletter_data['featured_papers'])} featured papers "
                   f"and {len(newsletter_data['additional_papers'])} additional papers")
        logger.info(f"Overview: {newsletter_data['overview'][:100]}...")

        return newsletter_data

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {str(e)}")
        logger.error(f"Raw content received (first 1000 chars): {original_content[:1000]}...")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing newsletter: {str(e)}")
        logger.error(f"Content that caused error (first 1000 chars): {original_content[:1000]}...")
        raise ValueError(f"Newsletter processing error: {str(e)}")

def validate_paper_format(paper: Dict, is_featured: bool = True) -> bool:
    """Validate paper format matches requirements"""
    required_fields = ['title', 'link']
    if is_featured:
        required_fields.append('summary')
    
    return all(field in paper for field in required_fields)

def get_previously_included_papers() -> Set[str]:
    """
    Retrieve all paper titles and links from previous newsletters stored in S3.
    Returns a set of strings (titles and links) for quick duplicate checking.
    """
    logger.info("Retrieving previously included papers from S3")
    s3_client = boto3.client('s3')
    bucket_name = os.environ['NEWSLETTER_BUCKET']
    
    try:
        # List all objects in the newsletters folder
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix='newsletters/'
        )
        
        if 'Contents' not in response:
            logger.info("No previous newsletters found in S3")
            return set()
        
        # Log what files are being processed
        files = response.get('Contents', [])
        logger.info(f"Found {len(files)} newsletter files in S3")
        for obj in files[:5]:  # Show first 5 files
            logger.info(f"  Processing file: {obj['Key']} (modified: {obj['LastModified']})")
        
        # Track all paper titles and links we've seen before
        previously_included = set()
        
        for obj in files:
            try:
                # Get the newsletter JSON file
                file_response = s3_client.get_object(
                    Bucket=bucket_name,
                    Key=obj['Key']
                )
                
                newsletter_data = json.loads(file_response['Body'].read().decode('utf-8'))
                
                # Extract paper titles and links from featured papers
                for paper in newsletter_data.get('featured_papers', []):
                    if 'title' in paper:
                        previously_included.add(paper['title'].lower())
                    if 'link' in paper:
                        previously_included.add(paper['link'].lower())
                
                # Extract paper titles and links from additional papers
                for paper in newsletter_data.get('additional_papers', []):
                    if 'title' in paper:
                        previously_included.add(paper['title'].lower())
                    if 'link' in paper:
                        previously_included.add(paper['link'].lower())
                
            except Exception as e:
                logger.warning(f"Error processing newsletter file {obj['Key']}: {str(e)}")
                continue
        
        logger.info(f"Found {len(previously_included)} previously included paper titles/links")
        if len(previously_included) > 0:
            logger.info(f"Sample previously included papers: {list(previously_included)[:5]}")
            # Show any links specifically
            links = [p for p in previously_included if p.startswith('http')]
            logger.info(f"Sample links in previously included: {links[:3]}")
        return previously_included
        
    except Exception as e:
        logger.error(f"Error retrieving previously included papers: {str(e)}")
        return set()

def filter_duplicate_papers(json_content: Dict, previously_included: Set[str],
                          similarity_threshold: float = 0.9) -> Dict:
    """
    Filter out papers that have been included in previous newsletters using enhanced duplicate detection.
    Returns the updated json_content with duplicates removed.

    Args:
        json_content: The newsletter content to filter
        previously_included: Set of previously included paper titles and links
        similarity_threshold: Threshold for fuzzy title matching (0-1), default 0.9
    """
    logger.info("Filtering out previously included papers with enhanced duplicate detection")

    if not previously_included:
        logger.info("No previously included papers found, skipping filter")
        return json_content

    # Initialize duplicate detector
    detector = DuplicateDetector(previously_included, similarity_threshold)

    # Track duplicate statistics
    duplicate_stats = {
        'exact_title': 0,
        'exact_link': 0,
        'fuzzy_title': 0
    }

    # Filter featured papers
    filtered_featured = []
    for paper in json_content.get('featured_papers', []):
        title = paper.get('title', '')
        link = paper.get('link', '')

        is_dup, reason = detector.is_duplicate(title, link)

        if is_dup:
            logger.info(f"Filtering out duplicate featured paper: {title[:80]}... ({reason})")
            # Track the type of duplicate
            if 'Exact title' in reason:
                duplicate_stats['exact_title'] += 1
            elif 'Exact link' in reason:
                duplicate_stats['exact_link'] += 1
            elif 'Fuzzy' in reason:
                duplicate_stats['fuzzy_title'] += 1
            continue

        filtered_featured.append(paper)

    # Filter additional papers
    filtered_additional = []
    for paper in json_content.get('additional_papers', []):
        title = paper.get('title', '')
        link = paper.get('link', '')

        is_dup, reason = detector.is_duplicate(title, link)

        if is_dup:
            logger.info(f"Filtering out duplicate additional paper: {title[:80]}... ({reason})")
            # Track the type of duplicate
            if 'Exact title' in reason:
                duplicate_stats['exact_title'] += 1
            elif 'Exact link' in reason:
                duplicate_stats['exact_link'] += 1
            elif 'Fuzzy' in reason:
                duplicate_stats['fuzzy_title'] += 1
            continue

        filtered_additional.append(paper)

    # Log duplicate statistics
    total_duplicates = sum(duplicate_stats.values())
    if total_duplicates > 0:
        logger.info(f"Duplicate detection stats - Total: {total_duplicates}, "
                   f"Exact title: {duplicate_stats['exact_title']}, "
                   f"Exact link: {duplicate_stats['exact_link']}, "
                   f"Fuzzy title: {duplicate_stats['fuzzy_title']}")

    # Update the json_content with filtered papers
    json_content['featured_papers'] = filtered_featured
    json_content['additional_papers'] = filtered_additional

    # Update metadata counts
    if 'metadata' in json_content:
        json_content['metadata']['featured_papers_count'] = len(filtered_featured)
        json_content['metadata']['additional_papers_count'] = len(filtered_additional)
        json_content['metadata']['total_papers_analyzed'] = len(filtered_featured) + len(filtered_additional)
        json_content['metadata']['duplicates_filtered'] = total_duplicates
        json_content['metadata']['duplicate_stats'] = duplicate_stats

    filtered_count = len(json_content.get('featured_papers', [])) + len(json_content.get('additional_papers', []))
    logger.info(f"After filtering: {len(filtered_featured)} featured papers, {len(filtered_additional)} additional papers")

    # If all papers were filtered out, add a note to the overview
    if filtered_count == 0:
        json_content['overview'] = "All papers were filtered out as they were already included in previous newsletters."
        logger.warning("All papers were filtered out as duplicates")

    return json_content

def lambda_handler(event, context):
    try:
        validate_environment_variables()
        
        start_time = datetime.now()
        logger.info(f"Starting newsletter generation at {start_time}")
        
        anthropic_api_key = get_secret()
        
        # Get previously included papers first
        previously_included = get_previously_included_papers()
        logger.info(f"Found {len(previously_included)} previously included papers")

        # Get model configurations from environment variables (with defaults)
        model_cheap = os.environ.get('CLAUDE_MODEL_CHEAP')
        model_expensive = os.environ.get('CLAUDE_MODEL_EXPENSIVE')
        model_fallback = os.environ.get('CLAUDE_MODEL_FALLBACK')
        model_emergency = os.environ.get('CLAUDE_MODEL_EMERGENCY')

        logger.info(f"Model configuration from environment: "
                   f"Cheap={model_cheap or 'default'}, "
                   f"Expensive={model_expensive or 'default'}, "
                   f"Fallback={model_fallback or 'default'}, "
                   f"Emergency={model_emergency or 'default'}")

        analyzer = PaperAnalyzer(
            anthropic_api_key=anthropic_api_key,
            eval_prompt=EVAL_PROMPT,
            newsletter_prompt=NEWSLETTER_PROMPT,
            previously_included_papers=previously_included,
            model_cheap=model_cheap,
            model_expensive=model_expensive,
            model_fallback=model_fallback,
            model_emergency=model_emergency
        )
        
        # Initial feed sources
        feed_urls = [
            "https://rss.arxiv.org/rss/cs.AI",
            "https://hedgehog.den.dev/feeds/toprecent-week.xml"
        ]
        
        # Backup feed sources if we don't have enough papers
        backup_feed_urls = [
            "https://hedgehog.den.dev/feeds/home.xml",
            "https://hedgehog.den.dev/feeds/random-last-week.xml"
        ]
        
        logger.info("Starting paper analysis and newsletter generation")
        newsletter_content = analyzer.process_multiple_feeds(feed_urls)
        
        try:
            json_content = process_newsletter_content(newsletter_content)
            
            # Validate paper formats
            for paper in json_content.get('featured_papers', []):
                if not validate_paper_format(paper, is_featured=True):
                    raise ValueError(f"Invalid featured paper format: {paper}")
                    
            for paper in json_content.get('additional_papers', []):
                if not validate_paper_format(paper, is_featured=False):
                    raise ValueError(f"Invalid additional paper format: {paper}")
            
            # If we have fewer than 3 featured papers, try using backup feeds
            min_featured_papers = 3
            if len(json_content['featured_papers']) < min_featured_papers and backup_feed_urls:
                logger.info(f"Found only {len(json_content['featured_papers'])} featured papers. Trying backup feeds.")
                
                # Process backup feeds
                backup_content = analyzer.process_multiple_feeds(backup_feed_urls)
                backup_json = process_newsletter_content(backup_content)
                
                # Add backup papers to our existing content
                if backup_json.get('featured_papers'):
                    json_content['featured_papers'].extend(backup_json['featured_papers'])
                    logger.info(f"Added {len(backup_json['featured_papers'])} more featured papers from backup feeds")
                
                if backup_json.get('additional_papers'):
                    json_content['additional_papers'].extend(backup_json['additional_papers'])
                    logger.info(f"Added {len(backup_json['additional_papers'])} more additional papers from backup feeds")
                
                # Update metadata counts
                if 'metadata' in json_content:
                    json_content['metadata']['featured_papers_count'] = len(json_content['featured_papers'])
                    json_content['metadata']['additional_papers_count'] = len(json_content['additional_papers'])
                    json_content['metadata']['total_papers_analyzed'] = len(json_content['featured_papers']) + len(json_content['additional_papers'])
                
                # If we added more papers, update the overview
                if len(json_content['featured_papers']) > 0 or len(json_content['additional_papers']) > 0:
                    json_content['overview'] = f"This newsletter includes papers from our regular and extended feeds. {json_content.get('overview', '')}"
            
        except Exception as e:
            logger.error(f"Error processing newsletter content: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Newsletter generation failed',
                    'message': str(e)
                })
            }
        
        if not json_content['featured_papers'] and not json_content['additional_papers']:
            logger.warning("No papers met the criteria for inclusion in the newsletter")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'No papers available for the newsletter - all filtered or none met criteria',
                    'metadata': json_content['metadata']
                })
            }
        
        logger.info("Sending newsletter to subscribers")
        email_results = send_newsletter(
            json_content=json_content,
            table_name=os.environ['SUBSCRIBERS_TABLE'],
            sender=os.environ['SENDER_EMAIL'],
        )

        s3_location = store_newsletter(json_content)
        logger.info(f"Newsletter stored at: {s3_location}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            'message': 'Newsletter generated and sent successfully',
            'execution_time_seconds': execution_time,
            'paper_stats': {
                'featured_papers': len(json_content['featured_papers']),
                'additional_papers': len(json_content['additional_papers']),
                'total_papers': json_content['metadata']['total_papers_analyzed']
            },
            'email_results': email_results,
            'storage_location': s3_location,
            'filtered_duplicates': True
        }
        
        logger.info(f"Newsletter process completed in {execution_time} seconds")
        logger.info(f"Email results: {email_results}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
        
    except Exception as e:
        logger.error(f"Error in lambda execution: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }