# functions/newsletter_generator.py
import json
import os
from .paper_analyzer import PaperAnalyzer
from .email_sender import send_newsletter
import logging
from datetime import datetime
import codecs
from typing import Dict, List, Set
import boto3
from botocore.exceptions import ClientError

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

def process_newsletter_content(content: str) -> Dict:
    """Process newsletter content returned from API"""
    logger.info("Processing newsletter content")
    try:
        # Handle TextBlock formatting if present
        if "TextBlock(text='" in content:
            content = content.split("TextBlock(text='", 1)[1].rsplit("')", 1)[0]
            content = content.split("', type='text", 1)[0]
            content = codecs.escape_decode(content)[0].decode('utf-8')
        
        # Parse JSON from the string
    
        newsletter_data = json.loads(content)
        
        # Initialize missing sections if they don't exist
        if 'overview' not in newsletter_data:
            newsletter_data['overview'] = "No papers met the criteria for this week's newsletter."
            
        if 'featured_papers' not in newsletter_data:
            newsletter_data['featured_papers'] = []
            
        if 'additional_papers' not in newsletter_data:
            newsletter_data['additional_papers'] = []
            
        if 'metadata' not in newsletter_data:
            newsletter_data['metadata'] = {}
            
        # Update metadata with current date if not present
        if 'generated_date' not in newsletter_data['metadata']:
            newsletter_data['metadata']['generated_date'] = datetime.now().strftime('%Y-%m-%d')
            
        # Update paper counts
        newsletter_data['metadata'].update({
            'featured_papers_count': len(newsletter_data['featured_papers']),
            'additional_papers_count': len(newsletter_data['additional_papers']),
            'total_papers_analyzed': len(newsletter_data['featured_papers']) + len(newsletter_data['additional_papers'])
        })
        
        logger.info(f"Processed newsletter with {len(newsletter_data['featured_papers'])} featured papers")
        logger.info(f"Overview: {newsletter_data['overview'][:100]}...")
        
        return newsletter_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {str(e)}")
        logger.error(f"Raw content received: {content[:500]}...")
        raise ValueError(f"Invalid JSON format: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing newsletter: {str(e)}")
        logger.error(f"Content that caused error: {content[:500]}...")
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
        
        # Track all paper titles and links we've seen before
        previously_included = set()
        
        for obj in response.get('Contents', []):
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
        return previously_included
        
    except Exception as e:
        logger.error(f"Error retrieving previously included papers: {str(e)}")
        return set()

def filter_duplicate_papers(json_content: Dict, previously_included: Set[str]) -> Dict:
    """
    Filter out papers that have been included in previous newsletters.
    Returns the updated json_content with duplicates removed.
    """
    logger.info("Filtering out previously included papers")
    
    if not previously_included:
        logger.info("No previously included papers found, skipping filter")
        return json_content
    
    # Filter featured papers
    filtered_featured = []
    for paper in json_content.get('featured_papers', []):
        title = paper.get('title', '').lower()
        link = paper.get('link', '').lower()
        
        if title in previously_included or link in previously_included:
            logger.info(f"Filtering out duplicate featured paper: {title}")
            continue
        
        filtered_featured.append(paper)
    
    # Filter additional papers
    filtered_additional = []
    for paper in json_content.get('additional_papers', []):
        title = paper.get('title', '').lower()
        link = paper.get('link', '').lower()
        
        if title in previously_included or link in previously_included:
            logger.info(f"Filtering out duplicate additional paper: {title}")
            continue
        
        filtered_additional.append(paper)
    
    # Update the json_content with filtered papers
    json_content['featured_papers'] = filtered_featured
    json_content['additional_papers'] = filtered_additional
    
    # Update metadata counts
    if 'metadata' in json_content:
        json_content['metadata']['featured_papers_count'] = len(filtered_featured)
        json_content['metadata']['additional_papers_count'] = len(filtered_additional)
        json_content['metadata']['total_papers_analyzed'] = len(filtered_featured) + len(filtered_additional)
    
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
        
        analyzer = PaperAnalyzer(
            anthropic_api_key=anthropic_api_key,
            eval_prompt=EVAL_PROMPT,
            newsletter_prompt=NEWSLETTER_PROMPT,
            previously_included_papers=previously_included
        )
        
        # Initial feed sources
        feed_urls = [
            "https://hedgehog.den.dev/feeds/toprecent-week.xml",
            "https://hedgehog.den.dev/feeds/home.xml",
            "https://hedgehog.den.dev/feeds/random-last-week.xml"
        ]
        
        # Backup feed sources if we don't have enough papers
        backup_feed_urls = [
            "https://hedgehog.den.dev/feeds/recent.xml",
            "https://hedgehog.den.dev/feeds/recent-month.xml"
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