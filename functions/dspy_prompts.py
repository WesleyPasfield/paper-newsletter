"""
DSPy Integration for Prompt Optimization
Provides structured prompts and optimization for paper evaluation and newsletter generation
"""

import logging
import os
from typing import List, Dict, Any, Optional
logger = logging.getLogger(__name__)
# Try to import DSPy, but don't fail if it's not available
try:
    import dspy
    from dspy import InputField, OutputField, Signature
    from dspy.teleprompt import BootstrapFewShot
    DSPY_AVAILABLE = True
    logger.info("DSPy is available")
except ImportError:
    DSPY_AVAILABLE = False
    logger.info("DSPy not installed - using legacy LLM evaluation")
    # Create dummy classes for when DSPy is not available
    class Signature:
        pass
    
    class InputField:
        def __init__(self, desc="", default=""):
            pass
    
    class OutputField:
        def __init__(self, desc=""):
            pass
    
    class BootstrapFewShot:
        def __init__(self, metric=None):
            pass
        
        def compile(self, *args, **kwargs):
            return args[0]
    
    # Create a dummy LM class for when DSPy is not available
    class DummyLM:
        def __init__(self, model=None):
            self.model = model

from .llm_providers import LLMManager, LLMProvider, LLMConfig, DEFAULT_CONFIGS

class DSPyLLMAdapter(DummyLM if not DSPY_AVAILABLE else dspy.LM):
    """DSPy adapter for our multi-LLM system"""
    
    def __init__(self, llm_manager: LLMManager, config: LLMConfig):
        self.llm_manager = llm_manager
        self.config = config
        if DSPY_AVAILABLE:
            super().__init__(model=config.model)
        else:
            super().__init__(model=config.model)
    
    def basic_request(self, prompt: str, **kwargs) -> str:
        """Basic request implementation for DSPy"""
        try:
            # Split prompt into system and user parts
            # DSPy typically sends everything as one prompt
            system_prompt = "You are a helpful AI assistant."
            user_prompt = prompt
            
            # Try to extract system prompt if it exists
            if "System:" in prompt:
                parts = prompt.split("System:", 1)
                if len(parts) == 2:
                    system_part, user_part = parts
                    system_prompt = system_part.strip()
                    user_prompt = user_part.strip()
            
            response = self.llm_manager.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                config=self.config
            )
            return response.content
            
        except Exception as e:
            logger.error(f"Error in DSPy LLM adapter: {str(e)}")
            raise

class PaperEvaluationSignature(Signature):
    """DSPy signature for paper evaluation"""
    paper_title: str = InputField(desc="Title of the research paper")
    paper_abstract: str = InputField(desc="Abstract of the research paper", default="")
    evaluation_criteria: str = InputField(desc="Evaluation criteria and focus areas")
    
    score: float = OutputField(desc="Interest score between 0.0 and 1.0")
    reasoning: str = OutputField(desc="Brief reasoning for the score")

class NewsletterGenerationSignature(Signature):
    """DSPy signature for newsletter generation"""
    featured_papers: str = InputField(desc="Featured papers with full content")
    additional_papers: str = InputField(desc="Additional papers with titles and links")
    newsletter_guidelines: str = InputField(desc="Newsletter formatting guidelines")
    
    newsletter_json: str = OutputField(desc="Complete newsletter in JSON format")

class PaperEvaluator:
    """DSPy module for paper evaluation"""
    
    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
        if DSPY_AVAILABLE:
            self.evaluate = dspy.ChainOfThought(PaperEvaluationSignature)
        else:
            self.evaluate = None
    
    def __call__(self, paper_title: str, paper_abstract: str = "", evaluation_criteria: str = ""):
        if DSPY_AVAILABLE and self.evaluate:
            return self.evaluate(
                paper_title=paper_title,
                paper_abstract=paper_abstract,
                evaluation_criteria=evaluation_criteria
            )
        else:
            # Fallback to legacy LLM-based evaluation
            try:
                from .llm_providers import DEFAULT_CONFIGS, LLMProvider
                
                # Use Claude for evaluation if available
                if LLMProvider.CLAUDE in self.llm_manager.get_available_providers():
                    config = DEFAULT_CONFIGS[LLMProvider.CLAUDE]['cheap']
                    user_prompt = f"Based on the title, rate interest between 0 and 1 on a numeric scale: {paper_title}"
                    
                    # Use the original detailed evaluation prompt
                    system_prompt = """You are evaluating academic papers for an expert specializing in:

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
                    
                    response = self.llm_manager.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        config=config
                    )
                    
                    try:
                        score = float(response.content.strip())
                        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                        reasoning = f"Legacy LLM evaluation using {response.provider.value}"
                    except (ValueError, TypeError):
                        score = 0.5
                        reasoning = "Legacy LLM evaluation failed to parse score"
                    
                    return type('obj', (object,), {
                        'score': score,
                        'reasoning': reasoning
                    })()
                else:
                    # No LLM available, use default
                    return type('obj', (object,), {
                        'score': 0.5,
                        'reasoning': 'No LLM providers available for evaluation'
                    })()
                    
            except Exception as e:
                logger.error(f"Error in legacy evaluation fallback: {str(e)}")
                return type('obj', (object,), {
                    'score': 0.5,
                    'reasoning': f'Legacy evaluation error: {str(e)}'
                })()

class NewsletterGenerator:
    """DSPy module for newsletter generation"""
    
    def __init__(self, llm_manager=None):
        self.llm_manager = llm_manager
        if DSPY_AVAILABLE:
            self.generate = dspy.ChainOfThought(NewsletterGenerationSignature)
        else:
            self.generate = None
    
    def __call__(self, featured_papers: str, additional_papers: str = "", newsletter_guidelines: str = ""):
        if DSPY_AVAILABLE and self.generate:
            return self.generate(
                featured_papers=featured_papers,
                additional_papers=additional_papers,
                newsletter_guidelines=newsletter_guidelines
            )
        else:
            # Fallback to legacy LLM-based newsletter generation
            try:
                from .llm_providers import DEFAULT_CONFIGS, LLMProvider
                
                # Use Claude for newsletter generation if available
                if LLMProvider.CLAUDE in self.llm_manager.get_available_providers():
                    config = DEFAULT_CONFIGS[LLMProvider.CLAUDE]['expensive']
                    
                    prompt = f"""Create an AI/ML research newsletter in JSON format.

Featured Papers:
{featured_papers}

Additional Papers:
{additional_papers}

Return ONLY valid JSON in this format:
{{
  "overview": "Brief overview of common themes and key findings",
  "featured_papers": [
    {{
      "title": "Full title of the paper",
      "summary": "2-3 paragraphs summarizing technical contributions and key takeaways",
      "link": "https://arxiv.org/abs/paper-id"
    }}
  ],
  "additional_papers": [
    {{
      "title": "Title of paper",
      "link": "https://arxiv.org/abs/paper-id"
    }}
  ]
}}"""
                    
                    response = self.llm_manager.generate(
                        system_prompt="You are a research assistant creating AI/ML research newsletters. Return only valid JSON.",
                        user_prompt=prompt,
                        config=config
                    )
                    
                    return type('obj', (object,), {
                        'newsletter_json': response.content
                    })()
                else:
                    # No LLM available, return empty newsletter
                    return type('obj', (object,), {
                        'newsletter_json': '{"overview": "No LLM providers available for newsletter generation", "featured_papers": [], "additional_papers": []}'
                    })()
                    
            except Exception as e:
                logger.error(f"Error in legacy newsletter generation: {str(e)}")
                return type('obj', (object,), {
                    'newsletter_json': f'{{"overview": "Legacy newsletter generation error: {str(e)}", "featured_papers": [], "additional_papers": []}}'
                })()

class DSPyPromptManager:
    """Manages DSPy-optimized prompts for the newsletter system"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.evaluation_criteria = self._get_updated_evaluation_criteria()
        self.newsletter_guidelines = self._get_newsletter_guidelines()
        
        # Initialize DSPy modules
        self.paper_evaluator = PaperEvaluator(llm_manager)
        self.newsletter_generator = NewsletterGenerator(llm_manager)
        
        # Configure DSPy with our LLM adapter
        if DSPY_AVAILABLE:
            self._configure_dspy()
    
    def _configure_dspy(self):
        """Configure DSPy with our multi-LLM system"""
        if not DSPY_AVAILABLE:
            logger.warning("DSPy not available, skipping configuration")
            return
            
        try:
            # Use Claude as primary for DSPy
            if LLMProvider.CLAUDE in self.llm_manager.get_available_providers():
                config = DEFAULT_CONFIGS[LLMProvider.CLAUDE]['expensive']
                lm = DSPyLLMAdapter(self.llm_manager, config)
                dspy.settings.configure(lm=lm)
                logger.info("DSPy configured with Claude")
            elif LLMProvider.OPENAI in self.llm_manager.get_available_providers():
                config = DEFAULT_CONFIGS[LLMProvider.OPENAI]['expensive']
                lm = DSPyLLMAdapter(self.llm_manager, config)
                dspy.settings.configure(lm=lm)
                logger.info("DSPy configured with OpenAI")
            else:
                logger.warning("No suitable LLM provider available for DSPy")
        except Exception as e:
            logger.error(f"Error configuring DSPy: {str(e)}")
    
    def _get_updated_evaluation_criteria(self) -> str:
        """Get updated evaluation criteria with agentic AI focus"""
        return """You are evaluating academic papers for an expert specializing in:

1. Practical LLM/Generative AI regulation and evaluation:
- Data-driven evaluation frameworks over compute-based metrics
- Domain-specific testing methodologies  
- Production deployment challenges
- User experience validation
- Certification processes for AI systems
- Impact of test or inference time optimizations on model performance and governance strategies

2. Agentic AI systems and implementations:
- Multi-agent architectures and coordination mechanisms
- Tool-using AI agents and function calling
- Autonomous decision-making systems
- Agent-based workflows and orchestration
- Human-AI collaboration in agentic systems
- Safety and control mechanisms for autonomous agents
- Real-world deployment of AI agents in business processes
- Agent memory systems and state management
- Planning and reasoning capabilities in AI agents

3. Societal and economic impacts of generative AI:
- Knowledge work transformation through AI agents
- Education system adaptation to AI-assisted learning
- Business model viability in an AI-first world
- Implementation challenges at scale
- Impact of autonomous agent-based systems on labor markets
- Human and AI collaboration frameworks
- Economic implications of widespread AI automation

4. Technical focus areas:
- LLM evaluation and benchmarking
- Hybrid systems combining LLMs with deterministic approaches
- Data curation and quality assessment for agent training
- Converting research demos to production applications
- Real-world deployment architectures for AI agents
- Agent-based LLM applications and multi-step reasoning
- Test or inference time optimization
- Tool integration and API orchestration for agents

Here are some example paper titles that align with these interests:

- On the Limitations of Compute Thresholds as a Governance Strategy for Large Language Models
- Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- Evaluating Synthetic Data for Tool-Using LLMs
- Agent-as-a-Judge: Evaluate Agents with Agents
- Turn Every Application into an Agent: Towards Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents
- Multi-Agent Systems for Complex Task Orchestration
- Autonomous Planning in Large Language Model Agents
- Tool Learning with Foundation Models
- ReAct: Synergizing Reasoning and Acting in Language Models
- AutoGPT: An Autonomous GPT-4 Experiment
- LangChain: Building Applications with LLMs through Composability
- Multi-Agent Debate for Factual Accuracy
- Toolformer: Language Models Can Teach Themselves to Use Tools
- WebGPT: Browser-assisted Question-answering with Human Feedback

Rate papers 0-1 based on alignment with:
- Relevance to agentic AI systems and practical implementations (high weight)
- Emphasis on practical implementation over pure theory  
- Concrete frameworks, tools, or solutions
- Data-driven approaches and empirical evaluation
- Relevance to real-world applications and deployment
- Similarity to example papers (strong positive signal if similar)

Scoring Guidelines:
0.0: No relevance to focus areas
0.25: Some relevance in one area, theoretical focus
0.5: Significant relevance in one area OR relevance across multiple areas
0.75+: Strong alignment with agentic AI focus AND practical implementation
0.9+: Exceptional alignment with agentic systems, practical deployment, and evaluation

A score above 0.5 indicates worth including in the newsletter."""

    def _get_newsletter_guidelines(self) -> str:
        """Get newsletter generation guidelines"""
        return """Create an AI/ML research newsletter focusing on agentic AI and practical implementations.

Return ONLY valid JSON in this exact format:
{
  "overview": "Brief overview emphasizing agentic AI developments, practical applications, and methodological advances",
  "featured_papers": [
    {
      "title": "Full title of the paper",
      "summary": "2-3 paragraphs focusing on: 1) Technical contributions to agentic systems, 2) Practical implementation insights, 3) Key implications for autonomous AI deployment. Emphasize agent architectures, tool usage, planning capabilities, and real-world applications.",
      "link": "https://arxiv.org/abs/paper-id"
    }
  ],
  "additional_papers": [
    {
      "title": "Title of paper",
      "link": "https://arxiv.org/abs/paper-id"
    }
  ],
  "metadata": {
    "generated_date": "YYYY-MM-DD",
    "total_papers_analyzed": 10,
    "featured_papers_count": 7,
    "additional_papers_count": 3
  }
}

Guidelines:
- Featured papers receive full summaries focusing on agentic AI aspects
- Additional papers only get title and link
- Emphasize practical deployment, agent architectures, and tool integration
- Highlight papers that advance autonomous AI systems
- Focus on real-world applications and implementation challenges"""

    def evaluate_paper(self, title: str, abstract: str = "") -> tuple[float, str]:
        """
        Evaluate a paper using DSPy-optimized prompts
        
        Returns:
            tuple: (score, reasoning)
        """
        try:
            result = self.paper_evaluator(
                paper_title=title,
                paper_abstract=abstract,
                evaluation_criteria=self.evaluation_criteria
            )
            
            # Extract score and ensure it's a float between 0 and 1
            try:
                score = float(result.score)
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except (ValueError, TypeError):
                logger.warning(f"Invalid score format: {result.score}, defaulting to 0.0")
                score = 0.0
            
            reasoning = result.reasoning if hasattr(result, 'reasoning') else "No reasoning provided"
            
            return score, reasoning
            
        except Exception as e:
            logger.error(f"Error evaluating paper {title}: {str(e)}")
            return 0.0, f"Evaluation error: {str(e)}"
    
    def generate_newsletter(self, featured_papers_content: str, additional_papers_content: str = "") -> str:
        """
        Generate newsletter using DSPy-optimized prompts
        
        Returns:
            str: Newsletter content in JSON format
        """
        try:
            result = self.newsletter_generator(
                featured_papers=featured_papers_content,
                additional_papers=additional_papers_content,
                newsletter_guidelines=self.newsletter_guidelines
            )
            
            return result.newsletter_json
            
        except Exception as e:
            logger.error(f"Error generating newsletter: {str(e)}")
            raise
    
    def optimize_prompts(self, training_examples: List[Dict[str, Any]]):
        """
        Optimize prompts using DSPy's few-shot learning
        
        Args:
            training_examples: List of training examples for optimization
        """
        if not DSPY_AVAILABLE:
            logger.warning("DSPy not available, skipping prompt optimization")
            return
            
        try:
            logger.info("Starting prompt optimization with DSPy")
            
            # Create training set for paper evaluation
            evaluation_trainset = []
            for example in training_examples:
                if 'evaluation' in example:
                    eval_data = example['evaluation']
                    evaluation_trainset.append(
                        dspy.Example(
                            paper_title=eval_data.get('title', ''),
                            paper_abstract=eval_data.get('abstract', ''),
                            evaluation_criteria=self.evaluation_criteria,
                            score=eval_data.get('score', 0.0),
                            reasoning=eval_data.get('reasoning', '')
                        ).with_inputs('paper_title', 'paper_abstract', 'evaluation_criteria')
                    )
            
            if evaluation_trainset:
                # Optimize paper evaluator
                optimizer = BootstrapFewShot(metric=lambda x, y: abs(x.score - y.score) < 0.1)
                self.paper_evaluator = optimizer.compile(
                    PaperEvaluator(), 
                    trainset=evaluation_trainset[:min(10, len(evaluation_trainset))]
                )
                logger.info("Paper evaluator optimized")
            
            # Create training set for newsletter generation
            newsletter_trainset = []
            for example in training_examples:
                if 'newsletter' in example:
                    news_data = example['newsletter']
                    newsletter_trainset.append(
                        dspy.Example(
                            featured_papers=news_data.get('featured_papers', ''),
                            additional_papers=news_data.get('additional_papers', ''),
                            newsletter_guidelines=self.newsletter_guidelines,
                            newsletter_json=news_data.get('expected_output', '')
                        ).with_inputs('featured_papers', 'additional_papers', 'newsletter_guidelines')
                    )
            
            if newsletter_trainset:
                # Optimize newsletter generator  
                optimizer = BootstrapFewShot(metric=lambda x, y: len(x.newsletter_json) > 100)
                self.newsletter_generator = optimizer.compile(
                    NewsletterGenerator(),
                    trainset=newsletter_trainset[:min(5, len(newsletter_trainset))]
                )
                logger.info("Newsletter generator optimized")
            
            logger.info("Prompt optimization completed")
            
        except Exception as e:
            logger.error(f"Error during prompt optimization: {str(e)}")
            # Continue with default prompts if optimization fails

# Example training data for prompt optimization
TRAINING_EXAMPLES = [
    {
        'evaluation': {
            'title': 'Agent-as-a-Judge: Evaluate Agents with Agents',
            'abstract': 'We propose using AI agents to evaluate other AI agents...',
            'score': 0.85,
            'reasoning': 'High relevance to agentic AI systems with practical evaluation framework'
        }
    },
    {
        'evaluation': {
            'title': 'Theoretical Analysis of Quantum Computing Algorithms',
            'abstract': 'This paper provides theoretical bounds...',
            'score': 0.1,
            'reasoning': 'No relevance to AI agents or practical LLM applications'
        }
    }
]