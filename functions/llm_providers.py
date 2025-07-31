"""
Multi-LLM Provider Interface
Provides unified interface for multiple LLM providers (Claude, OpenAI, etc.)
"""

import os
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

import boto3
from anthropic import Anthropic, APIError as AnthropicAPIError
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"

@dataclass
class LLMResponse:
    content: str
    provider: LLMProvider
    model: str
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None

@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    max_tokens: int
    temperature: float = 0.0
    timeout: int = 30

class LLMProviderInterface(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available and configured"""
        pass

class ClaudeProvider(LLMProviderInterface):
    """Claude/Anthropic LLM Provider"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Claude client with API key from AWS Secrets Manager"""
        try:
            api_key = self._get_secret()
            if api_key:
                self.client = Anthropic(api_key=api_key)
                logger.info("Claude client initialized successfully")
            else:
                logger.error("Failed to retrieve Claude API key")
        except Exception as e:
            logger.error(f"Error initializing Claude client: {str(e)}")
            self.client = None
    
    def _get_secret(self) -> Optional[str]:
        """Retrieve Claude API key from AWS Secrets Manager"""
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
                secret = json.loads(get_secret_value_response['SecretString'])
                return secret.get('anthropic_key')
            
            return None
        except Exception as e:
            logger.error(f"Error retrieving Claude secret: {str(e)}")
            return None
    
    def generate(self, system_prompt: str, user_prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate response using Claude"""
        if not self.client:
            raise RuntimeError("Claude client not initialized")
        
        try:
            response = self.client.messages.create(
                model=config.model,
                system=system_prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            content = response.content[0].text if response.content else ""
            
            # Safely extract token usage information
            tokens_used = None
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'total_tokens'):
                    tokens_used = response.usage.total_tokens
                elif hasattr(response.usage, 'input_tokens') and hasattr(response.usage, 'output_tokens'):
                    tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.CLAUDE,
                model=config.model,
                tokens_used=tokens_used
            )
            
        except AnthropicAPIError as e:
            logger.error(f"Claude API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Claude: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if Claude provider is available"""
        return self.client is not None

class OpenAIProvider(LLMProviderInterface):
    """OpenAI LLM Provider"""
    
    def __init__(self):
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client with API key from environment or AWS Secrets Manager"""
        try:
            api_key = self._get_secret()
            if api_key:
                # Initialize OpenAI client with minimal parameters
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized successfully")
            else:
                logger.error("Failed to retrieve OpenAI API key")
        except TypeError as e:
            logger.error(f"OpenAI client initialization error (likely version issue): {str(e)}")
            self.client = None
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            self.client = None
    
    def _get_secret(self) -> Optional[str]:
        """Retrieve OpenAI API key from AWS Secrets Manager or environment"""
        # First try environment variable
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            return api_key
        
        # Then try AWS Secrets Manager
        try:
            secret_name = "openai/api_key"
            region_name = os.environ.get('AWS_REGION', 'us-west-2')
            
            session = boto3.session.Session()
            client = session.client(
                service_name='secretsmanager',
                region_name=region_name
            )

            get_secret_value_response = client.get_secret_value(SecretId=secret_name)
            
            if 'SecretString' in get_secret_value_response:
                secret = json.loads(get_secret_value_response['SecretString'])
                return secret.get('openai_key')
            
            return None
        except Exception as e:
            logger.debug(f"OpenAI secret not found in AWS: {str(e)}")
            return None
    
    def generate(self, system_prompt: str, user_prompt: str, config: LLMConfig) -> LLMResponse:
        """Generate response using OpenAI"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=config.model,
                messages=messages,
                max_tokens=config.max_tokens,
                temperature=config.temperature
            )
            
            content = response.choices[0].message.content or ""
            
            # Safely extract token usage information
            tokens_used = None
            if hasattr(response, 'usage') and response.usage:
                if hasattr(response.usage, 'total_tokens'):
                    tokens_used = response.usage.total_tokens
                elif hasattr(response.usage, 'prompt_tokens') and hasattr(response.usage, 'completion_tokens'):
                    tokens_used = response.usage.prompt_tokens + response.usage.completion_tokens
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.OPENAI,
                model=config.model,
                tokens_used=tokens_used
            )
            
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        """Check if OpenAI provider is available"""
        return self.client is not None

class LLMManager:
    """Manages multiple LLM providers with fallback and load balancing"""
    
    def __init__(self):
        self.providers: Dict[LLMProvider, LLMProviderInterface] = {}
        self.provider_priority: List[LLMProvider] = []
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available LLM providers"""
        # Initialize Claude
        claude_provider = ClaudeProvider()
        if claude_provider.is_available():
            self.providers[LLMProvider.CLAUDE] = claude_provider
            self.provider_priority.append(LLMProvider.CLAUDE)
            logger.info("Claude provider initialized and available")
        
        # Initialize OpenAI
        openai_provider = OpenAIProvider()
        if openai_provider.is_available():
            self.providers[LLMProvider.OPENAI] = openai_provider
            self.provider_priority.append(LLMProvider.OPENAI)
            logger.info("OpenAI provider initialized and available")
        
        logger.info(f"LLM Manager initialized with {len(self.providers)} providers: {list(self.providers.keys())}")
    
    def get_available_providers(self) -> List[LLMProvider]:
        """Get list of available providers"""
        return list(self.providers.keys())
    
    def generate_with_fallback(self, 
                             system_prompt: str, 
                             user_prompt: str, 
                             configs: Dict[LLMProvider, LLMConfig],
                             preferred_provider: Optional[LLMProvider] = None,
                             max_retries: int = 3) -> LLMResponse:
        """
        Generate response with fallback to other providers if primary fails
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM
            configs: Configuration for each provider
            preferred_provider: Preferred provider to try first
            max_retries: Maximum retry attempts per provider
        """
        
        # Determine provider order
        provider_order = []
        if preferred_provider and preferred_provider in self.providers:
            provider_order.append(preferred_provider)
        
        # Add remaining providers in priority order
        for provider in self.provider_priority:
            if provider not in provider_order and provider in configs:
                provider_order.append(provider)
        
        if not provider_order:
            raise RuntimeError("No available providers or configurations")
        
        last_error = None
        
        for provider in provider_order:
            if provider not in self.providers:
                logger.warning(f"Provider {provider} not available, skipping")
                continue
            
            if provider not in configs:
                logger.warning(f"No configuration provided for {provider}, skipping")
                continue
            
            provider_instance = self.providers[provider]
            config = configs[provider]
            
            # Retry logic for each provider
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting generation with {provider} (attempt {attempt + 1}/{max_retries})")
                    response = provider_instance.generate(system_prompt, user_prompt, config)
                    logger.info(f"Successfully generated response with {provider}")
                    return response
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Error with {provider} (attempt {attempt + 1}): {str(e)}")
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff
                        wait_time = 2 ** attempt
                        logger.info(f"Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All retry attempts failed for {provider}")
        
        # If we get here, all providers failed
        raise RuntimeError(f"All LLM providers failed. Last error: {str(last_error)}")
    
    def generate(self, 
                system_prompt: str, 
                user_prompt: str, 
                config: LLMConfig,
                max_retries: int = 3) -> LLMResponse:
        """
        Generate response using specific provider configuration
        
        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for the LLM  
            config: LLM configuration specifying provider and model
            max_retries: Maximum retry attempts
        """
        if config.provider not in self.providers:
            raise ValueError(f"Provider {config.provider} not available")
        
        provider_instance = self.providers[config.provider]
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating with {config.provider} (attempt {attempt + 1}/{max_retries})")
                response = provider_instance.generate(system_prompt, user_prompt, config)
                logger.info(f"Successfully generated response with {config.provider}")
                return response
                
            except Exception as e:
                logger.warning(f"Error with {config.provider} (attempt {attempt + 1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for {config.provider}")
                    raise

# Default model configurations
DEFAULT_CONFIGS = {
    LLMProvider.CLAUDE: {
        'cheap': LLMConfig(
            provider=LLMProvider.CLAUDE,
            model="claude-3-haiku-20240307",
            max_tokens=4,
            temperature=0.0
        ),
        'expensive': LLMConfig(
            provider=LLMProvider.CLAUDE,
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            temperature=0.0
        )
    },
    LLMProvider.OPENAI: {
        'cheap': LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o-mini",
            max_tokens=4,
            temperature=0.0
        ),
        'expensive': LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            max_tokens=8000,
            temperature=0.0
        )
    }
}