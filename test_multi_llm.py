#!/usr/bin/env python3
"""
Test script for multi-LLM and DSPy integration
"""

import os
import sys
import json
import logging
from typing import Set

# Add functions directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'functions'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_providers():
    """Test LLM provider initialization"""  
    print("Testing LLM Provider Initialization...")
    
    # Set dummy API keys for testing
    os.environ['ANTHROPIC_API_KEY'] = 'test-key-123'  # Will fail auth but provider will initialize
    os.environ['OPENAI_API_KEY'] = 'test-key-456'     # Will fail auth but provider will initialize
    
    try:
        from functions.llm_providers import LLMManager, LLMProvider, DEFAULT_CONFIGS
        
        # Initialize LLM manager
        llm_manager = LLMManager()
        available_providers = llm_manager.get_available_providers()
        
        print(f"Available LLM providers: {available_providers}")
        
        if not available_providers:
            print("⚠️  No LLM providers available. Check your API keys.")
            return False
        
        # Test configuration
        for provider in available_providers:
            if provider in DEFAULT_CONFIGS:
                config = DEFAULT_CONFIGS[provider]['cheap']
                print(f"✅ {provider.value} configured with model: {config.model}")
            else:
                print(f"❌ {provider.value} missing configuration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing LLM providers: {str(e)}")
        return False

def test_dspy_integration():
    """Test DSPy integration"""
    print("\nTesting DSPy Integration...")
    
    try:
        from functions.llm_providers import LLMManager
        from functions.dspy_prompts import DSPyPromptManager
        
        llm_manager = LLMManager()
        dspy_manager = DSPyPromptManager(llm_manager)
        
        # Test paper evaluation
        test_title = "Agent-as-a-Judge: Evaluate Agents with Agents"
        test_abstract = "We propose using AI agents to evaluate other AI agents, enabling scalable evaluation of autonomous systems."
        
        print(f"Testing evaluation for: {test_title}")
        score, reasoning = dspy_manager.evaluate_paper(test_title, test_abstract)
        
        print(f"Score: {score:.3f}")
        print(f"Reasoning: {reasoning[:200]}...")
        
        if 0 <= score <= 1:
            print("✅ DSPy evaluation working correctly")
            return True
        else:
            print(f"❌ Invalid score returned: {score}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing DSPy integration: {str(e)}")
        return False

def test_paper_analyzer():
    """Test PaperAnalyzer with new interface"""
    print("\nTesting PaperAnalyzer...")
    
    try:
        from functions.paper_analyzer import PaperAnalyzer, Paper
        from functions.llm_providers import LLMProvider
        
        # Create analyzer
        analyzer = PaperAnalyzer(
            previously_included_papers=set(),
            preferred_provider=LLMProvider.CLAUDE
        )
        
        # Test paper evaluation
        test_paper = Paper(
            title="ReAct: Synergizing Reasoning and Acting in Language Models",
            link="https://arxiv.org/abs/2210.03629",
            abstract="We present ReAct, a general paradigm to combine reasoning and acting with language models for solving diverse language reasoning and decision making tasks."
        )
        
        print(f"Testing paper: {test_paper.title}")
        
        # Test title evaluation
        title_score = analyzer.evaluate_title(test_paper)
        print(f"Title evaluation score: {title_score:.3f}")
        
        # Test abstract evaluation
        abstract_score = analyzer.evaluate_abstract(test_paper)
        print(f"Abstract evaluation score: {abstract_score:.3f}")
        
        if test_paper.evaluation_reasoning:
            print(f"Reasoning: {test_paper.evaluation_reasoning[:150]}...")
        
        if test_paper.provider_used:
            print(f"Provider used: {test_paper.provider_used}")
        
        if 0 <= title_score <= 1 and 0 <= abstract_score <= 1:
            print("✅ PaperAnalyzer evaluation working correctly")
            return True
        else:
            print(f"❌ Invalid scores: title={title_score}, abstract={abstract_score}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing PaperAnalyzer: {str(e)}")
        return False

def test_newsletter_generation():
    """Test newsletter generation with sample data"""
    print("\nTesting Newsletter Generation...")
    
    try:
        from functions.paper_analyzer import PaperAnalyzer, Paper
        from functions.llm_providers import LLMProvider
        
        analyzer = PaperAnalyzer(
            previously_included_papers=set(),
            preferred_provider=LLMProvider.CLAUDE
        )
        
        # Create sample papers
        sample_papers = [
            Paper(
                title="Agent-as-a-Judge: Evaluate Agents with Agents",
                link="https://arxiv.org/abs/2024.example1",
                abstract="We propose using AI agents to evaluate other AI agents...",
                full_text="Full content would be here...",
                interest_score=0.85
            ),
            Paper(
                title="Multi-Agent Systems for Complex Task Orchestration",
                link="https://arxiv.org/abs/2024.example2", 
                abstract="This paper presents a framework for coordinating multiple AI agents...",
                full_text="Full content would be here...",
                interest_score=0.78
            )
        ]
        
        additional_papers = [
            Paper(
                title="Tool Learning with Foundation Models",
                link="https://arxiv.org/abs/2024.example3",
                interest_score=0.65
            )
        ]
        
        print(f"Testing newsletter generation with {len(sample_papers)} featured papers and {len(additional_papers)} additional papers")
        
        newsletter_content = analyzer.create_newsletter(sample_papers, additional_papers)
        
        # Try to parse as JSON
        try:
            newsletter_json = json.loads(newsletter_content)
            
            required_fields = ['overview', 'featured_papers', 'additional_papers', 'metadata']
            missing_fields = [field for field in required_fields if field not in newsletter_json]
            
            if missing_fields:
                print(f"❌ Missing required fields: {missing_fields}")
                return False
            
            print(f"✅ Newsletter generated successfully")
            print(f"   Featured papers: {len(newsletter_json.get('featured_papers', []))}")
            print(f"   Additional papers: {len(newsletter_json.get('additional_papers', []))}")
            print(f"   Overview length: {len(newsletter_json.get('overview', ''))}")
            
            return True
            
        except json.JSONDecodeError:
            print(f"❌ Newsletter content is not valid JSON")
            print(f"Content preview: {newsletter_content[:300]}...")
            return False
            
    except Exception as e:
        print(f"❌ Error testing newsletter generation: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Starting Multi-LLM and DSPy Integration Tests\n")
    
    # Set environment variables for testing
    os.environ.setdefault('AWS_REGION', 'us-west-2')
    os.environ.setdefault('PREFERRED_LLM_PROVIDER', 'claude')
    
    tests = [
        ("LLM Providers", test_llm_providers),
        ("DSPy Integration", test_dspy_integration), 
        ("PaperAnalyzer", test_paper_analyzer),
        ("Newsletter Generation", test_newsletter_generation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test {test_name} failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Test Results Summary:")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print("-"*50)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)