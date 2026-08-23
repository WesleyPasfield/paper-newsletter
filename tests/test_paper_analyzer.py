import unittest
from unittest.mock import MagicMock, patch

from functions.newsletter_generator import EVAL_PROMPT, NEWSLETTER_PROMPT
from functions.paper_analyzer import Paper, PaperAnalyzer


class PaperTests(unittest.TestCase):
    def test_equality_and_hash_use_the_same_fields(self):
        first = Paper(title='Same title', link='https://example.com/one')
        equal = Paper(title='Same title', link='https://example.com/one')
        different_link = Paper(title='Same title', link='https://example.com/two')

        self.assertEqual(first, equal)
        self.assertEqual(hash(first), hash(equal))
        self.assertNotEqual(first, different_link)


class PaperAnalyzerModelTests(unittest.TestCase):
    @patch('functions.paper_analyzer.OpenAI')
    def test_default_model_routing_uses_requested_models(self, openai_client):
        analyzer = PaperAnalyzer(
            openai_api_key='test-key',
            eval_prompt='eval',
            newsletter_prompt='newsletter',
        )

        self.assertEqual(analyzer.openai_model_cheap, 'gpt-5.6-luna')
        self.assertEqual(analyzer.openai_model_pricey, 'gpt-5.6-luna')
        self.assertEqual(analyzer.openai_model_fallback, 'gpt-5.6-sol')
        self.assertEqual(analyzer.openai_model_emergency, 'gpt-5.5')
        openai_client.assert_called_once_with(api_key='test-key')

    def test_repeated_rate_limits_advance_through_fallbacks(self):
        analyzer = PaperAnalyzer.__new__(PaperAnalyzer)
        analyzer.fallback_level = 0
        analyzer.consecutive_rate_limit_failures = 0
        analyzer.openai_model_fallback = 'gpt-5.6-sol'
        analyzer.openai_model_emergency = 'gpt-5.5'

        for _ in range(3):
            analyzer._record_standard_rate_limit()
        self.assertEqual(analyzer.fallback_level, 1)
        self.assertEqual(analyzer.consecutive_rate_limit_failures, 0)

        for _ in range(3):
            analyzer._record_standard_rate_limit()
        self.assertEqual(analyzer.fallback_level, 2)

    def test_input_token_acceleration_takes_precedence_over_429(self):
        error = MagicMock(status_code=429)
        error.__str__.return_value = 'Input tokens acceleration limit exceeded'
        analyzer = PaperAnalyzer.__new__(PaperAnalyzer)

        self.assertEqual(
            analyzer._classify_error(error),
            'input_token_acceleration',
        )

    def test_text_response_passes_reasoning_effort(self):
        analyzer = PaperAnalyzer.__new__(PaperAnalyzer)
        analyzer.client = MagicMock()
        analyzer.client.responses.create.return_value.output_text = '0.8'

        result = analyzer._create_text_response(
            model='gpt-5.6-luna',
            instructions='instructions',
            prompt='prompt',
            max_output_tokens=16,
            reasoning_effort='none',
        )

        self.assertEqual(result, '0.8')
        analyzer.client.responses.create.assert_called_once_with(
            model='gpt-5.6-luna',
            instructions='instructions',
            input='prompt',
            max_output_tokens=16,
            reasoning={'effort': 'none'},
        )

    def test_newsletter_generation_uses_medium_reasoning(self):
        analyzer = PaperAnalyzer.__new__(PaperAnalyzer)
        analyzer.processed_papers = set()
        analyzer.previously_included_papers = set()
        analyzer.newsletter_prompt = 'newsletter instructions'
        analyzer.api_call_with_retry = lambda func: func()
        analyzer._get_current_model = MagicMock(return_value='gpt-5.6-luna')
        analyzer._create_text_response = MagicMock(return_value='{}')
        paper = Paper(
            title='Loop Engineering for Reliable Agents',
            link='https://arxiv.org/abs/1234.5678',
            full_text='Paper content',
            interest_score=0.9,
        )

        self.assertEqual(analyzer.create_newsletter([paper], []), '{}')
        self.assertEqual(
            analyzer._create_text_response.call_args.kwargs['reasoning_effort'],
            'medium',
        )


class NewsletterPromptTests(unittest.TestCase):
    def test_interest_prompt_prioritizes_loop_engineering(self):
        self.assertIn('Agent evaluation, optimization, and loop engineering (30%)', EVAL_PROMPT)
        self.assertIn('closed feedback systems', EVAL_PROMPT)

    def test_newsletter_prompt_has_one_consistent_length_contract(self):
        self.assertIn('Overview: 2-3 sentences and 50-100 words', NEWSLETTER_PROMPT)
        self.assertIn('exactly 3 sentences and no more than 150 words each', NEWSLETTER_PROMPT)
        self.assertNotIn('Maximum 4-6 sentences', NEWSLETTER_PROMPT)
        self.assertNotIn('2024-11-21', NEWSLETTER_PROMPT)


if __name__ == '__main__':
    unittest.main()
