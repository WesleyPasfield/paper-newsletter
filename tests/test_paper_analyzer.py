import json
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import requests
from openai import RateLimitError

from functions.newsletter_generator import (
    ABSTRACT_EVAL_PROMPT,
    NEWSLETTER_PROMPT,
    TITLE_EVAL_PROMPT,
)
from functions.paper_analyzer import (
    FINAL_RELEVANCE_FLOOR,
    AbstractEvaluation,
    CircuitState,
    Paper,
    PaperAnalyzer,
    canonical_arxiv_id,
)


def evaluation(score: float, bucket: str = "loop_engineering") -> AbstractEvaluation:
    return AbstractEvaluation(
        primary_bucket=bucket,
        direct_fit=score,
        evidence_quality=score,
        production_transferability=score,
        domain_expert_relevance=score,
        novelty=score,
    )


def rate_limit_error(retry_after: str = "0") -> RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx.Response(
        429,
        request=request,
        headers={"Retry-After": retry_after},
    )
    return RateLimitError("rate limit", response=response, body=None)


class PaperTests(unittest.TestCase):
    def test_equality_and_hash_use_the_same_fields(self):
        first = Paper(title="Same title", link="https://example.com/one")
        equal = Paper(title="Same title", link="https://example.com/one")
        different_link = Paper(title="Same title", link="https://example.com/two")

        self.assertEqual(first, equal)
        self.assertEqual(hash(first), hash(equal))
        self.assertNotEqual(first, different_link)

    def test_canonical_arxiv_id_removes_link_format_and_version(self):
        self.assertEqual(
            canonical_arxiv_id("https://arxiv.org/pdf/2608.12345v2.pdf"),
            "2608.12345",
        )
        self.assertEqual(
            canonical_arxiv_id("https://export.arxiv.org/abs/cs.AI/9901001v3"),
            "cs.ai/9901001",
        )


class PaperAnalyzerModelTests(unittest.TestCase):
    @patch("functions.paper_analyzer.OpenAI")
    def setUp(self, openai_client):
        self.analyzer = PaperAnalyzer(
            openai_api_key="test-key",
            eval_prompt="eval",
            title_eval_prompt="title",
            abstract_eval_prompt="abstract",
            newsletter_prompt="newsletter",
        )
        self.analyzer._enforce_rate_limit = lambda: None

    def test_default_model_routing_uses_requested_models(self):
        self.assertEqual(self.analyzer.openai_model_cheap, "gpt-5.6-luna")
        self.assertEqual(self.analyzer.openai_model_pricey, "gpt-5.6-luna")
        self.assertEqual(self.analyzer.openai_model_fallback, "gpt-5.6-sol")
        self.assertEqual(self.analyzer.openai_model_emergency, "gpt-5.5")

    @patch("functions.paper_analyzer.time.sleep")
    @patch("functions.paper_analyzer.random.uniform", return_value=1.0)
    def test_real_retry_loop_reaches_sol_then_emergency(self, _, sleep):
        models = []
        error = rate_limit_error("7")

        def call():
            models.append(self.analyzer._get_current_model(is_expensive_operation=True))
            if len(models) < 3:
                raise error
            return "ok"

        result = self.analyzer.api_call_with_retry(max_retries=3, func=call)

        self.assertEqual(result, "ok")
        self.assertEqual(models, ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.5"])
        self.assertEqual(self.analyzer.circuit_breaker.state, CircuitState.CLOSED)
        self.assertGreaterEqual(sleep.call_args_list[0].args[0], 7.0)

    @patch("functions.paper_analyzer.time.sleep")
    def test_retry_loop_respects_total_attempt_budget(self, _):
        attempts = []
        error = rate_limit_error()

        def call():
            attempts.append(self.analyzer._get_current_model(True))
            raise error

        result = self.analyzer.api_call_with_retry(max_retries=3, func=call)

        self.assertIsNone(result)
        self.assertEqual(
            attempts,
            ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.5"],
        )

    @patch("functions.paper_analyzer.time.sleep")
    def test_each_logical_call_starts_on_luna(self, _):
        attempts = []

        def first_call():
            attempts.append(self.analyzer._get_current_model(True))
            if len(attempts) == 1:
                raise rate_limit_error()
            return "first"

        self.assertEqual(self.analyzer.api_call_with_retry(func=first_call), "first")
        self.assertEqual(
            self.analyzer.api_call_with_retry(
                func=lambda: self.analyzer._get_current_model(True)
            ),
            "gpt-5.6-luna",
        )

    @patch("functions.paper_analyzer.random.uniform", return_value=1.0)
    def test_retry_after_is_not_truncated_by_backoff_cap(self, _):
        delay = self.analyzer._calculate_retry_delay(
            attempt=0,
            base_delay=1.0,
            error=rate_limit_error("480"),
        )
        self.assertEqual(delay, 480.0)

    def test_input_token_acceleration_takes_precedence_over_429(self):
        error = MagicMock(status_code=429)
        error.__str__.return_value = "Input tokens acceleration limit exceeded"
        self.assertEqual(
            self.analyzer._classify_error(error),
            "input_token_acceleration",
        )

    def test_title_uses_no_reasoning_and_abstract_uses_low(self):
        self.analyzer.api_call_with_retry = lambda func, **_: func()
        self.analyzer._create_text_response = MagicMock(
            side_effect=[
                "0.8",
                json.dumps(
                    {
                        "primary_bucket": "loop_engineering",
                        "direct_fit": 0.9,
                        "evidence_quality": 0.8,
                        "production_transferability": 0.7,
                        "domain_expert_relevance": 0.6,
                        "novelty": 0.5,
                        "uncertainty": 0.1,
                        "exclusion_reason": "",
                    }
                ),
            ]
        )
        paper = Paper(
            title="Evaluation Loops",
            link="https://arxiv.org/abs/2608.12345",
            paper_id="2608.12345",
            abstract="An evaluated feedback loop.",
        )

        self.assertEqual(self.analyzer.evaluate_title(paper), 0.8)
        abstract_result = self.analyzer.evaluate_abstract_details(paper)

        self.assertAlmostEqual(abstract_result.total, 0.785)
        self.assertEqual(
            self.analyzer._create_text_response.call_args_list[0].kwargs[
                "reasoning_effort"
            ],
            "none",
        )
        self.assertEqual(
            self.analyzer._create_text_response.call_args_list[1].kwargs[
                "reasoning_effort"
            ],
            "low",
        )

    def test_text_response_passes_reasoning_effort(self):
        self.analyzer.client = MagicMock()
        self.analyzer.client.responses.create.return_value.output_text = "0.8"
        result = self.analyzer._create_text_response(
            model="gpt-5.6-luna",
            instructions="instructions",
            prompt="prompt",
            max_output_tokens=16,
            reasoning_effort="none",
        )
        self.assertEqual(result, "0.8")
        self.analyzer.client.responses.create.assert_called_once_with(
            model="gpt-5.6-luna",
            instructions="instructions",
            input="prompt",
            max_output_tokens=16,
            reasoning={"effort": "none"},
        )


class PaperPipelineTests(unittest.TestCase):
    @patch("functions.paper_analyzer.OpenAI")
    def setUp(self, _):
        self.analyzer = PaperAnalyzer(
            openai_api_key="test",
            eval_prompt="eval",
            newsletter_prompt="newsletter",
            candidate_pool_size=20,
            abstract_pool_size=20,
            max_featured=3,
            max_additional=1,
            max_paper_age_days=30,
        )
        self.analyzer._enforce_rate_limit = lambda: None

    @staticmethod
    def paper(paper_id: str, score: float = 0.0) -> Paper:
        return Paper(
            title=f"Paper {paper_id}",
            link=f"https://arxiv.org/abs/{paper_id}",
            paper_id=paper_id,
            abstract="Relevant abstract",
            published_date="2026-08-20",
            interest_score=score,
        )

    def test_canonical_duplicates_are_removed_before_scoring(self):
        first = self.paper("2608.11111")
        second = self.paper("2608.11111")
        second.link = "https://arxiv.org/pdf/2608.11111v2.pdf"
        second.paper_id = "2608.11111"
        second.abstract = "A richer abstract with more detail."
        mapping = {"feed-a": [first], "feed-b": [second]}
        self.analyzer._fetch_feed_candidates = lambda url: mapping[url]

        candidates, errors = self.analyzer.collect_candidates(
            ["feed-a", "feed-b"],
            now=datetime(2026, 8, 23, tzinfo=timezone.utc),
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].abstract, second.abstract)

    def test_stale_candidates_are_filtered(self):
        fresh = self.paper("2608.10001")
        stale = self.paper("2502.10001")
        stale.published_date = "2025-02-04"
        self.analyzer._fetch_feed_candidates = lambda _: [stale, fresh]

        candidates, _ = self.analyzer.collect_candidates(
            ["feed"],
            now=datetime(2026, 8, 23, tzinfo=timezone.utc),
        )

        self.assertEqual([paper.paper_id for paper in candidates], ["2608.10001"])

    def test_balanced_candidate_pool_is_source_diverse_and_feed_order_independent(self):
        high_volume = [self.paper(f"2608.11{index:03d}") for index in range(8)]
        low_volume = [self.paper(f"2608.22{index:03d}") for index in range(2)]
        mapping = {"feed-a": high_volume, "feed-b": low_volume}

        def collect(feed_order):
            analyzer = PaperAnalyzer(
                openai_api_key="test",
                eval_prompt="eval",
                newsletter_prompt="newsletter",
                candidate_pool_size=4,
                abstract_pool_size=4,
            )
            analyzer._fetch_feed_candidates = lambda url: [
                Paper(**vars(paper)) for paper in mapping[url]
            ]
            candidates, errors = analyzer.collect_candidates(
                feed_order,
                now=datetime(2026, 8, 23, tzinfo=timezone.utc),
            )
            return [paper.paper_id for paper in candidates], errors

        forward, forward_errors = collect(["feed-a", "feed-b"])
        reverse, reverse_errors = collect(["feed-b", "feed-a"])

        self.assertEqual(forward_errors, [])
        self.assertEqual(reverse_errors, [])
        self.assertEqual(forward, reverse)
        self.assertEqual(sum(paper_id.startswith("2608.22") for paper_id in forward), 2)

    def test_scoring_keeps_valid_below_floor_candidates_for_final_fallback(self):
        below = self.paper("2608.00001")
        above = self.paper("2608.00002")
        self.analyzer.evaluate_title = lambda _: 0.9
        scores = {below.paper_id: 0.59, above.paper_id: 0.61}
        self.analyzer.evaluate_abstract_details = lambda paper: evaluation(
            scores[paper.paper_id]
        )

        ranked, _, failed = self.analyzer._score_and_rank_candidates(
            [below, above], FINAL_RELEVANCE_FLOOR
        )

        self.assertFalse(failed)
        self.assertEqual(
            [paper.paper_id for paper in ranked],
            [above.paper_id, below.paper_id],
        )

    def test_title_screen_preserves_minimum_recall_pool(self):
        papers = [
            self.paper("2608.01001"),
            self.paper("2608.01002"),
            self.paper("2608.01003"),
        ]
        self.analyzer.evaluate_title = lambda _: 0.2
        self.analyzer.evaluate_abstract_details = lambda _: evaluation(0.7)

        ranked, counters, failed = self.analyzer._score_and_rank_candidates(
            papers, FINAL_RELEVANCE_FLOOR
        )

        self.assertFalse(failed)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(counters["title_screened"], 3)

    def test_global_ranking_is_independent_of_feed_order_and_html(self):
        paper_a = self.paper("2608.00001")
        paper_b = self.paper("2608.00002")
        paper_c = self.paper("2608.00003")
        mapping = {"feed-a": [paper_a], "feed-b": [paper_b, paper_c]}
        scores = {
            paper_a.paper_id: 0.70,
            paper_b.paper_id: 0.95,
            paper_c.paper_id: 0.80,
        }

        def run(feed_order):
            analyzer = PaperAnalyzer(
                openai_api_key="test",
                eval_prompt="eval",
                newsletter_prompt="newsletter",
                max_featured=2,
                min_featured=2,
                max_additional=1,
            )
            analyzer._fetch_feed_candidates = lambda url: [
                Paper(**vars(paper)) for paper in mapping[url]
            ]
            analyzer.evaluate_title = lambda _: 0.9
            analyzer.evaluate_abstract_details = lambda paper: evaluation(
                scores[paper.paper_id]
            )
            analyzer.get_paper_content = lambda _: ""
            captured = {}

            def create(featured, additional, metadata):
                captured["featured"] = [paper.paper_id for paper in featured]
                captured["grounding"] = [paper.grounding_source for paper in featured]
                captured["additional"] = [paper.paper_id for paper in additional]
                return json.dumps(captured)

            analyzer.create_newsletter = create
            return json.loads(analyzer.process_multiple_feeds(feed_order))

        forward = run(["feed-a", "feed-b"])
        reverse = run(["feed-b", "feed-a"])

        self.assertEqual(forward, reverse)
        self.assertEqual(forward["featured"], [paper_b.paper_id, paper_c.paper_id])
        self.assertEqual(forward["grounding"], ["abstract", "abstract"])

    def test_section_extraction_stops_before_references_and_deduplicates(self):
        html = """
        <main class="ltx_page_main">
          <div class="ltx_abstract"><p>Abstract evidence.</p></div>
          <h2>2 Method</h2><p>Method detail.</p><p>Method detail.</p>
          <h2>3 Evaluation and Results</h2><p>Measured result.</p>
          <h2>4 Limitations</h2><p>Known limitation.</p>
          <h2>5 Conclusion</h2><p>Supported conclusion.</p>
          <h2>References</h2><p>Should never appear.</p>
        </main>
        """
        extracted = self.analyzer.extract_relevant_sections(html)
        self.assertIn("Abstract evidence", extracted)
        self.assertIn("Measured result", extracted)
        self.assertIn("Known limitation", extracted)
        self.assertEqual(extracted.count("Method detail"), 1)
        self.assertNotIn("Should never appear", extracted)

    @patch("functions.paper_analyzer.time.sleep")
    def test_summary_identity_mismatch_is_rejected(self, _):
        paper = self.paper("2608.12345")
        paper.grounding_source = "abstract"
        self.analyzer._create_text_response = MagicMock(
            return_value=json.dumps(
                {
                    "paper_id": "2608.99999",
                    "contribution": "Contribution.",
                    "evidence": "Evidence.",
                    "relevance_and_limitation": "Relevance.",
                }
            )
        )

        self.assertIsNone(self.analyzer._summarize_paper(paper))
        self.assertEqual(self.analyzer._create_text_response.call_count, 5)

    def test_summary_uses_medium_and_code_owns_identity_metadata(self):
        paper = self.paper("2608.12345")
        paper.authors = ["Ada Researcher"]
        paper.primary_bucket = "loop_engineering"
        paper.grounding_source = "abstract"
        self.analyzer.api_call_with_retry = lambda func, **_: func()
        self.analyzer._create_text_response = MagicMock(
            side_effect=[
                json.dumps(
                    {
                        "paper_id": paper.paper_id,
                        "contribution": "The paper introduces a feedback loop",
                        "evidence": "The authors evaluate it on held-out traces",
                        "relevance_and_limitation": "It supports deployment gating but covers one domain",
                    }
                ),
                json.dumps(
                    {"overview": "The issue examines evaluated feedback loops."}
                ),
            ]
        )

        result = json.loads(self.analyzer.create_newsletter([paper], []))

        featured = result["featured_papers"][0]
        self.assertEqual(featured["paper_id"], paper.paper_id)
        self.assertEqual(featured["title"], paper.title)
        self.assertEqual(featured["authors"], paper.authors)
        self.assertEqual(featured["published_date"], paper.published_date)
        self.assertEqual(featured["grounding_source"], "abstract")
        self.assertTrue(
            all(
                call.kwargs["reasoning_effort"] == "medium"
                for call in self.analyzer._create_text_response.call_args_list
            )
        )

    def test_pipeline_failure_is_distinct_from_low_quality(self):
        self.analyzer._fetch_feed_candidates = MagicMock(
            side_effect=requests.exceptions.Timeout("down")
        )
        failure = json.loads(self.analyzer.process_multiple_feeds(["feed-a"]))
        self.assertEqual(failure["metadata"]["pipeline_status"], "pipeline_failure")

        healthy = PaperAnalyzer(
            openai_api_key="test",
            eval_prompt="eval",
            newsletter_prompt="newsletter",
        )
        candidates = [
            self.paper("2608.20001"),
            self.paper("2608.20002"),
            self.paper("2608.20003"),
        ]
        healthy._fetch_feed_candidates = lambda _: candidates
        healthy.evaluate_title = lambda _: 0.9
        healthy.evaluate_abstract_details = lambda _: evaluation(0.4)
        healthy.get_paper_content = lambda _: ""
        healthy.create_newsletter = lambda featured, additional, metadata: json.dumps(
            {
                "featured": [paper.paper_id for paper in featured],
                "metadata": metadata,
            }
        )
        best_available = json.loads(
            healthy.process_multiple_feeds(
                ["feed-a"],
                now=datetime(2026, 8, 23, tzinfo=timezone.utc),
            )
        )
        self.assertEqual(best_available["metadata"]["selection_mode"], "best_available")
        self.assertEqual(best_available["metadata"]["selected"], 3)

    def test_weekly_selection_expands_from_seven_to_fourteen_days(self):
        papers = [
            self.paper("2608.30001"),
            self.paper("2608.30002"),
            self.paper("2608.30003"),
        ]
        papers[0].published_date = "2026-08-21"
        papers[1].published_date = "2026-08-14"
        papers[2].published_date = "2026-08-12"
        self.analyzer._fetch_feed_candidates = lambda _: papers
        self.analyzer.evaluate_title = lambda _: 0.9
        self.analyzer.evaluate_abstract_details = lambda _: evaluation(0.8)
        self.analyzer.get_paper_content = lambda _: ""
        self.analyzer.create_newsletter = lambda featured, additional, metadata: (
            json.dumps(metadata)
        )

        metadata = json.loads(
            self.analyzer.process_multiple_feeds(
                ["feed"], now=datetime(2026, 8, 23, tzinfo=timezone.utc)
            )
        )

        self.assertEqual(metadata["selection_mode"], "expanded_window")
        self.assertEqual(metadata["source_window_days"], 14)
        self.assertEqual(metadata["valid_candidates"], 3)
        self.assertEqual(metadata["selected"], 3)

    def test_weekly_selection_expands_to_thirty_days(self):
        papers = [
            self.paper("2608.31001"),
            self.paper("2608.31002"),
            self.paper("2608.31003"),
        ]
        papers[0].published_date = "2026-08-21"
        papers[1].published_date = "2026-08-14"
        papers[2].published_date = "2026-08-02"
        self.analyzer._fetch_feed_candidates = lambda _: papers
        self.analyzer.evaluate_title = lambda _: 0.9
        self.analyzer.evaluate_abstract_details = lambda _: evaluation(0.8)
        self.analyzer.get_paper_content = lambda _: ""
        self.analyzer.create_newsletter = lambda featured, additional, metadata: (
            json.dumps(metadata)
        )

        metadata = json.loads(
            self.analyzer.process_multiple_feeds(
                ["feed"], now=datetime(2026, 8, 23, tzinfo=timezone.utc)
            )
        )

        self.assertEqual(metadata["selection_mode"], "expanded_window")
        self.assertEqual(metadata["source_window_days"], 30)

    def test_healthy_pipeline_uses_normal_quality_floor_at_seven_days(self):
        papers = [
            self.paper("2608.32001"),
            self.paper("2608.32002"),
            self.paper("2608.32003"),
        ]
        self.analyzer._fetch_feed_candidates = lambda _: papers
        self.analyzer.evaluate_title = lambda _: 0.9
        self.analyzer.evaluate_abstract_details = lambda _: evaluation(0.8)
        self.analyzer.get_paper_content = lambda _: ""
        self.analyzer.create_newsletter = lambda featured, additional, metadata: (
            json.dumps(metadata)
        )

        metadata = json.loads(
            self.analyzer.process_multiple_feeds(
                ["feed"], now=datetime(2026, 8, 23, tzinfo=timezone.utc)
            )
        )

        self.assertEqual(metadata["selection_mode"], "quality_floor")
        self.assertEqual(metadata["source_window_days"], 7)

    def test_fewer_than_three_valid_candidates_is_pipeline_failure(self):
        papers = [self.paper("2608.33001"), self.paper("2608.33002")]
        self.analyzer._fetch_feed_candidates = lambda _: papers
        self.analyzer.evaluate_title = lambda _: 0.9
        self.analyzer.evaluate_abstract_details = lambda _: evaluation(0.8)

        result = json.loads(
            self.analyzer.process_multiple_feeds(
                ["feed"], now=datetime(2026, 8, 23, tzinfo=timezone.utc)
            )
        )

        self.assertEqual(result["metadata"]["pipeline_status"], "pipeline_failure")
        self.assertNotEqual(
            result["metadata"]["pipeline_status"], "no_qualified_papers"
        )

    def test_recent_submission_fallback_recovers_an_empty_rss_pool(self):
        papers = [
            self.paper("2609.01001"),
            self.paper("2609.01002"),
            self.paper("2609.01003"),
        ]
        for paper in papers:
            paper.published_date = "2026-09-04"

        def fetch(feed_url):
            return papers if feed_url == "fallback" else []

        self.analyzer._fetch_feed_candidates = fetch
        self.analyzer.evaluate_title = lambda _: 0.9
        self.analyzer.evaluate_abstract_details = lambda _: evaluation(0.8)
        self.analyzer.get_paper_content = lambda _: ""
        self.analyzer.create_newsletter = lambda featured, additional, metadata: (
            json.dumps(metadata)
        )

        metadata = json.loads(
            self.analyzer.process_multiple_feeds(
                ["rss-a", "rss-b"],
                now=datetime(2026, 9, 8, tzinfo=timezone.utc),
                fallback_feed_url="fallback",
            )
        )

        self.assertTrue(metadata["fallback_used"])
        self.assertEqual(metadata["fallback_failures"], 0)
        self.assertEqual(metadata["selected"], 3)

    def test_recent_submission_fallback_keeps_the_minimum_safety_gate(self):
        papers = [self.paper("2609.02001"), self.paper("2609.02002")]
        for paper in papers:
            paper.published_date = "2026-09-04"
        self.analyzer._fetch_feed_candidates = lambda url: (
            papers if url == "fallback" else []
        )

        result = json.loads(
            self.analyzer.process_multiple_feeds(
                ["rss"],
                now=datetime(2026, 9, 8, tzinfo=timezone.utc),
                fallback_feed_url="fallback",
            )
        )

        self.assertEqual(result["metadata"]["pipeline_status"], "pipeline_failure")
        self.assertIn("Fewer than three", result["metadata"]["error"])

    def test_one_model_evaluation_failure_is_skipped_when_three_score(self):
        papers = [
            self.paper("2608.34001"),
            self.paper("2608.34002"),
            self.paper("2608.34003"),
            self.paper("2608.34004"),
        ]
        self.analyzer._fetch_feed_candidates = lambda _: papers
        self.analyzer.evaluate_title = lambda paper: (
            -1.0 if paper.paper_id.endswith("2") else 0.9
        )
        self.analyzer.evaluate_abstract_details = lambda _: evaluation(0.8)
        self.analyzer.get_paper_content = lambda _: ""
        self.analyzer.create_newsletter = lambda featured, additional, metadata: (
            json.dumps(metadata)
        )

        metadata = json.loads(
            self.analyzer.process_multiple_feeds(
                ["feed"], now=datetime(2026, 8, 23, tzinfo=timezone.utc)
            )
        )

        self.assertEqual(metadata["degraded_evaluations"], 1)
        self.assertEqual(metadata["selected"], 3)

    def test_stage_wide_model_failure_fails_the_pipeline(self):
        papers = [
            self.paper("2608.34501"),
            self.paper("2608.34502"),
            self.paper("2608.34503"),
        ]
        self.analyzer._fetch_feed_candidates = lambda _: papers
        self.analyzer.evaluate_title = lambda _: -1.0

        result = json.loads(
            self.analyzer.process_multiple_feeds(
                ["feed"], now=datetime(2026, 8, 23, tzinfo=timezone.utc)
            )
        )

        self.assertEqual(result["metadata"]["pipeline_status"], "pipeline_failure")
        self.assertIn("All model evaluations", result["metadata"]["error"])

    def test_one_abstract_failure_is_skipped_when_three_score(self):
        papers = [
            self.paper("2608.34701"),
            self.paper("2608.34702"),
            self.paper("2608.34703"),
            self.paper("2608.34704"),
        ]
        self.analyzer._fetch_feed_candidates = lambda _: papers
        self.analyzer.evaluate_title = lambda _: 0.9
        self.analyzer.evaluate_abstract_details = lambda paper: (
            None if paper.paper_id.endswith("2") else evaluation(0.8)
        )
        self.analyzer.get_paper_content = lambda _: ""
        self.analyzer.create_newsletter = lambda featured, additional, metadata: (
            json.dumps(metadata)
        )

        metadata = json.loads(
            self.analyzer.process_multiple_feeds(
                ["feed"], now=datetime(2026, 8, 23, tzinfo=timezone.utc)
            )
        )

        self.assertEqual(metadata["degraded_evaluations"], 1)
        self.assertEqual(metadata["selected"], 3)

    def test_one_feed_failure_is_degraded_but_publishable(self):
        papers = [
            self.paper("2608.35001"),
            self.paper("2608.35002"),
            self.paper("2608.35003"),
        ]

        def fetch(feed_url):
            if feed_url == "broken":
                raise requests.exceptions.Timeout("down")
            return papers

        self.analyzer._fetch_feed_candidates = fetch
        self.analyzer.evaluate_title = lambda _: 0.9
        self.analyzer.evaluate_abstract_details = lambda _: evaluation(0.8)
        self.analyzer.get_paper_content = lambda _: ""
        self.analyzer.create_newsletter = lambda featured, additional, metadata: (
            json.dumps(metadata)
        )

        metadata = json.loads(
            self.analyzer.process_multiple_feeds(
                ["healthy", "broken"],
                now=datetime(2026, 8, 23, tzinfo=timezone.utc),
            )
        )

        self.assertEqual(metadata["feed_failures"], 1)
        self.assertEqual(metadata["degraded_sources"], ["broken"])
        self.assertEqual(metadata["selected"], 3)

    def test_all_feeds_failed_is_pipeline_failure(self):
        self.analyzer._fetch_feed_candidates = MagicMock(
            side_effect=requests.exceptions.Timeout("down")
        )

        result = json.loads(self.analyzer.process_multiple_feeds(["feed-a", "feed-b"]))

        self.assertEqual(result["metadata"]["pipeline_status"], "pipeline_failure")
        self.assertEqual(result["metadata"]["feed_failures"], 2)


class NewsletterPromptTests(unittest.TestCase):
    def test_title_prompt_is_recall_oriented(self):
        self.assertIn("recall-oriented", TITLE_EVAL_PROMPT)
        self.assertIn("Do not infer evidence quality", TITLE_EVAL_PROMPT)

    def test_abstract_prompt_prioritizes_loop_engineering_and_is_structured(self):
        self.assertIn("loop engineering (30%, highest priority)", ABSTRACT_EVAL_PROMPT)
        self.assertIn("closed feedback systems", ABSTRACT_EVAL_PROMPT)
        self.assertIn('"evidence_quality"', ABSTRACT_EVAL_PROMPT)

    def test_newsletter_prompt_has_one_consistent_length_contract(self):
        self.assertIn("Overview: 2-3 sentences and 50-100 words", NEWSLETTER_PROMPT)
        self.assertIn(
            "exactly 3 sentences and no more than 150 words each", NEWSLETTER_PROMPT
        )
        self.assertNotIn("Maximum 4-6 sentences", NEWSLETTER_PROMPT)


if __name__ == "__main__":
    unittest.main()
