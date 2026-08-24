import json
import unittest
from unittest.mock import MagicMock, patch

from functions import newsletter_generator


def publishable_issue() -> dict:
    return {
        "overview": "A grounded weekly overview.",
        "featured_papers": [
            {
                "title": f"Paper {index}",
                "link": f"https://arxiv.org/abs/2608.0000{index}",
                "summary": "Contribution. Evidence. Practical relevance.",
            }
            for index in range(1, 4)
        ],
        "additional_papers": [],
        "metadata": {"pipeline_status": "ok"},
    }


class NewsletterValidationTests(unittest.TestCase):
    def test_rejects_no_qualified_papers_as_a_normal_outcome(self):
        issue = publishable_issue()
        issue["featured_papers"] = []
        issue["metadata"]["pipeline_status"] = "no_qualified_papers"

        with self.assertRaises(RuntimeError):
            newsletter_generator._validate_generated_issue(issue)

    def test_rejects_fewer_than_three_featured_papers(self):
        issue = publishable_issue()
        issue["featured_papers"] = issue["featured_papers"][:2]

        with self.assertRaisesRegex(RuntimeError, "at least 3 featured papers"):
            newsletter_generator._validate_generated_issue(issue)

    def test_accepts_a_publishable_weekly_issue(self):
        newsletter_generator._validate_generated_issue(publishable_issue())


class NewsletterHandlerTests(unittest.TestCase):
    def setUp(self):
        self.environment = {
            "EMAIL_TEST_MODE": "false",
            "NEWSLETTER_BUCKET": "newsletter-bucket",
            "NEWSLETTER_RUNS_TABLE": "newsletter-runs",
            "SUBSCRIBERS_TABLE": "subscribers",
            "SENDER_EMAIL": "newsletter@example.com",
        }
        self.record = {
            "issue_id": "newsletter-2026-W34",
            "recipient_id": "ISSUE",
            "status": "ready",
            "s3_key": "newsletters/newsletter-2026-W34.json",
        }

    def configured_store(self):
        store = MagicMock()
        store.get.return_value = self.record
        store.load_content.return_value = publishable_issue()
        return store

    @patch("functions.newsletter_generator.publish_issue_metric")
    @patch("functions.newsletter_generator.send_newsletter")
    @patch("functions.newsletter_generator.NewsletterIssueStore")
    @patch("functions.newsletter_generator.validate_environment_variables")
    def test_preview_never_claims_or_sends(
        self,
        _validate_environment,
        issue_store_class,
        send_newsletter,
        _publish_metric,
    ):
        store = self.configured_store()
        issue_store_class.return_value = store

        with patch.dict("os.environ", self.environment, clear=False):
            response = newsletter_generator.lambda_handler(
                {"issue_id": "newsletter-2026-W34"},
                None,
            )

        self.assertEqual(response["statusCode"], 200)
        self.assertIn("no email was sent", json.loads(response["body"])["message"])
        store.claim_delivery.assert_not_called()
        send_newsletter.assert_not_called()

    @patch("functions.newsletter_generator.publish_issue_metric")
    @patch("functions.newsletter_generator.send_newsletter")
    @patch("functions.newsletter_generator.NewsletterIssueStore")
    @patch("functions.newsletter_generator.validate_environment_variables")
    def test_confirmed_run_claims_and_submits_once(
        self,
        _validate_environment,
        issue_store_class,
        send_newsletter,
        _publish_metric,
    ):
        store = self.configured_store()
        store.claim_delivery.return_value = True
        issue_store_class.return_value = store
        send_newsletter.return_value = {"accepted": 2, "failed": 0, "skipped": 0}

        with patch.dict("os.environ", self.environment, clear=False):
            response = newsletter_generator.lambda_handler(
                {
                    "issue_id": "newsletter-2026-W34",
                    "confirm_send": True,
                    "api_url": "https://example.com/prod",
                },
                None,
            )

        self.assertEqual(json.loads(response["body"])["issue_status"], "submitted")
        store.claim_delivery.assert_called_once_with(
            "newsletter-2026-W34",
            allow_active_resume=False,
        )
        send_newsletter.assert_called_once()
        store.mark.assert_called_once_with(
            "newsletter-2026-W34",
            "submitted",
            {"accepted": 2, "failed": 0, "skipped": 0},
        )

    @patch("functions.newsletter_generator.publish_issue_metric")
    @patch("functions.newsletter_generator.send_newsletter")
    @patch("functions.newsletter_generator.NewsletterIssueStore")
    @patch("functions.newsletter_generator.validate_environment_variables")
    def test_replayed_claim_does_not_send_again(
        self,
        _validate_environment,
        issue_store_class,
        send_newsletter,
        _publish_metric,
    ):
        store = self.configured_store()
        store.claim_delivery.return_value = False
        store.get.side_effect = [self.record, {**self.record, "status": "submitted"}]
        issue_store_class.return_value = store

        with patch.dict("os.environ", self.environment, clear=False):
            response = newsletter_generator.lambda_handler(
                {"issue_id": "newsletter-2026-W34", "confirm_send": True},
                None,
            )

        self.assertEqual(json.loads(response["body"])["status"], "submitted")
        send_newsletter.assert_not_called()

    @patch("functions.newsletter_generator._generate_issue")
    @patch(
        "functions.newsletter_generator.get_previously_included_papers",
        return_value=set(),
    )
    @patch("functions.newsletter_generator.get_secret", return_value="secret")
    @patch("functions.newsletter_generator.publish_issue_metric")
    @patch("functions.newsletter_generator.NewsletterIssueStore")
    @patch("functions.newsletter_generator.validate_environment_variables")
    def test_generation_failure_does_not_create_a_phantom_issue_record(
        self,
        _validate_environment,
        issue_store_class,
        publish_metric,
        _get_secret,
        _get_previous,
        generate_issue,
    ):
        store = MagicMock()
        store.get.return_value = None
        issue_store_class.return_value = store
        generate_issue.side_effect = RuntimeError("model stage unavailable")

        with (
            patch.dict("os.environ", self.environment, clear=False),
            self.assertRaisesRegex(RuntimeError, "model stage unavailable"),
        ):
            newsletter_generator.lambda_handler(
                {"issue_id": "newsletter-2026-W34", "confirm_send": True},
                None,
            )

        store.create.assert_not_called()
        store.mark.assert_not_called()
        publish_metric.assert_called_with("IssueGenerationFailed")


if __name__ == "__main__":
    unittest.main()
