import hashlib
import json
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from functions.newsletter_issue_store import NewsletterIssueStore, build_issue_id


class NewsletterIssueStoreTests(unittest.TestCase):
    def test_weekly_issue_id_separates_test_mode(self):
        event = {"time": "2026-08-23T12:00:00Z"}
        self.assertEqual(build_issue_id(event), "newsletter-2026-W34")
        self.assertEqual(
            build_issue_id(event, test_mode=True), "newsletter-2026-W34-test"
        )

    def test_invalid_manual_issue_id_is_rejected(self):
        with self.assertRaises(ValueError):
            build_issue_id({"issue_id": "../../unsafe"})

    @patch("functions.newsletter_issue_store.boto3.client")
    @patch("functions.newsletter_issue_store.boto3.resource")
    def test_create_archives_before_creating_ready_record(self, resource, client):
        table = MagicMock()
        resource.return_value.Table.return_value = table
        s3 = MagicMock()
        client.return_value = s3
        operations = MagicMock()
        operations.attach_mock(s3.put_object, "archive")
        operations.attach_mock(table.put_item, "record")
        store = NewsletterIssueStore("runs", "issues")
        content = {"overview": "Test", "featured_papers": [], "additional_papers": []}

        result = store.create("newsletter-2026-W34-test", content)

        s3.put_object.assert_called_once()
        table.put_item.assert_called_once()
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["s3_key"], "newsletters/newsletter-2026-W34-test.json")
        self.assertEqual(
            [call[0] for call in operations.mock_calls], ["archive", "record"]
        )

    @patch("functions.newsletter_issue_store.boto3.client")
    @patch("functions.newsletter_issue_store.boto3.resource")
    def test_load_content_checks_integrity(self, resource, client):
        table = MagicMock()
        resource.return_value.Table.return_value = table
        body = json.dumps({"overview": "Stored"}, sort_keys=True, indent=2).encode()
        stream = MagicMock()
        stream.read.return_value = body
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": stream}
        client.return_value = s3
        store = NewsletterIssueStore("runs", "issues")
        record = {
            "issue_id": "newsletter-2026-W34",
            "s3_key": "newsletters/newsletter-2026-W34.json",
            "content_sha256": hashlib.sha256(body).hexdigest(),
        }

        self.assertEqual(store.load_content(record), {"overview": "Stored"})

    @patch("functions.newsletter_issue_store.boto3.client")
    @patch("functions.newsletter_issue_store.boto3.resource")
    def test_completed_issue_cannot_be_claimed_again(self, resource, client):
        table = MagicMock()
        table.update_item.side_effect = ClientError(
            {
                "Error": {
                    "Code": "ConditionalCheckFailedException",
                    "Message": "claimed",
                }
            },
            "UpdateItem",
        )
        resource.return_value.Table.return_value = table
        client.return_value = MagicMock()
        store = NewsletterIssueStore("runs", "issues")

        self.assertFalse(store.claim_delivery("newsletter-2026-W34"))


if __name__ == "__main__":
    unittest.main()
