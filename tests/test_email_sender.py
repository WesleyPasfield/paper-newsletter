import json
import os
import unittest
from datetime import timedelta
from unittest.mock import patch

from functions import email_sender


class FakeTable:
    def __init__(self, item=None):
        self.item = item
        self.update_calls = []

    def get_item(self, **kwargs):
        return {'Item': self.item} if self.item else {}

    def update_item(self, **kwargs):
        self.update_calls.append(kwargs)
        return {}


class FakeDynamoDB:
    def __init__(self, table):
        self.table = table

    def Table(self, table_name):
        return self.table


class EmailSenderTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                'SUBSCRIBERS_TABLE': 'subscribers',
                'SENDER_EMAIL': 'newsletter@example.com',
            },
            clear=False,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_verification_url_encodes_email_and_token(self):
        url = email_sender.build_verification_url(
            'https://api.example.com/prod',
            'person+tag@example.com',
            'a token&value',
        )

        self.assertEqual(
            url,
            'https://api.example.com/prod/verify?email=person%2Btag%40example.com&token=a+token%26value',
        )

    def test_expired_token_is_rejected(self):
        subscriber = {
            'verification_expires_at': (
                email_sender.utc_now() - timedelta(seconds=1)
            ).isoformat(),
        }

        self.assertTrue(email_sender.token_is_expired(subscriber))

    def test_expired_token_is_rotated_before_a_reminder(self):
        table = FakeTable()
        subscriber = {
            'email': 'person@example.com',
            'status': 'pending',
            'verification_token': 'expired-token',
            'verification_expires_at': (
                email_sender.utc_now() - timedelta(seconds=1)
            ).isoformat(),
        }

        with patch.object(
            email_sender.secrets,
            'token_urlsafe',
            return_value='replacement-token',
        ):
            token = email_sender.ensure_active_verification_token(table, subscriber)

        self.assertEqual(token, 'replacement-token')
        self.assertEqual(len(table.update_calls), 1)
        self.assertEqual(
            table.update_calls[0]['ExpressionAttributeValues'][':token'],
            'replacement-token',
        )

    def test_bulk_retry_selects_only_records_with_expired_tokens(self):
        valid = {
            'email': 'valid@example.com',
            'verification_token': 'valid-token',
            'verification_expires_at': (
                email_sender.utc_now() + timedelta(days=1)
            ).isoformat(),
        }
        expired = {
            'email': 'expired@example.com',
            'verification_token': 'expired-token',
            'verification_expires_at': (
                email_sender.utc_now() - timedelta(seconds=1)
            ).isoformat(),
        }

        selected = email_sender.select_bulk_verification_subscribers(
            [valid, expired],
            expired_only=True,
        )

        self.assertEqual(selected, [expired])

    def test_pending_subscriber_can_be_verified(self):
        table = FakeTable(
            {
                'email': 'person@example.com',
                'status': 'pending',
                'verification_token': 'valid-token',
                'verification_expires_at': (
                    email_sender.utc_now() + timedelta(days=1)
                ).isoformat(),
            }
        )
        event = {
            'httpMethod': 'GET',
            'queryStringParameters': {
                'email': 'person@example.com',
                'token': 'valid-token',
            },
        }

        with patch.object(
            email_sender.boto3,
            'resource',
            return_value=FakeDynamoDB(table),
        ):
            response = email_sender.verify_subscriber(event, None)

        self.assertEqual(response['statusCode'], 200)
        self.assertEqual(len(table.update_calls), 1)
        self.assertIn('#status = :pending', table.update_calls[0]['ConditionExpression'])

    def test_unsubscribed_record_cannot_be_reverified(self):
        table = FakeTable(
            {
                'email': 'person@example.com',
                'status': 'unsubscribed',
                'verification_token': 'old-token',
            }
        )
        event = {
            'httpMethod': 'GET',
            'queryStringParameters': {
                'email': 'person@example.com',
                'token': 'old-token',
            },
        }

        with patch.object(
            email_sender.boto3,
            'resource',
            return_value=FakeDynamoDB(table),
        ):
            response = email_sender.verify_subscriber(event, None)

        self.assertEqual(response['statusCode'], 409)
        self.assertEqual(table.update_calls, [])

    def test_failed_reminder_does_not_mark_reminder_sent(self):
        subscriber = {
            'email': 'person@example.com',
            'status': 'pending',
            'verification_token': 'still-valid',
            'verification_requested_at': (
                email_sender.utc_now() - timedelta(days=3)
            ).isoformat(),
            'verification_expires_at': (
                email_sender.utc_now() + timedelta(days=1)
            ).isoformat(),
        }
        table = FakeTable(subscriber)

        with (
            patch.object(
                email_sender.boto3,
                'resource',
                return_value=FakeDynamoDB(table),
            ),
            patch.object(
                email_sender,
                'query_subscribers_by_status',
                return_value=[subscriber],
            ),
            patch.object(email_sender, 'send_verification_email', return_value=False),
            patch.object(email_sender, 'publish_verification_metrics') as metrics,
        ):
            response = email_sender.check_and_resend_verifications(
                {'api_url': 'https://api.example.com/prod'},
                None,
            )

        body = json.loads(response['body'])
        self.assertEqual(body['failure_count'], 1)
        self.assertEqual(table.update_calls, [])
        metrics.assert_called_once_with(0, 1, 1)

    def test_unsubscribe_get_only_shows_confirmation(self):
        table = FakeTable(
            {
                'email': 'person@example.com',
                'status': 'verified',
                'unsubscribe_token': 'unsubscribe-token',
            }
        )
        event = {
            'httpMethod': 'GET',
            'queryStringParameters': {
                'email': 'person@example.com',
                'token': 'unsubscribe-token',
            },
            'requestContext': {
                'domainName': 'api.example.com',
                'stage': 'prod',
            },
        }

        with patch.object(
            email_sender.boto3,
            'resource',
            return_value=FakeDynamoDB(table),
        ):
            response = email_sender.unsubscribe_handler(event, None)

        self.assertEqual(response['statusCode'], 200)
        self.assertIn('Confirm unsubscribe', response['body'])
        self.assertEqual(table.update_calls, [])

    def test_unsubscribe_post_changes_verified_record(self):
        table = FakeTable(
            {
                'email': 'person@example.com',
                'status': 'verified',
                'unsubscribe_token': 'unsubscribe-token',
            }
        )
        event = {
            'httpMethod': 'POST',
            'headers': {'content-type': 'application/x-www-form-urlencoded'},
            'body': 'email=person%40example.com&token=unsubscribe-token',
        }

        with patch.object(
            email_sender.boto3,
            'resource',
            return_value=FakeDynamoDB(table),
        ):
            response = email_sender.unsubscribe_handler(event, None)

        self.assertEqual(response['statusCode'], 200)
        self.assertEqual(len(table.update_calls), 1)
        self.assertIn('#status = :verified', table.update_calls[0]['ConditionExpression'])

    def test_html_newsletter_data_escapes_model_content_and_urls(self):
        data = email_sender.build_html_newsletter_data(
            {
                'overview': '<script>alert(1)</script>',
                'featured_papers': [
                    {
                        'title': '<b>Title</b>',
                        'summary': 'A & B',
                        'link': 'javascript:alert(1)',
                    }
                ],
            }
        )

        self.assertEqual(data['overview_html'], '&lt;script&gt;alert(1)&lt;/script&gt;')
        self.assertEqual(data['featured_papers_html'][0]['title'], '&lt;b&gt;Title&lt;/b&gt;')
        self.assertEqual(data['featured_papers_html'][0]['summary'], 'A &amp; B')
        self.assertEqual(data['featured_papers_html'][0]['link'], '')

    def test_email_test_mode_fails_closed_without_an_address(self):
        with patch.dict(
            os.environ,
            {'EMAIL_TEST_MODE': 'true', 'EMAIL_TEST_ADDRESS': ''},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, 'EMAIL_TEST_ADDRESS'):
                email_sender.test_prod_check(
                    [{'email': 'subscriber@example.com', 'unsubscribe_token': 'token'}],
                    '',
                )


if __name__ == '__main__':
    unittest.main()
