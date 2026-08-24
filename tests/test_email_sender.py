import json
import os
import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from functions import delivery_state, email_sender, subscriber_reminders


def conditional_error(operation="UpdateItem"):
    return ClientError(
        {
            "Error": {
                "Code": "ConditionalCheckFailedException",
                "Message": "condition failed",
            }
        },
        operation,
    )


class FakeTable:
    def __init__(self, item=None):
        self.item = item
        self.update_calls = []
        self.put_calls = []

    def get_item(self, **kwargs):
        return {"Item": self.item} if self.item else {}

    def update_item(self, **kwargs):
        self.update_calls.append(kwargs)
        return {"Attributes": self.item or {}}

    def put_item(self, **kwargs):
        self.put_calls.append(kwargs)
        self.item = kwargs["Item"]
        return {}


class FakeDynamoDB:
    def __init__(self, tables):
        self.tables = tables if isinstance(tables, dict) else {"subscribers": tables}

    def Table(self, table_name):
        return self.tables[table_name]


class DeliveryLedgerTable:
    def __init__(self):
        self.records = {}
        self.update_calls = []

    def put_item(self, **kwargs):
        item = kwargs["Item"]
        key = (item["issue_id"], item["recipient_id"])
        if key in self.records:
            raise conditional_error("PutItem")
        self.records[key] = dict(item)

    def update_item(self, **kwargs):
        self.update_calls.append(kwargs)
        key_data = kwargs["Key"]
        key = (key_data["issue_id"], key_data["recipient_id"])
        record = self.records.setdefault(key, dict(key_data))
        values = kwargs.get("ExpressionAttributeValues", {})
        if ":status" in values:
            record["status"] = values[":status"]
        if ":message_id" in values:
            record["ses_message_id"] = values[":message_id"]
        return {}


class ReminderClaimTable(FakeTable):
    def __init__(self, item):
        super().__init__(item)
        self.claimed_campaigns = set()

    def update_item(self, **kwargs):
        values = kwargs.get("ExpressionAttributeValues", {})
        campaign_id = values.get(":campaign_id")
        if "verification_reminder_claimed_campaign_id" in kwargs.get(
            "UpdateExpression", ""
        ):
            if campaign_id in self.claimed_campaigns:
                raise conditional_error()
            self.claimed_campaigns.add(campaign_id)
        return super().update_item(**kwargs)


class EmailSenderTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "SUBSCRIBERS_TABLE": "subscribers",
                "NEWSLETTER_RUNS_TABLE": "deliveries",
                "SENDER_EMAIL": "newsletter@example.com",
                "SUBSCRIBE_LANDING_PAGE": "https://example.com/newsletter",
                "EMAIL_CONFIGURATION_SET": "newsletter-events",
                "NEWSLETTER_TEMPLATE_NAME": "newsletter-template",
                "EMAIL_TEST_MODE": "false",
                "BULK_VERIFICATION_MAX_BATCH": "100",
            },
            clear=False,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @staticmethod
    def newsletter_content():
        return {
            "overview": "Overview",
            "featured_papers": [],
            "additional_papers": [],
            "metadata": {
                "generated_date": "2026-08-23",
                "featured_papers_count": 0,
            },
        }

    @staticmethod
    def api_event(method="POST", body=None):
        event = {
            "httpMethod": method,
            "headers": {"content-type": "application/json"},
            "requestContext": {"domainName": "api.example.com", "stage": "prod"},
        }
        if body is not None:
            event["body"] = json.dumps(body)
        return event

    def test_verification_url_encodes_email_and_token(self):
        url = email_sender.build_verification_url(
            "https://api.example.com/prod",
            "person+tag@example.com",
            "a token&value",
        )
        self.assertEqual(
            url,
            "https://api.example.com/prod/verify?email=person%2Btag%40example.com&token=a+token%26value",
        )

    def test_verification_reminder_is_tagged_for_delivery_tracking(self):
        ses = MagicMock()
        with patch.object(email_sender.boto3, "client", return_value=ses):
            accepted = email_sender.send_verification_email(
                "person@example.com",
                "token",
                "newsletter@example.com",
                "https://api.example.com/prod",
                email_type="verification_reminder",
                campaign_id="pending-2026-08",
            )

        self.assertTrue(accepted)
        tags = ses.send_templated_email.call_args.kwargs["Tags"]
        self.assertIn({"Name": "EmailType", "Value": "verification_reminder"}, tags)
        self.assertIn({"Name": "CampaignId", "Value": "pending-2026-08"}, tags)
        self.assertIn(
            {
                "Name": "SubscriberId",
                "Value": email_sender.subscriber_id("person@example.com"),
            },
            tags,
        )

    def test_expired_token_is_rejected(self):
        subscriber = {
            "verification_expires_at": (
                email_sender.utc_now() - timedelta(seconds=1)
            ).isoformat(),
        }
        self.assertTrue(email_sender.token_is_expired(subscriber))

    def test_expired_token_is_rotated_before_a_reminder(self):
        table = FakeTable(
            {
                "email": "person@example.com",
                "status": "pending",
                "verification_token": "expired-token",
                "verification_expires_at": (
                    email_sender.utc_now() - timedelta(seconds=1)
                ).isoformat(),
            }
        )
        with patch.object(
            email_sender.secrets, "token_urlsafe", return_value="replacement-token"
        ):
            token = email_sender.ensure_active_verification_token(table, table.item)
        self.assertEqual(token, "replacement-token")
        self.assertIn(
            "verification_token = :old_token",
            table.update_calls[0]["ConditionExpression"],
        )
        self.assertIsInstance(
            table.update_calls[0]["ExpressionAttributeValues"][":pending_expires_at"],
            int,
        )

    def test_verification_get_renders_confirmation_without_state_change(self):
        table = FakeTable(
            {
                "email": "person@example.com",
                "status": "pending",
                "verification_token": "valid-token",
                "verification_expires_at": (
                    email_sender.utc_now() + timedelta(days=1)
                ).isoformat(),
            }
        )
        event = {
            "httpMethod": "GET",
            "queryStringParameters": {
                "email": "person@example.com",
                "token": "valid-token",
            },
            "requestContext": {"domainName": "api.example.com", "stage": "prod"},
        }
        with patch.object(
            email_sender.boto3, "resource", return_value=FakeDynamoDB(table)
        ):
            response = email_sender.verify_subscriber(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertIn("Confirm subscription", response["body"])
        self.assertEqual(table.update_calls, [])

    def test_verification_post_performs_conditional_state_change(self):
        table = FakeTable(
            {
                "email": "person@example.com",
                "status": "pending",
                "verification_token": "valid-token",
                "verification_expires_at": (
                    email_sender.utc_now() + timedelta(days=1)
                ).isoformat(),
                "subscription_history": [],
            }
        )
        event = {
            "httpMethod": "POST",
            "headers": {"content-type": "application/x-www-form-urlencoded"},
            "body": "email=person%40example.com&token=valid-token",
        }
        with patch.object(
            email_sender.boto3, "resource", return_value=FakeDynamoDB(table)
        ):
            response = email_sender.verify_subscriber(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(len(table.update_calls), 1)
        update = table.update_calls[0]
        self.assertIn("#status = :pending", update["ConditionExpression"])
        self.assertIn("REMOVE pending_expires_at", update["UpdateExpression"])

    def test_unsubscribed_record_cannot_be_reverified(self):
        table = FakeTable(
            {
                "email": "person@example.com",
                "status": "unsubscribed",
                "verification_token": "old-token",
            }
        )
        event = {
            "httpMethod": "GET",
            "queryStringParameters": {
                "email": "person@example.com",
                "token": "old-token",
            },
        }
        with patch.object(
            email_sender.boto3, "resource", return_value=FakeDynamoDB(table)
        ):
            response = email_sender.verify_subscriber(event, None)
        self.assertEqual(response["statusCode"], 409)
        self.assertEqual(table.update_calls, [])

    def test_subscribe_cooldown_reuses_token_without_send_claim(self):
        token = "valid-token"
        table = FakeTable(
            {
                "email": "person@example.com",
                "status": "pending",
                "verification_token": token,
                "verification_expires_at": (
                    email_sender.utc_now() + timedelta(days=1)
                ).isoformat(),
                "verification_last_attempted_at": email_sender.utc_now().isoformat(),
            }
        )
        should_send, returned_token = email_sender._claim_subscription_verification(
            table,
            "person@example.com",
            email_sender.utc_now(),
        )
        self.assertFalse(should_send)
        self.assertEqual(returned_token, token)
        self.assertEqual(table.update_calls, [])

    def test_new_subscribe_sets_bounded_history_and_pending_ttl(self):
        table = FakeTable()
        with patch.dict(os.environ, {"MAX_SUBSCRIPTION_HISTORY": "2"}, clear=False):
            should_send, token = email_sender._claim_subscription_verification(
                table,
                "person@example.com",
                email_sender.utc_now(),
            )
        self.assertTrue(should_send)
        self.assertTrue(token)
        item = table.put_calls[0]["Item"]
        self.assertIsInstance(item["pending_expires_at"], int)
        self.assertLessEqual(len(item["subscription_history"]), 2)
        self.assertEqual(
            table.put_calls[0]["ConditionExpression"], "attribute_not_exists(email)"
        )

    def test_public_subscribe_response_is_neutral_for_existing_states(self):
        event = self.api_event(body={"email": "person@example.com"})
        responses = []
        for item in (
            {"email": "person@example.com", "status": "verified"},
            {
                "email": "person@example.com",
                "status": "pending",
                "verification_token": "token",
                "verification_expires_at": (
                    email_sender.utc_now() + timedelta(days=1)
                ).isoformat(),
                "verification_last_attempted_at": email_sender.utc_now().isoformat(),
            },
        ):
            with (
                patch.object(
                    email_sender.boto3,
                    "resource",
                    return_value=FakeDynamoDB(FakeTable(item)),
                ),
                patch.object(email_sender, "send_verification_email") as send_email,
            ):
                responses.append(email_sender.add_subscriber(event, None))
                send_email.assert_not_called()
        self.assertEqual(responses[0]["statusCode"], 200)
        self.assertEqual(responses[0]["body"], responses[1]["body"])
        self.assertNotIn("person@example.com", responses[0]["body"])

    def test_bulk_verification_is_dry_run_by_default(self):
        subscriber = {
            "email": "person@example.com",
            "status": "pending",
            "verification_token": "token",
            "verification_requested_at": email_sender.utc_now().isoformat(),
            "verification_expires_at": (
                email_sender.utc_now() + timedelta(days=1)
            ).isoformat(),
        }
        with (
            patch.object(
                email_sender.boto3, "resource", return_value=FakeDynamoDB(FakeTable())
            ),
            patch.object(
                email_sender, "query_subscribers_by_status", return_value=[subscriber]
            ),
            patch.object(
                email_sender, "send_claimed_verification_reminder"
            ) as send_reminder,
        ):
            response = email_sender.bulk_verifier(self.api_event(body={}), None)
        body = json.loads(response["body"])
        self.assertTrue(body["dry_run"])
        self.assertEqual(body["candidate_count"], 1)
        send_reminder.assert_not_called()

    def test_bulk_verification_requires_campaign_and_parses_json_confirmation(self):
        response = email_sender.bulk_verifier(
            self.api_event(body={"confirm_send": True}),
            None,
        )
        self.assertEqual(response["statusCode"], 400)
        self.assertIn("campaign_id", response["body"])

    def test_bulk_verification_confirmed_campaign_sends_bounded_candidates(self):
        subscriber = {
            "email": "person@example.com",
            "status": "pending",
            "verification_token": "token",
            "verification_requested_at": email_sender.utc_now().isoformat(),
            "verification_expires_at": (
                email_sender.utc_now() + timedelta(days=1)
            ).isoformat(),
        }
        event = self.api_event(
            body={
                "confirm_send": True,
                "campaign_id": "campaign-1",
                "max_batch_size": 1,
            }
        )
        with (
            patch.object(
                email_sender.boto3, "resource", return_value=FakeDynamoDB(FakeTable())
            ),
            patch.object(
                email_sender, "query_subscribers_by_status", return_value=[subscriber]
            ),
            patch.object(
                email_sender,
                "send_claimed_verification_reminder",
                return_value="accepted",
            ) as send_reminder,
        ):
            response = email_sender.bulk_verifier(event, None)
        body = json.loads(response["body"])
        self.assertFalse(body["dry_run"])
        self.assertEqual(body["accepted_count"], 1)
        self.assertEqual(send_reminder.call_args.args[2], "campaign-1")

    def test_reminder_campaign_claim_prevents_replay(self):
        subscriber = {"email": "person@example.com", "status": "pending"}
        table = ReminderClaimTable(subscriber)
        first = email_sender.claim_verification_reminder(
            table, subscriber, "campaign-1"
        )
        second = email_sender.claim_verification_reminder(
            table, subscriber, "campaign-1"
        )
        self.assertTrue(first)
        self.assertIsNone(second)

    def test_scheduled_and_bulk_reminders_share_acceptance_state(self):
        subscriber = {
            "email": "person@example.com",
            "status": "pending",
            "verification_requested_at": (
                email_sender.utc_now() - timedelta(days=3)
            ).isoformat(),
            "verification_reminder_accepted_at": email_sender.utc_now().isoformat(),
        }
        self.assertTrue(subscriber_reminders.reminder_was_accepted(subscriber))
        with (
            patch.object(
                email_sender.boto3, "resource", return_value=FakeDynamoDB(FakeTable())
            ),
            patch.object(
                email_sender, "query_subscribers_by_status", return_value=[subscriber]
            ),
            patch.object(
                email_sender, "send_claimed_verification_reminder"
            ) as send_reminder,
            patch.object(email_sender, "publish_verification_metrics"),
        ):
            bulk = email_sender.bulk_verifier(self.api_event(body={}), None)
            scheduled = email_sender.check_and_resend_verifications(
                {"api_url": "https://api.example.com/prod"},
                None,
            )
        self.assertEqual(json.loads(bulk["body"])["candidate_count"], 0)
        self.assertEqual(json.loads(scheduled["body"])["batch_size"], 0)
        send_reminder.assert_not_called()

    def test_scheduled_reminders_do_not_reactivate_records_older_than_30_days(self):
        subscriber = {
            "email": "old-pending@example.com",
            "status": "pending",
            "verification_requested_at": (
                email_sender.utc_now() - timedelta(days=31)
            ).isoformat(),
        }

        candidates, invalid_count = email_sender.select_scheduled_candidates(
            [subscriber],
            email_sender.utc_now(),
            100,
        )

        self.assertEqual(candidates, [])
        self.assertEqual(invalid_count, 0)

    def test_bulk_invalid_timestamp_is_excluded(self):
        subscriber = {
            "email": "person@example.com",
            "status": "pending",
            "verification_requested_at": "bad",
        }
        with (
            patch.object(
                email_sender.boto3, "resource", return_value=FakeDynamoDB(FakeTable())
            ),
            patch.object(
                email_sender, "query_subscribers_by_status", return_value=[subscriber]
            ),
        ):
            response = email_sender.bulk_verifier(self.api_event(body={}), None)
        body = json.loads(response["body"])
        self.assertEqual(body["candidate_count"], 0)
        self.assertEqual(body["excluded_invalid_timestamp"], 1)

    def test_newsletter_delivery_claim_skips_replay(self):
        ses = MagicMock()
        ses.send_bulk_templated_email.return_value = {
            "Status": [{"Status": "Success", "MessageId": "message-1"}],
        }
        ledger = DeliveryLedgerTable()
        subscribers = [
            {"email": "person@example.com", "unsubscribe_token": "unsubscribe"}
        ]
        arguments = {
            "ses_client": ses,
            "source": "newsletter@example.com",
            "subscribers": subscribers,
            "json_content": self.newsletter_content(),
            "unsubscribe_url": "https://api.example.com/prod/unsubscribe",
            "subscribe_url": "https://example.com/newsletter",
            "issue_id": "issue-1",
            "delivery_table": ledger,
        }
        first = email_sender.send_email_batch(**arguments)
        second = email_sender.send_email_batch(**arguments)
        self.assertEqual(first["accepted"], 1)
        self.assertEqual(second["skipped"], 1)
        ses.send_bulk_templated_email.assert_called_once()
        tags = ses.send_bulk_templated_email.call_args.kwargs["Destinations"][0][
            "ReplacementTags"
        ]
        self.assertIn({"Name": "IssueId", "Value": "issue-1"}, tags)
        self.assertIn(
            {
                "Name": "SubscriberId",
                "Value": email_sender.subscriber_id("person@example.com"),
            },
            tags,
        )

    def test_newsletter_partial_recipient_failures_are_counted(self):
        ses = MagicMock()
        ses.send_bulk_templated_email.return_value = {
            "Status": [
                {"Status": "Success", "MessageId": "message-1"},
                {"Status": "MessageRejected", "Error": "bad recipient"},
            ],
        }
        ledger = DeliveryLedgerTable()
        subscribers = [
            {"email": "one@example.com", "unsubscribe_token": "one"},
            {"email": "two@example.com", "unsubscribe_token": "two"},
        ]
        result = email_sender.send_email_batch(
            ses,
            "newsletter@example.com",
            subscribers,
            self.newsletter_content(),
            "https://api.example.com/prod/unsubscribe",
            "https://example.com/newsletter",
            issue_id="issue-2",
            delivery_table=ledger,
        )
        self.assertEqual(result["accepted"], 1)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(len(result["failures"]), 1)

    def test_ambiguous_batch_attempt_is_not_retried(self):
        ses = MagicMock()
        ses.send_bulk_templated_email.side_effect = RuntimeError("connection lost")
        ledger = DeliveryLedgerTable()
        subscribers = [
            {"email": "person@example.com", "unsubscribe_token": "unsubscribe"}
        ]
        arguments = (
            ses,
            "newsletter@example.com",
            subscribers,
            self.newsletter_content(),
            "https://api.example.com/prod/unsubscribe",
            "https://example.com/newsletter",
        )
        first = email_sender.send_email_batch(
            *arguments, issue_id="issue-3", delivery_table=ledger
        )
        second = email_sender.send_email_batch(
            *arguments, issue_id="issue-3", delivery_table=ledger
        )
        self.assertEqual(first["failed"], 1)
        self.assertEqual(second["skipped"], 1)
        ses.send_bulk_templated_email.assert_called_once()

    def test_ses_complaint_and_permanent_bounce_suppress_subscriber(self):
        for event_type, details, reason in (
            ("Complaint", {"complaint": {}}, "complaint"),
            ("Bounce", {"bounce": {"bounceType": "Permanent"}}, "permanent_bounce"),
        ):
            delivery_table = DeliveryLedgerTable()
            subscriber_table = FakeTable({"email": "person@example.com"})
            payload = {
                "notificationType": event_type,
                "mail": {
                    "messageId": "message-1",
                    "destination": ["person@example.com"],
                    "tags": {"IssueId": ["issue-1"], "SubscriberId": ["recipient-1"]},
                },
                **details,
            }
            event = {"Records": [{"Sns": {"Message": json.dumps(payload)}}]}
            with patch.object(
                email_sender.boto3,
                "resource",
                return_value=FakeDynamoDB(
                    {
                        "deliveries": delivery_table,
                        "subscribers": subscriber_table,
                    }
                ),
            ):
                response = email_sender.ses_event_handler(event, None)
            self.assertEqual(response["statusCode"], 200)
            suppression = subscriber_table.update_calls[-1]["ExpressionAttributeValues"]
            self.assertEqual(suppression[":suppressed"], "suppressed")
            self.assertEqual(suppression[":reason"], reason)

    def test_ses_event_types_map_to_delivery_states(self):
        expected = {
            "Send": "accepted",
            "Delivery": "delivered",
            "Bounce": "bounced",
            "Complaint": "complained",
            "Reject": "rejected",
            "Rendering Failure": "rendering_failed",
        }
        for event_type, status in expected.items():
            self.assertEqual(
                delivery_state.ses_event_status({"eventType": event_type}),
                status,
            )

    def test_ses_delivery_updates_verification_reminder_state(self):
        subscriber_table = FakeTable(
            {"email": "person@example.com", "status": "pending"}
        )
        event = {
            "eventType": "Delivery",
            "mail": {
                "messageId": "message-1",
                "destination": ["person@example.com"],
                "tags": {"EmailType": ["verification_reminder"]},
            },
        }

        result = delivery_state.process_ses_event(
            event,
            DeliveryLedgerTable(),
            subscriber_table,
        )

        self.assertEqual(result["processed"], 1)
        update = subscriber_table.update_calls[0]
        self.assertEqual(
            update["ExpressionAttributeNames"]["#delivery_status"],
            "verification_reminder_delivery_status",
        )
        self.assertEqual(update["ExpressionAttributeValues"][":status"], "delivered")

    def test_log_redaction_replaces_addresses_with_stable_identifier(self):
        first = email_sender.redact_log_text("send failed for Person@example.com")
        second = email_sender.redact_log_text("retry Person@example.com")
        self.assertNotIn("Person@example.com", first)
        self.assertIn(email_sender.subscriber_log_id("Person@example.com"), first)
        self.assertIn(email_sender.subscriber_log_id("Person@example.com"), second)

    def test_get_subscribers_excludes_suppressed(self):
        table = FakeTable()
        subscribers = [
            {
                "email": "active@example.com",
                "status": "verified",
                "unsubscribe_token": "a",
            },
            {
                "email": "suppressed@example.com",
                "status": "verified",
                "unsubscribe_token": "s",
                "deliverability_status": "suppressed",
            },
        ]
        with (
            patch.object(
                email_sender.boto3, "resource", return_value=FakeDynamoDB(table)
            ),
            patch.object(
                email_sender, "query_subscribers_by_status", return_value=subscribers
            ),
        ):
            result = email_sender.get_subscribers("subscribers")
        self.assertEqual([item["email"] for item in result], ["active@example.com"])

    def test_test_mode_can_send_with_empty_verified_list_and_does_not_mutate_template(
        self,
    ):
        delivery_table = DeliveryLedgerTable()
        with (
            patch.dict(
                os.environ,
                {"EMAIL_TEST_MODE": "true", "EMAIL_TEST_ADDRESS": "test@example.com"},
                clear=False,
            ),
            patch.object(email_sender, "get_subscribers", return_value=[]),
            patch.object(
                email_sender,
                "send_email_batch",
                return_value={
                    "accepted": 1,
                    "successful": 1,
                    "failed": 0,
                    "skipped": 0,
                    "failures": [],
                },
            ) as send_batch,
            patch.object(email_sender.boto3, "client", return_value=MagicMock()),
            patch.object(
                email_sender.boto3,
                "resource",
                return_value=FakeDynamoDB({"deliveries": delivery_table}),
            ),
        ):
            result = email_sender.send_newsletter(
                self.newsletter_content(),
                "subscribers",
                "newsletter@example.com",
                "https://api.example.com/prod",
                issue_id="test-issue",
            )
        self.assertEqual(result["accepted"], 1)
        recipients = send_batch.call_args.kwargs["subscribers"]
        self.assertEqual(recipients[0]["email"], "test@example.com")
        self.assertFalse(hasattr(email_sender, "create_or_update_template"))

    def test_unsubscribe_get_only_shows_confirmation(self):
        table = FakeTable(
            {
                "email": "person@example.com",
                "status": "verified",
                "unsubscribe_token": "unsubscribe-token",
            }
        )
        event = {
            "httpMethod": "GET",
            "queryStringParameters": {
                "email": "person@example.com",
                "token": "unsubscribe-token",
            },
            "requestContext": {"domainName": "api.example.com", "stage": "prod"},
        }
        with patch.object(
            email_sender.boto3, "resource", return_value=FakeDynamoDB(table)
        ):
            response = email_sender.unsubscribe_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertIn("Confirm unsubscribe", response["body"])
        self.assertEqual(table.update_calls, [])

    def test_unsubscribe_post_changes_verified_record(self):
        table = FakeTable(
            {
                "email": "person@example.com",
                "status": "verified",
                "unsubscribe_token": "unsubscribe-token",
                "subscription_history": [],
            }
        )
        event = {
            "httpMethod": "POST",
            "headers": {"content-type": "application/x-www-form-urlencoded"},
            "body": "email=person%40example.com&token=unsubscribe-token",
        }
        with patch.object(
            email_sender.boto3, "resource", return_value=FakeDynamoDB(table)
        ):
            response = email_sender.unsubscribe_handler(event, None)
        self.assertEqual(response["statusCode"], 200)
        self.assertIn(
            "#status = :verified", table.update_calls[0]["ConditionExpression"]
        )

    def test_html_newsletter_data_escapes_model_content_and_urls(self):
        data = email_sender.build_html_newsletter_data(
            {
                "overview": "<script>alert(1)</script>",
                "featured_papers": [
                    {
                        "title": "<b>Title</b>",
                        "summary": "A & B",
                        "link": "javascript:alert(1)",
                    }
                ],
            }
        )
        self.assertEqual(data["overview_html"], "&lt;script&gt;alert(1)&lt;/script&gt;")
        self.assertEqual(
            data["featured_papers_html"][0]["title"], "&lt;b&gt;Title&lt;/b&gt;"
        )
        self.assertEqual(data["featured_papers_html"][0]["summary"], "A &amp; B")
        self.assertEqual(data["featured_papers_html"][0]["link"], "")

    def test_email_test_mode_fails_closed_without_an_address(self):
        with (
            patch.dict(
                os.environ,
                {"EMAIL_TEST_MODE": "true", "EMAIL_TEST_ADDRESS": ""},
                clear=False,
            ),
            self.assertRaisesRegex(ValueError, "EMAIL_TEST_ADDRESS"),
        ):
            email_sender.test_prod_check(
                [{"email": "subscriber@example.com", "unsubscribe_token": "token"}],
                "",
            )


if __name__ == "__main__":
    unittest.main()
