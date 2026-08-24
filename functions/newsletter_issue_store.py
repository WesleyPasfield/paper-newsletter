"""Durable issue state for idempotent newsletter generation and delivery."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

ISSUE_RECORD_ID = "ISSUE"
ISSUE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,80}$")
ISSUE_RETENTION_DAYS = 365


class IssueConflictError(RuntimeError):
    """Raised when an immutable issue ID is reused for different content."""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_issue_id(event: dict | None, test_mode: bool = False) -> str:
    """Build a stable weekly issue ID, with an isolated namespace for test sends."""
    event = event or {}
    requested_id = str(event.get("issue_id") or "").strip()
    if requested_id:
        if not ISSUE_ID_PATTERN.fullmatch(requested_id):
            raise ValueError(
                "issue_id must contain only letters, numbers, '.', '_' or '-'"
            )
        return requested_id

    event_time = event.get("time")
    if event_time:
        try:
            issue_date = datetime.fromisoformat(
                str(event_time).replace("Z", "+00:00")
            ).date()
        except ValueError as exc:
            raise ValueError("event time must be an ISO-8601 timestamp") from exc
    else:
        issue_date = utc_now().date()

    iso_year, iso_week, _ = issue_date.isocalendar()
    suffix = "-test" if test_mode else ""
    return f"newsletter-{iso_year}-W{iso_week:02d}{suffix}"


class NewsletterIssueStore:
    """Persist immutable issue content and coordinate resumable delivery state."""

    def __init__(self, table_name: str, bucket_name: str):
        self.table = boto3.resource("dynamodb").Table(table_name)
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name

    @staticmethod
    def _serialize(content: dict[str, Any]) -> bytes:
        return json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2).encode(
            "utf-8"
        )

    @staticmethod
    def _digest(body: bytes) -> str:
        return hashlib.sha256(body).hexdigest()

    def get(self, issue_id: str) -> dict[str, Any] | None:
        response = self.table.get_item(
            Key={"issue_id": issue_id, "recipient_id": ISSUE_RECORD_ID},
            ConsistentRead=True,
        )
        return response.get("Item")

    def load_content(self, record: dict[str, Any]) -> dict[str, Any]:
        response = self.s3.get_object(Bucket=self.bucket_name, Key=record["s3_key"])
        body = response["Body"].read()
        if self._digest(body) != record["content_sha256"]:
            raise IssueConflictError(
                f"Stored issue {record['issue_id']} failed its integrity check"
            )
        return json.loads(body.decode("utf-8"))

    def create(self, issue_id: str, content: dict[str, Any]) -> dict[str, Any]:
        """Store an immutable issue before any recipient delivery is attempted."""
        body = self._serialize(content)
        content_sha256 = self._digest(body)
        s3_key = f"newsletters/{issue_id}.json"
        now = utc_now()

        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=body,
                ContentType="application/json",
                IfNoneMatch="*",
            )
        except ClientError as exc:
            status_code = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            error_code = exc.response.get("Error", {}).get("Code")
            if status_code != 412 and error_code not in {
                "PreconditionFailed",
                "ConditionalRequestConflict",
            }:
                raise

            # A prior attempt may have archived the issue before DynamoDB was updated.
            # The first immutable S3 object remains authoritative on every replay.
            body = self.s3.get_object(Bucket=self.bucket_name, Key=s3_key)[
                "Body"
            ].read()
            content_sha256 = self._digest(body)

        item = {
            "issue_id": issue_id,
            "recipient_id": ISSUE_RECORD_ID,
            "status": "ready",
            "s3_key": s3_key,
            "content_sha256": content_sha256,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "expires_at": int((now + timedelta(days=ISSUE_RETENTION_DAYS)).timestamp()),
        }
        try:
            self.table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(issue_id) AND attribute_not_exists(recipient_id)",
            )
            return item
        except ClientError as exc:
            if (
                exc.response.get("Error", {}).get("Code")
                != "ConditionalCheckFailedException"
            ):
                raise
            existing = self.get(issue_id)
            if not existing:
                raise IssueConflictError(
                    f"Issue {issue_id} exists but its state could not be loaded"
                ) from exc
            return existing

    def claim_delivery(self, issue_id: str, allow_active_resume: bool = False) -> bool:
        """Claim a ready or resumable issue. Completed or active issues are skipped."""
        now = utc_now().isoformat()
        resumable_states = (
            "(:ready, :failed, :partial, :sending)"
            if allow_active_resume
            else "(:ready, :failed, :partial)"
        )
        try:
            self.table.update_item(
                Key={"issue_id": issue_id, "recipient_id": ISSUE_RECORD_ID},
                UpdateExpression="SET #status = :sending, updated_at = :now ADD delivery_attempts :one",
                ConditionExpression=f"#status IN {resumable_states}",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":sending": "sending",
                    ":ready": "ready",
                    ":failed": "failed",
                    ":partial": "partial",
                    ":now": now,
                    ":one": 1,
                },
            )
            return True
        except ClientError as exc:
            if (
                exc.response.get("Error", {}).get("Code")
                == "ConditionalCheckFailedException"
            ):
                return False
            raise

    def mark(
        self, issue_id: str, status: str, details: dict[str, Any] | None = None
    ) -> None:
        now = utc_now()
        values: dict[str, Any] = {
            ":status": status,
            ":now": now.isoformat(),
        }
        update_expression = "SET #status = :status, updated_at = :now"
        if status == "submitted":
            update_expression += ", submitted_at = :now"
        if details is not None:
            update_expression += ", delivery_results = :details"
            values[":details"] = details

        self.table.update_item(
            Key={"issue_id": issue_id, "recipient_id": ISSUE_RECORD_ID},
            UpdateExpression=update_expression,
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues=values,
        )
