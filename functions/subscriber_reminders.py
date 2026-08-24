"""Verification reminder request parsing, eligibility, and claim state."""

import json
import secrets
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from botocore.exceptions import ClientError


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_iso_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def parse_bulk_verification_options(event: dict) -> dict:
    """Parse both direct Lambda payloads and API Gateway JSON bodies."""
    options = dict(event or {})
    body = (event or {}).get("body")
    if body:
        parsed_body = json.loads(body)
        if not isinstance(parsed_body, dict):
            raise ValueError("Bulk verification request body must be a JSON object")
        options.update(parsed_body)
    return options


def reminder_was_accepted(subscriber: dict) -> bool:
    return bool(
        subscriber.get("verification_reminder_accepted_at")
        or subscriber.get("reminder_sent")
    )


def select_scheduled_candidates(
    subscribers: list[dict],
    now: datetime,
    max_batch: int,
) -> tuple[list[dict], int]:
    cutoff = now - timedelta(hours=48)
    oldest_eligible = now - timedelta(days=30)
    candidates = []
    invalid_timestamps = 0
    for subscriber in subscribers:
        if str(subscriber.get("deliverability_status", "")).lower() == "suppressed":
            continue
        if reminder_was_accepted(subscriber):
            continue
        requested_at = parse_iso_datetime(subscriber.get("verification_requested_at"))
        if not requested_at:
            invalid_timestamps += 1
            continue
        if oldest_eligible <= requested_at < cutoff:
            candidates.append(subscriber)
    candidates.sort(key=lambda item: item.get("verification_requested_at", ""))
    return candidates[:max_batch], invalid_timestamps


def select_bulk_candidates(
    subscribers: list[dict],
    now: datetime,
    force: bool,
    expired_tokens_only: bool,
    token_is_expired: Callable[[dict, datetime | None], bool],
) -> tuple[list[dict], dict]:
    cutoff = now - timedelta(days=30)
    candidates = []
    stats = {
        "excluded_invalid_timestamp": 0,
        "excluded_older_than_30_days": 0,
        "excluded_prior_reminder": 0,
        "excluded_suppressed": 0,
    }
    for subscriber in subscribers:
        if str(subscriber.get("deliverability_status", "")).lower() == "suppressed":
            stats["excluded_suppressed"] += 1
            continue
        requested_at = parse_iso_datetime(subscriber.get("verification_requested_at"))
        if not requested_at:
            stats["excluded_invalid_timestamp"] += 1
            continue
        if requested_at < cutoff:
            stats["excluded_older_than_30_days"] += 1
            continue
        if reminder_was_accepted(subscriber) and not force:
            stats["excluded_prior_reminder"] += 1
            continue
        if (
            expired_tokens_only
            and subscriber.get("verification_token")
            and not token_is_expired(subscriber, now)
        ):
            continue
        candidates.append(subscriber)
    candidates.sort(key=lambda item: item.get("verification_requested_at", ""))
    return candidates, stats


def claim_verification_reminder(
    table,
    subscriber: dict,
    campaign_id: str,
    force: bool = False,
) -> str | None:
    """Atomically claim one reminder attempt for a campaign."""
    claim_id = secrets.token_urlsafe(18)
    condition = (
        "#status = :pending AND "
        "(attribute_not_exists(deliverability_status) OR deliverability_status <> :suppressed) AND "
        "(attribute_not_exists(verification_reminder_claimed_campaign_id) OR "
        "verification_reminder_claimed_campaign_id <> :campaign_id) AND "
        "(attribute_not_exists(verification_reminder_campaign_id) OR "
        "verification_reminder_campaign_id <> :campaign_id)"
    )
    if not force:
        condition += (
            " AND attribute_not_exists(verification_reminder_accepted_at) "
            "AND attribute_not_exists(reminder_sent)"
        )
    try:
        table.update_item(
            Key={"email": subscriber["email"]},
            UpdateExpression="SET verification_reminder_claimed_at = :now, "
            "verification_reminder_claim_id = :claim_id, "
            "verification_reminder_claimed_campaign_id = :campaign_id, "
            "verification_reminder_delivery_status = :attempting",
            ConditionExpression=condition,
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":pending": "pending",
                ":suppressed": "suppressed",
                ":campaign_id": campaign_id,
                ":claim_id": claim_id,
                ":now": utc_now().isoformat(),
                ":attempting": "attempting",
            },
        )
        return claim_id
    except ClientError as error:
        if (
            error.response.get("Error", {}).get("Code")
            == "ConditionalCheckFailedException"
        ):
            return None
        raise


def record_verification_reminder_outcome(
    table,
    email: str,
    campaign_id: str,
    claim_id: str,
    accepted: bool,
) -> None:
    now = utc_now().isoformat()
    if accepted:
        table.update_item(
            Key={"email": email},
            UpdateExpression="SET reminder_sent = :now, verification_reminder_accepted_at = :now, "
            "verification_reminder_campaign_id = :campaign_id, "
            "verification_reminder_delivery_status = :accepted, "
            "verification_reminder_count = if_not_exists(verification_reminder_count, :zero) + :one",
            ConditionExpression="#status = :pending AND verification_reminder_claim_id = :claim_id",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":pending": "pending",
                ":claim_id": claim_id,
                ":campaign_id": campaign_id,
                ":now": now,
                ":accepted": "accepted",
                ":zero": 0,
                ":one": 1,
            },
        )
        return

    table.update_item(
        Key={"email": email},
        UpdateExpression="SET verification_reminder_delivery_status = :failed, "
        "verification_reminder_failed_at = :now",
        ConditionExpression="#status = :pending AND verification_reminder_claim_id = :claim_id",
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues={
            ":pending": "pending",
            ":claim_id": claim_id,
            ":failed": "failed",
            ":now": now,
        },
    )
