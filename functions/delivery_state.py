"""Newsletter delivery ledger and SES feedback state transitions."""

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone

from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def subscriber_id(email: str) -> str:
    """Return a stable non-reversible subscriber identifier."""
    return hashlib.sha256(normalize_email(email).encode("utf-8")).hexdigest()


def subscriber_log_id(email: str) -> str:
    return subscriber_id(email)[:16]


def delivery_expiry_epoch(now: datetime | None = None) -> int:
    return int(((now or utc_now()) + timedelta(days=180)).timestamp())


def safe_ses_tag_value(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(value))[:256]


def claim_newsletter_delivery(
    delivery_table,
    issue_id: str,
    recipient_id: str,
    retry_pre_ses_access_denied: bool = False,
) -> bool:
    if delivery_table is None:
        return True
    now = utc_now()
    try:
        delivery_table.put_item(
            Item={
                "issue_id": issue_id,
                "recipient_id": recipient_id,
                "status": "attempted",
                "attempted_at": now.isoformat(),
                "updated_at": now.isoformat(),
                "expires_at": delivery_expiry_epoch(now),
            },
            ConditionExpression="attribute_not_exists(issue_id) AND attribute_not_exists(recipient_id)",
        )
        return True
    except ClientError as error:
        if (
            error.response.get("Error", {}).get("Code")
            != "ConditionalCheckFailedException"
        ):
            raise
        if not retry_pre_ses_access_denied:
            return False

    # A caller may explicitly recover one known pre-SES IAM denial. The stored
    # error proves SES did not accept the request, and the recovery marker keeps
    # repeated invocations from claiming the same recipient again.
    now = utc_now()
    try:
        delivery_table.update_item(
            Key={"issue_id": issue_id, "recipient_id": recipient_id},
            UpdateExpression=(
                "SET #status = :attempted, attempted_at = :now, "
                "updated_at = :now, recovery_attempted_at = :now, "
                "expires_at = :expires_at"
            ),
            ConditionExpression=(
                "#status = :ambiguous AND contains(delivery_error, :access_denied) "
                "AND attribute_not_exists(ses_message_id) "
                "AND attribute_not_exists(recovery_attempted_at)"
            ),
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":attempted": "attempted",
                ":ambiguous": "ambiguous",
                ":access_denied": "AccessDenied",
                ":now": now.isoformat(),
                ":expires_at": delivery_expiry_epoch(now),
            },
        )
        return True
    except ClientError as error:
        if (
            error.response.get("Error", {}).get("Code")
            == "ConditionalCheckFailedException"
        ):
            return False
        raise


def update_newsletter_delivery(
    delivery_table,
    issue_id: str,
    recipient_id: str,
    status: str,
    message_id: str | None = None,
    error: str | None = None,
) -> None:
    if delivery_table is None:
        return
    values = {
        ":status": status,
        ":updated_at": utc_now().isoformat(),
        ":expires_at": delivery_expiry_epoch(),
    }
    expression = (
        "SET #status = :status, updated_at = :updated_at, expires_at = :expires_at"
    )
    if message_id:
        expression += ", ses_message_id = :message_id"
        values[":message_id"] = message_id
    if error:
        expression += ", delivery_error = :error"
        values[":error"] = str(error)[:1000]
    delivery_table.update_item(
        Key={"issue_id": issue_id, "recipient_id": recipient_id},
        UpdateExpression=expression,
        ExpressionAttributeNames={"#status": "status"},
        ExpressionAttributeValues=values,
    )


def ses_payloads(event: dict) -> list[dict]:
    records = (event or {}).get("Records")
    if not records:
        return [event or {}]
    payloads = []
    for record in records:
        message = (record.get("Sns") or {}).get("Message")
        if message is None:
            payloads.append(record)
            continue
        try:
            payloads.append(
                json.loads(message) if isinstance(message, str) else message
            )
        except json.JSONDecodeError:
            logger.warning("Skipping malformed SNS SES event payload")
    return payloads


def ses_tag(tags: dict, *names: str) -> str | None:
    for name in names:
        value = (tags or {}).get(name)
        if isinstance(value, list):
            value = value[0] if value else None
        if value:
            return str(value)
    return None


def ses_event_status(payload: dict) -> str | None:
    event_type = (
        str(payload.get("eventType") or payload.get("notificationType") or "")
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )
    return {
        "send": "accepted",
        "delivery": "delivered",
        "bounce": "bounced",
        "complaint": "complained",
        "reject": "rejected",
        "renderingfailure": "rendering_failed",
    }.get(event_type)


def update_delivery_from_ses_event(
    delivery_table,
    issue_id: str,
    recipient_id: str,
    status: str,
    message_id: str | None,
) -> None:
    values = {
        ":status": status,
        ":updated_at": utc_now().isoformat(),
        ":expires_at": delivery_expiry_epoch(),
    }
    expression = (
        "SET #status = :status, updated_at = :updated_at, expires_at = :expires_at"
    )
    if message_id:
        expression += ", ses_message_id = :message_id"
        values[":message_id"] = message_id

    condition = None
    if status == "accepted":
        condition = "attribute_not_exists(#status) OR #status IN (:attempted, :ambiguous, :accepted)"
        values.update(
            {
                ":attempted": "attempted",
                ":ambiguous": "ambiguous",
                ":accepted": "accepted",
            }
        )
    elif status == "delivered":
        condition = (
            "attribute_not_exists(#status) OR "
            "#status IN (:attempted, :ambiguous, :accepted, :delivered)"
        )
        values.update(
            {
                ":attempted": "attempted",
                ":ambiguous": "ambiguous",
                ":accepted": "accepted",
                ":delivered": "delivered",
            }
        )

    request = {
        "Key": {"issue_id": issue_id, "recipient_id": recipient_id},
        "UpdateExpression": expression,
        "ExpressionAttributeNames": {"#status": "status"},
        "ExpressionAttributeValues": values,
    }
    if condition:
        request["ConditionExpression"] = condition
    try:
        delivery_table.update_item(**request)
    except ClientError as error:
        if (
            error.response.get("Error", {}).get("Code")
            != "ConditionalCheckFailedException"
        ):
            raise


def suppress_subscriber(table, email: str, reason: str) -> None:
    try:
        table.update_item(
            Key={"email": normalize_email(email)},
            UpdateExpression="SET deliverability_status = :suppressed, suppression_reason = :reason, "
            "suppressed_at = :now, updated_at = :now",
            ConditionExpression="attribute_exists(email)",
            ExpressionAttributeValues={
                ":suppressed": "suppressed",
                ":reason": reason,
                ":now": utc_now().isoformat(),
            },
        )
    except ClientError as error:
        if (
            error.response.get("Error", {}).get("Code")
            != "ConditionalCheckFailedException"
        ):
            raise


def record_verification_delivery_event(
    table,
    email: str,
    email_type: str,
    status: str,
) -> None:
    """Persist provider delivery state for initial verification and reminders."""
    prefix = (
        "verification_reminder"
        if email_type == "verification_reminder"
        else "verification"
    )
    status_field = f"{prefix}_delivery_status"
    timestamp_field = f"{prefix}_{status}_at"
    now = utc_now().isoformat()
    values = {
        ":status": status,
        ":now": now,
    }
    condition = "attribute_exists(email)"
    if status == "accepted":
        condition += (
            " AND (attribute_not_exists(#delivery_status) OR "
            "#delivery_status IN (:attempting, :failed, :accepted))"
        )
        values.update(
            {
                ":attempting": "attempting",
                ":failed": "failed",
                ":accepted": "accepted",
            }
        )
    elif status == "delivered":
        condition += (
            " AND (attribute_not_exists(#delivery_status) OR "
            "#delivery_status IN (:attempting, :failed, :accepted, :delivered))"
        )
        values.update(
            {
                ":attempting": "attempting",
                ":failed": "failed",
                ":accepted": "accepted",
                ":delivered": "delivered",
            }
        )

    try:
        table.update_item(
            Key={"email": normalize_email(email)},
            UpdateExpression=(
                "SET #delivery_status = :status, #event_timestamp = :now, "
                "updated_at = :now"
            ),
            ConditionExpression=condition,
            ExpressionAttributeNames={
                "#delivery_status": status_field,
                "#event_timestamp": timestamp_field,
            },
            ExpressionAttributeValues=values,
        )
    except ClientError as error:
        if (
            error.response.get("Error", {}).get("Code")
            != "ConditionalCheckFailedException"
        ):
            raise


def process_ses_event(event: dict, delivery_table, subscriber_table) -> dict:
    processed = 0
    ignored = 0
    for payload in ses_payloads(event):
        status = ses_event_status(payload)
        if not status:
            ignored += 1
            continue
        mail = payload.get("mail") or {}
        tags = mail.get("tags") or {}
        issue_id = ses_tag(tags, "IssueId", "issue_id")
        recipient_id = ses_tag(tags, "SubscriberId", "subscriber_id", "recipient_id")
        email_type = ses_tag(tags, "EmailType", "email_type")
        destinations = mail.get("destination") or []
        if not recipient_id and destinations:
            recipient_id = subscriber_id(destinations[0])

        if issue_id and recipient_id:
            update_delivery_from_ses_event(
                delivery_table,
                issue_id,
                recipient_id,
                status,
                mail.get("messageId"),
            )

        if email_type in {"verification", "verification_reminder"}:
            for destination in destinations:
                record_verification_delivery_event(
                    subscriber_table,
                    destination,
                    email_type,
                    status,
                )

        should_suppress = status == "complained"
        suppression_reason = "complaint" if should_suppress else None
        if status == "bounced":
            bounce_type = str(
                (payload.get("bounce") or {}).get("bounceType", "")
            ).lower()
            should_suppress = bounce_type == "permanent"
            suppression_reason = "permanent_bounce" if should_suppress else None

        if should_suppress:
            for destination in destinations:
                suppress_subscriber(subscriber_table, destination, suppression_reason)
                logger.info(
                    f"Suppressed subscriber_id={subscriber_log_id(destination)} reason={suppression_reason}"
                )
        processed += 1

    return {"processed": processed, "ignored": ignored}
