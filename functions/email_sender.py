import html
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlencode, urlparse

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

from .delivery_state import (
    claim_newsletter_delivery,
    process_ses_event,
    safe_ses_tag_value,
    subscriber_id,
    subscriber_log_id,
    update_newsletter_delivery,
)
from .subscriber_reminders import (
    claim_verification_reminder,
    parse_bool,
    parse_bulk_verification_options,
    parse_iso_datetime,
    record_verification_reminder_outcome,
    select_bulk_candidates,
    select_scheduled_candidates,
)

logger = logging.getLogger(__name__)

VERIFICATION_TOKEN_TTL = timedelta(days=7)
DEFAULT_SUBSCRIBE_COOLDOWN_SECONDS = 15 * 60
DEFAULT_PENDING_SUBSCRIBER_TTL_DAYS = 30
DEFAULT_MAX_SUBSCRIPTION_HISTORY = 50
DEFAULT_BULK_VERIFICATION_MAX_BATCH = 100


def bounded_history(existing: list[dict], entry: dict) -> list[dict]:
    max_entries = max(
        1,
        int(
            os.environ.get("MAX_SUBSCRIPTION_HISTORY", DEFAULT_MAX_SUBSCRIPTION_HISTORY)
        ),
    )
    return [*(existing or [])[-(max_entries - 1) :], entry][-max_entries:]


def pending_expiry_epoch(now: datetime) -> int:
    ttl_days = max(
        1,
        int(
            os.environ.get(
                "PENDING_SUBSCRIBER_TTL_DAYS", DEFAULT_PENDING_SUBSCRIBER_TTL_DAYS
            )
        ),
    )
    return int((now + timedelta(days=ttl_days)).timestamp())


def neutral_subscription_response(headers: dict) -> dict:
    return {
        "statusCode": 200,
        "headers": headers,
        "body": json.dumps(
            {
                "message": "If this address can be subscribed, a verification email will be sent.",
            }
        ),
    }


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def redact_log_text(value) -> str:
    """Replace email addresses in diagnostic text with stable identifiers."""
    pattern = r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
    return re.sub(
        pattern,
        lambda match: f"subscriber_id={subscriber_log_id(match.group(0))}",
        str(value),
    )


def build_api_url(api_url: str, path: str, params: dict | None = None) -> str:
    """Build a validated API URL with encoded query parameters."""
    base_url = api_url.rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.netloc.startswith("."):
        raise ValueError(f"Invalid API URL configured: {api_url}")

    url = f"{base_url}/{path.lstrip('/')}" if path else base_url
    if params:
        url = f"{url}?{urlencode(params)}"
    return url


def build_verification_url(api_url: str, email: str, token: str) -> str:
    return build_api_url(api_url, "/verify", {"email": email, "token": token})


def token_is_expired(subscriber: dict, now: datetime | None = None) -> bool:
    """Check explicit expiry, falling back to the verification request timestamp."""
    now = now or utc_now()
    expiry_value = subscriber.get("verification_expires_at")

    if expiry_value:
        try:
            expiry = datetime.fromisoformat(expiry_value.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            return now >= expiry
        except (TypeError, ValueError):
            logger.warning("Subscriber has an invalid verification expiry timestamp")
            return True

    requested_value = subscriber.get("verification_requested_at") or subscriber.get(
        "created_at"
    )
    if not requested_value:
        return False

    try:
        requested = datetime.fromisoformat(requested_value.replace("Z", "+00:00"))
        if requested.tzinfo is None:
            requested = requested.replace(tzinfo=timezone.utc)
        return now >= requested + VERIFICATION_TOKEN_TTL
    except (TypeError, ValueError):
        logger.warning("Subscriber has an invalid verification request timestamp")
        return True


def ensure_active_verification_token(table, subscriber: dict) -> str:
    """Reuse a valid token or atomically replace an expired token once."""
    verification_token = subscriber.get("verification_token")
    if verification_token and not token_is_expired(subscriber):
        return verification_token

    old_token = verification_token
    replacement_token = secrets.token_urlsafe(32)
    now = utc_now()
    condition = "#status = :pending AND attribute_not_exists(verification_token)"
    values = {
        ":token": replacement_token,
        ":now": now.isoformat(),
        ":expires_at": (now + VERIFICATION_TOKEN_TTL).isoformat(),
        ":pending_expires_at": pending_expiry_epoch(now),
        ":pending": "pending",
    }
    if old_token:
        condition = "#status = :pending AND verification_token = :old_token"
        values[":old_token"] = old_token

    try:
        table.update_item(
            Key={"email": subscriber["email"]},
            UpdateExpression="SET verification_token = :token, verification_requested_at = :now, "
            "verification_expires_at = :expires_at, pending_expires_at = :pending_expires_at",
            ConditionExpression=condition,
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues=values,
        )
        subscriber.update(
            {
                "verification_token": replacement_token,
                "verification_requested_at": now.isoformat(),
                "verification_expires_at": (now + VERIFICATION_TOKEN_TTL).isoformat(),
            }
        )
        return replacement_token
    except ClientError as error:
        if (
            error.response.get("Error", {}).get("Code")
            != "ConditionalCheckFailedException"
        ):
            raise

    current = table.get_item(Key={"email": subscriber["email"]}).get("Item", {})
    current_token = current.get("verification_token")
    if (
        current.get("status") == "pending"
        and current_token
        and not token_is_expired(current)
    ):
        subscriber.update(current)
        return current_token
    raise RuntimeError(
        "Verification token changed concurrently and no active token is available"
    )


def select_bulk_verification_subscribers(
    subscribers: list[dict], expired_only: bool
) -> list[dict]:
    """Select all candidates or only records that still need token rotation."""
    if not expired_only:
        return subscribers
    return [
        subscriber
        for subscriber in subscribers
        if not subscriber.get("verification_token") or token_is_expired(subscriber)
    ]


def query_subscribers_by_status(table, status: str) -> list[dict]:
    """Read every subscriber for a status using the table's status index."""
    items = []
    query_args = {
        "IndexName": "StatusIndex",
        "KeyConditionExpression": Key("status").eq(status),
    }

    while True:
        response = table.query(**query_args)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            return items
        query_args["ExclusiveStartKey"] = last_key


def parse_request_body(event: dict) -> dict:
    body = event.get("body") or ""
    content_type = (event.get("headers") or {}).get("content-type", "")
    if not content_type:
        content_type = (event.get("headers") or {}).get("Content-Type", "")

    if "application/json" in content_type:
        return json.loads(body)
    return {key: values[0] for key, values in parse_qs(body).items() if values}


def html_page(title: str, heading: str, message: str, extra_html: str = "") -> dict:
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/html",
            "Access-Control-Allow-Origin": "*",
        },
        "body": f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 600px; margin: 40px auto; padding: 20px; text-align: center; }}
        .message {{ color: #666; margin-bottom: 20px; }}
        button {{ padding: 12px 24px; border: 0; border-radius: 4px; background: #007bff; color: white; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>{html.escape(heading)}</h1>
    <p class="message">{html.escape(message)}</p>
    {extra_html}
</body>
</html>""",
    }


def safe_http_url(value: str) -> str:
    """Return a safe absolute web URL or an empty string."""
    parsed = urlparse(str(value).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return parsed.geturl()


def build_html_newsletter_data(json_content: dict) -> dict:
    """Create escaped values used only by the newsletter HTML template."""

    def escaped_papers(papers: list[dict]) -> list[dict]:
        return [
            {
                "title": html.escape(str(paper.get("title", ""))),
                "summary": html.escape(str(paper.get("summary", ""))),
                "link": html.escape(safe_http_url(paper.get("link", "")), quote=True),
            }
            for paper in papers
        ]

    return {
        "overview_html": html.escape(str(json_content.get("overview", ""))),
        "featured_papers_html": escaped_papers(json_content.get("featured_papers", [])),
        "additional_papers_html": escaped_papers(
            json_content.get("additional_papers", [])
        ),
    }


def is_valid_email(email: str) -> bool:
    """Validate email format"""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def create_api_response(status_code: int, body: dict) -> dict:
    """Create standardized API Gateway response"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",  # For CORS support
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body),
    }


def _claim_subscription_verification(
    table, email: str, now: datetime
) -> tuple[bool, str | None]:
    """Atomically claim at most one verification send inside the cooldown window."""
    existing = table.get_item(Key={"email": email}).get("Item")
    timestamp = now.isoformat()
    expires_at = (now + VERIFICATION_TOKEN_TTL).isoformat()
    cooldown_seconds = max(
        1,
        int(
            os.environ.get(
                "SUBSCRIBE_COOLDOWN_SECONDS", DEFAULT_SUBSCRIBE_COOLDOWN_SECONDS
            )
        ),
    )
    cooldown_cutoff = (now - timedelta(seconds=cooldown_seconds)).isoformat()

    if existing and (
        existing.get("status") == "verified"
        or str(existing.get("deliverability_status", "")).lower() == "suppressed"
    ):
        return False, None

    if existing and existing.get("status") == "pending":
        active_token = existing.get("verification_token")
        if active_token and not token_is_expired(existing, now):
            last_attempt = parse_iso_datetime(
                existing.get("verification_last_attempted_at")
            )
            if last_attempt and last_attempt > now - timedelta(
                seconds=cooldown_seconds
            ):
                return False, active_token
            try:
                table.update_item(
                    Key={"email": email},
                    UpdateExpression="SET verification_last_attempted_at = :now, updated_at = :now, "
                    "verification_delivery_status = :attempting, pending_expires_at = :pending_expires_at",
                    ConditionExpression="#status = :pending AND verification_token = :token AND "
                    "(attribute_not_exists(verification_last_attempted_at) OR "
                    "verification_last_attempted_at <= :cooldown_cutoff)",
                    ExpressionAttributeNames={"#status": "status"},
                    ExpressionAttributeValues={
                        ":pending": "pending",
                        ":token": active_token,
                        ":now": timestamp,
                        ":attempting": "attempting",
                        ":cooldown_cutoff": cooldown_cutoff,
                        ":pending_expires_at": pending_expiry_epoch(now),
                    },
                )
                return True, active_token
            except ClientError as error:
                if (
                    error.response.get("Error", {}).get("Code")
                    == "ConditionalCheckFailedException"
                ):
                    return False, active_token
                raise

    verification_token = secrets.token_urlsafe(32)
    unsubscribe_token = secrets.token_urlsafe(32)
    prior_status = existing.get("status", "new") if existing else "new"
    history = bounded_history(
        (existing or {}).get("subscription_history", []),
        {
            "status": "pending",
            "timestamp": timestamp,
            "action": "initial_subscribe" if not existing else "resubscribe",
            "previous_status": prior_status,
        },
    )
    item_values = {
        "email": email,
        "status": "pending",
        "verification_token": verification_token,
        "unsubscribe_token": unsubscribe_token,
        "created_at": (existing or {}).get("created_at", timestamp),
        "updated_at": timestamp,
        "verification_requested_at": timestamp,
        "verification_expires_at": expires_at,
        "verification_last_attempted_at": timestamp,
        "verification_delivery_status": "attempting",
        "pending_expires_at": pending_expiry_epoch(now),
        "subscription_history": history,
    }

    try:
        if not existing:
            table.put_item(
                Item=item_values,
                ConditionExpression="attribute_not_exists(email)",
            )
        else:
            condition = "#status = :prior_status"
            expression_values = {
                ":pending": "pending",
                ":prior_status": prior_status,
                ":token": verification_token,
                ":unsubscribe_token": unsubscribe_token,
                ":timestamp": timestamp,
                ":expires_at": expires_at,
                ":pending_expires_at": item_values["pending_expires_at"],
                ":attempting": "attempting",
                ":history": history,
            }
            old_token = existing.get("verification_token")
            if prior_status == "pending":
                if old_token:
                    condition += " AND verification_token = :old_token"
                    expression_values[":old_token"] = old_token
                else:
                    condition += " AND attribute_not_exists(verification_token)"
            table.update_item(
                Key={"email": email},
                UpdateExpression="SET #status = :pending, verification_token = :token, "
                "unsubscribe_token = :unsubscribe_token, updated_at = :timestamp, "
                "verification_requested_at = :timestamp, verification_expires_at = :expires_at, "
                "verification_last_attempted_at = :timestamp, verification_delivery_status = :attempting, "
                "pending_expires_at = :pending_expires_at, subscription_history = :history "
                "REMOVE reminder_sent, reminder_token, verification_reminder_accepted_at, "
                "verification_reminder_campaign_id, verification_reminder_claimed_at, "
                "verification_reminder_claim_id, verification_reminder_count",
                ConditionExpression=condition,
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues=expression_values,
            )
        return True, verification_token
    except ClientError as error:
        if (
            error.response.get("Error", {}).get("Code")
            == "ConditionalCheckFailedException"
        ):
            return False, None
        raise


def add_subscriber(event, context):
    """Handle a public subscription request without disclosing subscriber state."""
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "OPTIONS,POST",
        "Content-Type": "application/json",
    }

    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": cors_headers, "body": ""}

    try:
        if not event.get("body"):
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Missing request body"}),
            }

        try:
            body = json.loads(event["body"])
        except (json.JSONDecodeError, TypeError):
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Invalid JSON body"}),
            }
        email = normalize_email(body.get("email", ""))
        if not email or not is_valid_email(email):
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "A valid email is required"}),
            }

        api_url = get_api_url_from_event(event) or os.environ.get("API_URL")
        if not api_url:
            logger.error("Could not determine API URL from subscription event")
            return {
                "statusCode": 500,
                "headers": cors_headers,
                "body": json.dumps({"error": "Could not determine API URL"}),
            }
        build_api_url(api_url, "")

        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(os.environ["SUBSCRIBERS_TABLE"])
        now = utc_now()
        try:
            should_send, verification_token = _claim_subscription_verification(
                table, email, now
            )
        except ClientError as e:
            logger.error(
                f"Subscriber database operation failed for subscriber_id={subscriber_log_id(email)}: "
                f"{redact_log_text(e)}"
            )
            return {
                "statusCode": 500,
                "headers": cors_headers,
                "body": json.dumps({"error": "Database operation failed"}),
            }

        if not should_send or not verification_token:
            return neutral_subscription_response(cors_headers)

        accepted = send_verification_email(
            email=email,
            verification_token=verification_token,
            sender=os.environ["SENDER_EMAIL"],
            api_url=api_url,
        )
        delivery_status = "accepted" if accepted else "failed"
        outcome_values = {
            ":status": delivery_status,
            ":updated_at": utc_now().isoformat(),
            ":pending": "pending",
            ":token": verification_token,
        }
        outcome_expression = (
            "SET verification_delivery_status = :status, updated_at = :updated_at"
        )
        if accepted:
            outcome_values[":accepted_at"] = utc_now().isoformat()
            outcome_expression += ", verification_last_accepted_at = :accepted_at"
        try:
            table.update_item(
                Key={"email": email},
                UpdateExpression=outcome_expression,
                ConditionExpression="#status = :pending AND verification_token = :token",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues=outcome_values,
            )
        except ClientError as update_error:
            if (
                update_error.response.get("Error", {}).get("Code")
                != "ConditionalCheckFailedException"
            ):
                logger.error(
                    f"Could not record verification outcome for subscriber_id={subscriber_log_id(email)}: "
                    f"{redact_log_text(update_error)}"
                )

        if not accepted:
            logger.warning(
                f"Verification email was not accepted for subscriber_id={subscriber_log_id(email)}"
            )

        return neutral_subscription_response(cors_headers)

    except Exception as e:
        logger.error(f"Unexpected subscription error: {redact_log_text(e)}")
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": json.dumps({"error": "Internal server error"}),
        }


def get_api_url_from_event(event):
    """Extract API URL from the event context"""
    request_context = event.get("requestContext", {})
    domain_name = request_context.get("domainName")
    stage = request_context.get("stage")
    if domain_name and stage:
        return f"https://{domain_name}/{stage}"
    return None


def send_verification_email(
    email: str,
    verification_token: str,
    sender: str,
    api_url: str | None = None,
    email_type: str = "verification",
    campaign_id: str | None = None,
) -> bool:
    """Send verification email to new subscriber"""
    ses_client = boto3.client("ses")

    if not api_url:
        logger.error("No API URL provided for verification link")
        return False

    try:
        verification_url = build_verification_url(api_url, email, verification_token)

        # Get configuration set name for better deliverability
        config_set_name = os.environ.get("EMAIL_CONFIGURATION_SET")

        send_params = {
            "Source": sender,
            "Destination": {"ToAddresses": [email]},
            "Template": os.environ.get(
                "VERIFICATION_TEMPLATE_NAME", "VerificationTemplate"
            ),
            "TemplateData": json.dumps(
                {"verification_url": verification_url, "email": email}
            ),
            "Tags": [
                {"Name": "EmailType", "Value": safe_ses_tag_value(email_type)},
                {"Name": "SubscriberId", "Value": subscriber_id(email)},
            ],
        }
        if campaign_id:
            send_params["Tags"].append(
                {
                    "Name": "CampaignId",
                    "Value": safe_ses_tag_value(campaign_id),
                }
            )

        # Add ConfigurationSetName if available (critical for deliverability)
        if config_set_name:
            send_params["ConfigurationSetName"] = config_set_name

        ses_client.send_templated_email(**send_params)
        logger.info(
            f"Verification email accepted for subscriber_id={subscriber_log_id(email)}"
        )
        return True
    except Exception as e:
        logger.error(
            f"Verification email failed for subscriber_id={subscriber_log_id(email)}: "
            f"{redact_log_text(e)}"
        )
        return False


def verify_subscriber(event, context) -> dict:
    """Render confirmation on GET and verify conditionally on POST."""
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Content-Type": "application/json",
    }

    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": cors_headers, "body": ""}

    try:
        method = event.get("httpMethod", "GET").upper()
        try:
            params = (
                parse_request_body(event)
                if method == "POST"
                else (event.get("queryStringParameters") or {})
            )
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Invalid request body"}),
            }
        if not params:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Missing query parameters"}),
            }

        requested_email = (params.get("email") or "").strip()
        email = normalize_email(requested_email)
        token = params.get("token")

        if not email or not token:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Missing email or token"}),
            }

        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(os.environ["SUBSCRIBERS_TABLE"])

        try:
            response = table.get_item(Key={"email": email})

            # Preserve compatibility with records created before email normalization.
            if "Item" not in response and requested_email != email:
                response = table.get_item(Key={"email": requested_email})

            if "Item" not in response:
                return {
                    "statusCode": 404,
                    "headers": cors_headers,
                    "body": json.dumps({"error": "Subscriber not found"}),
                }

            subscriber = response["Item"]
            subscriber_email = subscriber["email"]
            stored_token = subscriber.get("verification_token")

            if not stored_token or not secrets.compare_digest(
                str(stored_token), str(token)
            ):
                return {
                    "statusCode": 400,
                    "headers": cors_headers,
                    "body": json.dumps({"error": "Invalid verification token"}),
                }

            if subscriber.get("status") == "verified":
                return html_page(
                    "Already Verified",
                    "Email Already Verified",
                    "Your email subscription has already been verified.",
                )

            if subscriber.get("status") != "pending":
                return {
                    "statusCode": 409,
                    "headers": cors_headers,
                    "body": json.dumps(
                        {"error": "Subscription is not pending verification"}
                    ),
                }

            if token_is_expired(subscriber):
                return {
                    "statusCode": 410,
                    "headers": cors_headers,
                    "body": json.dumps(
                        {
                            "error": "Verification token has expired. Please subscribe again."
                        }
                    ),
                }

            if method == "GET":
                api_url = get_api_url_from_event(event) or os.environ.get("API_URL")
                if not api_url:
                    return {
                        "statusCode": 500,
                        "headers": cors_headers,
                        "body": json.dumps({"error": "Could not determine API URL"}),
                    }
                action_url = html.escape(build_api_url(api_url, "/verify"), quote=True)
                form = f'''<form method="post" action="{action_url}">
    <input type="hidden" name="email" value="{html.escape(subscriber_email, quote=True)}">
    <input type="hidden" name="token" value="{html.escape(str(token), quote=True)}">
    <button type="submit">Confirm subscription</button>
</form>'''
                return html_page(
                    "Confirm Subscription",
                    "Confirm Subscription",
                    "Confirm that you want to receive the AI Papers Newsletter.",
                    form,
                )

            if method != "POST":
                return {
                    "statusCode": 405,
                    "headers": cors_headers,
                    "body": json.dumps({"error": "Method not allowed"}),
                }

            verified_at = utc_now().isoformat()
            history = bounded_history(
                subscriber.get("subscription_history", []),
                {
                    "status": "verified",
                    "timestamp": verified_at,
                    "action": "verify",
                },
            )
            table.update_item(
                Key={"email": subscriber_email},
                UpdateExpression="SET #status = :verified, verified_at = :verified_at, updated_at = :verified_at, "
                "subscription_history = :history REMOVE pending_expires_at",
                ConditionExpression="#status = :pending AND verification_token = :token",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":verified": "verified",
                    ":pending": "pending",
                    ":token": token,
                    ":verified_at": verified_at,
                    ":history": history,
                },
            )

            return html_page(
                "Subscription Verified",
                "Subscription Verified!",
                "Thank you for subscribing. You'll start receiving newsletters soon.",
            )

        except ClientError as e:
            if (
                e.response.get("Error", {}).get("Code")
                == "ConditionalCheckFailedException"
            ):
                return {
                    "statusCode": 409,
                    "headers": cors_headers,
                    "body": json.dumps(
                        {
                            "error": "Subscription state changed. Please request a new link."
                        }
                    ),
                }
            logger.error(
                f"Subscriber verification database error: {redact_log_text(e)}"
            )
            return {
                "statusCode": 500,
                "headers": cors_headers,
                "body": json.dumps({"error": "Database error"}),
            }

    except Exception as e:
        logger.error(f"Error verifying subscriber: {redact_log_text(e)}")
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": json.dumps({"error": "Internal server error"}),
        }


def publish_verification_metrics(
    success_count: int, failure_count: int, batch_size: int
) -> None:
    """Publish the custom metrics referenced by the verification dashboard."""
    try:
        cloudwatch = boto3.client("cloudwatch")
        cloudwatch.put_metric_data(
            Namespace="Newsletter/Verification",
            MetricData=[
                {
                    "MetricName": "ReminderEmailSuccess",
                    "Value": success_count,
                    "Unit": "Count",
                },
                {
                    "MetricName": "ReminderEmailFailure",
                    "Value": failure_count,
                    "Unit": "Count",
                },
                {
                    "MetricName": "ReminderBatchSize",
                    "Value": batch_size,
                    "Unit": "Count",
                },
            ],
        )
    except Exception as metric_error:
        logger.error(
            f"Could not publish verification metrics: {redact_log_text(metric_error)}"
        )


def publish_verification_system_error() -> None:
    try:
        boto3.client("cloudwatch").put_metric_data(
            Namespace="Newsletter/Verification",
            MetricData=[
                {"MetricName": "ReminderSystemError", "Value": 1, "Unit": "Count"}
            ],
        )
    except Exception as metric_error:
        logger.error(
            f"Could not publish verification system error metric: {redact_log_text(metric_error)}"
        )


def send_claimed_verification_reminder(
    table,
    subscriber: dict,
    campaign_id: str,
    sender: str,
    api_url: str,
    force: bool = False,
) -> str:
    """Return accepted, failed, or skipped for one reminder candidate."""
    verification_token = ensure_active_verification_token(table, subscriber)
    claim_id = claim_verification_reminder(table, subscriber, campaign_id, force=force)
    if not claim_id:
        return "skipped"

    accepted = send_verification_email(
        email=subscriber["email"],
        verification_token=verification_token,
        sender=sender,
        api_url=api_url,
        email_type="verification_reminder",
        campaign_id=campaign_id,
    )
    try:
        record_verification_reminder_outcome(
            table,
            subscriber["email"],
            campaign_id,
            claim_id,
            accepted,
        )
    except ClientError as error:
        if (
            error.response.get("Error", {}).get("Code")
            != "ConditionalCheckFailedException"
        ):
            raise
        logger.warning(
            f"Reminder outcome state changed for subscriber_id={subscriber_log_id(subscriber['email'])}"
        )
    return "accepted" if accepted else "failed"


def check_and_resend_verifications(event, context):
    """
    Send a single verification reminder to subscribers who:
    - Have been unverified for > 48 hours
    - Have never received a reminder
    """
    try:
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(os.environ["SUBSCRIBERS_TABLE"])

        now = utc_now()
        pending_subscribers = query_subscribers_by_status(table, "pending")
        max_batch = max(
            1,
            int(
                os.environ.get(
                    "BULK_VERIFICATION_MAX_BATCH", DEFAULT_BULK_VERIFICATION_MAX_BATCH
                )
            ),
        )
        subscribers_to_remind, invalid_timestamp_count = select_scheduled_candidates(
            pending_subscribers,
            now,
            max_batch,
        )
        if invalid_timestamp_count:
            logger.warning(
                f"Skipped {invalid_timestamp_count} scheduled reminders with invalid timestamps"
            )

        api_url = (event or {}).get("api_url") or os.environ.get("API_URL")
        if not api_url:
            raise ValueError("No API URL was provided to the verification reminder")
        sender_email = os.environ["SENDER_EMAIL"]
        build_api_url(api_url, "")

        accepted_count = 0
        failure_count = 0
        skipped_count = 0
        campaign_id = "scheduled-verification-reminder"

        for item in subscribers_to_remind:
            try:
                outcome = send_claimed_verification_reminder(
                    table,
                    item,
                    campaign_id,
                    sender_email,
                    api_url,
                )
                if outcome == "accepted":
                    accepted_count += 1
                elif outcome == "failed":
                    failure_count += 1
                else:
                    skipped_count += 1

            except Exception as e:
                failure_count += 1
                logger.error(
                    f"Reminder processing failed for subscriber_id={subscriber_log_id(item.get('email', ''))}: {e!s}"
                )

        batch_size = len(subscribers_to_remind)
        publish_verification_metrics(accepted_count, failure_count, batch_size)
        logger.info(
            f"Reminder process completed. Accepted: {accepted_count}, Failures: {failure_count}, Skipped: {skipped_count}"
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Verification reminder process completed",
                    "accepted_count": accepted_count,
                    "success_count": accepted_count,
                    "failure_count": failure_count,
                    "skipped_count": skipped_count,
                    "batch_size": batch_size,
                }
            ),
        }

    except Exception as e:
        logger.error(f"Error in check_and_resend_verifications: {redact_log_text(e)}")
        publish_verification_system_error()
        raise


def unsubscribe_handler(event, context) -> dict:
    """Show an unsubscribe confirmation on GET and apply it on POST."""
    cors_headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Content-Type": "application/json",
    }

    # Handle OPTIONS preflight request
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": cors_headers, "body": ""}

    try:
        method = event.get("httpMethod", "GET").upper()
        try:
            params = (
                parse_request_body(event)
                if method == "POST"
                else (event.get("queryStringParameters") or {})
            )
        except json.JSONDecodeError:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Invalid request body"}),
            }

        requested_email = (params.get("email") or "").strip()
        email = normalize_email(requested_email)
        token = params.get("token")
        if not email or not token:
            return {
                "statusCode": 400,
                "headers": cors_headers,
                "body": json.dumps({"error": "Missing email or unsubscribe token"}),
            }

        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(os.environ["SUBSCRIBERS_TABLE"])

        try:
            response = table.get_item(Key={"email": email})
            if "Item" not in response and requested_email != email:
                response = table.get_item(Key={"email": requested_email})

            if "Item" not in response:
                return {
                    "statusCode": 404,
                    "headers": cors_headers,
                    "body": json.dumps({"error": "Subscriber not found"}),
                }

            subscriber = response["Item"]
            subscriber_email = subscriber["email"]
            stored_token = subscriber.get("unsubscribe_token")
            if not stored_token or not secrets.compare_digest(
                str(stored_token), str(token)
            ):
                return {
                    "statusCode": 400,
                    "headers": cors_headers,
                    "body": json.dumps({"error": "Invalid unsubscribe token"}),
                }

            if subscriber.get("status") == "unsubscribed":
                return html_page(
                    "Already Unsubscribed",
                    "Email Already Unsubscribed",
                    "This email has already been unsubscribed from the newsletter.",
                )

            if method == "GET":
                api_url = get_api_url_from_event(event)
                if not api_url:
                    return {
                        "statusCode": 500,
                        "headers": cors_headers,
                        "body": json.dumps({"error": "Could not determine API URL"}),
                    }

                action_url = html.escape(
                    build_api_url(api_url, "/unsubscribe"), quote=True
                )
                form = f'''<form method="post" action="{action_url}">
    <input type="hidden" name="email" value="{html.escape(subscriber_email, quote=True)}">
    <input type="hidden" name="token" value="{html.escape(token, quote=True)}">
    <button type="submit">Confirm unsubscribe</button>
</form>'''
                return html_page(
                    "Confirm Unsubscribe",
                    "Confirm Unsubscribe",
                    "Confirm that you want to stop receiving the AI Papers Newsletter.",
                    form,
                )

            if subscriber.get("status") != "verified":
                return {
                    "statusCode": 409,
                    "headers": cors_headers,
                    "body": json.dumps(
                        {"error": "Only verified subscriptions can be unsubscribed"}
                    ),
                }

            unsubscribed_at = utc_now().isoformat()
            history = bounded_history(
                subscriber.get("subscription_history", []),
                {
                    "status": "unsubscribed",
                    "timestamp": unsubscribed_at,
                    "action": "unsubscribe",
                },
            )
            table.update_item(
                Key={"email": subscriber_email},
                UpdateExpression="SET #status = :unsubscribed, unsubscribed_at = :unsubscribed_at, updated_at = :unsubscribed_at, "
                "subscription_history = :history",
                ConditionExpression="#status = :verified AND unsubscribe_token = :token",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":unsubscribed": "unsubscribed",
                    ":verified": "verified",
                    ":token": token,
                    ":unsubscribed_at": unsubscribed_at,
                    ":history": history,
                },
            )
            return html_page(
                "Unsubscribed",
                "Unsubscribed Successfully!",
                "You have been unsubscribed from the AI Papers Newsletter.",
            )

        except ClientError as e:
            if (
                e.response.get("Error", {}).get("Code")
                == "ConditionalCheckFailedException"
            ):
                return {
                    "statusCode": 409,
                    "headers": cors_headers,
                    "body": json.dumps(
                        {"error": "Subscription state changed. Refresh and try again."}
                    ),
                }
            logger.error(f"DynamoDB error: {redact_log_text(e)}")
            return {
                "statusCode": 500,
                "headers": cors_headers,
                "body": json.dumps({"error": "Database error"}),
            }

    except Exception as e:
        logger.error(f"Error unsubscribing: {redact_log_text(e)}")
        return {
            "statusCode": 500,
            "headers": cors_headers,
            "body": json.dumps({"error": "Internal server error"}),
        }


def get_subscribers(table_name: str) -> list[dict]:
    """Retrieve every verified subscriber and ensure an unsubscribe token exists."""
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    try:
        subscribers = [
            subscriber
            for subscriber in query_subscribers_by_status(table, "verified")
            if str(subscriber.get("deliverability_status", "")).lower() != "suppressed"
        ]
        for subscriber in subscribers:
            if subscriber.get("unsubscribe_token"):
                continue

            token = secrets.token_urlsafe(32)
            response = table.update_item(
                Key={"email": subscriber["email"]},
                UpdateExpression="SET unsubscribe_token = if_not_exists(unsubscribe_token, :token)",
                ConditionExpression="#status = :verified",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={":token": token, ":verified": "verified"},
                ReturnValues="ALL_NEW",
            )
            subscriber["unsubscribe_token"] = response["Attributes"][
                "unsubscribe_token"
            ]
        return subscribers
    except ClientError as e:
        logger.error(f"Error querying DynamoDB table: {redact_log_text(e)}")
        raise


def test_prod_check(subscribers: list[dict], test_email: str) -> list[dict]:
    """
    Returns appropriate recipient list based on test mode settings
    """
    test_mode = os.environ.get("EMAIL_TEST_MODE", "false").lower() == "true"

    if test_mode:
        if not test_email or not is_valid_email(test_email):
            raise ValueError(
                "EMAIL_TEST_ADDRESS must contain a valid email when test mode is enabled"
            )
        normalized_test_email = normalize_email(test_email)
        logger.info(
            f"Test mode enabled: Sending single test email instead of {len(subscribers)} subscriber emails"
        )
        existing = next(
            (
                subscriber
                for subscriber in subscribers
                if subscriber["email"] == normalized_test_email
            ),
            None,
        )
        return [
            existing
            or {
                "email": normalized_test_email,
                "unsubscribe_token": secrets.token_urlsafe(32),
            }
        ]

    return subscribers


def send_email_batch(
    ses_client,
    source: str,
    subscribers: list[dict],
    json_content: dict,
    unsubscribe_url: str,
    subscribe_url: str,
    issue_id: str | None = None,
    delivery_table=None,
) -> dict:
    results = {
        "accepted": 0,
        "successful": 0,
        "failed": 0,
        "skipped": 0,
        "failures": [],
    }
    issue_id = issue_id or json_content.get("metadata", {}).get("generated_date")
    if not issue_id:
        raise ValueError(
            "issue_id or metadata.generated_date is required for delivery tracking"
        )
    issue_id = safe_ses_tag_value(issue_id)

    default_template_data = {
        "date": utc_now().strftime("%B %d, %Y"),
        "generated_date": json_content["metadata"]["generated_date"],
        "featured_count": json_content["metadata"]["featured_papers_count"],
        "overview": json_content["overview"],
        "featured_papers": json_content["featured_papers"],
        "additional_papers": json_content.get("additional_papers", []),
        "unsubscribe_url": unsubscribe_url,
        "subscribe_url": subscribe_url,
        "email": "",
    }
    default_template_data.update(build_html_newsletter_data(json_content))
    default_template_data.update(
        {
            "unsubscribe_url_html": html.escape(unsubscribe_url, quote=True),
            "subscribe_url_html": html.escape(subscribe_url, quote=True),
        }
    )

    batch_size = 50
    for i in range(0, len(subscribers), batch_size):
        batch_subscribers = subscribers[i : i + batch_size]
        destinations = []
        claimed_recipients = []

        for subscriber in batch_subscribers:
            email = subscriber["email"]
            recipient_id = subscriber_id(email)
            if not claim_newsletter_delivery(delivery_table, issue_id, recipient_id):
                results["skipped"] += 1
                continue
            unsubscribe_link = build_api_url(
                unsubscribe_url,
                "",
                {
                    "email": email,
                    "token": subscriber["unsubscribe_token"],
                },
            )
            destinations.append(
                {
                    "Destination": {"ToAddresses": [email]},
                    "ReplacementTemplateData": json.dumps(
                        {
                            "email": email,
                            "unsubscribe_url": unsubscribe_link,
                            "subscribe_url": subscribe_url,
                            "unsubscribe_url_html": html.escape(
                                unsubscribe_link, quote=True
                            ),
                            "subscribe_url_html": html.escape(
                                subscribe_url, quote=True
                            ),
                        }
                    ),
                    "ReplacementTags": [
                        {"Name": "EmailType", "Value": "newsletter"},
                        {
                            "Name": "IssueId",
                            "Value": safe_ses_tag_value(issue_id),
                        },
                        {
                            "Name": "SubscriberId",
                            "Value": recipient_id,
                        },
                    ],
                }
            )
            claimed_recipients.append(
                {
                    "recipient_id": recipient_id,
                    "email": email,
                }
            )

        if not destinations:
            continue

        try:
            config_set_name = os.environ.get("EMAIL_CONFIGURATION_SET")
            if not config_set_name:
                raise ValueError("EMAIL_CONFIGURATION_SET environment variable not set")

            response = ses_client.send_bulk_templated_email(
                Source=source,
                Template=os.environ.get(
                    "NEWSLETTER_TEMPLATE_NAME", "NewsletterTemplate"
                ),
                DefaultTemplateData=json.dumps(default_template_data),
                Destinations=destinations,
                ConfigurationSetName=config_set_name,
            )

            statuses = response.get("Status", [])
            for index, recipient in enumerate(claimed_recipients):
                status = statuses[index] if index < len(statuses) else {}
                status_name = status.get("Status")
                accepted = status_name == "Success" or (
                    not status_name
                    and not status.get("Error")
                    and bool(status.get("MessageId"))
                )
                if accepted:
                    results["accepted"] += 1
                    results["successful"] += 1
                    update_newsletter_delivery(
                        delivery_table,
                        issue_id,
                        recipient["recipient_id"],
                        "accepted",
                        message_id=status.get("MessageId"),
                    )
                else:
                    error_message = redact_log_text(
                        status.get("Error")
                        or status_name
                        or "Missing SES recipient status"
                    )
                    results["failed"] += 1
                    results["failures"].append(
                        {
                            "subscriber_id": recipient["recipient_id"][:16],
                            "error": error_message,
                        }
                    )
                    update_newsletter_delivery(
                        delivery_table,
                        issue_id,
                        recipient["recipient_id"],
                        "rejected",
                        error=error_message,
                    )

        except Exception as e:
            error_message = redact_log_text(e)
            logger.error(f"Error sending batch: {error_message}")
            results["failed"] += len(claimed_recipients)
            for recipient in claimed_recipients:
                results["failures"].append(
                    {
                        "subscriber_id": recipient["recipient_id"][:16],
                        "error": error_message,
                    }
                )
                try:
                    update_newsletter_delivery(
                        delivery_table,
                        issue_id,
                        recipient["recipient_id"],
                        "ambiguous",
                        error=error_message,
                    )
                except Exception as update_error:
                    logger.error(
                        f"Could not mark ambiguous delivery for subscriber_id={recipient['recipient_id'][:16]}: "
                        f"{redact_log_text(update_error)}"
                    )

    return results


def send_newsletter(
    json_content: dict,
    table_name: str,
    sender: str,
    api_url: str,
    issue_id: str | None = None,
    delivery_table_name: str | None = None,
) -> dict:
    """Main function to send newsletter to all subscribers"""
    try:
        ses_client = boto3.client("ses")

        # Construct unsubscribe URL using AWS environment variables
        unsubscribe_url = build_api_url(api_url, "/unsubscribe")
        subscribe_url = build_api_url(os.environ["SUBSCRIBE_LANDING_PAGE"], "")

        logger.info(f"unsubscribe url: {unsubscribe_url}")
        logger.info(f"subscribe url: {subscribe_url}")

        # Get subscribers
        subscribers = get_subscribers(table_name)
        subscribers = test_prod_check(
            subscribers, os.environ.get("EMAIL_TEST_ADDRESS", "")
        )
        if not subscribers:
            logger.warning("No subscribers found")
            return {
                "message": "No subscribers found",
                "accepted": 0,
                "successful": 0,
                "failed": 0,
                "skipped": 0,
                "failures": [],
            }

        issue_id = issue_id or json_content.get("metadata", {}).get("generated_date")
        if not issue_id:
            raise ValueError("issue_id or metadata.generated_date is required")
        resolved_delivery_table_name = delivery_table_name or os.environ.get(
            "NEWSLETTER_RUNS_TABLE"
        )
        delivery_table = (
            boto3.resource("dynamodb").Table(resolved_delivery_table_name)
            if resolved_delivery_table_name
            else None
        )
        if delivery_table is None:
            logger.warning(
                "NEWSLETTER_RUNS_TABLE is not configured; delivery replay protection is disabled"
            )

        # Send emails using the template
        results = send_email_batch(
            ses_client=ses_client,
            source=sender,
            subscribers=subscribers,
            json_content=json_content,
            unsubscribe_url=unsubscribe_url,
            subscribe_url=subscribe_url,
            issue_id=issue_id,
            delivery_table=delivery_table,
        )

        logger.info(f"Newsletter sending complete. Results: {results}")
        return results

    except Exception as e:
        logger.error(f"Error sending newsletter: {redact_log_text(e)}")
        raise


def ses_event_handler(event, context):
    """Record SNS-wrapped SES delivery events and suppress unsafe recipients."""
    dynamodb = boto3.resource("dynamodb")
    delivery_table = dynamodb.Table(os.environ["NEWSLETTER_RUNS_TABLE"])
    subscriber_table = dynamodb.Table(os.environ["SUBSCRIBERS_TABLE"])
    result = process_ses_event(event, delivery_table, subscriber_table)
    return {
        "statusCode": 200,
        "body": json.dumps(result),
    }


def get_subscriber_counts(table_name: str) -> dict:
    """Query complete subscriber counts, including paginated results."""
    dynamodb = boto3.resource("dynamodb")
    table = dynamodb.Table(table_name)

    try:

        def count_status(status: str) -> int:
            count = 0
            query_args = {
                "IndexName": "StatusIndex",
                "KeyConditionExpression": Key("status").eq(status),
                "Select": "COUNT",
            }
            while True:
                response = table.query(**query_args)
                count += response.get("Count", 0)
                last_key = response.get("LastEvaluatedKey")
                if not last_key:
                    return count
                query_args["ExclusiveStartKey"] = last_key

        status_counts = {
            "verified": count_status("verified"),
            "pending": count_status("pending"),
            "unsubscribed": count_status("unsubscribed"),
        }

        metrics = {
            "total_subscribers": sum(status_counts.values()),
            "status_counts": status_counts,
            "metadata": {
                "timestamp": utc_now().isoformat(),
                "table_name": os.environ["SUBSCRIBERS_TABLE"],
            },
        }

        logger.info(f"Retrieved metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    except ClientError as e:
        logger.error(f"Error querying DynamoDB: {redact_log_text(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {redact_log_text(e)}")
        raise


def sub_status_checker(event, context):
    """
    Handler for subscriber metrics Lambda
    Returns subscriber counts by status with CORS headers
    """
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "GET,OPTIONS",
        "Content-Type": "application/json",
    }

    # Handle OPTIONS request
    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": ""}

    dynamo_table_name = os.environ["SUBSCRIBERS_TABLE"]

    try:
        metrics = get_subscriber_counts(dynamo_table_name)

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(metrics, indent=2),
        }

    except Exception as e:
        logger.error(f"Error processing request: {redact_log_text(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": "Internal server error"}),
        }


def bulk_verifier(event, context):
    """Preview or send a bounded campaign of bulk verification reminders."""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
        "Content-Type": "application/json",
    }

    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": ""}

    try:
        try:
            options = parse_bulk_verification_options(event)
        except (json.JSONDecodeError, ValueError) as error:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({"error": redact_log_text(error)}),
            }

        confirm_send = parse_bool(options.get("confirm_send"))
        force = parse_bool(options.get("force"))
        expired_tokens_only = parse_bool(options.get("expired_tokens_only"))
        campaign_id = str(options.get("campaign_id") or "").strip()
        if confirm_send and not campaign_id:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps(
                    {
                        "error": "campaign_id is required when confirm_send=true",
                    }
                ),
            }

        now = utc_now()
        dynamodb = boto3.resource("dynamodb")
        table = dynamodb.Table(os.environ["SUBSCRIBERS_TABLE"])
        pending_subscribers = query_subscribers_by_status(table, "pending")
        candidates, exclusion_stats = select_bulk_candidates(
            pending_subscribers,
            now,
            force,
            expired_tokens_only,
            token_is_expired,
        )

        configured_max = max(
            1,
            int(
                os.environ.get(
                    "BULK_VERIFICATION_MAX_BATCH", DEFAULT_BULK_VERIFICATION_MAX_BATCH
                )
            ),
        )
        try:
            requested_max = int(options.get("max_batch_size", configured_max))
        except (TypeError, ValueError):
            requested_max = configured_max
        batch_limit = min(configured_max, max(1, requested_max))
        candidates.sort(key=lambda item: item.get("verification_requested_at", ""))
        eligible_before_cap = len(candidates)
        candidates = candidates[:batch_limit]

        summary = {
            "total_pending": len(pending_subscribers),
            "eligible_before_cap": eligible_before_cap,
            "candidate_count": len(candidates),
            "batch_limit": batch_limit,
            **exclusion_stats,
            "force": force,
            "expired_tokens_only": expired_tokens_only,
        }

        if not confirm_send:
            return {
                "statusCode": 200,
                "headers": headers,
                "body": json.dumps(
                    {
                        "message": "Bulk verification reminder dry run completed. No email was sent.",
                        "dry_run": True,
                        **summary,
                    },
                    indent=2,
                ),
            }

        api_url = get_api_url_from_event(event) or os.environ.get("API_URL")
        if not api_url:
            raise ValueError("Could not determine the API URL for verification links")
        build_api_url(api_url, "")
        accepted_count = 0
        failed_count = 0
        skipped_count = 0
        failures = []
        send_rate = max(1.0, float(os.environ.get("BULK_VERIFICATION_SEND_RATE", "10")))
        send_interval = 1 / send_rate

        for index, subscriber in enumerate(candidates):
            try:
                outcome = send_claimed_verification_reminder(
                    table,
                    subscriber,
                    campaign_id,
                    os.environ["SENDER_EMAIL"],
                    api_url,
                    force=force,
                )
                if outcome == "accepted":
                    accepted_count += 1
                elif outcome == "failed":
                    failed_count += 1
                    failures.append(
                        {
                            "subscriber_id": subscriber_log_id(subscriber["email"]),
                            "error": "SES did not accept the verification reminder",
                        }
                    )
                else:
                    skipped_count += 1
            except Exception as e:
                failed_count += 1
                failures.append(
                    {
                        "subscriber_id": subscriber_log_id(subscriber.get("email", "")),
                        "error": redact_log_text(e),
                    }
                )
                logger.error(
                    f"Bulk verification reminder failed for subscriber_id={subscriber_log_id(subscriber.get('email', ''))}: {e}"
                )
            finally:
                if index < len(candidates) - 1:
                    time.sleep(send_interval)

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps(
                {
                    "message": "Bulk verification reminders processed.",
                    "dry_run": False,
                    "campaign_id": campaign_id,
                    "accepted_count": accepted_count,
                    "failed_count": failed_count,
                    "skipped_count": skipped_count,
                    "failures": failures,
                    **summary,
                },
                indent=2,
            ),
        }

    except Exception as e:
        logger.error(
            f"Error processing bulk verification reminders: {redact_log_text(e)}"
        )
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": "Internal server error"}),
        }
