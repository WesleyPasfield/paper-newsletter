import html
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

VERIFICATION_TOKEN_TTL = timedelta(days=7)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def build_api_url(api_url: str, path: str, params: Optional[dict] = None) -> str:
    """Build a validated API URL with encoded query parameters."""
    base_url = api_url.rstrip('/')
    parsed = urlparse(base_url)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.netloc.startswith('.'):
        raise ValueError(f"Invalid API URL configured: {api_url}")

    url = f"{base_url}/{path.lstrip('/')}" if path else base_url
    if params:
        url = f"{url}?{urlencode(params)}"
    return url


def build_verification_url(api_url: str, email: str, token: str) -> str:
    return build_api_url(api_url, '/verify', {'email': email, 'token': token})


def token_is_expired(subscriber: dict, now: Optional[datetime] = None) -> bool:
    """Check explicit expiry, falling back to the verification request timestamp."""
    now = now or utc_now()
    expiry_value = subscriber.get('verification_expires_at')

    if expiry_value:
        try:
            expiry = datetime.fromisoformat(expiry_value.replace('Z', '+00:00'))
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            return now >= expiry
        except (TypeError, ValueError):
            logger.warning("Subscriber has an invalid verification expiry timestamp")
            return True

    requested_value = subscriber.get('verification_requested_at') or subscriber.get('created_at')
    if not requested_value:
        return False

    try:
        requested = datetime.fromisoformat(requested_value.replace('Z', '+00:00'))
        if requested.tzinfo is None:
            requested = requested.replace(tzinfo=timezone.utc)
        return now >= requested + VERIFICATION_TOKEN_TTL
    except (TypeError, ValueError):
        logger.warning("Subscriber has an invalid verification request timestamp")
        return True


def ensure_active_verification_token(table, subscriber: dict) -> str:
    """Reuse a valid token or atomically replace a missing or expired token."""
    verification_token = subscriber.get('verification_token')
    if verification_token and not token_is_expired(subscriber):
        return verification_token

    verification_token = secrets.token_urlsafe(32)
    now = utc_now()
    table.update_item(
        Key={'email': subscriber['email']},
        UpdateExpression='SET verification_token = :token, verification_requested_at = :now, '
                         'verification_expires_at = :expires_at',
        ConditionExpression='#status = :pending',
        ExpressionAttributeNames={'#status': 'status'},
        ExpressionAttributeValues={
            ':token': verification_token,
            ':now': now.isoformat(),
            ':expires_at': (now + VERIFICATION_TOKEN_TTL).isoformat(),
            ':pending': 'pending',
        },
    )
    return verification_token


def select_bulk_verification_subscribers(subscribers: List[dict], expired_only: bool) -> List[dict]:
    """Select all candidates or only records that still need token rotation."""
    if not expired_only:
        return subscribers
    return [
        subscriber
        for subscriber in subscribers
        if not subscriber.get('verification_token') or token_is_expired(subscriber)
    ]


def query_subscribers_by_status(table, status: str) -> List[dict]:
    """Read every subscriber for a status using the table's status index."""
    items = []
    query_args = {
        'IndexName': 'StatusIndex',
        'KeyConditionExpression': Key('status').eq(status),
    }

    while True:
        response = table.query(**query_args)
        items.extend(response.get('Items', []))
        last_key = response.get('LastEvaluatedKey')
        if not last_key:
            return items
        query_args['ExclusiveStartKey'] = last_key


def parse_request_body(event: dict) -> dict:
    body = event.get('body') or ''
    content_type = (event.get('headers') or {}).get('content-type', '')
    if not content_type:
        content_type = (event.get('headers') or {}).get('Content-Type', '')

    if 'application/json' in content_type:
        return json.loads(body)
    return {key: values[0] for key, values in parse_qs(body).items() if values}


def html_page(title: str, heading: str, message: str, extra_html: str = '') -> dict:
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'text/html',
            'Access-Control-Allow-Origin': '*',
        },
        'body': f'''<!DOCTYPE html>
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
</html>''',
    }


def safe_http_url(value: str) -> str:
    """Return a safe absolute web URL or an empty string."""
    parsed = urlparse(str(value).strip())
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        return ''
    return parsed.geturl()


def build_html_newsletter_data(json_content: Dict) -> dict:
    """Create escaped values used only by the newsletter HTML template."""
    def escaped_papers(papers: List[dict]) -> List[dict]:
        return [
            {
                'title': html.escape(str(paper.get('title', ''))),
                'summary': html.escape(str(paper.get('summary', ''))),
                'link': html.escape(safe_http_url(paper.get('link', '')), quote=True),
            }
            for paper in papers
        ]

    return {
        'overview_html': html.escape(str(json_content.get('overview', ''))),
        'featured_papers_html': escaped_papers(json_content.get('featured_papers', [])),
        'additional_papers_html': escaped_papers(json_content.get('additional_papers', [])),
    }


def is_valid_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def create_api_response(status_code: int, body: dict) -> dict:
    """Create standardized API Gateway response"""
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',  # For CORS support
            'Access-Control-Allow-Methods': 'POST, GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps(body)
    }

def add_subscriber(event, context):
    """Handle new subscriber requests from API Gateway"""
    # Extract API URL from the event
    # Standard CORS headers
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST',
        'Content-Type': 'application/json'
    }
    
    api_url = get_api_url_from_event(event)
    if not api_url:
        logger.error("Could not determine API URL from event")
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({'error': 'Could not determine API URL'})
        }
    
    # Handle OPTIONS preflight request
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': ''
        }

    try:
        # Parse request body
        if event.get('body'):
            try:
                body = json.loads(event['body'])
                email = normalize_email(body.get('email', ''))
                if not email:
                    return {
                        'statusCode': 400,
                        'headers': cors_headers,
                        'body': json.dumps({'error': 'Email is required'})
                    }
                if not is_valid_email(email):
                    return {
                        'statusCode': 400,
                        'headers': cors_headers,
                        'body': json.dumps({'error': 'Invalid email format'})
                    }
            except json.JSONDecodeError:
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Invalid JSON body'})
                }
        else:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Missing request body'})
            }

        # Initialize DynamoDB client
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['SUBSCRIBERS_TABLE'])
        
        verification_token = secrets.token_urlsafe(32)
        unsubscribe_token = secrets.token_urlsafe(32)
        now = utc_now()
        timestamp = now.isoformat()
        verification_expires_at = (now + VERIFICATION_TOKEN_TTL).isoformat()

        try:
            response = table.get_item(Key={'email': email})
            if 'Item' in response:
                existing_subscriber = response['Item']
                if existing_subscriber.get('status') == 'verified':
                    return {
                        'statusCode': 400,
                        'headers': cors_headers,
                        'body': json.dumps({'error': 'Email already subscribed'})
                    }
                
                # Update existing record
                table.update_item(
                    Key={'email': email},
                    UpdateExpression='SET #status = :status, verification_token = :token, '
                                    'unsubscribe_token = :unsubscribe_token, updated_at = :timestamp, '
                                    'verification_requested_at = :timestamp, verification_expires_at = :expires_at, '
                                    'subscription_history = list_append(if_not_exists(subscription_history, :empty_list), :history) '
                                    'REMOVE reminder_sent, reminder_token',
                    ExpressionAttributeNames={
                        '#status': 'status'
                    },
                    ExpressionAttributeValues={
                        ':status': 'pending',
                        ':token': verification_token,
                        ':unsubscribe_token': unsubscribe_token,
                        ':timestamp': timestamp,
                        ':expires_at': verification_expires_at,
                        ':history': [{
                            'status': 'pending',
                            'timestamp': timestamp,
                            'action': 'resubscribe',
                            'previous_status': existing_subscriber.get('status', 'unknown')
                        }],
                        ':empty_list': []
                    }
                )
            else:
                # Create new subscriber
                table.put_item(
                    Item={
                        'email': email,
                        'status': 'pending',
                        'verification_token': verification_token,
                        'unsubscribe_token': unsubscribe_token,
                        'created_at': timestamp,
                        'updated_at': timestamp,
                        'verification_requested_at': timestamp,
                        'verification_expires_at': verification_expires_at,
                        'subscription_history': [{
                            'status': 'pending',
                            'timestamp': timestamp,
                            'action': 'initial_subscribe'
                        }]
                    }
                )
        except ClientError as e:
            logger.error(f"Error with DynamoDB operation: {str(e)}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Database operation failed'})
            }

        # Send verification email
        if not send_verification_email(
            email=email,
            verification_token=verification_token,
            sender=os.environ['SENDER_EMAIL'],
            api_url=api_url
        ):
            # Preserve the pending record so the scheduled reminder can retry delivery.
            try:
                table.update_item(
                    Key={'email': email},
                    UpdateExpression='SET verification_delivery_status = :status, updated_at = :timestamp',
                    ExpressionAttributeValues={
                        ':status': 'failed',
                        ':timestamp': utc_now().isoformat(),
                    },
                )
            except ClientError as update_error:
                logger.error(f"Could not record verification delivery failure: {update_error}")
            return {
                'statusCode': 502,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Failed to send verification email'})
            }

        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': json.dumps({
                'message': 'Subscription pending verification',
                'email': email
            })
        }

    except Exception as e:
        logger.error(f"Unexpected error in add_subscriber: {str(e)}")
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }
    
def get_api_url_from_event(event):
    """Extract API URL from the event context"""
    request_context = event.get('requestContext', {})
    domain_name = request_context.get('domainName')
    stage = request_context.get('stage')
    if domain_name and stage:
        return f"https://{domain_name}/{stage}"
    return None

def send_verification_email(email: str, verification_token: str, sender: str, api_url: str = None) -> bool:
    """Send verification email to new subscriber"""
    ses_client = boto3.client('ses')
    
    if not api_url:
        logger.error("No API URL provided for verification link")
        return False
    
    try:
        verification_url = build_verification_url(api_url, email, verification_token)

        # Get configuration set name for better deliverability
        config_set_name = os.environ.get('EMAIL_CONFIGURATION_SET')
        
        send_params = {
            'Source': sender,
            'Destination': {
                'ToAddresses': [email]
            },
            'Template': os.environ.get('VERIFICATION_TEMPLATE_NAME', 'VerificationTemplate'),
            'TemplateData': json.dumps({
                'verification_url': verification_url,
                'email': email
            })
        }
        
        # Add ConfigurationSetName if available (critical for deliverability)
        if config_set_name:
            send_params['ConfigurationSetName'] = config_set_name
        
        ses_client.send_templated_email(**send_params)
        logger.info(f"Verification email sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Error sending verification email: {str(e)}")
        return False
    
def verify_subscriber(event, context) -> dict:
    """Handle email verification requests"""
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,OPTIONS',
        'Content-Type': 'application/json'
    }
    
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': ''
        }

    try:
        params = event.get('queryStringParameters') or {}
        if not params:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Missing query parameters'})
            }
            
        requested_email = (params.get('email') or '').strip()
        email = normalize_email(requested_email)
        token = params.get('token')
        
        if not email or not token:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Missing email or token'})
            }
        
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['SUBSCRIBERS_TABLE'])
        
        try:
            response = table.get_item(Key={'email': email})

            # Preserve compatibility with records created before email normalization.
            if 'Item' not in response and requested_email != email:
                response = table.get_item(Key={'email': requested_email})
            
            if 'Item' not in response:
                return {
                    'statusCode': 404,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Subscriber not found'})
                }
                
            subscriber = response['Item']
            subscriber_email = subscriber['email']
            stored_token = subscriber.get('verification_token')
            
            if not stored_token or not secrets.compare_digest(str(stored_token), str(token)):
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Invalid verification token'})
                }
            
            if subscriber.get('status') == 'verified':
                return html_page(
                    'Already Verified',
                    'Email Already Verified',
                    'Your email subscription has already been verified.',
                )

            if subscriber.get('status') != 'pending':
                return {
                    'statusCode': 409,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Subscription is not pending verification'}),
                }

            if token_is_expired(subscriber):
                return {
                    'statusCode': 410,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Verification token has expired. Please subscribe again.'}),
                }

            verified_at = utc_now().isoformat()
            table.update_item(
                Key={'email': subscriber_email},
                UpdateExpression='SET #status = :verified, verified_at = :verified_at, updated_at = :verified_at, '
                                 'subscription_history = list_append(if_not_exists(subscription_history, :empty_list), :history)',
                ConditionExpression='#status = :pending AND verification_token = :token',
                ExpressionAttributeNames={
                    '#status': 'status'
                },
                ExpressionAttributeValues={
                    ':verified': 'verified',
                    ':pending': 'pending',
                    ':token': token,
                    ':verified_at': verified_at,
                    ':empty_list': [],
                    ':history': [{
                        'status': 'verified',
                        'timestamp': verified_at,
                        'action': 'verify',
                    }],
                }
            )
            
            return html_page(
                'Subscription Verified',
                'Subscription Verified!',
                "Thank you for subscribing. You'll start receiving newsletters soon.",
            )
            
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'ConditionalCheckFailedException':
                return {
                    'statusCode': 409,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Subscription state changed. Please request a new link.'}),
                }
            logger.error(f"DynamoDB error: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error verifying subscriber: {e}")
        raise

def publish_verification_metrics(success_count: int, failure_count: int, batch_size: int) -> None:
    """Publish the custom metrics referenced by the verification dashboard."""
    try:
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='Newsletter/Verification',
            MetricData=[
                {'MetricName': 'ReminderEmailSuccess', 'Value': success_count, 'Unit': 'Count'},
                {'MetricName': 'ReminderEmailFailure', 'Value': failure_count, 'Unit': 'Count'},
                {'MetricName': 'ReminderBatchSize', 'Value': batch_size, 'Unit': 'Count'},
            ],
        )
    except Exception as metric_error:
        logger.error(f"Could not publish verification metrics: {metric_error}")


def publish_verification_system_error() -> None:
    try:
        boto3.client('cloudwatch').put_metric_data(
            Namespace='Newsletter/Verification',
            MetricData=[{'MetricName': 'ReminderSystemError', 'Value': 1, 'Unit': 'Count'}],
        )
    except Exception as metric_error:
        logger.error(f"Could not publish verification system error metric: {metric_error}")


def check_and_resend_verifications(event, context):
    """
    Send a single verification reminder to subscribers who:
    - Have been unverified for > 48 hours
    - Have never received a reminder
    """
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['SUBSCRIBERS_TABLE'])
        
        cutoff_time = utc_now() - timedelta(hours=48)
        pending_subscribers = query_subscribers_by_status(table, 'pending')
        subscribers_to_remind = []
        for subscriber in pending_subscribers:
            if subscriber.get('reminder_sent'):
                continue

            requested_at = subscriber.get('verification_requested_at') or subscriber.get('created_at')
            if not requested_at:
                continue

            try:
                requested_time = datetime.fromisoformat(requested_at.replace('Z', '+00:00'))
                if requested_time.tzinfo is None:
                    requested_time = requested_time.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                logger.warning(f"Skipping subscriber with invalid verification timestamp: {subscriber.get('email', 'unknown')}")
                continue

            if requested_time < cutoff_time:
                subscribers_to_remind.append(subscriber)
        
        api_url = (event or {}).get('api_url') or os.environ.get('API_URL')
        if not api_url:
            raise ValueError('No API URL was provided to the verification reminder')
        sender_email = os.environ['SENDER_EMAIL']
        build_api_url(api_url, '')
        
        success_count = 0
        failure_count = 0
        
        for item in subscribers_to_remind:
            email = item.get('email', 'unknown')
            try:
                email = item['email']
                verification_token = ensure_active_verification_token(table, item)

                logger.info(f"Processing reminder for {email}")

                if send_verification_email(
                    email=email,
                    verification_token=verification_token,
                    sender=sender_email,
                    api_url=api_url
                ):
                    table.update_item(
                        Key={'email': email},
                        UpdateExpression='SET reminder_sent = :now',
                        ConditionExpression='#status = :pending AND verification_token = :token',
                        ExpressionAttributeNames={'#status': 'status'},
                        ExpressionAttributeValues={
                            ':now': utc_now().isoformat(),
                            ':pending': 'pending',
                            ':token': verification_token,
                        },
                    )
                    success_count += 1
                    logger.info(f"Successfully sent verification reminder to {email}")
                else:
                    failure_count += 1
                    logger.error(f"Failed to send verification reminder to {email}")
                
            except Exception as e:
                failure_count += 1
                logger.error(f"Error processing subscriber {email}: {str(e)}")
        
        batch_size = len(subscribers_to_remind)
        publish_verification_metrics(success_count, failure_count, batch_size)
        logger.info(f"Reminder process completed. Successes: {success_count}, Failures: {failure_count}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Verification reminder process completed',
                'success_count': success_count,
                'failure_count': failure_count,
                'batch_size': batch_size
            })
        }
        
    except Exception as e:
        logger.error(f"Error in check_and_resend_verifications: {e}")
        publish_verification_system_error()
        raise

def unsubscribe_handler(event, context) -> dict:
    """Show an unsubscribe confirmation on GET and apply it on POST."""
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        'Content-Type': 'application/json'
    }
    
    # Handle OPTIONS preflight request
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': ''
        }

    try:
        method = event.get('httpMethod', 'GET').upper()
        try:
            params = parse_request_body(event) if method == 'POST' else (event.get('queryStringParameters') or {})
        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Invalid request body'}),
            }

        requested_email = (params.get('email') or '').strip()
        email = normalize_email(requested_email)
        token = params.get('token')
        if not email or not token:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Missing email or unsubscribe token'}),
            }

        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['SUBSCRIBERS_TABLE'])

        try:
            response = table.get_item(Key={'email': email})
            if 'Item' not in response and requested_email != email:
                response = table.get_item(Key={'email': requested_email})
            
            if 'Item' not in response:
                return {
                    'statusCode': 404,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Subscriber not found'})
                }
                
            subscriber = response['Item']
            subscriber_email = subscriber['email']
            stored_token = subscriber.get('unsubscribe_token')
            if not stored_token or not secrets.compare_digest(str(stored_token), str(token)):
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Invalid unsubscribe token'}),
                }

            if subscriber.get('status') == 'unsubscribed':
                return html_page(
                    'Already Unsubscribed',
                    'Email Already Unsubscribed',
                    'This email has already been unsubscribed from the newsletter.',
                )

            if method == 'GET':
                api_url = get_api_url_from_event(event)
                if not api_url:
                    return {
                        'statusCode': 500,
                        'headers': cors_headers,
                        'body': json.dumps({'error': 'Could not determine API URL'}),
                    }

                action_url = html.escape(build_api_url(api_url, '/unsubscribe'), quote=True)
                form = f'''<form method="post" action="{action_url}">
    <input type="hidden" name="email" value="{html.escape(subscriber_email, quote=True)}">
    <input type="hidden" name="token" value="{html.escape(token, quote=True)}">
    <button type="submit">Confirm unsubscribe</button>
</form>'''
                return html_page(
                    'Confirm Unsubscribe',
                    'Confirm Unsubscribe',
                    'Confirm that you want to stop receiving the AI Papers Newsletter.',
                    form,
                )

            if subscriber.get('status') != 'verified':
                return {
                    'statusCode': 409,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Only verified subscriptions can be unsubscribed'}),
                }

            unsubscribed_at = utc_now().isoformat()
            table.update_item(
                Key={'email': subscriber_email},
                UpdateExpression='SET #status = :unsubscribed, unsubscribed_at = :unsubscribed_at, updated_at = :unsubscribed_at, '
                                 'subscription_history = list_append(if_not_exists(subscription_history, :empty_list), :history)',
                ConditionExpression='#status = :verified AND unsubscribe_token = :token',
                ExpressionAttributeNames={
                    '#status': 'status'
                },
                ExpressionAttributeValues={
                    ':unsubscribed': 'unsubscribed',
                    ':verified': 'verified',
                    ':token': token,
                    ':unsubscribed_at': unsubscribed_at,
                    ':empty_list': [],
                    ':history': [{
                        'status': 'unsubscribed',
                        'timestamp': unsubscribed_at,
                        'action': 'unsubscribe',
                    }],
                }
            )
            return html_page(
                'Unsubscribed',
                'Unsubscribed Successfully!',
                'You have been unsubscribed from the AI Papers Newsletter.',
            )
            
        except ClientError as e:
            if e.response.get('Error', {}).get('Code') == 'ConditionalCheckFailedException':
                return {
                    'statusCode': 409,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Subscription state changed. Refresh and try again.'}),
                }
            logger.error(f"DynamoDB error: {e}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Database error'})
            }
            
    except Exception as e:
        logger.error(f"Error unsubscribing: {str(e)}")
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({'error': 'Internal server error'})
        }
    
def create_or_update_template(ses_client):
    """Create or update SES template with subscribe button"""
    template_name = os.environ.get('NEWSLETTER_TEMPLATE_NAME', 'NewsletterTemplate')
    template_data = {
        'Template': {
            'TemplateName': template_name,
            'SubjectPart': 'AI Papers Newsletter: {{featured_count}} New Papers This Week',
            'HtmlPart': '''
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="color-scheme" content="light dark">
    <meta name="supported-color-schemes" content="light dark">
    <!--[if !mso]><!-->
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <!--<![endif]-->
    <title>AI Papers Newsletter</title>
    <style type="text/css">
        /* Reset styles */
        body, #bodyTable { margin:0; padding:0; width:100% !important; }
        img { border:0; height:auto; line-height:100%; outline:none; text-decoration:none; }
        table { border-collapse:collapse !important; }
        
        /* Client-specific styles */
        #outlook a { padding:0; }
        .ReadMsgBody { width:100%; }
        .ExternalClass { width:100%; }
        .ExternalClass, .ExternalClass p, .ExternalClass span, .ExternalClass font, 
        .ExternalClass td, .ExternalClass div { line-height:100%; }
        
        /* Dark mode styles */
        @media (prefers-color-scheme: dark) {
            .dark-bg { background-color: #1a1a1a !important; }
            .dark-text { color: #ffffff !important; }
            .paper { border-left-color: #4a9eff !important; background-color: #232323 !important; }
            .paper h3 a { color: #4a9eff !important; }
            .header { background-color: #1a1a1a !important; }
            .body { background-color: #1a1a1a !important; }
            .container { background-color: #232323 !important; }
            .footer { border-color: #333333 !important; color: #ffffff !important; }
            .metadata { color: #cccccc !important; }
        }
        
        /* Responsive styles */
        @media screen and (max-width: 600px) {
            .container { width: 100% !important; }
            .paper { margin: 15px !important; }
            .header { padding: 15px !important; }
            .social-button { display: block !important; margin: 10px auto !important; }
        }

        /* Base styles */
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333333;
            background-color: #f8f9fa;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
            background-color: #ffffff;
        }
        .header {
            background-color: #f8f9fa;
            padding: 30px;
            text-align: center;
            border-bottom: 1px solid #eeeeee;
        }
        .paper {
            margin: 20px;
            padding: 20px;
            border-left: 4px solid #007bff;
            background-color: #f8f9fa;
        }
        .paper h3 {
            margin-top: 0;
            margin-bottom: 15px;
            color: #007bff;
        }
        .paper a {
            color: #007bff;
            text-decoration: none;
        }
        .paper a:hover {
            text-decoration: underline;
        }
        .footer {
            margin-top: 30px;
            padding: 20px;
            border-top: 1px solid #eeeeee;
            font-size: 14px;
            text-align: center;
            color: #666666;
        }
        .metadata {
            font-size: 14px;
            color: #666666;
            margin-bottom: 10px;
        }
        .social-section {
            padding: 20px;
            text-align: center;
            border-top: 1px solid #eeeeee;
            margin-top: 20px;
        }
        .social-button {
            display: inline-block;
            padding: 6px 12px;
            margin: 0 5px;
            font-size: 12px;
            border-radius: 4px;
            text-decoration: none;
        }
    </style>
    <!--[if mso]>
    <style type="text/css">
        .fallback-font {
            font-family: Arial, sans-serif;
        }
    </style>
    <![endif]-->
</head>
<body class="dark-bg">
    <!-- Preview Text -->
    <div style="display: none; max-height: 0px; overflow: hidden;">
        Your weekly AI papers digest - {{featured_count}} new papers on practical AI implementation and regulation
        &nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;&nbsp;&zwnj;
    </div>

    <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="100%" id="bodyTable">
        <tr>
            <td align="center" valign="top">
                <table role="presentation" border="0" cellpadding="0" cellspacing="0" width="600" class="container">
                    <!-- Header -->
                    <tr>
                        <td align="center" valign="top" class="header">
                            <h1 class="dark-text fallback-font">AI Papers Newsletter</h1>
                        </td>
                    </tr>

                    <!-- Overview -->
                    <tr>
                        <td align="left" valign="top" style="padding: 20px;" class="dark-text fallback-font">
                            {{{overview_html}}}
                        </td>
                    </tr>

                    <!-- Featured Papers -->
                    <tr>
                        <td align="left" valign="top" style="padding: 0 20px;">
                            <h2 class="dark-text fallback-font">Featured Research</h2>
                            {{#each featured_papers_html}}
                            <div class="paper">
                                <h3 class="fallback-font"><a href="{{{link}}}">{{{title}}}</a></h3>
                                <p class="dark-text fallback-font">{{{summary}}}</p>
                            </div>
                            {{/each}}
                        </td>
                    </tr>

                    <!-- Additional Papers -->
                    {{#if additional_papers_html}}
                    <tr>
                        <td align="left" valign="top" style="padding: 0 20px;">
                            <h2 class="dark-text fallback-font">Additional Papers</h2>
                            {{#each additional_papers_html}}
                            <div class="metadata fallback-font">
                                • <a href="{{{link}}}">{{{title}}}</a>
                            </div>
                            {{/each}}
                        </td>
                    </tr>

                    <!-- Social Sharing Section -->
                    <tr>
                        <td align="center" valign="top" class="social-section">
                            <p style="margin-bottom: 15px; color: #666666; font-size: 14px;">Share this newsletter:</p>
                            <a href="https://twitter.com/intent/tweet?text=Check%20out%20this%20week%27s%20AI%20Papers%20Newsletter%20featuring%20{{featured_count}}%20new%20papers%20on%20AI%20research%20and%20implementation&amp;url={{{subscribe_url_html}}}" target="_blank" class="social-button" style="background-color: #000000; color: #ffffff;">Share on X</a>
                            <a href="https://bsky.app/intent/compose?text=Check%20out%20this%20week%27s%20AI%20Papers%20Newsletter%20featuring%20{{featured_count}}%20new%20papers%20on%20AI%20research%20and%20implementation%20{{{subscribe_url_html}}}" target="_blank" class="social-button" style="background-color: #0085ff; color: #ffffff;">Share on Bluesky</a>
                            <a href="https://www.linkedin.com/sharing/share-offsite/?url={{{subscribe_url_html}}}" target="_blank" class="social-button" style="background-color: #0077b5; color: #ffffff;">Share on LinkedIn</a>
                            <a href="mailto:?subject=AI%20Papers%20Newsletter&amp;body=I%20thought%20you%20might%20be%20interested%20in%20this%20AI%20research%20newsletter.%20Check%20it%20out%20at%20{{{subscribe_url_html}}}" class="social-button" style="background-color: #666666; color: #ffffff;">Share via Email</a>
                        </td>
                    </tr>
                    {{/if}}

                    <!-- Footer -->
                    <tr>
                        <td align="center" valign="top" class="footer">
                            <p class="dark-text fallback-font">
                                <a href="{{{unsubscribe_url_html}}}" style="color: #666666;">Unsubscribe</a>
                            </p>
                            <p class="dark-text fallback-font">
                                Was this forwarded to you? <a href="{{{subscribe_url_html}}}" style="color: #666666;">Subscribe or contact here</a>
                            </p>
                            <p class="dark-text fallback-font">
                                Check out the <a href="https://wesleypasfield.substack.com/" style="color: #666666;">blog</a> for more AI related content
                            </p>
                            <p class="dark-text fallback-font">
                                You received this because you subscribed to the AI Papers Newsletter.<br>
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>''',
            'TextPart': '''
AI Papers Newsletter

{{overview}}

Featured Papers:
{{#each featured_papers}}
* {{title}}
{{summary}}
Link: {{{link}}}

{{/each}}
{{#if additional_papers}}
Additional Papers:
{{#each additional_papers}}
* {{title}} - {{{link}}}
{{/each}}
{{/if}}

Share this newsletter:
- Twitter/X: https://twitter.com/intent/tweet?text=Check%20out%20this%20week%27s%20AI%20Papers%20Newsletter&url={{subscribe_url}}
- Bluesky: https://bsky.app/intent/compose?text=Check%20out%20this%20week%27s%20AI%20Papers%20Newsletter%20{{subscribe_url}}
- LinkedIn: https://www.linkedin.com/sharing/share-offsite/?url={{subscribe_url}}

Check out the blog for more AI related content: https://wesleypasfield.substack.com/
Was this forwarded to you? Subscribe or contact at: {{{subscribe_url}}}
To unsubscribe: {{{unsubscribe_url}}}'''
        }
    }

    try:
        ses_client.create_template(Template=template_data['Template'])
    except ClientError as e:
        if e.response['Error']['Code'] == 'AlreadyExists':
            ses_client.update_template(Template=template_data['Template'])
        else:
            raise

def get_subscribers(table_name: str) -> List[dict]:
    """Retrieve every verified subscriber and ensure an unsubscribe token exists."""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    try:
        subscribers = query_subscribers_by_status(table, 'verified')
        for subscriber in subscribers:
            if subscriber.get('unsubscribe_token'):
                continue

            token = secrets.token_urlsafe(32)
            response = table.update_item(
                Key={'email': subscriber['email']},
                UpdateExpression='SET unsubscribe_token = if_not_exists(unsubscribe_token, :token)',
                ExpressionAttributeValues={':token': token},
                ReturnValues='ALL_NEW',
            )
            subscriber['unsubscribe_token'] = response['Attributes']['unsubscribe_token']
        return subscribers
    except ClientError as e:
        logger.error(f"Error querying DynamoDB table: {e}")
        raise

def test_prod_check(subscribers: List[dict], test_email: str) -> List[dict]:
    """
    Returns appropriate recipient list based on test mode settings
    """
    test_mode = os.environ.get('EMAIL_TEST_MODE', 'false').lower() == 'true'

    if test_mode:
        if not test_email or not is_valid_email(test_email):
            raise ValueError('EMAIL_TEST_ADDRESS must contain a valid email when test mode is enabled')
        normalized_test_email = normalize_email(test_email)
        logger.info(f"Test mode enabled: Sending single test email instead of {len(subscribers)} subscriber emails")
        existing = next(
            (subscriber for subscriber in subscribers if subscriber['email'] == normalized_test_email),
            None,
        )
        return [existing or {'email': normalized_test_email, 'unsubscribe_token': 'test-mode'}]

    return subscribers

def send_email_batch(ses_client, source: str, subscribers: List[dict], json_content: Dict, unsubscribe_url: str, subscribe_url: str) -> Dict:
    results = {'successful': 0, 'failed': 0, 'failures': []}

    subscribers = test_prod_check(subscribers, os.environ.get('EMAIL_TEST_ADDRESS', ''))
    
    default_template_data = {
        'date': utc_now().strftime('%B %d, %Y'),
        'generated_date': json_content['metadata']['generated_date'],
        'featured_count': json_content['metadata']['featured_papers_count'],
        'overview': json_content['overview'],
        'featured_papers': json_content['featured_papers'],
        'additional_papers': json_content.get('additional_papers', []),
        'unsubscribe_url': unsubscribe_url,
        'subscribe_url': subscribe_url,
        'email': ''
    }
    default_template_data.update(build_html_newsletter_data(json_content))
    default_template_data.update({
        'unsubscribe_url_html': html.escape(unsubscribe_url, quote=True),
        'subscribe_url_html': html.escape(subscribe_url, quote=True),
    })

    batch_size = 50
    for i in range(0, len(subscribers), batch_size):
        batch_subscribers = subscribers[i:i + batch_size]
        destinations = []
        
        for subscriber in batch_subscribers:
            email = subscriber['email']
            unsubscribe_link = build_api_url(
                unsubscribe_url,
                '',
                {
                    'email': email,
                    'token': subscriber['unsubscribe_token'],
                },
            )
            destinations.append({
                'Destination': {
                    'ToAddresses': [email]
                },
                'ReplacementTemplateData': json.dumps({
                    'email': email,
                    'unsubscribe_url': unsubscribe_link,
                    'subscribe_url': subscribe_url,
                    'unsubscribe_url_html': html.escape(unsubscribe_link, quote=True),
                    'subscribe_url_html': html.escape(subscribe_url, quote=True),
                }),
                'ReplacementTags': [
                    {
                        'Name': 'EmailType',
                        'Value': 'newsletter'
                    }
                ]
            })

        try:
            config_set_name = os.environ.get('EMAIL_CONFIGURATION_SET')
            if not config_set_name:
                raise ValueError("EMAIL_CONFIGURATION_SET environment variable not set")

            response = ses_client.send_bulk_templated_email(
                Source=source,
                Template=os.environ.get('NEWSLETTER_TEMPLATE_NAME', 'NewsletterTemplate'),
                DefaultTemplateData=json.dumps(default_template_data),
                Destinations=destinations,
                ConfigurationSetName=config_set_name
            )

            for status in response.get('Status', []):
                if 'Error' in status:
                    results['failed'] += 1
                    results['failures'].append({
                        'email': status.get('Destination', {}).get('ToAddresses', ['Unknown'])[0],
                        'error': status['Error']
                    })
                else:
                    results['successful'] += 1

        except Exception as e:
            logger.error(f"Error sending batch: {str(e)}")
            results['failed'] += len(batch_subscribers)
            results['failures'].append({
                'batch': [subscriber['email'] for subscriber in batch_subscribers],
                'error': str(e)
            })
            
    return results

def send_newsletter(json_content: Dict, table_name: str, sender: str, api_url: str) -> Dict:
    """Main function to send newsletter to all subscribers"""
    try:
        ses_client = boto3.client('ses')

        # Construct unsubscribe URL using AWS environment variables
        unsubscribe_url = build_api_url(api_url, '/unsubscribe')
        subscribe_url = build_api_url(os.environ['SUBSCRIBE_LANDING_PAGE'], '')

        logger.info(f'unsubscribe url: {unsubscribe_url}')
        logger.info(f'subscribe url: {subscribe_url}')
        
        # Get subscribers
        subscribers = get_subscribers(table_name)
        if not subscribers:
            logger.warning("No subscribers found")
            return {'message': 'No subscribers found'}
            
        # Create/update SES template with the new format
        create_or_update_template(ses_client)
        
        # Send emails using the template
        results = send_email_batch(
            ses_client=ses_client,
            source=sender,
            subscribers=subscribers,
            json_content=json_content,
            unsubscribe_url=unsubscribe_url,
            subscribe_url= subscribe_url
        )
        
        logger.info(f"Newsletter sending complete. Results: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Error sending newsletter: {str(e)}")
        raise

def get_subscriber_counts(table_name: str) -> dict:
    """Query complete subscriber counts, including paginated results."""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    try:
        def count_status(status: str) -> int:
            count = 0
            query_args = {
                'IndexName': 'StatusIndex',
                'KeyConditionExpression': Key('status').eq(status),
                'Select': 'COUNT',
            }
            while True:
                response = table.query(**query_args)
                count += response.get('Count', 0)
                last_key = response.get('LastEvaluatedKey')
                if not last_key:
                    return count
                query_args['ExclusiveStartKey'] = last_key

        status_counts = {
            'verified': count_status('verified'),
            'pending': count_status('pending'),
            'unsubscribed': count_status('unsubscribed'),
        }
        
        metrics = {
            'total_subscribers': sum(status_counts.values()),
            'status_counts': status_counts,
            'metadata': {
                'timestamp': utc_now().isoformat(),
                'table_name': os.environ['SUBSCRIBERS_TABLE']
            }
        }
        
        logger.info(f"Retrieved metrics: {json.dumps(metrics, indent=2)}")
        return metrics
        
    except ClientError as e:
        logger.error(f"Error querying DynamoDB: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def sub_status_checker(event, context):
    """
    Handler for subscriber metrics Lambda
    Returns subscriber counts by status with CORS headers
    """
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,OPTIONS',
        'Content-Type': 'application/json'
    }
    
    # Handle OPTIONS request
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }

    dynamo_table_name = os.environ['SUBSCRIBERS_TABLE']
    
    try:
        metrics = get_subscriber_counts(dynamo_table_name)
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(metrics, indent=2)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def bulk_verifier(event, context):
    """
    Manually triggered function to send verification reminders to pending subscribers
    who have subscribed within the last 30 days
    """
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'POST,OPTIONS',
        'Content-Type': 'application/json'
    }
    
    # Handle OPTIONS request
    if event.get('httpMethod') == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    try:
        # Calculate cutoff date (30 days ago) in UTC
        cutoff_date = utc_now() - timedelta(days=30)
        
        # Get all pending subscribers
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['SUBSCRIBERS_TABLE'])
        
        pending_subscribers = query_subscribers_by_status(table, 'pending')
        
        # Filter to only include subscribers who subscribed within the last 30 days
        recent_subscribers = []
        filtered_count = 0
        
        for subscriber in pending_subscribers:
            created_at = subscriber.get('created_at')
            if created_at:
                try:
                    # Parse ISO timestamp and compare
                    # Handle both 'Z' suffix and timezone offset formats
                    created_str = created_at.replace('Z', '+00:00')
                    created_datetime = datetime.fromisoformat(created_str)
                    
                    # Ensure timezone-aware datetime
                    if created_datetime.tzinfo is None:
                        created_datetime = created_datetime.replace(tzinfo=timezone.utc)
                    
                    # Only include if created within last 30 days
                    if created_datetime >= cutoff_date:
                        recent_subscribers.append(subscriber)
                    else:
                        filtered_count += 1
                        logger.info(f"Filtered out {subscriber.get('email', 'unknown')} - subscribed {created_at} (older than 30 days)")
                except (ValueError, AttributeError) as e:
                    # If date parsing fails, log and skip
                    logger.warning(f"Could not parse created_at for {subscriber.get('email', 'unknown')}: {created_at}, error: {str(e)}")
                    # Include it to be safe (better to send than miss)
                    recent_subscribers.append(subscriber)
            else:
                # If no created_at, include it to be safe
                logger.warning(f"Subscriber {subscriber.get('email', 'unknown')} missing created_at field")
                recent_subscribers.append(subscriber)
        
        if not recent_subscribers:
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'message': 'No pending subscribers found within the last 30 days',
                    'count': 0,
                    'total_pending': len(pending_subscribers),
                    'filtered_out': filtered_count
                })
            }

        recent_subscribers = select_bulk_verification_subscribers(
            recent_subscribers,
            bool((event or {}).get('expired_tokens_only')),
        )

        if not recent_subscribers:
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'message': 'No matching pending subscribers require verification emails',
                    'sent_count': 0,
                    'total_pending': len(pending_subscribers),
                    'recent_subscribers': 0,
                    'filtered_out_older_than_30_days': filtered_count,
                    'failures': [],
                }, indent=2),
            }
        
        # Send emails
        api_url = get_api_url_from_event(event) or os.environ.get('API_URL')
        if not api_url:
            raise ValueError('Could not determine the API URL for verification links')
        build_api_url(api_url, '')
        sent_count = 0
        failures = []
        send_rate = max(1.0, float(os.environ.get('BULK_VERIFICATION_SEND_RATE', '10')))
        send_interval = 1 / send_rate
        
        for subscriber in recent_subscribers:
            try:
                verification_token = ensure_active_verification_token(table, subscriber)
                if send_verification_email(
                    email=subscriber['email'],
                    verification_token=verification_token,
                    sender=os.environ['SENDER_EMAIL'],
                    api_url=api_url,
                ):
                    sent_count += 1
                    logger.info(f"Sent reminder to {subscriber['email']}")
                else:
                    failures.append({
                        'email': subscriber['email'],
                        'error': 'SES rejected the verification email',
                    })
                
            except Exception as e:
                failures.append({
                    'email': subscriber['email'],
                    'error': str(e)
                })
                logger.error(f"Failed to send reminder to {subscriber['email']}: {str(e)}")
            finally:
                time.sleep(send_interval)
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'message': 'Reminder emails processed',
                'sent_count': sent_count,
                'total_pending': len(pending_subscribers),
                'recent_subscribers': len(recent_subscribers),
                'filtered_out_older_than_30_days': filtered_count,
                'failures': failures if failures else None
            }, indent=2)
        }
        
    except Exception as e:
        logger.error(f"Error processing reminders: {str(e)}")
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }
