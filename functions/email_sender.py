import boto3
from botocore.exceptions import ClientError
import json
import os
from typing import Dict, List
import logging
from datetime import datetime, timedelta, timezone
import secrets
import re
import time

logger = logging.getLogger(__name__)

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
                email = body.get('email')
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
        timestamp = datetime.now().isoformat()

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
                    UpdateExpression='SET #status = :status, verification_token = :token, updated_at = :timestamp, ' 
                                    'subscription_history = list_append(if_not_exists(subscription_history, :empty_list), :history)',
                    ExpressionAttributeNames={
                        '#status': 'status'
                    },
                    ExpressionAttributeValues={
                        ':status': 'pending',
                        ':token': verification_token,
                        ':timestamp': timestamp,
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
                        'created_at': timestamp,
                        'updated_at': timestamp,
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
            # If email fails, delete the subscriber record
            try:
                table.delete_item(Key={'email': email})
            except ClientError:
                pass
            return {
                'statusCode': 500,
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
    
    verification_url = f"{api_url}/verify?email={email}&token={verification_token}"
    
    try:
        # Get configuration set name for better deliverability
        config_set_name = os.environ.get('EMAIL_CONFIGURATION_SET')
        
        send_params = {
            'Source': sender,
            'Destination': {
                'ToAddresses': [email]
            },
            'Template': 'VerificationTemplate',
            'TemplateData': json.dumps({
                'verification_url': verification_url,
                'email': email
            })
        }
        
        # Add ConfigurationSetName if available (critical for deliverability)
        if config_set_name:
            send_params['ConfigurationSetName'] = config_set_name
        
        response = ses_client.send_templated_email(**send_params)
        logger.info(f"Verification email sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Error sending verification email: {str(e)}")
        return False
    
def verify_subscriber(event, context) -> dict:
    """Handle email verification requests"""
    # Use same CORS headers as working add_subscriber function
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,OPTIONS',
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
        # Extract parameters
        params = event.get('queryStringParameters', {})
        if not params:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Missing query parameters'})
            }
            
        email = params.get('email')
        token = params.get('token')
        
        if not email or not token:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Missing email or token'})
            }
        
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['SUBSCRIBERS_TABLE'])
        
        # Get subscriber
        try:
            response = table.get_item(
                Key={'email': email}
            )
            
            if 'Item' not in response:
                return {
                    'statusCode': 404,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Subscriber not found'})
                }
                
            subscriber = response['Item']
            
            # Verify token
            if subscriber['verification_token'] != token:
                return {
                    'statusCode': 400,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Invalid verification token'})
                }
            
            # Check if already verified
            if subscriber.get('status') == 'verified':
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'text/html',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': '''
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>Already Verified</title>
                            <style>
                                body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 600px; margin: 40px auto; padding: 20px; text-align: center; }
                                .message { color: #666; margin-bottom: 20px; }
                            </style>
                        </head>
                        <body>
                            <h1>Email Already Verified</h1>
                            <p class="message">Your email subscription has already been verified.</p>
                            <p>You should be receiving our newsletters. Check your spam folder if you haven't seen them.</p>
                        </body>
                        </html>
                    '''
                }
                
            # Update status to verified
            table.update_item(
                Key={'email': email},
                UpdateExpression='SET #status = :status, verified_at = :verified_at',
                ExpressionAttributeNames={
                    '#status': 'status'
                },
                ExpressionAttributeValues={
                    ':status': 'verified',
                    ':verified_at': datetime.now().isoformat()
                }
            )
            
            # Return success with HTML
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'text/html',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': '''
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Subscription Verified</title>
                        <style>
                            body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 600px; margin: 40px auto; padding: 20px; text-align: center; }
                            .success { color: #28a745; font-size: 24px; margin-bottom: 20px; }
                            .message { color: #666; margin-bottom: 20px; }
                        </style>
                    </head>
                    <body>
                        <h1 class="success">Subscription Verified!</h1>
                        <p class="message">Thank you for subscribing to the AI Papers Newsletter.</p>
                        <p>You'll start receiving newsletters in your inbox soon.</p>
                    </body>
                    </html>
                '''
            }
            
        except ClientError as e:
            logger.error(f"DynamoDB error: {str(e)}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Database error'})
            }
            
    except Exception as e:
        logger.error(f"Error verifying subscriber: {str(e)}")
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({'error': 'Internal server error'})
        }

def check_and_resend_verifications(event, context):
    """
    Send a single verification reminder to subscribers who:
    - Have been unverified for > 48 hours
    - Have never received a reminder
    """
    try:
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['SUBSCRIBERS_TABLE'])
        
        # Calculate cutoff time (48 hours ago)
        cutoff_time = (datetime.now() - timedelta(hours=48)).isoformat()
        
        # Query for subscribers who need reminders
        response = table.scan(
            FilterExpression='#status = :status AND created_at < :cutoff AND attribute_not_exists(reminder_sent)',
            ExpressionAttributeNames={
                '#status': 'status'
            },
            ExpressionAttributeValues={
                ':status': 'pending',
                ':cutoff': cutoff_time
            }
        )
        
        api_url = os.environ['API_URL']
        sender_email = os.environ['SENDER_EMAIL']
        
        success_count = 0
        failure_count = 0
        
        for item in response.get('Items', []):
            try:
                email = item['email']
                # Generate new verification token for security
                verification_token = secrets.token_urlsafe(32)
                
                logger.info(f"Processing reminder for {email}")
                
                # Update token before sending email
                table.update_item(
                    Key={'email': email},
                    UpdateExpression='SET verification_token = :token, reminder_sent = :now, reminder_token = :token',
                    ExpressionAttributeValues={
                        ':token': verification_token,
                        ':now': datetime.now().isoformat()
                    }
                )
                
                # Send verification reminder
                if send_verification_email(
                    email=email,
                    verification_token=verification_token,
                    sender=sender_email,
                    api_url=api_url
                ):
                    success_count += 1
                    logger.info(f"Successfully sent verification reminder to {email}")
                else:
                    failure_count += 1
                    logger.error(f"Failed to send verification reminder to {email}")
                
            except Exception as e:
                failure_count += 1
                logger.error(f"Error processing subscriber {email}: {str(e)}")
        
        logger.info(f"Reminder process completed. Successes: {success_count}, Failures: {failure_count}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Verification reminder process completed',
                'success_count': success_count,
                'failure_count': failure_count,
                'batch_size': len(response.get('Items', []))
            })
        }
        
    except Exception as e:
        logger.error(f"Error in check_and_resend_verifications: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def unsubscribe_handler(event, context) -> dict:
    """Handle unsubscribe requests"""
    # Use same CORS headers as working add_subscriber function
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,OPTIONS',
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
        # Extract parameters
        params = event.get('queryStringParameters', {})
        if not params:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Missing query parameters'})
            }
            
        email = params.get('email')
        
        if not email:
            return {
                'statusCode': 400,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Missing email parameter'})
            }
        
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['SUBSCRIBERS_TABLE'])
        
        # Get subscriber
        try:
            response = table.get_item(
                Key={'email': email}
            )
            
            if 'Item' not in response:
                return {
                    'statusCode': 404,
                    'headers': cors_headers,
                    'body': json.dumps({'error': 'Subscriber not found'})
                }
                
            subscriber = response['Item']
            
            # Check if already unsubscribed
            if subscriber.get('status') == 'unsubscribed':
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'text/html',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': '''
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>Already Unsubscribed</title>
                            <style>
                                body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 600px; margin: 40px auto; padding: 20px; text-align: center; }
                                .message { color: #666; margin-bottom: 20px; }
                            </style>
                        </head>
                        <body>
                            <h1>Email Already Unsubscribed</h1>
                            <p class="message">Your email has already been unsubscribed from our newsletter.</p>
                            <p>If you'd like to subscribe again, you can do so on our website.</p>
                        </body>
                        </html>
                    '''
                }
                
            # Update status to unsubscribed
            table.update_item(
                Key={'email': email},
                UpdateExpression='SET #status = :status, unsubscribed_at = :unsubscribed_at',
                ExpressionAttributeNames={
                    '#status': 'status'
                },
                ExpressionAttributeValues={
                    ':status': 'unsubscribed',
                    ':unsubscribed_at': datetime.now().isoformat()
                }
            )
            
            # Return success with HTML
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'text/html',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': '''
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Unsubscribed</title>
                        <style>
                            body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 600px; margin: 40px auto; padding: 20px; text-align: center; }
                            .success { color: #28a745; font-size: 24px; margin-bottom: 20px; }
                            .message { color: #666; margin-bottom: 20px; }
                        </style>
                    </head>
                    <body>
                        <h1 class="success">Unsubscribed Successfully!</h1>
                        <p class="message">You have been unsubscribed from the AI Papers Newsletter.</p>
                        <p>If you change your mind, you can always subscribe again on our website.</p>
                    </body>
                    </html>
                '''
            }
            
        except ClientError as e:
            logger.error(f"DynamoDB error: {str(e)}")
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
    
def create_or_update_template(ses_client, json_content: Dict, unsubscribe_url: str, subscribe_url: str):
    """Create or update SES template with subscribe button"""
    template_name = 'NewsletterTemplate'
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
                            {{overview}}
                        </td>
                    </tr>

                    <!-- Featured Papers -->
                    <tr>
                        <td align="left" valign="top" style="padding: 0 20px;">
                            <h2 class="dark-text fallback-font">Featured Research</h2>
                            {{#each featured_papers}}
                            <div class="paper">
                                <h3 class="fallback-font"><a href="{{{link}}}">{{title}}</a></h3>
                                <p class="dark-text fallback-font">{{summary}}</p>
                            </div>
                            {{/each}}
                        </td>
                    </tr>

                    <!-- Additional Papers -->
                    {{#if additional_papers}}
                    <tr>
                        <td align="left" valign="top" style="padding: 0 20px;">
                            <h2 class="dark-text fallback-font">Additional Papers</h2>
                            {{#each additional_papers}}
                            <div class="metadata fallback-font">
                                • <a href="{{{link}}}">{{title}}</a>
                            </div>
                            {{/each}}
                        </td>
                    </tr>

                    <!-- Social Sharing Section -->
                    <tr>
                        <td align="center" valign="top" class="social-section">
                            <p style="margin-bottom: 15px; color: #666666; font-size: 14px;">Share this newsletter:</p>
                            <a href="https://twitter.com/intent/tweet?text=Check%20out%20this%20week%27s%20AI%20Papers%20Newsletter%20featuring%20{{featured_count}}%20new%20papers%20on%20AI%20research%20and%20implementation&url={{subscribe_url}}" target="_blank" class="social-button" style="background-color: #000000; color: #ffffff;">Share on X</a>
                            <a href="https://bsky.app/intent/compose?text=Check%20out%20this%20week%27s%20AI%20Papers%20Newsletter%20featuring%20{{featured_count}}%20new%20papers%20on%20AI%20research%20and%20implementation%20{{subscribe_url}}" target="_blank" class="social-button" style="background-color: #0085ff; color: #ffffff;">Share on Bluesky</a>
                            <a href="https://www.linkedin.com/sharing/share-offsite/?url={{subscribe_url}}" target="_blank" class="social-button" style="background-color: #0077b5; color: #ffffff;">Share on LinkedIn</a>
                            <a href="mailto:?subject=AI%20Papers%20Newsletter&body=I%20thought%20you%20might%20be%20interested%20in%20this%20AI%20research%20newsletter.%20Check%20it%20out%20at%20{{subscribe_url}}" class="social-button" style="background-color: #666666; color: #ffffff;">Share via Email</a>
                        </td>
                    </tr>
                    {{/if}}

                    <!-- Footer -->
                    <tr>
                        <td align="center" valign="top" class="footer">
                            <p class="dark-text fallback-font">
                                <a href="{{{unsubscribe_url}}}" style="color: #666666;">Unsubscribe</a>
                            </p>
                            <p class="dark-text fallback-font">
                                Was this forwarded to you? <a href="{{{subscribe_url}}}" style="color: #666666;">Subscribe or contact here</a>
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

def get_subscribers(table_name: str) -> List[str]:
    """Retrieve verified subscriber emails from DynamoDB"""
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    try:
        response = table.scan(
            ProjectionExpression='email',
            FilterExpression='#status = :status',
            ExpressionAttributeNames={
                '#status': 'status'
            },
            ExpressionAttributeValues={
                ':status': 'verified'
            }
        )
        return [item['email'] for item in response['Items']]
    except ClientError as e:
        logger.error(f"Error scanning DynamoDB table: {str(e)}")
        raise
# Note need to update send_email_batch to enable testing

def test_prod_check(emails: List[str], test_email: str) -> List[str]:
    """
    Returns appropriate recipient list based on test mode settings
    """
    test_mode = os.environ.get('EMAIL_TEST_MODE', 'false').lower() == 'true'
    
    if test_mode and test_email:
        logger.info(f"Test mode enabled: Sending single test email instead of {len(emails)} subscriber emails")
        return [test_email]

    return emails

def send_email_batch(ses_client, source: str, emails: List[str], json_content: Dict, unsubscribe_url: str, subscribe_url: str) -> Dict:
    results = {'successful': 0, 'failed': 0, 'failures': []}

    # Get appropriate recipient list based on test mode
    emails = test_prod_check(emails, os.environ.get('EMAIL_TEST_ADDRESS', ''))
    
    default_template_data = {
        'date': datetime.now().strftime('%B %d, %Y'),
        'generated_date': json_content['metadata']['generated_date'],
        'featured_count': json_content['metadata']['featured_papers_count'],
        'overview': json_content['overview'],
        'featured_papers': json_content['featured_papers'],
        'additional_papers': json_content.get('additional_papers', []),
        'unsubscribe_url': unsubscribe_url,
        'subscribe_url': subscribe_url,
        'email': ''
    }

    batch_size = 50
    for i in range(0, len(emails), batch_size):
        batch_emails = emails[i:i + batch_size]
        destinations = []
        
        for email in batch_emails:
            destinations.append({
                'Destination': {
                    'ToAddresses': [email]
                },
                'ReplacementTemplateData': json.dumps({
                    'email': email,
                    'unsubscribe_url': f"{unsubscribe_url}?email={email}",
                    'subscribe_url': subscribe_url
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
                Template='NewsletterTemplate',
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
            results['failed'] += len(batch_emails)
            results['failures'].append({
                'batch': batch_emails,
                'error': str(e)
            })
            
    return results

def send_newsletter(json_content: Dict, table_name: str, sender: str) -> Dict:
    """Main function to send newsletter to all subscribers"""
    try:
        ses_client = boto3.client('ses')

        # Construct unsubscribe URL using AWS environment variables
        unsubscribe_url = f"{os.environ['API_URL']}/unsubscribe"
        subscribe_url = f"{os.environ['SUBSCRIBE_LANDING_PAGE']}"

        logger.info(f'unsubscribe url: {unsubscribe_url}')
        logger.info(f'subscribe url: {subscribe_url}')
        
        # Get subscribers
        subscriber_emails = get_subscribers(table_name)
        if not subscriber_emails:
            logger.warning("No subscribers found")
            return {'message': 'No subscribers found'}
            
        # Create/update SES template with the new format
        create_or_update_template(ses_client, json_content, unsubscribe_url, subscribe_url)  
        
        # Send emails using the template
        results = send_email_batch(
            ses_client=ses_client,
            source=sender,
            emails=subscriber_emails,
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
    """
    Query DynamoDB to get counts of subscribers by status using direct scan approach
    """
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    try:
        # Get verified subscribers
        verified = table.scan(
            FilterExpression='#status = :verified',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':verified': 'verified'},
        )
        
        # Get pending subscribers
        pending = table.scan(
            FilterExpression='#status = :verified',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':verified': 'pending'},
        )
        
        # Get unsubscribed users
        unsubscribed = table.scan(
            FilterExpression='#status = :unsubscribed',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':unsubscribed': 'unsubscribed'},
        )
        
        # Get all records for total count
        all_records = table.scan()
        
        metrics = {
            'total_subscribers': all_records['Count'],
            'status_counts': {
                'verified': verified['Count'],
                'pending': pending['Count'],
                'unsubscribed': unsubscribed['Count']
            },
            'metadata': {
                'timestamp': datetime.now(timezone.utc).isoformat(),
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
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
        
        # Get all pending subscribers
        dynamodb = boto3.resource('dynamodb')
        table = dynamodb.Table(os.environ['SUBSCRIBERS_TABLE'])
        
        response = table.scan(
            FilterExpression='#status = :pending',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':pending': 'pending'}
        )
        
        pending_subscribers = response['Items']
        
        # Handle pagination if necessary
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='#status = :pending',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':pending': 'pending'},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            pending_subscribers.extend(response['Items'])
        
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
        
        # Send emails
        ses_client = boto3.client('ses')
        sent_count = 0
        failures = []
        
        for subscriber in recent_subscribers:
            try:
                verification_url = f"{os.environ['API_URL']}/verify?email={subscriber['email']}&token={subscriber['verification_token']}"
                
                response = ses_client.send_templated_email(
                    Source=os.environ['SENDER_EMAIL'],
                    Destination={
                        'ToAddresses': [subscriber['email']]
                    },
                    Template='VerificationTemplate',
                    TemplateData=json.dumps({
                        'verification_url': verification_url,
                        'email': subscriber['email']
                    }),
                    ConfigurationSetName=os.environ['EMAIL_CONFIGURATION_SET']
                )
                sent_count += 1
                logger.info(f"Sent reminder to {subscriber['email']}")
                
            except Exception as e:
                failures.append({
                    'email': subscriber['email'],
                    'error': str(e)
                })
                logger.error(f"Failed to send reminder to {subscriber['email']}: {str(e)}")
        
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