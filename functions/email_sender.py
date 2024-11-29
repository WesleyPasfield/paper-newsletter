import boto3
from botocore.exceptions import ClientError
import json
import os
from typing import Dict, List
import logging
from datetime import datetime
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
    api_url = get_api_url_from_event(event)
    if not api_url:
        logger.error("Could not determine API URL from event")
        return {
            'statusCode': 500,
            'headers': cors_headers,
            'body': json.dumps({'error': 'Could not determine API URL'})
        }
    # Standard CORS headers
    cors_headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'OPTIONS,POST',
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
        
        # Check if email already exists
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
        except ClientError as e:
            logger.error(f"Error checking existing subscriber: {str(e)}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Failed to check subscription status'})
            }

        # Generate verification token
        verification_token = secrets.token_urlsafe(32)
        timestamp = datetime.now().isoformat()

        # Store new subscriber
        try:
            table.put_item(
                Item={
                    'email': email,
                    'status': 'pending',
                    'verification_token': verification_token,
                    'created_at': timestamp,
                    'updated_at': timestamp
                }
            )
        except ClientError as e:
            logger.error(f"Error storing subscriber: {str(e)}")
            return {
                'statusCode': 500,
                'headers': cors_headers,
                'body': json.dumps({'error': 'Failed to store subscription'})
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
        logger.error(f"Unexpected error in add_subscriber: {str(e)}")  # Added logging
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
        response = ses_client.send_templated_email(
            Source=sender,
            Destination={
                'ToAddresses': [email]
            },
            Template='VerificationTemplate',
            TemplateData=json.dumps({
                'verification_url': verification_url,
                'email': email
            })
        )
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
            .dark-bg { background-color: #2d2d2d !important; }
            .dark-text { color: #ffffff !important; }
            .paper { border-left-color: #4a9eff !important; background-color: #363636 !important; }
            .paper h3 a { color: #4a9eff !important; }
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
                            <p class="dark-text fallback-font">{{generated_date}}</p>
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
                                Was this forwarded to you? <a href="{{{subscribe_url}}}" style="color: #666666;">Subscribe here</a>
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
AI Papers Newsletter - {{generated_date}}

{{overview}}

Featured Research:
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

Was this forwarded to you? Subscribe at: {{{subscribe_url}}}
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

def send_email_batch(ses_client, source: str, emails: List[str], json_content: Dict, unsubscribe_url: str, subscribe_url: str) -> Dict:
    results = {'successful': 0, 'failed': 0, 'failures': []}
    
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
        create_or_update_template(ses_client, json_content, unsubscribe_url, subscribe_url)  # Removed unsubscribe_url argument
        
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
