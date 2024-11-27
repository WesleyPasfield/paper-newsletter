# AI Research Paper Newsletter

Automatically generates and sends newsletters about AI research papers using AWS Lambda and Claude. Create your own prompts that fit your desired use case.

## Features

- 🤖 Uses Claude to evaluate and summarize AI research papers
- 📊 Customizable paper evaluation criteria
- 📧 Automated email delivery via AWS SES
- ✅ Double opt-in subscriber management
- 📈 Built-in analytics and monitoring

Keep in mind there might be some required customization in your account, but this should get you really far along the way

## Architecture

The system consists of several key components:
- Paper Analyzer: Evaluates papers using Claude
- Newsletter Generator: Creates formatted newsletters
- Email Management: Handles subscriptions and delivery
- Monitoring: Tracks delivery and engagement metrics

## Prerequisites

- AWS Account with SES in production mode
- Anthropic API Key
- Python 3.11+
- AWS SAM CLI
- Verified SES sender email or domain

## Setup

1. **Configure AWS Credentials**
   ```bash
   aws configure
   ```

2. **Create Required Secrets**
   ```bash
   aws secretsmanager create-secret \
       --name anthropic/api_key \
       --secret-string '{"anthropic_key":"your-key-here"}'
   ```

3. **Configure Deployment**
   - Copy samconfig.example.toml to samconfig.toml
   - Update with your values
   ```bash
   cp samconfig.example.toml samconfig.toml
   # Edit samconfig.toml with your values
   ```

4. **Deploy Infrastructure**
   ```bash
   sam build
   sam deploy --guided
   ```

## Configuration

### Paper Evaluation

The system uses two key prompts that can be customized:

1. EVAL_PROMPT: Determines how papers are scored
2. NEWSLETTER_PROMPT: Controls newsletter formatting

Example evaluation criteria:
```python
EVAL_PROMPT = """
You are evaluating academic papers for an expert specializing in:
1. Practical LLM/Generative AI regulation
2. Societal and economic impacts
3. Technical implementation
...
"""
```

### Email Templates

Templates can be customized in the CloudFormation template:
- Verification Email
- Welcome Email
- Newsletter Template
- Unsubscribe Confirmation

## API Endpoints

The system exposes several REST endpoints:

- POST /subscribe: Subscribe new email
- GET /verify: Verify subscription
- GET /unsubscribe: Remove subscription

## Monitoring

Built-in CloudWatch dashboard provides metrics for:
- Email delivery rates
- Bounce/complaint rates
- Lambda execution metrics
- Paper processing statistics

## Development

### Adding New Features

1. Update SAM template
2. Add required Lambda functions
3. Update email templates if needed
4. Test locally
5. Deploy changes
6. Configure needed S3 resourcs for website hosting/landing page if interested

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Security

Review the security checklist before deployment:
- SES domain verification
- API authentication
- IAM permissions
- Rate limiting
- CORS configuration

I definitely made some updates manually in the console, and had some resources already in my account so everything might not be perfectly smooth, particilarly with DNS/SES