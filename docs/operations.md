# Operations runbook

Use this runbook for production administration. Commands use placeholders so an operator has to resolve the intended stack, Region, function, and issue before invoking anything.

## Safety rules

1. Keep both schedules disabled during the first deployment or a risky upgrade.
2. Keep `EmailTestMode=true` until a test-address submission succeeds.
3. Treat `accepted` as SES submission acceptance. Confirm delivery through feedback events and the recipient ledger.
4. Preview every manual newsletter or bulk reminder campaign before confirming it.
5. Use a unique `issue_id` for intentional newsletter content changes. Reusing an issue ID reuses its immutable archived content.
6. Do not retry a failed issue until you understand whether recipient submissions started.

## Resolve stack resources

Start with read-only inspection:

```bash
aws cloudformation describe-stacks \
  --stack-name <stack-name> \
  --region <region> \
  --query 'Stacks[0].Outputs'

aws cloudformation describe-stack-resources \
  --stack-name <stack-name> \
  --region <region>
```

Record the API endpoint, newsletter function, bulk verifier function, metrics function, subscribers table, newsletter-runs table, bucket, and alarm topic before continuing.

## Deploy without sending

Set these overrides before the first deployment or a risky infrastructure migration:

```toml
NewsletterScheduleEnabled="false"
VerificationReminderScheduleEnabled="false"
EmailTestMode="true"
```

Then validate, build, and review the CloudFormation change set:

```bash
python -m pytest -q
sam validate --lint --template-file template.yaml
sam build --parallel
sam deploy
```

Stop if the change set replaces the subscribers table, newsletter-runs table, or newsletter bucket. Those resources have retention policies, but an unexpected replacement still creates application and operational risk.

After deployment:

- Confirm the alarm-topic email subscription.
- Confirm SES identity verification and production access in the deployment Region.
- Confirm `${stack}-newsletter-template-v2` exists.
- Confirm each schedule is in its intended state. Production newsletter delivery should be enabled only after test-mode verification.
- Confirm the newsletter generation and submission alarms are healthy.

## Preview a newsletter

Direct Lambda invocation is the simplest administrative path. Omit `confirm_send` or set it to `false`:

```bash
aws lambda invoke \
  --function-name <newsletter-function-name> \
  --region <region> \
  --cli-binary-format raw-in-base64-out \
  --payload '{"issue_id":"newsletter-YYYY-Www-preview","api_url":"https://<api-id>.execute-api.<region>.amazonaws.com/<stage>"}' \
  preview-response.json
```

Check that the response says no email was sent. Review the archived JSON at the returned S3 location. Inspect title relevance, evidence claims, limitations, links, paper counts, and the issue overview.

## Submit a newsletter to the test address

Confirm that the deployed `EmailTestMode` parameter is `true` and `EmailTestAddress` is correct. Then reuse the preview issue ID so the exact archived content is submitted:

```bash
aws lambda invoke \
  --function-name <newsletter-function-name> \
  --region <region> \
  --cli-binary-format raw-in-base64-out \
  --payload '{"issue_id":"newsletter-YYYY-Www-preview","api_url":"https://<api-id>.execute-api.<region>.amazonaws.com/<stage>","confirm_send":true}' \
  test-submit-response.json
```

The issue ledger prevents another ordinary submission for the same issue ID. Check the test recipient record and SES event stream for `delivered`. An `accepted` response alone is not delivery proof.

## Enable the weekly schedule

Only enable the schedule after a successful preview and test-address delivery. Change `NewsletterScheduleEnabled` to `true` and decide whether `EmailTestMode` should remain enabled. Review the change set before deployment.

The scheduled payload includes `confirm_send=true`. The default schedule is Tuesday at 12:00 UTC. Changing `NewsletterCadenceDescription` updates subscriber-facing cadence copy but does not alter EventBridge timing.

## Inspect subscriber counts

Invoke the IAM-protected metrics endpoint with SigV4 credentials, or invoke its Lambda directly:

```bash
aws lambda invoke \
  --function-name <subscriber-metrics-function-name> \
  --region <region> \
  --cli-binary-format raw-in-base64-out \
  --payload '{}' \
  subscriber-metrics.json
```

The response includes complete verified, pending, and unsubscribed counts. Suppressed recipients can remain verified in subscription state but are excluded from newsletter delivery.

## Verification reminders

The scheduled job checks hourly and sends at most one accepted reminder to a pending subscriber after 48 hours. It processes up to `BulkVerificationMaxBatch` records per run. Pending records expire according to `PendingSubscriberTtlDays`.

Before enabling the schedule, inspect subscriber counts and invoke the code in a non-production environment. The scheduled reminder handler is a sending action and does not have a dry-run mode.

### Preview a bulk campaign

Bulk verification defaults to a dry run:

```bash
aws lambda invoke \
  --function-name <bulk-verifier-function-name> \
  --region <region> \
  --cli-binary-format raw-in-base64-out \
  --payload '{"api_url":"https://<api-id>.execute-api.<region>.amazonaws.com/<stage>","max_batch_size":25,"expired_tokens_only":true}' \
  bulk-preview.json
```

Review `total_pending`, `eligible_before_cap`, `candidate_count`, and every exclusion count. The function excludes requests older than 30 days and previously accepted reminders unless `force=true`.

### Confirm the same bulk campaign

Use a durable campaign name and the same selection options:

```bash
aws lambda invoke \
  --function-name <bulk-verifier-function-name> \
  --region <region> \
  --cli-binary-format raw-in-base64-out \
  --payload '{"api_url":"https://<api-id>.execute-api.<region>.amazonaws.com/<stage>","confirm_send":true,"campaign_id":"verification-reminder-YYYY-MM-DD","max_batch_size":25,"expired_tokens_only":true}' \
  bulk-submit.json
```

Compare `accepted_count`, `failed_count`, and `skipped_count` with the preview. Reusing the campaign ID does not resend records already claimed for that campaign. Avoid `force=true` unless the purpose and recipient population have been reviewed again.

## Issue and delivery states

The issue-level `recipient_id=ISSUE` record can move through `ready`, `sending`, `submitted`, `partial`, or `failed` states. A generation error emits the separate `IssueGenerationFailed` metric. A `submitted` issue means all attempted recipient submissions were accepted by SES.

Recipient records can move through:

```text
attempted -> accepted -> delivered
          -> ambiguous
          -> bounced
          -> complained
          -> rejected
          -> rendering_failed
```

Permanent bounces and complaints suppress the matching subscriber. Transient bounces are recorded without automatic suppression.

## Alarm and failure triage

### Missing issue

The missing-issue alarm evaluates `Newsletter/Issues:IssueReady` only when the schedule is enabled. Check:

1. EventBridge schedule state and last invocation.
2. Newsletter Lambda errors and duration.
3. arXiv feed errors.
4. OpenAI authentication, rate limits, and structured-output failures.
5. The issue record and `IssueGenerationFailed` or Lambda error alarm history.

`IssueGenerationFailed` means no valid issue was produced. It is an error path and does not send an empty newsletter.

### Partial or failed submission

Inspect the issue record, then query recipient records for the issue. Determine which recipients reached `accepted` or `delivered` before considering a resume. Use `resume_delivery=true` only when the issue is in a resumable state and the ledger confirms that already-claimed recipients will be skipped.

### Bounce or complaint

Check the SES feedback Lambda logs, the recipient ledger, and the subscriber `deliverability_status`. A complaint should suppress the recipient immediately. A permanent bounce should also suppress. Investigate any feedback-handler failure before the next newsletter.

## Rollback

For a code-only regression, deploy the last known-good commit through SAM with schedules disabled. Inspect the change set before execution.

For an infrastructure regression:

1. Disable both schedules.
2. Preserve the failed change set and CloudFormation event history.
3. Deploy the last known-good template if it does not remove the new state resources.
4. Validate subscribers, issue state, S3 archives, SES configuration, and alarms.
5. Re-enable schedules only after a preview and test-address delivery.

CloudFormation retains the subscribers table, newsletter-runs table, bucket, and custom Lambda log groups when the stack or those resources are removed. `sam delete` therefore does not erase all data. Inventory retained resources before deleting them manually.

## Secret rotation

Rotate the existing secret value without changing its name:

```bash
export OPENAI_PAPER_NEWSLETTER_KEY="replace-me"
aws secretsmanager put-secret-value \
  --secret-id openai/paper_newsletter_key \
  --secret-string "{\"openai_key\":\"$OPENAI_PAPER_NEWSLETTER_KEY\"}"
unset OPENAI_PAPER_NEWSLETTER_KEY
```

Run a newsletter preview after rotation. A Lambda redeploy is not required when the secret name stays the same.
