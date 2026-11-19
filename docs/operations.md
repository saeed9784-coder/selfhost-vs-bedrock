# Keepsake operations guide

Keepsake is organized around the evidence gates in [`proposal-roadmap.md`](proposal-roadmap.md).
The default CDK mode deploys only Stage 0 log analysis. The former customer-facing application is
retained as an optional prototype and does not implement Bedrock's serving-side persistent KV
cache.

## Prerequisites

- Python 3.11+ locally (Lambda runs Python 3.13), Node.js (for the CDK CLI and jsii).
- AWS credentials for the target account and AWS CDK CLI v2. Stage 0 does not need Bedrock model
  invocation permissions; it reads logs already written to S3.

## Install and test

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest tests/unit -q
```

Tests use moto and a botocore Stubber for Bedrock; they never call AWS.

## Stage 0: run the analyzer

For an offline report, download the Bedrock invocation logs you are authorized to analyze and run:

```bash
PYTHONPATH=src .venv/bin/python -m keepsake.analyzer --input logs/*.json.gz --output phase0-report
```

To run analysis in AWS, synthesize and deploy the minimal Phase 0 stack. It reads an existing log
bucket; it does not create a Bedrock logging configuration or change account-level settings.

```bash
cdk synth -c keepsake:mode=phase0
cdk deploy KeepsakePhase0 -c keepsake:mode=phase0 \
  -c keepsake:sourceInvocationLogsBucket=<existing-log-bucket> \
  -c keepsake:sourceInvocationLogsPrefix=AWSLogs/
```

If the source bucket uses a customer-managed KMS key, also pass
`-c keepsake:sourceInvocationLogsKmsKeyArn=<key-arn>`. The source bucket must already contain
Bedrock invocation logs with text delivery enabled. Reports are written to a separate retained,
KMS-encrypted bucket. The weekly schedule is off by default; enable it with
`-c keepsake:phase0WeeklySchedule=true` after confirming the observation window and log prefix.
The stack outputs `AnalyzerFunctionName` and `ReportsBucketName`.

For an optional CDK pipeline, configure the GitHub repository, CodeConnections ARN, and source log
bucket context in `cdk.json`, then synthesize/deploy with `-c keepsake:mode=pipeline`. The pipeline
deploys the Stage 0 analyzer only.

## Later-stage customer prototype (optional)

The legacy document store, model gateway, dashboard, and exact-response cache are retained for
comparison, but are not the proposal's internal Bedrock cache experiment. Deploy it only by
explicitly selecting `customer-poc` mode. It requires a CDK bootstrap and Bedrock model access.

```bash
cdk synth -c keepsake:mode=customer-poc
cdk deploy Keepsake -c keepsake:mode=customer-poc -c keepsake:alarmEmail=ops@example.com
```

### After deploying the optional prototype

1. Create staff users in the Cognito user pool and add them to groups: `doc-admin` (manage all
   documents, approve quarantined documents, admin API), `finance` (company-wide usage, audit),
   `team-reviewer` / `team-helpdesk` / `team-atlas` (team leads: own team only).
2. Turn on Bedrock model-invocation logging to the `InvocationLogsBucket` output (account-level
   setting, so the stack does not change it for you):
   ```bash
   aws bedrock put-model-invocation-logging-configuration --region us-east-1 \
     --logging-config '{"s3Config":{"bucketName":"<InvocationLogsBucket>","keyPrefix":"AWSLogs"},
                        "textDataDeliveryEnabled":true}'
   ```
3. Give each agent runtime permission to assume its role: `keepsake-agent-reviewer`,
   `keepsake-agent-helpdesk`, `keepsake-agent-atlas`. Agents sign requests to `POST /invoke`
   with SigV4 (service `execute-api`).

## Prototype operations (not proposal stages)

The following API, model-switching, and cache-management procedures apply only after deploying the
optional `customer-poc` stack.

### Call the gateway (agents)

```json
POST {ApiUrl}invoke
{
  "customer_id": "cust-42",
  "document_ids": ["policy-manual", "runbook"],
  "history": [{"role": "user", "text": "Hi"}, {"role": "assistant", "text": "Hello"}],
  "messages": [{"role": "user", "text": "How do refunds work?"}]
}
```

The response envelope is `{"success": true, "data": {...}, "error": null}`; `data` holds the answer,
`request_id`, `model_id`, `cache_outcome` (`provider_hit`, `response_reuse` or `miss`), the
`document_versions` used and token usage. The agent identity comes from the signing role, never
from the body.

### Switch an agent's model (no deployment)

Each agent's configuration is one SSM parameter. Change `model_id` (and optionally
`shadow_model_id` for shadow mode):

```bash
aws ssm get-parameter --name /keepsake/agents/helpdesk/config --query Parameter.Value --output text > helpdesk.json
# edit model_id, e.g. "us.amazon.nova-pro-v1:0"
aws ssm put-parameter --name /keepsake/agents/helpdesk/config --type String --overwrite --value file://helpdesk.json
```

The gateway caches configuration for 30 seconds, so the switch is live within one minute
(NFR5). Rolling back is the same command with the previous file. The model must be listed in
`/keepsake/models/capabilities` and `/keepsake/models/prices`, and IAM allows only the models in
`config/model_capabilities.json`; adding a brand-new model id needs that file updated and a deploy.

### Block a model for a customer group

As a `doc-admin` user: `PUT {ApiUrl}models/blocks` with

```json
{"groups": {"eu-regulated": {"customers": ["cust-7", "cust-9"], "blocked_models": ["deepseek.v3.2"]}}}
```

`"customers": ["*"]` applies to every customer. Blocked requests return HTTP 403 and an audit
event `model.blocked`.

### Roll back a Lambda function (NFR6)

Every function is called through its `live` alias. To roll back without interrupting in-flight
requests (they finish on the version they started on):

```bash
aws lambda list-versions-by-function --function-name <function name> --query 'Versions[].Version'
aws lambda update-alias --function-name <function name> --name live --function-version <previous>
```

The next `cdk deploy` points `live` at the newly built version again, so fix forward in code
before the next deploy.

### Delete cache entries

As `doc-admin`: `DELETE {ApiUrl}cache?customer=<id>`, `?agent=<id>` or `?all=true`. The response
returns the number of entries removed (FR4.7).

### Erase a customer (FR6.2)

As `doc-admin`: `POST {ApiUrl}erasure/<customer_id>`. Keepsake deletes the customer's cache
entries (which also hold their conversation history) and usage records, writes a deletion report
to the reports bucket under `erasure-reports/` and records an audit event. The audit log stores a
SHA-256 customer reference rather than the raw id, so it stays append-only.

### Invoke the Phase 0 analyzer in AWS

The analyzer runs only when invoked unless the optional weekly schedule is enabled. Invoke it with
the source prefix configured at deployment:

```bash
aws lambda invoke --function-name <AnalyzerFunctionName> \
  --payload '{"prefix":"AWSLogs/"}' --cli-binary-format raw-in-base64-out phase0-response.json
```

Reports are stored under `phase0/` in the `ReportsBucketName` output.

### Prototype monitoring

- CloudWatch dashboard `Keepsake`: per-agent requests, cache hits, tokens, cost, saving, p99
  latency, errors and degraded calls.
- Alarms (to the `Keepsake alerts` SNS topic): error rate above 5 % per component, any degraded
  gateway call, any Lambda error.
- Daily budget job: SNS alert when an agent passes 80 % of its monthly budget or when its cache
  storage cost exceeds its savings.
- Staff dashboard page: `{SiteUrl}/dashboard.html`.
