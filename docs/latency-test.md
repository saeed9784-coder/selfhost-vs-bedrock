# Latency and cost checks against the deployed stack (NFR1, NFR2)

Unit tests cannot measure these. Run them after deployment, with AWS credentials for an
identity that can assume an agent role and call Bedrock directly.

## NFR1: gateway overhead at 10 requests per second

Target: Keepsake overhead (excluding model time) is at most 150 ms at p99, and p99 end-to-end
latency through Keepsake is no worse than the direct-Bedrock baseline in the same test.

Method: send the same workload two ways in the same time window, then compare.

1. **Workload.** Use the Helpdesk shape from Phase 0: the policy manual uploaded as a document,
   a fixed set of 50 different customer questions, 10 requests per second for 10 minutes
   (6,000 requests per path). Use one `customer_id` per question so response reuse is exercised
   the way production traffic would be, and record the outcome mix.
2. **Direct baseline.** Call `bedrock-runtime` `Converse` with the same model, the same assembled
   prompt (system text + manual + question) and the same `maxTokens`, from the same host.
3. **Through Keepsake.** Call `POST /invoke` signed with SigV4 as `keepsake-agent-helpdesk`.
4. **Measure** client-side wall-clock latency for every request on both paths. Report p50, p95
   and p99 for each path, split by `cache_outcome` for the Keepsake path.
5. **Overhead.** For `miss` requests, overhead = Keepsake latency minus direct latency at the
   same percentile. For `response_reuse` requests, the full Keepsake latency is overhead (no
   model call) and is expected to be far below the baseline.

Pass if the p99 overhead for `miss` requests is at most 150 ms and the overall Keepsake p99 is
at or below the baseline p99.

A load generator such as Locust or a small Python script using `botocore.auth.SigV4Auth` with a
thread pool is enough; keep the client in `us-east-1` to avoid adding network distance. Watch the
`Keepsake` CloudWatch dashboard during the run for errors and degraded calls, and record the
Lambda concurrency so cold starts can be separated from steady-state latency.

## NFR2: cost per million input tokens versus the Phase 0 baseline

1. Run Phase 0 on generated Helpdesk and Reviewer traffic
   (`python -m keepsake.analyzer --sample --output phase0-report`). The report's
   `cost_today_usd` divided by `total_input_tokens` (times 1,000,000) is the baseline cost per
   million input tokens per agent.
2. Replay the same generated traffic through `POST /invoke` (same documents, same order and
   spacing compressed as needed, retention set to cover the replay).
3. Read the dashboard API: `GET /usage?scope=team&team=helpdesk&from=<start>&to=<end>` and the
   same for `reviewer`. Cost per million input tokens through Keepsake =
   (`spend_usd` + `storage_cost_usd`) / `tokens_sent` * 1,000,000.

Pass if the Keepsake figure is lower than the Phase 0 baseline for both agents.
