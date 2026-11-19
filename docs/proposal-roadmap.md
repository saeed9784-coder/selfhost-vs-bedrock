# Keepsake roadmap aligned to the Bedrock persistent-cache proposal

Keepsake follows the staged investment plan in [the proposal](https://medium.com/@saeed_themedaibiref/deepseek-open-sourced-its-inference-engine-the-interesting-part-is-what-it-kept). Each stage has a decision gate. Passing one stage does not imply that a later stage is approved or built.

## Stage 0 — Measure before building

**Purpose:** estimate repeated-prefix demand and establish whether a persistent cache could pay for itself.

**Current implementation:** the local analyzer reads Bedrock invocation logs and reports repeated text, reuse intervals, model prices, and estimated savings for 1 hour, 1 day, 3 days, and 7 days. It also summarizes common leading prompt chunks by model across agents in one AWS account; it refuses to combine model reuse across accounts. A minimal optional CDK stack can read an existing invocation-log bucket and write encrypted aggregate reports. The default CDK mode is `phase0`; it does not deploy the customer-facing Keepsake prototype.

**Limits:** the per-agent cost screen matches repeated text segments; the model summary compares common leading chunks in 2,000-character windows. Both are proxies, not observed Bedrock KV-cache hits. The 4-byte-per-token storage size is a planning assumption. Customer invocation logs do not expose serving-side GPU time or actual KV-cache bytes, so those figures must come from Bedrock platform telemetry before Stage 0 can be considered conclusive. Do not treat the analyzer's `WORTH BUILDING` verdict alone as the investment decision.

**Gate:** Bedrock platform owners validate reuse distribution, prefix lengths, GPU time avoided, and storage estimates. Stop if the best-case savings do not clearly exceed cache and operating costs.

## Stage 1 — Internal optimization on one model family

**Purpose:** test a shared persistent KV cache behind one high-reuse model, first on shadow traffic and then on a small production slice.

**Required platform integration:** a Bedrock serving-side cache adapter, cache storage and routing, safe shadow traffic, and platform telemetry. These interfaces are not exposed by the public Bedrock runtime API. The customer-account Keepsake prototype cannot implement or prove this stage by itself. Do not substitute exact-response reuse or the public prompt-cache feature and label it a persistent KV-cache experiment.

**Measure:** GPU-hours avoided per terabyte-day; prefix-length-weighted hit rate; P50/P95/P99 time to first token versus control; cost per million tokens split into cached input, new input, and output; cache read failures and validation errors; availability during storage-node failures; and security findings.

**Gate:** require a meaningful reduction in input cost with no P99 latency regression and no unresolved security finding. If the gate fails, stop or revise the design before exposing a customer feature.

## Stage 2 — Managed customer feature

**Purpose:** expose persistent context caching only after Stage 1 succeeds.

**Feature requirements:** customer-controlled retention, deletion, usage and savings visibility, account/Region isolation, encryption, model-version invalidation, and clear billing for cached versus new input. Measure adoption, cost per successful task, and net revenue after lower cached-token prices.

**Gate:** proceed only if adoption and net revenue justify the service and the security and deletion requirements are met.

## Kept as a separate prototype

The former customer-facing Keepsake stack remains available as an explicitly selected `customer-poc` mode for its document store, Bedrock gateway, and exact-response reuse. It is not the proposal's shared Bedrock KV cache, and it is not the default deployment. The self-hosted EC2/vLLM/LMCache analysis is an alternative architecture, not a planned stage of this proposal.
