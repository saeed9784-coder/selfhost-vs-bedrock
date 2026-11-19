# Optional architecture comparison: self-hosted KV cache

This document compares an EC2-based alternative. It is not the architecture selected by the
Bedrock persistent-cache proposal. The proposal's Stage 1 is an internal Bedrock serving-side
experiment; it does not require this customer-managed EC2 design. See
[`proposal-roadmap.md`](proposal-roadmap.md) for the active stage plan.

**Question.** Would running our own inference with a persistent KV cache (vLLM + LMCache on EC2,
KV tiered GPU -> CPU RAM -> local NVMe -> S3) be a viable alternative for Groundhog Labs' agents?

**Answer.** The old planning estimate found self-hosting uneconomic at the assumed POC volume. It
does not compare against or disprove the proposed Bedrock platform cache, whose shared-storage,
GPU-time, and fleet-cost figures require Bedrock platform telemetry.

All figures below are planning estimates. Prices are approximate public on-demand list prices
and must be checked against the current AWS price list before any decision; nothing here was
measured. Assumptions are marked *(assumption)*.

## Workload used for the comparison

Helpdesk, from the SOW and the Phase 0 sample generator: a 100,000-token support manual sent 50
times over 3 days, about 1.7 million input tokens a day, so about **50 million input tokens a
month** *(assumption: the sample rate holds for a month)*.

## Option A: Bedrock baseline (what Keepsake does today)

| Item | Estimate |
|---|---|
| DeepSeek V3.2 input at about $0.62 per million tokens (`config/prices.json`, to be confirmed) | about $31 a month with no caching at all |
| Same traffic on Claude Sonnet 4.5 ($3.00 input / $0.30 cache read per million) | about $150 a month uncached; well under half that when most manual tokens are cache reads |
| Keepsake response reuse | removes the model call entirely for identical requests; storage in DynamoDB at $0.25 per GB-month is cents |
| Keepsake infrastructure (Lambda, API Gateway, DynamoDB, S3, CloudWatch) | tens of dollars a month at this volume *(assumption)* |

Latency: Bedrock provider caching cuts time-to-first-token for long cached prefixes; response
reuse answers in the gateway's own overhead (target at most 150 ms p99, NFR1).

## Option B: self-hosted vLLM + LMCache

### Hardware needed

- **DeepSeek V3.2** is a very large mixture-of-experts model (hundreds of billions of
  parameters; FP8 weights in the order of 700 GB). It needs a full 8-GPU H100/H200-class node
  (EC2 `p5` family) at minimum, and realistically two nodes or aggressive quantization for
  headroom *(assumption based on published model size)*.
- **EC2 `g6`** (NVIDIA L4, 24 GB per GPU) cannot hold DeepSeek V3.2, Claude or Nova Pro. It can
  serve small open models (for example 7-8B parameters), which means changing the agents' model,
  not just their hosting.

### KV-cache tiering (what LMCache adds)

| Tier | Typical capacity | Role |
|---|---|---|
| GPU HBM | a few GB of free memory after weights | hot prefixes, current requests |
| CPU RAM | hundreds of GB to 2 TB on `p5` | recently used prefixes, sub-second reload |
| Local NVMe | several TB on `p5` | day-scale retention for repeated documents |
| S3 | unbounded | multi-day retention and sharing across instances; slowest reload |

Size of one cached 100,000-token prefix *(estimates)*: about 7 GB for DeepSeek V3's compressed
(multi-head latent) attention cache, about 13 GB for a dense 8B model. A handful of manuals and
repository snapshots fit in CPU RAM and NVMe; reload from S3 costs seconds per prefix.

### Cost

| Configuration | Approximate on-demand cost |
|---|---|
| One `p5.48xlarge` running all month (730 h) | roughly $20,000-$40,000 a month depending on current pricing, before storage and engineering time |
| One `g6.12xlarge` (4x L4) running all month, small model only | roughly $3,000-$4,000 a month |

Break-even against DeepSeek V3.2 on Bedrock (about $0.62 per million input tokens): a single
`p5` node at about $25,000 a month only pays off above roughly **40 billion input tokens a month**,
which is several hundred times the Helpdesk volume. Savings plans or Spot reduce the instance
price but not by enough to close that gap, and Spot interrupts long-lived caches.

### Latency

With a warm KV cache in GPU or CPU memory, time-to-first-token for a 100,000-token prefix drops
sharply, comparable to Bedrock's provider cache. Cold reloads from NVMe or S3 add seconds.
Throughput and tail latency then depend on our own capacity planning and autoscaling, which
Bedrock currently handles for us.

### Operations complexity

- GPU capacity reservations, AMIs and drivers, vLLM/LMCache upgrades, model weight management.
- Autoscaling and cache warm-up across instances; cache placement and eviction tuning.
- Security and isolation work Bedrock gives us today: tenant separation inside a shared KV cache,
  encryption of NVMe and S3 cache tiers with our KMS key, deletion guarantees for customer data
  held in KV form (FR6.2 would have to cover GPU, RAM, NVMe and S3 tiers).
- On-call ownership for an inference service.

### Model-choice limits

Self-hosting only works with open-weight models. Claude and Amazon Nova are not available as
weights, so the Reviewer and Atlas agents could not move. Keepsake's config-only model switching
(FR3.5) would become a capacity-planning exercise per model.

## Comparison

| Criterion | Bedrock + Keepsake | Self-hosted vLLM + LMCache |
|---|---|---|
| Monthly cost at Helpdesk volume | tens to low hundreds of dollars | thousands (`g6`, small model) to tens of thousands (`p5`) |
| Latency for repeated context | provider cache plus response reuse | similar when warm; worse on cold reload |
| Model choice | Claude, Nova, DeepSeek and more, by configuration | open-weight models only |
| Operations | managed | GPU fleet, caching tiers, upgrades, on-call |
| Data deletion and isolation | handled by Keepsake on cache entries and logs | must be built for every KV tier |

## Recommendation

Use this only as a separate option study if the decision is to operate model inference outside
Bedrock. It is not a substitute for Stage 0 measurement or the Stage 1 internal Bedrock experiment.
