# Self-hosted inference with a live Bedrock cost comparison

This builds `architecture/full-inference-architecture.drawio` in two sizes, and adds a chat page
("Inference Cost Lab") where every message is answered by the self-hosted model and by the
same model on Amazon Bedrock at the same time, with tokens, latency and cost shown for each.

| Size | Code | Status |
|---|---|---|
| Minimal slice: one GPU node | `infra/selfhost_stack.py`, `src/selfhost`, `web/chat` | Deployable (`keepsake:mode=selfhost`) |
| Full diagram: prefill and decode pools, L2 KV tier, EKS | `infra/selfhost_full_stack.py`, `deploy/k8s/inference.yaml` | Synthesizes only. Not deployed, not tested on a cluster |

## Default: CPU proof of concept

New AWS accounts start with a GPU (G-family) vCPU quota of 0, so the default deploy runs on CPU:

- **Hardware:** `c7i.2xlarge` (8 vCPU, 16 GiB, about $0.36 an hour) on Ubuntu 24.04.
- **Engine:** llama.cpp server with Llama 3.1 8B Instruct (bartowski Q4_K_M GGUF, about 4.9 GB, not gated, so no Hugging Face token is needed). It runs 2 parallel slots of 8k context with prompt caching, and `--metrics` feeds the router. Llama 3.2 3B was the first choice, but it has reached end of life on Bedrock.
- **Comparison:** Bedrock `us.meta.llama3-1-8b-instruct-v1:0` at $0.22 per million tokens. A new account starts with a Bedrock tokens-per-minute quota of 0 for this model, and AWS has to raise it.

The gateway, router, chat page and AWS stack are the same as in the GPU build. Only the engine and the AMI change. The cost panel's GPU instance picker uses separate throughput assumptions per instance kind. You can therefore compare the CPU proof of concept with the same model on a `g6e` GPU without deploying one.

When the GPU quota is approved, deploy the GPU version with
`-c selfhost:instanceType=g6e.xlarge`. The same model then runs on vLLM.

## How the diagram maps to the code

| Diagram block | Minimal slice | Full reference |
|---|---|---|
| API ingress (TLS, WAF, auth, quotas, streaming) | CloudFront (HTTPS) → ALB that only accepts the CloudFront prefix list and a secret header; the gateway checks a bearer token and applies a per-client rate limit; responses stream over SSE | Same gateway, as a Deployment on the system nodes |
| Inference gateway and cache-aware router | `src/selfhost/router.py` and `prefix_index.py`: chained prefix-block hashes keyed by model and tokenizer version, longest-prefix affinity, failover on connection errors, 429 backpressure | Same, routing to llm-d / the decode service |
| Global scheduler (context tiers, KV pressure) | The router filters replicas by context length and scores load from each vLLM `/metrics` (running, waiting, KV-cache use) | Same, plus llm-d's scheduler for prefill/decode pairing |
| Prefill pool and decode pool | One vLLM process does both (aggregated) | Two p5.48xlarge node groups, NIXL KV transfer |
| L1 GPU/host KV | vLLM prefix caching in GPU memory | Plus LMCache CPU offload |
| L2 shared prefix KV tier | not in the slice | FSx for Lustre mounted at `/fsx`, LMCache disk backend |
| Cache manager and metadata, TTL by workload | Prefix index with retention per workload: agent 72 h, chat 168 h (`catalog.json`) | Same |
| Amazon S3 model artifacts and registry | Versioned bucket; the first boot publishes the weights, later boots load that pinned copy | Versioned bucket |
| Observability and SLOs | CloudWatch dashboard `self-hosted-inference`: TTFT p50/p90, tokens/s, cost per 1M tokens, cached vs total input tokens | Container Insights add-on |
| Security and governance | Private subnet, IMDSv2, encrypted EBS, Session Manager (no SSH), Secrets Manager, Bedrock IAM scoped to one model | Private EKS endpoint, audit logs |

## Run it locally (no GPU, no AWS)

```bash
pip install -r requirements-dev.txt
export PYTHONPATH=src
uvicorn selfhost.devserver:app --port 8000 &   # stands in for vLLM, imitates prefix caching
uvicorn selfhost.app:app --port 8080           # gateway + chat page at http://127.0.0.1:8080
```

The Bedrock column calls the real Bedrock with your local AWS credentials. Without credentials
it shows the error, and the self-hosted column keeps working.

## Deploy the minimal slice

Prerequisites:
- AWS credentials for the target account.
- A G-and-VT on-demand vCPU quota of at least 4 in us-east-1 (`g6e.xlarge` has 4 vCPUs).
- Bedrock model access to Llama 3.1 8B Instruct.
- A Hugging Face token with the Llama 3.1 license accepted. The `qwen3-32b` option needs no token.

```bash
cdk deploy SelfHostedInference --app ".venv/bin/python app.py" -c keepsake:mode=selfhost \
  [-c selfhost:model=qwen3-32b] [-c selfhost:instanceType=g6e.2xlarge] \
  [-c selfhost:vllmImage=vllm/vllm-openai:<tag>]
```

After the deploy:
1. Run the `SetHuggingFaceToken` output command, putting your token in place of `hf_...`. vLLM
   retries every 30 seconds, so it picks up the token without a restart.
2. Run the `GetAccessToken` output command, open `ChatUrl`, and paste the token when the page
   asks for it.
3. Allow 10 to 20 minutes for the first answer while the vLLM image (about 10 GB) and the model
   weights download. Follow progress with
   `aws ssm start-session --target <GpuInstanceId>` and then
   `journalctl -u selfhost-vllm -f`.
4. **Stop the GPU when you are not demoing** with the `StopGpuToSaveMoney` output command. Start
   it again with `aws ec2 start-instances`. Run `cdk destroy` to remove everything.

Running costs (planning prices, check them against the AWS price list): `g6e.xlarge` is about
$1.86 an hour (about $1,359 a month if it never stops). The NAT gateway, ALB, EBS and secrets
add about $66 a month, plus traffic.

## How the costs are compared

For each answer the page shows:

- **Bedrock**: tokens × the on-demand price (`catalog.json`). Llama 3.1 8B has no Bedrock prompt
  cache, so each turn pays for the whole conversation again.
- **Self-hosted, at target load**: the GPU time the tokens need when the GPU is kept busy at the
  target utilization (default 60%). Prompt tokens served from the KV cache skip prefill, which is
  why long conversations get cheaper per turn.
- **Self-hosted, this GPU now**: the GPU time the answer actually held, divided by the number of
  requests running at the same moment. With one person chatting this is much higher than
  Bedrock. That is the real cost of an idle GPU, and the page shows it on purpose.

The monthly projection charges whole instances (an instance costs the same whether busy or idle),
adds instances when the load needs more GPU hours, and shows the break-even volume. With the
default assumptions (Llama 3.1 8B, one `g6e.xlarge`, 60% prompt reuse), Bedrock is cheaper below
roughly 5.6 billion input tokens a month. Above that, self-hosting wins, and it wins by more as
prompt reuse grows.

**The throughput figures are assumptions** (`throughput_assumption` in `catalog.json`). After
deploying, measure the real figures and enter them in the catalog:

```bash
docker exec vllm vllm bench serve --model meta-llama/Llama-3.1-8B-Instruct \
  --served-model-name llama-3.1-8b --dataset-name random --random-input-len 2000 \
  --random-output-len 300 --num-prompts 200 --request-rate inf
```

## Known limits

- One GPU node in one AZ, with no autoscaling: this is a demo, not production. The full reference
  covers the scale-out shape.
- `g6e` capacity varies by AZ. If the deploy fails with `InsufficientInstanceCapacity`, deploy
  again, or try a different instance type.
- The prefix index lives in memory and is lost when the gateway restarts. vLLM's own cache is
  unaffected.
- CloudFront to ALB is HTTP inside AWS. Add a custom domain with an ACM certificate on the ALB,
  and AWS WAF, before any real use.
- Without a Hugging Face token, the default Llama model cannot download. The vLLM service then
  logs a `SELFHOST: ... is gated` line every 30 seconds, and the chat page says the self-hosted
  model is still starting. Bedrock keeps answering.
- The CloudFront origin header value can be read by anyone with
  `cloudfront:GetDistributionConfig`, and rotating it takes a redeploy. That is acceptable for a
  demo; a production setup would use CloudFront VPC origins instead of a public ALB.
- The default vLLM image tag and its flags (`--enable-prompt-tokens-details`) have not been
  checked on the GPU yet. They are checked on the first deploy.
