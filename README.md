# Self-hosted inference vs Amazon Bedrock

When is Amazon Bedrock the cheaper way to serve an open model, and when does a self-hosted
service with long-lived KV caching win? This repo builds a working slice of a self-hosted
inference architecture on AWS and compares it, live and on the **same model**, with Bedrock.

Every chat message goes to both sides at once. The page shows each answer's tokens, cached
tokens, time to first token, speed and cost, plus a monthly cost projection with the
break-even volume.

## What's in it

| Part | Path | What it does |
| --- | --- | --- |
| Gateway | `src/selfhost/app.py`, `service.py` | FastAPI app: sends each turn to the self-hosted engine and to Bedrock, streams both over SSE, prices each answer |
| Cache-aware router | `src/selfhost/router.py`, `prefix_index.py` | Chained prefix-block hashes per model and tokenizer version; routes to the replica holding the longest matching prefix; admission control and failover |
| Backends | `src/selfhost/backends.py` | OpenAI-compatible engine client (llama.cpp or vLLM) and Bedrock ConverseStream, including reasoning tokens |
| Cost model | `src/selfhost/pricing.py`, `projection.py`, `catalog.json` | Per-answer cost, monthly projection by instance and purchase option (On-Demand, Spot, Savings Plan), break-even solver |
| Chat UI | `web/chat/` | Side-by-side chat, a cache showcase (a long shared document over three turns) and a cost panel |
| Infrastructure | `infra/selfhost_stack.py` | AWS CDK (Python): CloudFront → ALB → an Auto Scaling group of one inference node, with Spot, S3 model registry, Secrets Manager and a CloudWatch dashboard |
| Full reference design | `infra/selfhost_full_stack.py`, `deploy/k8s/inference.yaml`, `architecture/full-inference-architecture.drawio` | EKS with separate prefill and decode GPU pools and an FSx L2 KV tier. Synthesizes only; not deployed |

The repo also contains Keepsake (`src/keepsake`, `infra/keepsake_stack.py`,
`infra/phase0_stack.py`), an earlier tool for measuring Bedrock invocation logs. See
`docs/operations.md`.

## How it's deployed today

The account this was built in has a GPU vCPU quota of 0, so the default deploy runs on CPU:

- **Model:** MiniMax M2.5 on both sides. Self-hosted: the unsloth `UD-Q3_K_XL` GGUF (about
  3.5 bits per weight, 101 GB) served by llama.cpp. Bedrock: `minimax.minimax-m2.5`.
- **Node:** `r7a.4xlarge` (16 vCPU, 128 GiB) on Spot, falling back to `r7i`, `r6a` or `r6i`
  4xlarge. About $0.30 an hour.
- **Engine choice:** llama.cpp on CPU instance types, vLLM on GPU instance types, picked from
  `catalog.json`.

## Findings so far

| Measurement | Self-hosted | Bedrock |
| --- | --- | --- |
| Time to first token, cached turns (Llama 3.1 8B, 4-vCPU node) | 20.9 s first turn → 0.98 s, 97% of the prompt from cache | 2.1 s average |
| Decode speed, MiniMax M2.5 | 5.8 tokens/s (CPU) | 59–66 tokens/s |
| Cost of one answer, MiniMax M2.5 | $0.0026 | $0.00017 |

On CPU, Bedrock is far cheaper. The projection, using **assumed** GPU throughput that has not
been benchmarked, puts the break-even at about 12 billion input tokens a month on 4× L40S
(`g6e.12xlarge`) with a Savings Plan and 90% prompt reuse. For a single agent task, the
crossover is about 25 minutes when 16 tasks share the node. Bedrock's own prompt caching, where
it applies, pushes both further out.

## Run it locally (no GPU, no AWS)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
export PYTHONPATH=src
uvicorn selfhost.devserver:app --port 8000 &   # stands in for the engine, imitates prefix caching
uvicorn selfhost.app:app --port 8080           # gateway and chat page at http://127.0.0.1:8080
```

The Bedrock column uses your local AWS credentials. Without them it shows an error, and the
self-hosted column keeps working.

## Deploy to AWS

Prerequisites: AWS credentials, the AWS CDK CLI, and Bedrock access to the comparison model.
Bedrock's tokens-per-minute quota for a new model can start low; raise it in Service Quotas if
you see `ThrottlingException`.

```bash
cdk deploy SelfHostedInference --app ".venv/bin/python app.py" -c keepsake:mode=selfhost \
  [-c selfhost:model=minimax-m2.5] [-c selfhost:instanceType=r7a.4xlarge] \
  [-c selfhost:purchase=spot|on_demand] [-c selfhost:bedrockModel=<model id>]
```

Pass `--app ".venv/bin/python app.py"`: `cdk.json` calls the system `python3`, which may not
have the dependencies.

After the deploy, use the stack outputs:

1. Run `GetAccessToken`, open `ChatUrl`, and paste the token when the page asks for it.
2. The first boot downloads about 100 GB of weights and publishes them to the S3 model bucket;
   later boots restore from S3. Allow time before the first answer.
3. Run `StopNodeToSaveMoney` when you're not using it, and `StartNode` to bring it back.
   `cdk destroy` removes everything.
4. For gated Hugging Face models, run `SetHuggingFaceToken` with your token. MiniMax M2.5 isn't
   gated.

When GPU quota is available, deploy with a GPU instance type (for example
`-c selfhost:instanceType=g6e.12xlarge`) and the node runs vLLM instead. Before that path is
used, it needs a vLLM release that supports the MiniMax M2 architecture, tensor parallelism
across the GPUs, and a 4-bit build of the model to fit 4× L40S.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The self-hosted tests (`tests/unit/test_selfhost_*.py`) pass. Five Keepsake tests currently
fail: four pipeline tests need `keepsake:sourceInvocationLogsBucket` set in `cdk.json`, and one
analyzer test fails on the sample logs.

## Further reading

- `docs/self-hosted-inference.md`: how the diagram maps to the code and how costs are compared
  (parts of it describe the earlier Llama 3.1 8B build).
- `docs/self-hosted-kv-cache-evaluation.md`, `docs/latency-test.md`: evaluation notes.
- `architecture/full-inference-architecture.drawio`: the full reference architecture.

## Known limits

- One node in one region; the router's prefix index is in memory and resets on restart.
- Spot capacity can be reclaimed; the node is replaced within minutes and restores the model
  from S3.
- CloudFront to the ALB is HTTP inside AWS, protected by the CloudFront prefix list and a secret
  origin header. Add a custom domain, ACM certificate and AWS WAF before real use.
- All GPU cost figures are projections until benchmarked.
