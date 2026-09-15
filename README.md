# prefixlens

**A cache-attribution analyzer for LLM prefix caches. Tells you *which prompts* are killing your hit rate, *at which token*, and *what to change* to fix it.**

Every serving engine (vLLM, SGLang, TRT-LLM, LMCache) reports prefix-cache hit rate as a single number. When that number is bad, the engine can't tell you why. `prefixlens` can.

---

## The problem this exists to solve

You run vLLM in production. Two tenants share the same instance. Tenant A gets a 68% prefix-cache hit rate. Tenant B gets 0.3%. Everything looks fine in the metrics — the engine is working, both tenants have similar traffic. Something in Tenant B's prompts is breaking cache reuse, but the metrics can't tell you what.

Today, finding it means eyeballing prompts side-by-side until you spot a UUID or timestamp somewhere near the front. In a real workload that's hours of manual diffing per incident. The signal you actually want — *"the first divergent token across Tenant B's traffic is at position 47, and it's a session ID injected into the system prompt"* — doesn't exist in any tool.

`prefixlens` produces that signal, and goes one step further: it tells you what to change.

## What it does (v0.1)

The primary output is **attribution and remediation**, not raw statistics. Statistics are the substrate; the report is the product.

**Attribution — where the cache is being broken:**
- **Position-of-first-divergent-token histogram**, per tag: for each tenant/route/session, the distribution of *how far into the prompt* cache reuse is failing. Fat tail at position 47 = something near the top of every prompt is unique.
- **Cache-killer substrings**: recurring high-entropy fragments (UUIDs, timestamps, per-request IDs) that break block boundaries. Ranked by how many misses they explain.

**Remediation — what to change:**
- **Reordering suggestions**: given the killer substrings, propose prompt structure changes ("move `session_id` from system prompt into user message") and identify which divergent positions each change would eliminate.
- **Simulated-vs-actual reconciler**: point it at a vLLM `/metrics` endpoint + a request log; it compares its simulation against what the real engine reported. Trust the analysis before acting on it.

**Substrate — how it computes any of this:**
- CPU simulation of a radix-tree prefix cache with LRU eviction, block-aligned to a configurable size (defaults to vLLM's 16). Given a JSONL of prompts + a tokenizer, it produces the same hit/miss decisions the engine would.
- Engine-agnostic: the simulator is a general radix-tree model. Calibrated against vLLM in v0.1; SGLang and TRT-LLM parity in v0.2.

Everything runs on CPU. No GPU required.

## Usage

```bash
$ pip install -e .
$ prefixlens analyze examples/chen_multitenant.jsonl --block-size 16 --capacity-blocks 100

prefixlens analyze — 10 requests, 40 blocks total

  overall hit rate:  40.0%  (16 / 40 blocks)

  by tenant:
    acme      80.0%  (5 req, 16/20 blk)
    widgets    0.0%  (5 req, 0/20 blk)

  divergent positions for tenant=widgets:
    block   0      5 miss  (100.0% unique — unique content per request)
```

Read that bottom line the way an operator would: *every widgets request first-misses at block 0, and the token content at block 0 is different every time.* Diagnosis: something at the top of the prompt is per-request-unique — the classic session-UUID-at-position-0 shape. Restructure to move it into the user message, not the system prompt.

The overall 40% is exactly what vLLM's `/metrics` gives you — and exactly what obscured the real story in Chen's post. `prefixlens` pulls the tenants apart, and then within `widgets` locates *where* and *what kind* of divergence you're looking at:

- **`100% unique content per request`** — variable field (UUID, timestamp, request ID) at that block. Fix: restructure the prompt.
- **`shared content, thrashing`** — same tokens repeatedly missed. Fix: grow the cache; the working set is bigger than capacity.

The bundled `examples/chen_multitenant.jsonl` reproduces this in 10 requests, on CPU, in milliseconds. Add `--json` for machine-readable output.

**Coming next (see [Roadmap](#roadmap))**: text-space substring mining once the tokenizer path is wired up — turns "block 0 is unique per request" into `"session_id=<uuid>"`, plus reordering suggestions that project the hit-rate lift of each proposed fix.

Corpora with a `"prompt"` field instead of `"token_ids"` need a tokenizer — the v0.1 CLI has no `--tokenizer` flag, so pre-tokenize your corpus. See `src/prefixlens/loader.py` for the library API.

### `prefixlens explain` — per-request diagnosis

When someone gets paged with *"request `req_abc123` was slow, why?"*:

```bash
$ prefixlens explain traces.jsonl --request-id req_abc123 --block-size 16 --capacity-blocks 4096

prefixlens explain — req_abc123

  tags:    tenant=widgets, route=/v1/chat/completions
  blocks:  0 hit, 4 miss, 4 total
  first divergence: block 0

  block-by-block trace:
    block   0  MISS  [9000002, 2000, 2001, 2002, …, 2012, 2013, 2014]
    block   1  MISS  [2015, 2016, 2017, 2018, …, 2028, 2029, 2030]
    block   2  MISS  [2031, 2032, 2033, 2034, …, 2044, 2045, 2046]
    block   3  MISS  [2047, 2048, 2049, 2050, …, 2060, 2061, 2062]

  divergence context (aggregate signal at this position):
    tenant=widgets: 5 miss at block 0 (100.0% unique)
```

Two views bolted together: the per-request block-by-block trace *and* the aggregate signal for the request's tag buckets at that same position. So the reader knows both "your request diverged at block 0" *and* "you're one of 5 widgets requests with the exact same shape — this is systemic, not a one-off." Add `--json` for scripting.

### `prefixlens validate` — the credibility anchor

Everything above is trustworthy only if our CPU sim agrees with what real vLLM would do on the same request stream. `validate` is that check:

```bash
$ curl http://vllm:8000/metrics > metrics.txt   # after your workload runs
$ prefixlens validate requests.jsonl \
    --metrics-file metrics.txt \
    --block-size 16 \
    --capacity-blocks 4096

prefixlens validate — sim vs real vLLM /metrics

  sim hit rate:    42.1%   (8,412 blocks processed)
  real hit rate:   43.7%   (3,678 hits / 8,412 queries)
  delta:           1.60 pp   (tolerance ±3.0 pp)

  verdict: OK
           Sim is calibrated on this workload; downstream
           attribution is trustworthy.
```

Exit code 0 if `OK`, 1 if `DIVERGED` (useful in CI). `--block-size` and `--capacity-blocks` must match the real engine's config; a wrong config is the most common source of false-diverge. Add `--tolerance-pp N` to widen the threshold, or `--json` for machine output.

## What it is not

- Not a GPU profiler. Doesn't measure kernel time. If your bottleneck is compute, not cache, this won't help.
- Not a benchmark harness. It analyzes what happened, not what would happen at 10× QPS.
- Not a replacement for the engine's built-in tooling. vLLM is building an in-tree analyzer (see [Related work](#related-work)); `prefixlens` sits on top of and around it, focused on the *interpretation* layer the engine won't ship.

## Roadmap

- **v0.1** — CPU simulator, JSONL input, per-tag stats, position-of-first-divergent-token histogram, cache-killer substrings, reordering suggestions, simulated-vs-actual reconciler. **Target: 6 weeks from first commit.**
- **v0.2** — SGLang and TRT-LLM parity (engine-specific radix/block semantics). Live engine hooks: vLLM plugin + TRT-LLM KV event API adapter, so you can analyze production traffic without exporting first.
- **v0.3** — Reordering *validator*: given a proposed prompt change, simulate the new hit rate and produce a confidence interval before you ship the change.

## Related work

**vLLM RFC [#47993](https://github.com/vllm-project/vllm/issues/47993)** proposes an in-tree offline prefix-cache workload analyzer (`vllm analyze-prefix-cache`), currently being implemented across PRs [#48369](https://github.com/vllm-project/vllm/pull/48369) and [#48838](https://github.com/vllm-project/vllm/pull/48838) by @harsh543 and @raravind007. The in-tree tool will provide the block-hash simulation and basic statistics — the *substrate* — using vLLM's exact hashing semantics, which is a real advantage no external tool can match.

`prefixlens` is complementary, not competitive:

| Feature | vLLM in-tree analyzer | prefixlens |
|---|---|---|
| Block-hash simulation matching engine semantics | ✅ (by construction) | ✅ (calibrated) |
| Per-tag / per-tenant breakdown | ✅ | ✅ |
| Position-of-first-divergent-token histogram | (not in v1 scope) | ✅ |
| Cache-killer substring mining | (not in scope) | ✅ |
| Reordering suggestions | (not in scope) | ✅ |
| Simulated-vs-actual reconciliation | (not in scope) | ✅ |
| Engine-agnostic (SGLang, TRT-LLM) | ❌ (vLLM only) | v0.2 |

**Related tooling:** [LMCache](https://github.com/lmcache/lmcache) exposes rich per-request cache observability and a Redis-CLI-based cache introspection tool, but not workload-level attribution. [llm-d's precise-prefix-cache-routing](https://github.com/llm-d/llm-d) and [Truefoundry's cache-aware routing](https://www.truefoundry.com/blog/kv-cache-routing-why-standard-load-balancers-break-prefix-caching-and-how-to-fix-it) address the routing layer (a different fix — moving requests, not restructuring prompts).

## Why this tool

Multi-tenant workload observability is a solved craft in backend systems and a wide-open gap in LLM serving. The engines expose the counters they built for themselves — averages, rates, totals. What operators actually need to *fix* things are attribution and per-request breakdowns. `prefixlens` brings that discipline to the prefix cache layer.

## Status

**v0.0.x in progress.** Radix simulator + LRU eviction + JSONL loader + per-tag aggregation + divergent-position attribution + `prefixlens analyze` + `prefixlens validate` + `prefixlens explain` (per-request block-by-block trace + aggregate divergence context) are all in and tested (115+ tests). The Chen scenario reproduces in one command against a bundled fixture. Next up: `prefixlens lint` (CI-friendly prompt-template hazard check) and text-space substring mining.

If this problem is one you also have, or if you've solved it a different way, open an issue. Real workload traces (anonymized) welcome.
