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
# Analyze a trace of prompts
$ prefixlens analyze traces.jsonl --tokenizer meta-llama/Llama-3-8B

Cache hit rate: 34.2%
By tag:
  tenant=acme      68.1%    (12,433 requests)
  tenant=widgets    0.3%    (8,912 requests)

Top divergent positions (tenant=widgets):
  position 47:  6,204 requests   ← 69% of misses
  position 92:  1,880 requests
  position 118:   612 requests

Top cache-killer substrings (tenant=widgets):
  1. "session_id: <uuid>"        appears at pos 41-58 in 6,204 requests
  2. "generated_at: <iso8601>"   appears at pos 88-104 in 1,880 requests

Suggestion:
  Moving `session_id` and `generated_at` fields from the system prompt
  into the user message would eliminate the top two divergent positions
  for tenant=widgets. Rerun to measure the actual hit-rate lift.
```

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

**Spec, not built yet.** README-first — this document is the contract; the code will follow.

If this problem is one you also have, or if you've solved it a different way, open an issue. Real workload traces (anonymized) welcome.
