# prefixlens

**Diagnose why your LLM prefix cache is missing.** Given a JSONL of prompts, `prefixlens` simulates a radix-tree prefix cache, breaks the hit rate down per-tenant / per-route, and pinpoints the block position where cache reuse fails — plus whether that failure is unique-content-per-request (a UUID injection) or shared-content-repeatedly-evicted (capacity thrashing).

[![PyPI](https://img.shields.io/pypi/v/prefixlens.svg)](https://pypi.org/project/prefixlens/)
[![Python](https://img.shields.io/pypi/pyversions/prefixlens.svg)](https://pypi.org/project/prefixlens/)
[![License](https://img.shields.io/pypi/l/prefixlens.svg)](https://github.com/starparvinai/prefixlens/blob/main/LICENSE)

```bash
pip install prefixlens
```

---

## The problem

You run vLLM in production. Two tenants share the same instance. Tenant A gets a 68% prefix-cache hit rate. Tenant B gets 0.3%. Everything looks fine in `/metrics` — the engine is working, both tenants have similar traffic. Something in Tenant B's prompts is breaking cache reuse.

Today, finding it means eyeballing prompts side-by-side until you spot a UUID or timestamp somewhere near the front — hours of manual diffing per incident. The signal you actually want — *"the first divergent block for Tenant B's traffic is at position 3, with 100%-unique content"* — doesn't exist in any tool. vLLM's `/metrics` gives you one number per engine, and that's the whole story it can tell.

`prefixlens` produces the signal the engine can't.

## What it does (v0.1)

Three CLI modes, all CPU-only (no GPU required):

- **`prefixlens analyze`** — corpus-wide report from a JSONL prompt trace. Overall hit rate, per-tag breakdown (tenant / route / model / any custom axis), and per-tag *divergent-position attribution* with a UUID-vs-thrashing diagnosis label.
- **`prefixlens explain`** — per-request block-by-block trace. Answers *"why was request X slow?"* with both the walk through that request's chain and the aggregate divergence context for its tag buckets.
- **`prefixlens validate`** — differential check against a real vLLM `/metrics` scrape. Verdict `OK` (within tolerance) or `DIVERGED` — the credibility anchor for everything downstream.

---

## 30-second demo

```bash
$ pip install prefixlens
$ curl -sO https://raw.githubusercontent.com/starparvinai/prefixlens/main/examples/chen_multitenant.jsonl
$ prefixlens analyze chen_multitenant.jsonl --block-size 16 --capacity-blocks 100

prefixlens analyze — 10 requests, 40 blocks total

  overall hit rate:  40.0%  (16 / 40 blocks)

  by tenant:
    acme      80.0%  (5 req, 16/20 blk)
    widgets    0.0%  (5 req, 0/20 blk)

  divergent positions for tenant=widgets:
    block   0      5 miss  (100.0% unique — unique content per request)
```

Read the bottom line the way an operator would: *every widgets request first-misses at block 0, and the content at block 0 is different every time.* That's the classic **session-UUID-at-position-0** shape — a variable field at the top of the prompt poisoning downstream cache reuse. Restructure to move it into the user message.

The overall **40% is what vLLM's `/metrics` would report** — and exactly what obscured the real story. The tool pulls the tenants apart and locates *where* and *what kind* of divergence you're looking at:

- **`100% unique content per request`** — a variable field (UUID, timestamp, request ID) at that block. Fix: **restructure the prompt**.
- **`shared content, thrashing`** — same tokens repeatedly evicted. Fix: **grow the cache** — the working set exceeds capacity.

Two symptoms that look identical in the aggregate hit rate; opposite fixes.

---

## Real-workload evidence

The tool has been run against three real corpora with different shapes. Same tool, same flags, three consistent diagnoses. Full reports live in [`reports/`](reports/).

| Workload | Shape | Requests | Overall hit rate | Divergence pattern |
|---|---|---|---|---|
| [**WildChat-1M**](reports/wildchat_2k.txt) (2K conversations) | ad-hoc chat | 5,832 | 70.2% | block 0 unique (~97%) |
| [**LMSys-Chat-1M**](reports/lmsys_2k.txt) (2K conversations, 25+ models) | ad-hoc chat | 3,817 | 81.7% | block 0 unique (~97%) |
| [**MMLU 5-shot**](reports/mmlu_5shot.txt) (57 subjects × 50 questions) | batch eval | 2,850 | 84.0% | block K per subject, 100% unique (K = end of preamble) |

Both chat corpora produce the same shape: block 0 dominates because different users share no common prefix — only within-conversation history reuses. MMLU produces a completely different shape: dense shared-preamble hits blocks 0..K−1 across every question in a subject, divergence lands at exactly block K where the per-question stem starts. Three workloads, three diagnoses, one tool.

Regenerate any of these with the adapters in [`examples/`](examples/) and re-run analyze in a few seconds.

---

## The other two modes

### `prefixlens explain` — per-request diagnosis

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

Per-request trace **plus** aggregate context for the request's tag buckets. So the reader knows both *"your request diverged at block 0"* and *"you're one of 5 widgets requests with the same shape — this is systemic, not a one-off."*

### `prefixlens validate` — check the sim against a real vLLM

```bash
$ curl http://vllm:8000/metrics > metrics.txt   # after your workload runs
$ prefixlens validate requests.jsonl \
    --metrics-file metrics.txt \
    --block-size 16 \
    --capacity-blocks 4096

  sim hit rate:    42.1%   (8,412 blocks processed)
  real hit rate:   43.7%   (3,678 hits / 8,412 queries)
  delta:           1.60 pp   (tolerance ±3.0 pp)
  verdict: OK
```

Differential test at the top-line hit rate. `OK` (exit 0) or `DIVERGED` (exit 1) — CI-usable. If sim and engine agree within tolerance, downstream attribution is trustworthy. If they don't, the config is wrong before anything else — `--block-size` and `--capacity-blocks` must match the engine's real config.

---

## Input format

JSONL, one request per line:

```json
{"request_id": "abc123", "token_ids": [1, 2, 3, ...], "tenant": "acme", "route": "/v1/chat/completions"}
```

- `token_ids` (list of ints) required — pre-tokenized. The v0.1 CLI has no `--tokenizer` flag on purpose: `pip install prefixlens` stays dependency-free. Pre-tokenize your corpus with whatever tokenizer the engine used (`tiktoken.get_encoding("cl100k_base").encode(...)` for the OpenAI family). That way the block hashes match what the engine actually saw.
- Any tag fields (`tenant`, `route`, `model`, or a nested `tags: {...}`) become groupable axes in the report.
- `request_id` optional (defaults to `line-N`).

The library's `load_jsonl(path, tokenizer=<callable>)` accepts a `prompt` field + a bring-your-own tokenizer. See [`src/prefixlens/loader.py`](src/prefixlens/loader.py).

---

## What it is NOT

- **Not a GPU profiler.** Doesn't measure kernel time. If compute is your bottleneck, this won't help.
- **Not a benchmark harness.** Analyzes what happened, not what would happen at 10× QPS.
- **Not a replacement for the engine.** vLLM has an [in-tree analyzer in development](#related-work). `prefixlens` sits on the layer above — attribution and cross-request patterns the engine won't ship.
- **Not text-aware yet.** Divergence is reported at token-block granularity, not human-readable substrings. `"block 0 unique"` today; `"session_id=<uuid>"` in v0.2 once the tokenizer path lands.

---

## Roadmap

**v0.1 (this release)** ✅
- CPU radix simulator with parent-chained FNV-1a-64 hashing, LRU eviction (leaves-only + min-heap + lazy deletion).
- JSONL streaming loader with bring-your-own tokenizer.
- `analyze` / `explain` / `validate` CLI, human + JSON output modes.
- Per-tag divergent-position attribution with UUID-vs-thrashing diagnosis.
- Real-workload adapters for WildChat, LMSys, MMLU. Injection self-validation harness.
- 127 tests.

**v0.2 (next)**
- Optional `pip install prefixlens[tiktoken]` extra for a first-party tokenizer.
- **Text-space substring mining** — turn *"block 0 unique per request"* into *"session_id=<uuid>"*.
- **Text-space reordering-suggestion projections** — *"move field X to position Y, projected lift Δpp"* with counterfactual simulation.
- **`prefixlens lint`** — CI-friendly prompt-template hazard check.

**v0.3+**
- SGLang / TRT-LLM parity (engine-specific radix and block semantics).
- Live engine hooks (vLLM plugin, TRT-LLM KV event API).
- Attribution across evictions in `explain` — *"block 2 hit against a node originally inserted by req_98 at step 12."*

---

## Design notes

See [`docs/design.md`](docs/design.md) for the load-bearing decisions:

- Why the radix tree isn't path-compressed (block-aligned makes it a wash).
- Why eviction is leaves-only (parent-chained hashing orphans descendants).
- Why LRU uses a min-heap with lazy deletion instead of scanning leaves.
- The UUID-vs-thrashing diagnosis and why `unique_content_ratio` needs *raw token blocks*, not block hashes.
- Why `validate` compares only the top-line hit rate (aggregate as go/no-go, finer checks as diagnosis).
- Why the CLI is opinionated for humans (tag cardinality auto-suppress) but `--json` output stays unfiltered.

---

## Related work

**vLLM RFC [#47993](https://github.com/vllm-project/vllm/issues/47993)** proposes an in-tree offline prefix-cache workload analyzer (`vllm analyze-prefix-cache`), being implemented across PRs [#48369](https://github.com/vllm-project/vllm/pull/48369) and [#48838](https://github.com/vllm-project/vllm/pull/48838). The in-tree tool will provide block-hash simulation using vLLM's exact hashing semantics — a real advantage no external tool can match.

`prefixlens` is complementary, not competitive:

| Feature | vLLM in-tree analyzer | prefixlens |
|---|---|---|
| Block-hash simulation with engine's exact semantics | ✅ | ✅ (calibrated) |
| Per-tag breakdown | ✅ | ✅ |
| Per-position divergence attribution | (not in scope) | ✅ |
| UUID-vs-thrashing diagnosis | (not in scope) | ✅ |
| Per-request `explain` mode | (not in scope) | ✅ |
| Simulated-vs-actual reconciler | (not in scope) | ✅ |
| Text-space substring mining | (not in scope) | v0.2 |
| Engine-agnostic (SGLang, TRT-LLM) | ❌ (vLLM only) | v0.3 |

**Related tooling.** [LMCache](https://github.com/lmcache/lmcache) exposes rich per-request cache observability and a Redis-CLI-based cache introspection tool, but not workload-level attribution. [llm-d's precise-prefix-cache-routing](https://github.com/llm-d/llm-d) and [Truefoundry's cache-aware routing](https://www.truefoundry.com/blog/kv-cache-routing-why-standard-load-balancers-break-prefix-caching-and-how-to-fix-it) address the routing layer (a different fix: move requests, don't restructure prompts).

---

## Contributing

Real workload traces (anonymized) very welcome — open an issue with a description of the shape and I'll help wire an adapter.

Bug reports, especially "the sim says X but real vLLM says Y" pairs where `validate` produces `DIVERGED` on a workload it shouldn't — those are the highest-value reports.

## License

MIT. See [LICENSE](LICENSE).
