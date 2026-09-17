# Changelog

All notable changes to `prefixlens` are documented here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); the project follows [SemVer](https://semver.org/).

## [0.1.0] — 2026-09-16

First public release. The v0.1 milestone in [SPEC.md](docs/design.md) is closed.

### Added

- **`prefixlens analyze`** — corpus-wide report from a JSONL trace: overall hit rate, per-tag breakdown (hit rate, request count, cached/total blocks), and per-tag divergent-position attribution with the UUID-vs-thrashing diagnosis.
- **`prefixlens validate`** — differential check against a real vLLM `/metrics` scrape. Verdict `OK` (within tolerance, default ±3pp) or `DIVERGED`, exit code 0/1 for CI.
- **`prefixlens explain`** — per-request block-by-block trace + aggregate-divergence context for the request's tag buckets. The "flight recorder" answer to *"why was request X slow?"*
- **`--tag NAME` whitelist + cardinality auto-suppress** on `analyze` so real workloads with high-cardinality tags (`conversation_id`, per-user IDs) don't drown the report. `--json` output stays unfiltered.
- **Streaming JSONL loader** with bring-your-own tokenizer — no hard dep on `tiktoken` or `transformers`. Corpora with `token_ids` skip tokenization entirely (fast, exact-match path).
- **CPU radix-cache simulator** with parent-chained FNV-1a-64 block hashing (mirrors vLLM's semantics), LRU eviction via leaves-only + min-heap with lazy deletion.
- **Divergent-position attribution** — for each tag bucket, ranked list of block positions where misses cluster, each with a `unique_content_ratio` metric distinguishing UUID-injection ("unique content per request") from capacity thrashing ("shared content, thrashing").
- **Bundled examples**:
  - `examples/chen_multitenant.jsonl` — 10-request synthetic corpus reproducing Marcus Chen's 2026-05 scenario in one command.
  - `examples/build_wildchat_scenario.py` — adapter turning [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) into a prefixlens corpus.
  - `examples/build_injection_scenarios.py` — self-validation harness: apply known transforms to a real corpus, verify the report matches ground truth.
- **`reports/`** — saved analyze output for the WildChat and injection runs, both human-readable and JSON.
- **Docs**: `docs/design.md` covers the load-bearing design decisions (leaves-only eviction, heap with lazy deletion, why-no-path-compression, JSONL loader boundary, UUID-vs-thrashing diagnosis, validate-mode scope, explain-mode state preservation).
- **Test suite**: 127 tests covering hashing, tree, simulator, eviction, loader, aggregation, divergence, CLI, metrics parser, and explain.

### Known limitations (v0.1)

- No text-space substring mining — divergence is reported at token-block granularity, not text. Turning "block 0 is unique per request" into `"session_id=<uuid>"` requires the tokenizer path, planned for v0.2.
- No `lint` mode yet (proactive prompt-template check for CI). Planned for v0.2.
- No reordering-suggestion projections (SPEC v0.3) — the tool tells you *where* the divergence is, not how much lift you'd get from moving a specific field.
- No attribution across evictions — `explain` says "block 2 hit," not "block 2 hit against a node originally inserted by req_98." Planned for v0.2 with `first_inserted_by_request_id` on `RadixNode`.
- `validate` compares only the top-line hit rate. Block-lifetime histograms and eviction-rate comparisons (SPEC §8 later) are planned diagnosis tools for when the top-line disagrees.

### Related work

vLLM has an in-tree offline prefix-cache workload analyzer in development ([RFC #47993](https://github.com/vllm-project/vllm/issues/47993), PRs [#48369](https://github.com/vllm-project/vllm/pull/48369) and [#48838](https://github.com/vllm-project/vllm/pull/48838)) — the in-tree tool will provide block-hash simulation using vLLM's exact hashing semantics. `prefixlens` is complementary: attribution, cache-killer detection, and cross-engine simulation live here.
