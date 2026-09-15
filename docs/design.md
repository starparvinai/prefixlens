# Design notes

## Explain mode: two views, bolted together on purpose

`explain` answers "why did request X miss?" by combining two things that individually don't add up to a diagnosis:

1. **A per-request block-by-block trace** — for the target request, walk its chain position-by-position, marking each block HIT or MISS. Trivially derivable from `first_divergent_block` on that request's `ProcessResult`.
2. **The aggregate divergence context for the request's tag buckets** — from the corpus-wide report we already build, pull the divergence signal for whichever tag buckets this request belongs to, at exactly the position it diverged. This tells the operator "you're not a one-off; 142 requests in tenant=widgets have this same shape."

Neither view alone answers the operator's question. The trace shows *where* the request broke; the context shows *what class of problem* that is. Together they turn one paged incident into a systemic diagnosis.

### State preservation shape

The cost of `explain` is that we retain the *full token stream* of every request on `ProcessResult.token_ids`. Storing 500 tokens × 10K requests = ~40MB for a big corpus — cheap. And because `token_ids` is an immutable tuple, the reference is shared between `Request` and `ProcessResult` (no copy), so the memory hit is exactly one copy of each request's tokens, not two.

### What `explain` deliberately does not do in v0.1

**Attribution — "which earlier request seeded the block we hit?"** — genuinely useful, but requires tracking `first_inserted_by_request_id` on `RadixNode` and deciding what happens to that field across evict-then-reinsert cycles (keep the original inserter? overwrite with the latest?). Punted to v0.2. In v0.1, `explain` says "block 2 hit," not "block 2 hit against a node originally inserted by req_98 at step 12."

**Pure function boundary.** `explain_request(sim, request_id, report) -> RequestExplanation` takes both the simulator and the pre-computed report. It doesn't recompute the report on each call. Matters if the CLI ends up explaining several requests in one session (a `--request-ids` list flag, later); the corpus-wide analysis runs once, then O(N) lookups per request explained.

## Validate mode: what we're differentially testing, and why the top-line hit rate is enough

`prefixlens validate` compares one number — the aggregate prefix-cache hit rate — between our CPU sim and a real vLLM `/metrics` scrape taken after the same request stream ran on both. It's a differential test in the same sense a JIT is differentially tested against its interpreter or a new consensus implementation against a reference: the sim IS a reimplementation of vLLM's block-manager behavior, and if it can't reproduce the single most-aggregate output, the finer-grained outputs (per-tenant, per-position) are almost certainly wrong too.

**Why one number is enough for v0.1.**

The per-tenant, per-position, and per-request breakdowns all derive from the same simulation state — they're views over the identical set of `process()` decisions the sim already made. If the aggregate hit rate matches, the shared underlying decisions match. If the aggregate diverges, chasing which specific view diverged is a waste; the substrate is off, and no view is trustworthy.

Later modes (block-lifetime histograms, eviction rate) diagnose *which kind* of divergence when the aggregate disagrees — they answer "we're off by 5pp; is it the eviction policy, the hash function, or the request order?" But that's post-diagnosis. The top-line check is the go/no-go gate that determines whether any of the finer questions are worth asking.

**Why we compare against `/metrics`, not per-request state.**

vLLM exposes prefix-cache activity as two Prometheus counters — `vllm:prefix_cache_hits` and `vllm:prefix_cache_queries` — and no per-request hit/miss decision. That gap is exactly the one `prefixlens` is built to close: the engine can't tell you *which* requests missed. But it means our ground-truth reference is aggregate-only. So we compare aggregate-to-aggregate.

**The load-bearing constraints for the check to be meaningful:**

1. **`--block-size` and `--capacity-blocks` must match the real engine.** vLLM defaults to 16-token blocks; capacity is set from GPU memory config and varies. A wrong config produces a false DIVERGED verdict. Most user-side mistake we expect to see, so the "verdict: DIVERGED" output prints this as the first troubleshooting hint.
2. **The request log must be the exact stream, in order.** Skipped or reordered requests desync the sim's cache from real. vLLM's `--request-logging` gives this directly.
3. **The `/metrics` snapshot must be taken *after* the workload finishes.** Counters are monotonic totals; a snapshot taken during the workload measures partial state.

**Trust score.** Absolute difference in hit rate, in percentage points. Verdict is `OK` if `≤ tolerance-pp` (default 3.0 per SPEC §8), else `DIVERGED`. Exit code 0 on OK, 1 on DIVERGED — scripts and CI can use this directly.

**What we DON'T validate.**

- Semantic correctness of substring mining (structural, not sim-vs-engine).
- Divergent-position histograms — the sim IS the source of truth; vLLM exposes no comparable data.
- Per-request behavior — see above; this is the gap prefixlens fills, so being unable to check it is *the design*, not an oversight.

## Divergent-position attribution: the two diagnoses that share one symptom

The load-bearing idea behind cache-killer attribution is that **the same first-divergent-block position can mean two very different things**, with opposite fixes.

Say a tag bucket has 100 requests, all first-diverging at block position 3.

- **World A — UUID injection.** Each request has *different* token content at block 3: a session UUID, a per-request timestamp, a request ID sitting near the top of the prompt. Every miss is a new block-hash in the tree; there was nothing to hit. **Fix: restructure the prompt.** Move the variable field somewhere the cache doesn't care about (e.g. the user message, past the shared prefix).
- **World B — capacity thrashing.** Each request has *identical* token content at block 3, but they all still miss — because the block got evicted before the next request arrived. Same hash, wrong lifetime. **Fix: grow the cache.** The working set is bigger than what fits.

A tool that reports "top divergent position: block 3, 100 misses" without separating these two worlds is only half-useful; the two fixes are opposite.

`unique_content_ratio` is the discriminator. For the set of miss requests at a position, count the number of distinct token blocks appearing there and divide by the miss count:

- Ratio ≈ 1.0 → every miss's block content is distinct → World A.
- Ratio → 0 → every miss's block content is the same → World B.
- Anything in between → mixed cause; the CLI labels it "mixed content" and asks the reader to look closer.

### Why counting unique *token blocks* is the right question, not unique *block hashes*

Block hashes are parent-chained: `hash(parent_hash + block_tokens)`. Two requests with the same tokens at position P but different chain-history at 0..P−1 have different block hashes at P. Counting unique hashes conflates *"different content"* with *"different history."* Counting unique raw-token blocks separates them — which is what "was the block content per-request-unique" actually asks.

### Storage cost

To answer the question after `process()` returns, the simulator retains the token content of only the diverging block on each `ProcessResult` (`diverging_block_tokens: tuple[int, ...] | None`). Full-hit requests carry `None`. Memory cost is O(one 16-token tuple per miss), not O(full prompt per request) — targeted retention, not a general "keep the whole thing" hack.

## The JSONL loader boundary

The loader is where an on-disk trace becomes an in-memory `Request` stream. Three design decisions worth remembering:

**Bring-your-own tokenizer.** `load_jsonl` takes a `Callable[[str], Sequence[int]]` and calls it only when needed. There is no bundled tokenizer, no soft dependency on `tiktoken` or `transformers`. This keeps `pip install prefixlens` small and lets callers use whatever tokenizer they already send to the engine — guaranteeing the analysis matches what the engine actually saw. A corpus with `token_ids` already present skips the tokenizer entirely (the fast, exact-match path).

**Streaming, not load-all.** The loader is a generator. A 10K-prompt fixture fits in memory; a 10M-line production trace does not. Everything downstream (the simulator's `process()` loop, future aggregation, the CLI) consumes iterators, so nothing forces the corpus into RAM. Callers that want a list just wrap `list(...)`.

**Fail loud with line numbers, don't skip silently.** Malformed JSON, non-integer `token_ids`, missing required fields — all raise `ValueError` naming the offending line. This is a developer-facing tool; a silent skip masks corpus bugs that produce a wrong report, which is worse than any crash. A `--skip-invalid` mode is a plausible v0.2, but adding it before we know the failure shape would be premature.

Tag flattening is the last small thing: top-level `tenant`/`route`/`model` and any nested `tags: {...}` merge into a single sorted `(key, value)` tuple on the `Request`. Downstream aggregation sees one namespace, not two. Nested overrides top-level on collision (last-write-wins).

## The "radix tree" naming

`prefixlens` calls its core data structure a **radix cache** and its class `RadixTree` because that is the term the LLM-serving community uses (SGLang's "RadixAttention," vLLM's "radix cache" internals, LMCache's docs). Strictly, a radix tree is a **path-compressed trie** — chains of single-child nodes collapse into one edge labeled with the sequence of keys along that chain.

What we actually implement is a **trie of block hashes with a hash-indexed lookup dict**: each node holds exactly one block hash, insertion walks one level per block, and no path compression happens. This is the same shape vLLM uses internally (in vLLM's case, expressed as a flat `dict[block_hash → block_metadata]` with parent pointers — semantically equivalent to our tree). We keep the community naming to stay searchable and to match how practitioners talk about this system.

## Why no path compression

Path compression is essential for token-level caches like SGLang's RadixAttention, where every token is potentially a cache-key element. Without compression, a 4000-token prompt would allocate 4000 nodes. Compression collapses long unbranched chains into single labeled edges, saving memory proportional to `1 − (branch_count / token_count)`.

For a **block-aligned** cache (vLLM's model, and prefixlens v0.1's model), path compression buys nothing and costs at every other operation:

1. **Every block is independently evictable.** Eviction operates on individual blocks. A compressed edge holding `[H_A, H_B, H_C]` must split when any middle block is evicted. The split cost paid at eviction time cancels the insert-time savings.
2. **Block-level branching is common in real workloads.** Hundreds of prompts sharing a system prompt prefix but each with unique tails means compressed edges rarely stay compressed — you pay to compress and then pay again to split.
3. **The lookup index degrades.** Our `dict[block_hash → node]` gives O(1) node lookup, which every future feature (eviction, tree diffs, "find where block X lives") depends on. With compressed edges, the index becomes `dict[block_hash → (node, position_in_edge)]` — every mutation has to keep those positions consistent.
4. **Walk complexity is unchanged.** Matching against a compressed edge still requires comparing each hash in the edge against each hash in the request's chain. Same O(depth) work, more bookkeeping.

**Rule of thumb:** path compression is worth it when the atomic unit is small (per-character, per-token) and chains are long and thin. It hurts when the atomic unit is already coarse (16-token blocks) and branching is heavy.

## Consequences of choosing block-alignment

Because every hash in the tree corresponds to exactly one 16-token block:

- **The `RadixTree._index` is authoritative.** Given any block hash, we find the node in O(1). No need to walk paths or split edges.
- **Divergence position equals matched depth.** The walk exits at the first unmatched hash; the loop counter *is* the divergence position, so `ProcessResult.first_divergent_block` is computed for free.
- **Cache-safety is by construction.** Parent-chained hashing (`hash(parent_hash + block_tokens)`) means every node in the tree represents both a specific block content *and* a specific history that led to it. Sharing a node is legitimate sharing — same tokens, same K/V.

## Related: how vLLM does the same thing without a tree

vLLM's internal representation is a flat `dict[block_hash → BlockMetadata]` where each `BlockMetadata` carries a parent-hash pointer. It's the same structure as our trie, expressed differently — walking the "tree" means iteratively looking up successive block hashes in the dict, following parent pointers upward for LRU chain-of-ancestry queries. We keep the explicit tree form because a debugging tool benefits from an inspectable structure; a production serving engine benefits from the flat dict's cache locality. Different constraints, same underlying model.

## LRU eviction

### Leaves-only eviction

We only evict leaves of the tree, never internal nodes. This is the same rule vLLM uses, and the reason is a semantic one, not just a performance heuristic:

Each block's hash is `hash(parent_hash + block_tokens)`. A block's node is *meaningful only because its parent is still in the tree.* Evict a middle node and its descendants become an orphan chain — the hashes still refer to real content, but they're unreachable from the root, so no future request can walk into them. The block is functionally dead cache; it consumes an index slot without ever producing a hit.

Leaves-only eviction sidesteps this cleanly. When a leaf L is evicted:
- L's parent P may or may not have other children.
- If P has other children, P stays where it is (still an internal node).
- If P was L's only child, P *becomes* a leaf. It joins the eviction candidate set.

This is where the **cascade** comes from. A single "evict one block" call at the tree level can turn into "evict several blocks" at the simulator level if the request that inserted them made a long unbranched chain.

### The heap + lazy deletion

Naïve LRU on a tree needs, for each eviction, to find *the leaf with the smallest `last_used_at`.* Linear scan of `_index` filtered to leaves is O(n) per eviction — untenable for the 10k+ prompt corpora prefixlens targets.

We maintain a **min-heap of leaves keyed by `last_used_at`.** Standard idea, with two wrinkles this tree needs:

1. **Nodes get *touched* on hit** — a walk through the tree bumps `last_used_at` on every node it passes. If a leaf is touched, the heap entry for it now has a stale timestamp: it says "this node was last used at time 3" when the node itself is at time 12. Removing that stale entry means finding it (O(n)) and re-heapifying (O(log n)). We don't want to pay that on every hit.
2. **Nodes get *promoted* from leaf-ness** — a node that was a leaf can gain a child on the next insert. Its heap entry is now stale in a different way: not the timestamp, the leaf-ness.

Both are handled with **lazy deletion**. On pop:
- If the node's `last_used_at` > the entry's timestamp: the entry is stale (node was touched later). Push a fresh entry with the current timestamp and keep popping.
- If the node has children now: not a leaf anymore. Discard the entry and keep popping.
- If the node isn't in the index anymore: already evicted. Discard and keep popping.
- Otherwise: this is the true LRU leaf. Evict it.

This lets us do O(1) work per touch (nothing) and O(log n) amortized per eviction, with heap size bounded by the number of unique blocks ever inserted (stale entries clean themselves up as we walk past them).

### When eviction runs

After `match_and_insert`, not during. The tree briefly exceeds `capacity_blocks` mid-request — same as any real allocator with a high-water mark — and the caller-visible state after each `process()` call respects the bound. Doing it this way keeps the walk logic single-purpose (matching + inserting; no interaction with eviction bookkeeping) and matches how real block managers behave.

### The "earliest blocks survive" invariant

A subtle consequence worth naming: when a request's prompt is longer than the entire cache, leaves-only eviction preserves the *earliest* blocks of that chain, not the latest.

Trace: request has 5 blocks A→B→C→D→E, capacity is 2. All 5 blocks get inserted (chain: root→A→B→C→D→E, only E is a leaf). Eviction loop starts: E is the sole leaf, evicted. D becomes a leaf, evicted. C becomes a leaf, evicted. Tree ends with {A, B}, not {D, E}.

Not just a quirk — this is *correct* simulator behavior. Any future re-request has to walk from the root, so the useful blocks to keep are the ones a walk from the root can actually reach. Keeping E without A, B, C, D would produce a phantom cache that never hits.
