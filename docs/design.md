# Design notes

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
