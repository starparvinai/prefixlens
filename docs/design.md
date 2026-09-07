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
