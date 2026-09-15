from dataclasses import dataclass, field

from prefixlens.hashing import block_hash_chain
from prefixlens.request import Request
from prefixlens.tree import RadixTree


@dataclass(frozen=True)
class ProcessResult:
    request_id: str
    cached_prefix_blocks: int
    total_prompt_blocks: int
    first_divergent_block: int | None
    """0-indexed position of the first block that failed to hit, or None if the
    request was a full hit (or had zero complete blocks)."""
    tags: tuple[tuple[str, str], ...] = ()
    diverging_block_tokens: tuple[int, ...] | None = None
    """The actual token content at first_divergent_block, retained so downstream
    attribution can distinguish 'unique-content-per-request' (UUID injection)
    from 'same-content-per-request-but-evicted' (capacity thrashing). None when
    the request was a full hit or had no complete blocks."""


@dataclass(frozen=True)
class TagStats:
    """Aggregate stats for one tag value (e.g. tenant=acme).

    hit_rate is token-weighted — cached_blocks / total_blocks — matching what
    vLLM's `vllm:prefix_cache_hits / vllm:prefix_cache_queries` reports. This
    is the correct comparable when reconciling sim against a real engine.
    """
    hit_rate: float
    total_requests: int
    cached_blocks: int
    total_blocks: int


@dataclass(frozen=True)
class DivergentPosition:
    """A block position where cache reuse frequently fails in one tag bucket.

    unique_content_ratio measures whether the token content AT this block
    position varies across the miss requests:
      - 1.0  every miss has different tokens there → UUID/timestamp injection
             (fix: restructure the prompt, move the variable field elsewhere).
      - 0.0  every miss has the same tokens there → capacity thrashing
             (fix: grow the cache; the parent chain is what's diverging).
      - between → mixed causes, worth splitting the bucket further.

    Token-space, not text-space: v0.1 doesn't detokenize, so the diagnosis
    is "block 3 of tenant=widgets is where things break, and the content
    there is per-request-unique." Human-readable substring lift comes when
    the tokenizer path is wired up.
    """
    block_position: int
    miss_count: int
    unique_content_ratio: float


@dataclass(frozen=True)
class Report:
    hit_rate: float
    total_requests: int
    cached_blocks: int
    total_blocks: int
    by_tag: dict[str, dict[str, TagStats]] = field(default_factory=dict)
    """Nested breakdown: by_tag[tag_key][tag_value] -> TagStats.

    Every tag key that appears on any request gets a top-level entry. Within
    each key, only tag values actually present are keyed. A request contributes
    to a bucket only if it carries that (key, value) pair — requests without
    the key don't count against it.
    """
    by_tag_divergence: dict[str, dict[str, tuple[DivergentPosition, ...]]] = field(
        default_factory=dict
    )
    """Same shape as by_tag: outer key = tag name, inner key = tag value.
    Value is a tuple of DivergentPosition ranked by miss_count desc. Only
    positions where at least one request first-diverged are included; a bucket
    with zero misses gets an empty tuple.
    """


def _hit_rate(cached: int, total: int) -> float:
    return cached / total if total > 0 else 0.0


def _compute_divergent_positions(
    misses: list[tuple[int, tuple[int, ...]]],
) -> tuple[DivergentPosition, ...]:
    """Group miss records by block position, computing per-position miss count
    and unique-content ratio. Sorted by miss_count desc, then position asc.

    Input: list of (first_divergent_block, diverging_block_tokens) per miss.
    """
    by_pos: dict[int, list[tuple[int, ...]]] = {}
    for pos, tokens in misses:
        by_pos.setdefault(pos, []).append(tokens)

    positions: list[DivergentPosition] = []
    for pos, token_blocks in by_pos.items():
        distinct = len(set(token_blocks))
        ratio = distinct / len(token_blocks) if token_blocks else 0.0
        positions.append(
            DivergentPosition(
                block_position=pos,
                miss_count=len(token_blocks),
                unique_content_ratio=ratio,
            )
        )

    positions.sort(key=lambda p: (-p.miss_count, p.block_position))
    return tuple(positions)


class RadixCacheSimulator:
    def __init__(
        self,
        block_size: int,
        capacity_blocks: int,
        hash_fn: str = "fnv1a",
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        if hash_fn != "fnv1a":
            raise ValueError(f"unsupported hash_fn: {hash_fn!r}")
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        self.hash_fn = hash_fn
        self._tree = RadixTree()
        self._records: list[ProcessResult] = []
        self._step = 0
        self._evictions = 0

    @property
    def tree(self) -> RadixTree:
        return self._tree

    @property
    def evictions(self) -> int:
        return self._evictions

    def process(self, req: Request) -> ProcessResult:
        self._step += 1
        chain = block_hash_chain(req.token_ids, self.block_size)
        matched, _touched = self._tree.match_and_insert(chain, now=self._step)

        # Enforce capacity after insertion. Real allocators briefly exceed the
        # bound mid-request; we mirror that — the caller-visible state after
        # each process() respects capacity_blocks.
        while len(self._tree) > self.capacity_blocks:
            evicted = self._tree.evict_lru_leaf()
            if evicted is None:
                break  # nothing left to evict; capacity < 1 would be a config bug
            self._evictions += 1

        first_divergent = matched if matched < len(chain) else None
        if first_divergent is not None:
            start = first_divergent * self.block_size
            diverging_block = req.token_ids[start : start + self.block_size]
        else:
            diverging_block = None

        result = ProcessResult(
            request_id=req.request_id,
            cached_prefix_blocks=matched,
            total_prompt_blocks=len(chain),
            first_divergent_block=first_divergent,
            tags=req.tags,
            diverging_block_tokens=diverging_block,
        )
        self._records.append(result)
        return result

    def report(self) -> Report:
        cached = sum(r.cached_prefix_blocks for r in self._records)
        total = sum(r.total_prompt_blocks for r in self._records)

        # Per-tag accumulator: {tag_key: {tag_value: [cached, total, requests]}}
        by_tag_accum: dict[str, dict[str, list[int]]] = {}
        # Per-tag miss accumulator for divergence attribution:
        # {tag_key: {tag_value: [(first_divergent_block, diverging_block_tokens), ...]}}
        by_tag_misses: dict[str, dict[str, list[tuple[int, tuple[int, ...]]]]] = {}

        for r in self._records:
            for k, v in r.tags:
                bucket = by_tag_accum.setdefault(k, {}).setdefault(v, [0, 0, 0])
                bucket[0] += r.cached_prefix_blocks
                bucket[1] += r.total_prompt_blocks
                bucket[2] += 1

                if r.first_divergent_block is not None and r.diverging_block_tokens is not None:
                    by_tag_misses.setdefault(k, {}).setdefault(v, []).append(
                        (r.first_divergent_block, r.diverging_block_tokens)
                    )

        by_tag: dict[str, dict[str, TagStats]] = {
            k: {
                v: TagStats(
                    hit_rate=_hit_rate(c, t),
                    total_requests=n,
                    cached_blocks=c,
                    total_blocks=t,
                )
                for v, (c, t, n) in values.items()
            }
            for k, values in by_tag_accum.items()
        }

        # Divergence: include an entry for every tag bucket that appears in by_tag,
        # even if it has zero misses (empty tuple) — makes downstream rendering
        # simpler (no need to check both dicts).
        by_tag_divergence: dict[str, dict[str, tuple[DivergentPosition, ...]]] = {}
        for k, values in by_tag_accum.items():
            by_tag_divergence[k] = {}
            for v in values:
                misses = by_tag_misses.get(k, {}).get(v, [])
                by_tag_divergence[k][v] = _compute_divergent_positions(misses)

        return Report(
            hit_rate=_hit_rate(cached, total),
            total_requests=len(self._records),
            cached_blocks=cached,
            total_blocks=total,
            by_tag=by_tag,
            by_tag_divergence=by_tag_divergence,
        )
