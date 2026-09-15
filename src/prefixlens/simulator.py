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


def _hit_rate(cached: int, total: int) -> float:
    return cached / total if total > 0 else 0.0


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
        result = ProcessResult(
            request_id=req.request_id,
            cached_prefix_blocks=matched,
            total_prompt_blocks=len(chain),
            first_divergent_block=first_divergent,
            tags=req.tags,
        )
        self._records.append(result)
        return result

    def report(self) -> Report:
        cached = sum(r.cached_prefix_blocks for r in self._records)
        total = sum(r.total_prompt_blocks for r in self._records)

        # Per-tag accumulator: {tag_key: {tag_value: [cached, total, requests]}}
        by_tag_accum: dict[str, dict[str, list[int]]] = {}
        for r in self._records:
            for k, v in r.tags:
                bucket = by_tag_accum.setdefault(k, {}).setdefault(v, [0, 0, 0])
                bucket[0] += r.cached_prefix_blocks
                bucket[1] += r.total_prompt_blocks
                bucket[2] += 1

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

        return Report(
            hit_rate=_hit_rate(cached, total),
            total_requests=len(self._records),
            cached_blocks=cached,
            total_blocks=total,
            by_tag=by_tag,
        )
