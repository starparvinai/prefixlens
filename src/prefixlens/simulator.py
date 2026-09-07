from dataclasses import dataclass

from prefixlens.hashing import block_hash_chain
from prefixlens.request import Request
from prefixlens.tree import RadixTree


@dataclass(frozen=True)
class ProcessResult:
    request_id: str
    cached_prefix_blocks: int
    total_prompt_blocks: int


@dataclass(frozen=True)
class Report:
    hit_rate: float
    total_requests: int


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

    @property
    def tree(self) -> RadixTree:
        return self._tree

    def process(self, req: Request) -> ProcessResult:
        self._step += 1
        chain = block_hash_chain(req.token_ids, self.block_size)
        matched, _touched = self._tree.match_and_insert(chain, now=self._step)

        result = ProcessResult(
            request_id=req.request_id,
            cached_prefix_blocks=matched,
            total_prompt_blocks=len(chain),
        )
        self._records.append(result)
        return result

    def report(self) -> Report:
        cached = sum(r.cached_prefix_blocks for r in self._records)
        total = sum(r.total_prompt_blocks for r in self._records)
        hit_rate = cached / total if total > 0 else 0.0
        return Report(hit_rate=hit_rate, total_requests=len(self._records))
