from dataclasses import dataclass, field

from prefixlens.request import Request


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
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        self.hash_fn = hash_fn
        self._records: list[ProcessResult] = []

    @property
    def tree(self) -> object:
        raise NotImplementedError("tree inspection lands in a later commit")

    def process(self, req: Request) -> ProcessResult:
        total_prompt_blocks = len(req.token_ids) // self.block_size
        cached_prefix_blocks = 0  # empty cache — everything is a miss for now
        result = ProcessResult(
            request_id=req.request_id,
            cached_prefix_blocks=cached_prefix_blocks,
            total_prompt_blocks=total_prompt_blocks,
        )
        self._records.append(result)
        return result

    def report(self) -> Report:
        cached = sum(r.cached_prefix_blocks for r in self._records)
        total = sum(r.total_prompt_blocks for r in self._records)
        hit_rate = cached / total if total > 0 else 0.0
        return Report(hit_rate=hit_rate, total_requests=len(self._records))
