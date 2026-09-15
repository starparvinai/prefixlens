"""Per-request explanation: block-by-block trace of one request through
the simulated cache, plus the aggregate-divergence context for the tag
buckets that request belongs to.

Pure functions — no I/O, no globals. The CLI composes explain_request()
with rendering; tests exercise it directly against hand-built states.
"""

from __future__ import annotations

from dataclasses import dataclass

from prefixlens.simulator import DivergentPosition, RadixCacheSimulator, Report


@dataclass(frozen=True)
class BlockTrace:
    """One row of the per-request block-by-block walk."""
    position: int
    tokens: tuple[int, ...]
    verdict: str  # "HIT" or "MISS"


@dataclass(frozen=True)
class DivergenceContext:
    """The aggregate signal (from Report.by_tag_divergence) at THIS request's
    first-divergent-block, for one of its tag buckets. None when the tag bucket
    has no divergence entry at this position (e.g. this request is the only
    miss and we haven't cataloged it — shouldn't happen in practice, but the
    Optional keeps the type honest)."""
    tag_key: str
    tag_value: str
    position: DivergentPosition


@dataclass(frozen=True)
class RequestExplanation:
    request_id: str
    tags: tuple[tuple[str, str], ...]
    total_prompt_blocks: int
    cached_prefix_blocks: int
    first_divergent_block: int | None
    block_traces: tuple[BlockTrace, ...]
    divergence_context: tuple[DivergenceContext, ...]
    """For each tag bucket this request belongs to that has a divergence entry
    at first_divergent_block, an entry showing the aggregate signal. Empty
    when the request was a full hit or had no tags."""


def explain_request(
    sim: RadixCacheSimulator,
    request_id: str,
    report: Report,
) -> RequestExplanation | None:
    """Return a block-by-block explanation of one request, or None if the
    request_id isn't in the sim's records.

    The caller must have already run the corpus through `sim` (so records
    exist) and computed `report = sim.report()` (so divergence data exists).
    Separating these steps means explain doesn't recompute the report on
    every call, which matters if the CLI ends up explaining multiple
    requests in one session (a --request-ids flag, later).
    """
    record = sim.find_result(request_id)
    if record is None:
        return None

    block_size = sim.block_size
    traces: list[BlockTrace] = []
    for i in range(record.total_prompt_blocks):
        start = i * block_size
        block_tokens = record.token_ids[start : start + block_size]
        # HIT if it landed before the first divergence (or the whole request hit).
        if record.first_divergent_block is None or i < record.first_divergent_block:
            verdict = "HIT"
        else:
            verdict = "MISS"
        traces.append(BlockTrace(position=i, tokens=block_tokens, verdict=verdict))

    contexts: list[DivergenceContext] = []
    if record.first_divergent_block is not None:
        for k, v in record.tags:
            positions = report.by_tag_divergence.get(k, {}).get(v, ())
            match = next(
                (p for p in positions if p.block_position == record.first_divergent_block),
                None,
            )
            if match is not None:
                contexts.append(DivergenceContext(tag_key=k, tag_value=v, position=match))

    return RequestExplanation(
        request_id=record.request_id,
        tags=record.tags,
        total_prompt_blocks=record.total_prompt_blocks,
        cached_prefix_blocks=record.cached_prefix_blocks,
        first_divergent_block=record.first_divergent_block,
        block_traces=tuple(traces),
        divergence_context=tuple(contexts),
    )
