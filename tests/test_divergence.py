"""Divergent-position attribution: where in the prompt cache reuse fails,
and whether the content there is unique-per-request (UUID case) or
shared-but-still-missed (thrashing case).

The distinction matters because the two diagnoses have opposite fixes:
UUID injection → restructure the prompt; thrashing → grow the cache.
`unique_content_ratio` is the signal.
"""

import pytest

from prefixlens import DivergentPosition, RadixCacheSimulator, Request


BLOCK = 16


def _tokens(offset: int, blocks: int = 2) -> tuple[int, ...]:
    return tuple(range(offset, offset + blocks * BLOCK))


# ---- ProcessResult carries the diverging block ---------------------------


def test_full_miss_records_diverging_block_tokens():
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=100)
    tokens = _tokens(0, blocks=2)

    result = sim.process(Request("r1", tokens))

    assert result.first_divergent_block == 0
    assert result.diverging_block_tokens == tokens[:BLOCK]


def test_partial_miss_records_the_second_block_content():
    # r1 seeds blocks 0 and 1. r2 shares block 0, diverges at block 1.
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=100)
    sim.process(Request("r1", _tokens(0, blocks=2)))

    r2_tokens = _tokens(0, blocks=1) + _tokens(9000, blocks=1)
    result = sim.process(Request("r2", r2_tokens))

    assert result.first_divergent_block == 1
    assert result.diverging_block_tokens == r2_tokens[BLOCK : 2 * BLOCK]


def test_full_hit_leaves_diverging_block_tokens_as_none():
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=100)
    tokens = _tokens(0, blocks=2)
    sim.process(Request("r1", tokens))

    result = sim.process(Request("r2", tokens))

    assert result.first_divergent_block is None
    assert result.diverging_block_tokens is None


def test_zero_block_prompt_has_no_diverging_content():
    # Sub-block prompt: no complete blocks, so no "first divergent block" to
    # attribute anything to.
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=100)

    result = sim.process(Request("r1", tuple(range(5))))

    assert result.first_divergent_block is None
    assert result.diverging_block_tokens is None


# ---- Report.by_tag_divergence: UUID case (unique content per miss) ------


def test_uuid_case_yields_unique_content_ratio_one():
    """Each miss request has a unique first-block token. This is the
    session-UUID-at-position-0 scenario — restructure the prompt to fix.
    """
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=10_000)

    for i in range(5):
        # Distinct first token per request → distinct first block → all miss at block 0.
        toks = (9_000_000 + i,) + tuple(range(1001, 1001 + BLOCK - 1))
        sim.process(Request(f"r{i}", toks, (("tenant", "widgets"),)))

    report = sim.report()
    positions = report.by_tag_divergence["tenant"]["widgets"]

    assert len(positions) == 1
    p = positions[0]
    assert p.block_position == 0
    assert p.miss_count == 5
    assert p.unique_content_ratio == 1.0  # all-different content at pos 0


# ---- Report.by_tag_divergence: thrashing case (same content, repeated misses) ---


def test_thrashing_case_yields_low_unique_content_ratio():
    """Same block content repeatedly evicted from too-small a cache. All misses
    happen at the same position with identical token content — diagnosis:
    grow the cache, don't restructure the prompt.

    Setup: capacity=1 forces immediate eviction of anything that isn't in
    active use. Alternate between two distinct chains so each request
    evicts the other.
    """
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=1)

    chain_a = _tokens(1000, blocks=1)  # single-block chain
    chain_b = _tokens(2000, blocks=1)

    # Alternate: A, B, A, B, A, B. After the first two, every A and every B
    # misses because the other is what's in cache.
    for i in range(6):
        toks = chain_a if i % 2 == 0 else chain_b
        sim.process(Request(f"r{i}", toks, (("tenant", "thrash"),)))

    report = sim.report()
    positions = report.by_tag_divergence["tenant"]["thrash"]

    assert len(positions) == 1
    p = positions[0]
    assert p.block_position == 0
    assert p.miss_count == 6
    # 2 distinct block contents (chain_a and chain_b) across 6 misses = 2/6
    assert p.unique_content_ratio == pytest.approx(2 / 6)


# ---- ordering, empty buckets, multi-position -----------------------------


def test_positions_sorted_by_miss_count_descending():
    # Two positions with different miss counts — higher count first.
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=10_000)

    # Seed a shared block-0 with r0 so subsequent requests match block 0.
    shared_block0 = _tokens(5000, blocks=1)
    sim.process(Request("seed", shared_block0 + _tokens(6000, blocks=1), (("tenant", "t"),)))

    # 4 requests that share block 0 with the seed but diverge at block 1 (unique block-1).
    for i in range(4):
        toks = shared_block0 + tuple(range(7000 + i * 100, 7000 + i * 100 + BLOCK))
        sim.process(Request(f"r_pos1_{i}", toks, (("tenant", "t"),)))

    # 2 requests that miss at block 0 (fully new chains).
    for i in range(2):
        toks = tuple(range(8000 + i * 100, 8000 + i * 100 + 2 * BLOCK))
        sim.process(Request(f"r_pos0_{i}", toks, (("tenant", "t"),)))

    report = sim.report()
    positions = report.by_tag_divergence["tenant"]["t"]

    # Highest miss_count first: position 1 has 4 misses, position 0 has 3 (seed + 2 new).
    assert positions[0].block_position == 1
    assert positions[0].miss_count == 4
    assert positions[1].block_position == 0
    assert positions[1].miss_count == 3


def test_all_hits_bucket_yields_empty_divergence_tuple():
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=100)
    tokens = _tokens(0)
    sim.process(Request("r1", tokens, (("tenant", "acme"),)))
    sim.process(Request("r2", tokens, (("tenant", "acme"),)))  # full hit

    report = sim.report()

    # acme has misses (r1) — not the right test for empty. Set up better:
    sim2 = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=100)
    sim2.process(Request("r1", tokens, (("tenant", "acme"),)))
    # r2 tagged widgets — full hit, no misses for widgets bucket.
    sim2.process(Request("r2", tokens, (("tenant", "widgets"),)))

    report2 = sim2.report()

    assert report2.by_tag_divergence["tenant"]["widgets"] == ()
    # acme still has its one miss (r1)
    assert len(report2.by_tag_divergence["tenant"]["acme"]) == 1


def test_untagged_requests_produce_no_divergence_buckets():
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=100)
    sim.process(Request("r1", _tokens(0)))  # no tags

    report = sim.report()

    assert report.by_tag_divergence == {}


# ---- Chen story: divergence attribution across two tenants --------------


def test_chen_scenario_divergence_diagnosis():
    """Full end-to-end reproduction of the SPEC §1 signal.

    Tenant acme: shared prefix, repeated requests → misses only on r0, and
    all subsequent hits — the acme bucket has one miss at position 0 with
    unique_content_ratio 1.0 (trivially; sample size 1).

    Tenant widgets: unique first-block content per request → 5 misses all at
    position 0, unique_content_ratio 1.0 (each request has a distinct first
    block). This is the diagnostic pattern that says 'restructure the prompt.'
    """
    sim = RadixCacheSimulator(block_size=BLOCK, capacity_blocks=10_000)
    shared = _tokens(1000, blocks=4)

    for i in range(5):
        sim.process(Request(f"acme-{i}", shared, (("tenant", "acme"),)))

    for i in range(5):
        unique_toks = (9_000_000 + i,) + tuple(range(2001, 2001 + 4 * BLOCK - 1))
        sim.process(Request(f"widgets-{i}", unique_toks, (("tenant", "widgets"),)))

    report = sim.report()
    widgets_positions = report.by_tag_divergence["tenant"]["widgets"]

    assert len(widgets_positions) == 1
    assert widgets_positions[0].block_position == 0
    assert widgets_positions[0].miss_count == 5
    assert widgets_positions[0].unique_content_ratio == 1.0
