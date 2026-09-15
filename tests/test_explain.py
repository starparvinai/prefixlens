"""Per-request explain: block-by-block trace + aggregate-divergence context.

Tests exercise the pure explain_request() function against sim states built
by real process() calls. No file I/O — the CLI layer wraps this and gets
its own tests in test_cli.py.
"""

from prefixlens import (
    RadixCacheSimulator,
    Request,
    explain_request,
)


BLOCK = 16


def _run(requests, block_size=BLOCK, capacity_blocks=100):
    """Helper: process a list of Requests, return (sim, report)."""
    sim = RadixCacheSimulator(block_size=block_size, capacity_blocks=capacity_blocks)
    for r in requests:
        sim.process(r)
    return sim, sim.report()


# ---- basic shape ---------------------------------------------------------


def test_explain_returns_none_for_unknown_request_id():
    sim, report = _run([Request("r1", tuple(range(32)))])

    result = explain_request(sim, "nonexistent", report)

    assert result is None


def test_explain_returns_request_metadata():
    tokens = tuple(range(32))
    sim, report = _run([Request("r1", tokens, (("tenant", "acme"),))])

    result = explain_request(sim, "r1", report)

    assert result is not None
    assert result.request_id == "r1"
    assert result.tags == (("tenant", "acme"),)
    assert result.total_prompt_blocks == 2
    assert result.cached_prefix_blocks == 0  # fresh cache
    assert result.first_divergent_block == 0


# ---- block traces --------------------------------------------------------


def test_full_miss_traces_all_blocks_as_MISS():
    tokens = tuple(range(32))  # 2 blocks
    sim, report = _run([Request("r1", tokens)])

    result = explain_request(sim, "r1", report)

    assert len(result.block_traces) == 2
    assert all(t.verdict == "MISS" for t in result.block_traces)
    assert result.block_traces[0].position == 0
    assert result.block_traces[0].tokens == tokens[:BLOCK]
    assert result.block_traces[1].position == 1
    assert result.block_traces[1].tokens == tokens[BLOCK:]


def test_full_hit_traces_all_blocks_as_HIT():
    tokens = tuple(range(32))
    sim, report = _run([
        Request("r1", tokens),
        Request("r2", tokens),  # full hit
    ])

    result = explain_request(sim, "r2", report)

    assert all(t.verdict == "HIT" for t in result.block_traces)
    assert result.cached_prefix_blocks == 2
    assert result.first_divergent_block is None


def test_partial_hit_traces_split_at_divergence():
    # r1 seeds blocks 0 and 1. r2 shares block 0 (16 tokens) and diverges at block 1.
    sim, report = _run([
        Request("r1", tuple(range(32))),
        Request("r2", tuple(range(16)) + tuple(range(9000, 9016))),
    ])

    result = explain_request(sim, "r2", report)

    assert len(result.block_traces) == 2
    assert result.block_traces[0].verdict == "HIT"
    assert result.block_traces[1].verdict == "MISS"
    assert result.first_divergent_block == 1


def test_partial_prompt_produces_zero_traces():
    # Sub-block prompt: total_prompt_blocks = 0 → no traces.
    sim, report = _run([Request("r1", tuple(range(5)))])  # only 5 tokens

    result = explain_request(sim, "r1", report)

    assert result.block_traces == ()
    assert result.total_prompt_blocks == 0
    assert result.first_divergent_block is None


# ---- divergence context --------------------------------------------------


def test_divergence_context_pulls_signal_from_report():
    """Explain should surface the aggregate divergence signal for the
    request's tag buckets at the request's diverging position.
    """
    # 5 widgets requests all diverge at block 0 with unique content
    requests = []
    for i in range(5):
        toks = (9_000_000 + i,) + tuple(range(1001, 1001 + BLOCK - 1))
        requests.append(Request(f"w{i}", toks, (("tenant", "widgets"),)))

    sim, report = _run(requests)

    result = explain_request(sim, "w2", report)

    assert result.first_divergent_block == 0
    assert len(result.divergence_context) == 1
    ctx = result.divergence_context[0]
    assert ctx.tag_key == "tenant"
    assert ctx.tag_value == "widgets"
    assert ctx.position.block_position == 0
    assert ctx.position.miss_count == 5
    assert ctx.position.unique_content_ratio == 1.0


def test_divergence_context_empty_for_full_hit_requests():
    tokens = tuple(range(32))
    sim, report = _run([
        Request("r1", tokens, (("tenant", "acme"),)),
        Request("r2", tokens, (("tenant", "acme"),)),  # full hit
    ])

    result = explain_request(sim, "r2", report)

    assert result.divergence_context == ()


def test_divergence_context_covers_every_matching_tag_bucket():
    # A request with two tags → two context entries (assuming both buckets
    # have a divergence entry at this position).
    tokens = (99999,) + tuple(range(1, BLOCK))  # unique at block 0
    sim, report = _run([Request("r1", tokens, (("tenant", "t"), ("route", "/r")))])

    result = explain_request(sim, "r1", report)

    keys = {(ctx.tag_key, ctx.tag_value) for ctx in result.divergence_context}
    assert keys == {("tenant", "t"), ("route", "/r")}


def test_divergence_context_skips_buckets_without_matching_position():
    # Two tag axes, but the request's diverging position only appears in
    # one bucket's divergence data (contrived: mixing tags across requests).
    # Setup: one tenant=A request that diverges at block 0; one tenant=B request
    # that diverges at block 1. Explain the A request — it should NOT surface
    # a divergence context for tenant=B.
    sim, report = _run([
        Request("a1", (99999,) + tuple(range(1, BLOCK)), (("tenant", "A"),)),
        # tenant=B seeds a shared block-0 and diverges at block 1
        Request("seed", tuple(range(500, 500 + 32)), (("tenant", "B"),)),
        Request(
            "b1",
            tuple(range(500, 500 + BLOCK)) + (88888,) + tuple(range(1, BLOCK)),
            (("tenant", "B"),),
        ),
    ])

    result = explain_request(sim, "a1", report)

    # Only tenant=A context should be present, not tenant=B (a1 has no B tag anyway).
    assert len(result.divergence_context) == 1
    assert result.divergence_context[0].tag_value == "A"


# ---- Chen scenario end-to-end -------------------------------------------


def test_chen_scenario_explain_a_widgets_request():
    """Full-flavor test: build the Chen scenario, explain one widgets request,
    verify the trace shows all-miss and the divergence context surfaces the
    UUID diagnosis (100% unique)."""
    requests = []
    shared = tuple(range(1000, 1000 + 4 * BLOCK))

    for i in range(5):
        requests.append(Request(f"acme-{i}", shared, (("tenant", "acme"),)))

    for i in range(5):
        toks = (9_000_000 + i,) + tuple(range(2001, 2001 + 4 * BLOCK - 1))
        requests.append(Request(f"widgets-{i}", toks, (("tenant", "widgets"),)))

    sim, report = _run(requests)

    result = explain_request(sim, "widgets-2", report)

    assert result.total_prompt_blocks == 4
    assert result.cached_prefix_blocks == 0
    assert result.first_divergent_block == 0
    # All 4 blocks are MISS since block 0 diverges (cascading failure).
    assert all(t.verdict == "MISS" for t in result.block_traces)

    # Context should be a single entry (tenant=widgets) showing the aggregate signal.
    assert len(result.divergence_context) == 1
    ctx = result.divergence_context[0]
    assert ctx.tag_value == "widgets"
    assert ctx.position.miss_count == 5
    assert ctx.position.unique_content_ratio == 1.0
