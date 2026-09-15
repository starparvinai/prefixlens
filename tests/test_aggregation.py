"""Per-tag aggregation in Report.

The whole pitch of prefixlens turns on the tenant-A-vs-B breakdown — a
flat overall hit rate hides exactly the signal you want. These tests
lock in how tags on Requests flow through the simulator and land as a
nested by_tag dict on the Report.
"""

from prefixlens import RadixCacheSimulator, Report, Request, TagStats


def _tokens(offset: int, blocks: int = 2, block_size: int = 16) -> tuple[int, ...]:
    """Deterministic non-overlapping token stream for a fresh chain."""
    return tuple(range(offset, offset + blocks * block_size))


# ---- Report shape --------------------------------------------------------


def test_report_carries_raw_block_counts():
    # Every reader needs to see the numerator/denominator, not just the ratio —
    # the CLI displays them, and tests assert against them.
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)
    sim.process(Request(request_id="r1", token_ids=_tokens(0)))  # 2 blocks
    sim.process(Request(request_id="r2", token_ids=_tokens(0)))  # 2 hits

    report = sim.report()

    assert report.cached_blocks == 2
    assert report.total_blocks == 4
    assert report.hit_rate == 0.5


def test_empty_simulator_has_no_tags():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)

    report = sim.report()

    assert report.by_tag == {}
    assert report.total_requests == 0
    assert report.cached_blocks == 0
    assert report.total_blocks == 0


def test_untagged_requests_produce_no_tag_buckets():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)
    sim.process(Request(request_id="r1", token_ids=_tokens(0)))

    report = sim.report()

    assert report.by_tag == {}


# ---- basic tag grouping --------------------------------------------------


def test_single_tag_key_groups_requests_by_value():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)

    sim.process(Request("r1", _tokens(0), (("tenant", "acme"),)))    # 2 blk, 0 hit
    sim.process(Request("r2", _tokens(0), (("tenant", "acme"),)))    # 2 blk, 2 hit
    sim.process(Request("r3", _tokens(1000), (("tenant", "widgets"),)))  # 2 blk, 0 hit

    report = sim.report()

    assert set(report.by_tag.keys()) == {"tenant"}
    assert set(report.by_tag["tenant"].keys()) == {"acme", "widgets"}

    acme = report.by_tag["tenant"]["acme"]
    assert acme == TagStats(hit_rate=0.5, total_requests=2, cached_blocks=2, total_blocks=4)

    widgets = report.by_tag["tenant"]["widgets"]
    assert widgets == TagStats(hit_rate=0.0, total_requests=1, cached_blocks=0, total_blocks=2)


def test_multiple_tag_keys_each_get_their_own_grouping():
    # Same three requests, two tag axes: tenant AND route. Both should aggregate
    # independently — a request contributes to whatever buckets its own tags name.
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)
    sim.process(Request("r1", _tokens(0), (("route", "/chat"), ("tenant", "acme"))))
    sim.process(Request("r2", _tokens(0), (("route", "/chat"), ("tenant", "acme"))))
    sim.process(Request("r3", _tokens(1000), (("route", "/complete"), ("tenant", "acme"))))

    report = sim.report()

    assert set(report.by_tag.keys()) == {"tenant", "route"}

    # All three are tenant=acme
    assert report.by_tag["tenant"]["acme"].total_requests == 3
    assert report.by_tag["tenant"]["acme"].total_blocks == 6
    assert report.by_tag["tenant"]["acme"].cached_blocks == 2

    # Two on /chat (0 + 2 hits = 2), one on /complete (0 hits)
    assert report.by_tag["route"]["/chat"].cached_blocks == 2
    assert report.by_tag["route"]["/complete"].cached_blocks == 0


def test_request_without_a_tag_key_does_not_contribute_to_that_key():
    # r1 has tenant, r2 doesn't. The tenant bucket should reflect only r1.
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)

    sim.process(Request("r1", _tokens(0), (("tenant", "acme"),)))
    sim.process(Request("r2", _tokens(1000), ()))  # no tags

    report = sim.report()

    assert report.total_requests == 2
    assert report.by_tag["tenant"]["acme"].total_requests == 1
    assert report.by_tag["tenant"]["acme"].total_blocks == 2


# ---- the Chen story (mini) -----------------------------------------------


def test_chen_story_shape_two_tenants_diverging_hit_rates():
    """Mini reproduction of the SPEC §1 scenario.

    Two tenants share an infrastructure. Tenant A has cache-friendly prompts
    (repeated identical chain). Tenant B has cache-hostile prompts (every
    request starts with a unique chain — like a session UUID at position 0).
    Overall hit rate is mediocre; the per-tenant breakdown reveals A is fine
    and B is broken.
    """
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=10_000)

    shared = _tokens(0, blocks=4)  # 4 blocks — Tenant A's shared prompt

    # Tenant A: 5 identical requests → first is 0 hits, next 4 are 4 hits each.
    for i in range(5):
        sim.process(Request(f"a{i}", shared, (("tenant", "acme"),)))

    # Tenant B: 5 requests, each with a unique first block → no cache reuse.
    for i in range(5):
        b_tokens = _tokens(2000 + i * 100, blocks=4)
        sim.process(Request(f"b{i}", b_tokens, (("tenant", "widgets"),)))

    report = sim.report()

    acme = report.by_tag["tenant"]["acme"]
    widgets = report.by_tag["tenant"]["widgets"]

    # Acme: 5 requests × 4 blocks = 20 total; 0 + 4×4 = 16 cached → 80%
    assert acme.total_blocks == 20
    assert acme.cached_blocks == 16
    assert acme.hit_rate == 0.8

    # Widgets: 5 requests × 4 blocks = 20 total; 0 cached → 0%
    assert widgets.total_blocks == 20
    assert widgets.cached_blocks == 0
    assert widgets.hit_rate == 0.0

    # Overall: (16 + 0) / (20 + 20) = 40% — hides the tenant-level story.
    assert report.hit_rate == 0.4


# ---- tags flow through ProcessResult ------------------------------------


def test_process_result_carries_the_request_tags():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)
    tags = (("tenant", "acme"), ("route", "/chat"))

    result = sim.process(Request("r1", _tokens(0), tags))

    assert result.tags == tags


def test_process_result_has_empty_tags_when_request_has_none():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)

    result = sim.process(Request("r1", _tokens(0)))

    assert result.tags == ()
