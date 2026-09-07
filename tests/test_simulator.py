from prefixlens import RadixCacheSimulator, Request


def test_first_request_against_empty_cache_is_a_full_miss():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    req = Request(request_id="r1", token_ids=tuple(range(32)))  # exactly 2 blocks

    result = sim.process(req)

    assert result.request_id == "r1"
    assert result.total_prompt_blocks == 2
    assert result.cached_prefix_blocks == 0


def test_report_on_empty_simulator_returns_zero_hit_rate():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)

    report = sim.report()

    assert report.hit_rate == 0.0
    assert report.total_requests == 0


def test_report_after_one_full_miss_returns_zero_hit_rate():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    req = Request(request_id="r1", token_ids=tuple(range(32)))

    sim.process(req)
    report = sim.report()

    assert report.hit_rate == 0.0
    assert report.total_requests == 1


def test_partial_trailing_tokens_do_not_count_as_blocks():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    req = Request(request_id="r1", token_ids=tuple(range(20)))  # 1 full block + 4 loose tokens

    result = sim.process(req)

    assert result.total_prompt_blocks == 1


def test_second_identical_request_is_a_full_hit():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    tokens = tuple(range(32))
    sim.process(Request(request_id="r1", token_ids=tokens))

    result = sim.process(Request(request_id="r2", token_ids=tokens))

    assert result.cached_prefix_blocks == 2
    assert result.total_prompt_blocks == 2


def test_second_request_shares_only_prefix_of_first():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    sim.process(Request(request_id="r1", token_ids=tuple(range(32))))

    r2_tokens = tuple(range(16)) + tuple(range(1000, 1016))
    result = sim.process(Request(request_id="r2", token_ids=r2_tokens))

    assert result.cached_prefix_blocks == 1
    assert result.total_prompt_blocks == 2


def test_hit_rate_computes_across_multiple_requests():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    sim.process(Request(request_id="r1", token_ids=tuple(range(32))))  # 2 blocks, 0 hits
    sim.process(Request(request_id="r2", token_ids=tuple(range(32))))  # 2 blocks, 2 hits

    report = sim.report()
    # 2 hits out of 4 total blocks = 50%
    assert report.hit_rate == 0.5
    assert report.total_requests == 2


def test_totally_disjoint_requests_share_nothing():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    sim.process(Request(request_id="r1", token_ids=tuple(range(0, 32))))
    result = sim.process(Request(request_id="r2", token_ids=tuple(range(500, 532))))

    assert result.cached_prefix_blocks == 0


def test_first_divergent_block_is_zero_on_full_miss():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    result = sim.process(Request(request_id="r1", token_ids=tuple(range(32))))

    assert result.first_divergent_block == 0


def test_first_divergent_block_is_none_on_full_hit():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    tokens = tuple(range(32))
    sim.process(Request(request_id="r1", token_ids=tokens))

    result = sim.process(Request(request_id="r2", token_ids=tokens))

    assert result.first_divergent_block is None


def test_first_divergent_block_at_end_of_shared_prefix():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    sim.process(Request(request_id="r1", token_ids=tuple(range(48))))  # 3 blocks

    # First 32 tokens (2 blocks) shared, third block diverges
    r2_tokens = tuple(range(32)) + tuple(range(1000, 1016))
    result = sim.process(Request(request_id="r2", token_ids=r2_tokens))

    assert result.cached_prefix_blocks == 2
    assert result.first_divergent_block == 2  # 0-indexed: block 0, 1 matched; block 2 missed


def test_first_divergent_block_is_none_on_empty_prompt():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    result = sim.process(Request(request_id="r1", token_ids=()))

    assert result.total_prompt_blocks == 0
    assert result.cached_prefix_blocks == 0
    assert result.first_divergent_block is None


def test_first_divergent_block_is_none_on_sub_block_prompt():
    """A prompt with only a partial trailing block has 0 total blocks — no divergence to report."""
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    result = sim.process(Request(request_id="r1", token_ids=tuple(range(10))))

    assert result.total_prompt_blocks == 0
    assert result.first_divergent_block is None


def test_tree_grows_with_new_blocks():
    sim = RadixCacheSimulator(block_size=16, capacity_blocks=1024)
    assert len(sim.tree) == 0

    sim.process(Request(request_id="r1", token_ids=tuple(range(32))))
    assert len(sim.tree) == 2  # 2 new blocks inserted

    # Same request → no new nodes
    sim.process(Request(request_id="r2", token_ids=tuple(range(32))))
    assert len(sim.tree) == 2

    # Diverging suffix → 1 new node (shared first block)
    r3_tokens = tuple(range(16)) + tuple(range(1000, 1016))
    sim.process(Request(request_id="r3", token_ids=r3_tokens))
    assert len(sim.tree) == 3
