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
