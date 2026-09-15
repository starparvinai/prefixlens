"""Parser for vLLM's Prometheus /metrics text — the ground truth side of
validate mode. Tests hand-construct representative /metrics blobs and
assert the parser extracts the two counters we care about.
"""

import pytest

from prefixlens.metrics import VllmMetrics, parse_vllm_metrics


# ---- happy path ----------------------------------------------------------


def test_parse_single_sample_unlabeled():
    text = (
        "vllm:prefix_cache_hits 100\n"
        "vllm:prefix_cache_queries 250\n"
    )

    m = parse_vllm_metrics(text)

    assert m.prefix_cache_hits == 100
    assert m.prefix_cache_queries == 250
    assert m.hit_rate == 0.4


def test_parse_with_labels():
    text = (
        '# HELP vllm:prefix_cache_hits Total prefix-cache hits\n'
        '# TYPE vllm:prefix_cache_hits counter\n'
        'vllm:prefix_cache_hits{engine="0",model="llama-3"} 12345\n'
        '# HELP vllm:prefix_cache_queries Total prefix-cache queries\n'
        '# TYPE vllm:prefix_cache_queries counter\n'
        'vllm:prefix_cache_queries{engine="0",model="llama-3"} 30000\n'
    )

    m = parse_vllm_metrics(text)

    assert m.prefix_cache_hits == 12345
    assert m.prefix_cache_queries == 30000


def test_parse_sums_across_multiple_label_sets():
    # A multi-worker deployment exposes each worker's counters separately.
    # We report the whole-engine total.
    text = (
        'vllm:prefix_cache_hits{engine="0"} 100\n'
        'vllm:prefix_cache_hits{engine="1"} 200\n'
        'vllm:prefix_cache_hits{engine="2"} 300\n'
        'vllm:prefix_cache_queries{engine="0"} 500\n'
        'vllm:prefix_cache_queries{engine="1"} 500\n'
        'vllm:prefix_cache_queries{engine="2"} 500\n'
    )

    m = parse_vllm_metrics(text)

    assert m.prefix_cache_hits == 600
    assert m.prefix_cache_queries == 1500
    assert m.hit_rate == 0.4


def test_parse_accepts_float_values_and_truncates_to_int():
    # Prometheus values are technically floats; vLLM's counters are integer-
    # valued but emitted with a trailing '.0'.
    text = (
        "vllm:prefix_cache_hits 12345.0\n"
        "vllm:prefix_cache_queries 30000.0\n"
    )

    m = parse_vllm_metrics(text)

    assert m.prefix_cache_hits == 12345
    assert m.prefix_cache_queries == 30000


def test_parse_tolerates_extra_unrelated_metrics():
    # A real /metrics scrape has dozens of counters. We only care about ours.
    text = (
        "process_cpu_seconds_total 123.4\n"
        "python_gc_objects_collected_total{generation=\"0\"} 500\n"
        "vllm:prefix_cache_hits 10\n"
        "vllm:some_other_counter 999\n"
        "vllm:prefix_cache_queries 20\n"
        "vllm:kv_cache_usage_perc 0.8\n"
    )

    m = parse_vllm_metrics(text)

    assert m.prefix_cache_hits == 10
    assert m.prefix_cache_queries == 20


def test_parse_ignores_comments_and_blank_lines():
    text = (
        "\n"
        "# HELP vllm:prefix_cache_hits ...\n"
        "\n"
        "vllm:prefix_cache_hits 5\n"
        "\n"
        "# HELP vllm:prefix_cache_queries ...\n"
        "vllm:prefix_cache_queries 10\n"
        "\n"
    )

    m = parse_vllm_metrics(text)

    assert m.prefix_cache_hits == 5


def test_parse_ignores_optional_timestamp():
    # Prometheus samples optionally have a trailing unix-ms timestamp.
    text = (
        "vllm:prefix_cache_hits 100 1726300000000\n"
        "vllm:prefix_cache_queries 250 1726300000000\n"
    )

    m = parse_vllm_metrics(text)

    assert m.prefix_cache_hits == 100
    assert m.prefix_cache_queries == 250


# ---- hit_rate edge cases -------------------------------------------------


def test_zero_queries_yields_zero_hit_rate():
    """Fresh engine: no traffic yet. Don't divide by zero."""
    m = VllmMetrics(prefix_cache_hits=0, prefix_cache_queries=0)
    assert m.hit_rate == 0.0


def test_all_hit_yields_hit_rate_one():
    m = VllmMetrics(prefix_cache_hits=100, prefix_cache_queries=100)
    assert m.hit_rate == 1.0


# ---- missing metrics -----------------------------------------------------


def test_missing_hits_counter_raises():
    # Only queries present — this is a user error (prefix caching off, or
    # wrong endpoint scraped). Surface it, don't silently fall back.
    text = "vllm:prefix_cache_queries 100\n"

    with pytest.raises(ValueError, match="prefix_cache_hits.*not found"):
        parse_vllm_metrics(text)


def test_missing_queries_counter_raises():
    text = "vllm:prefix_cache_hits 100\n"

    with pytest.raises(ValueError, match="prefix_cache_queries.*not found"):
        parse_vllm_metrics(text)


def test_empty_text_raises():
    with pytest.raises(ValueError, match="not found"):
        parse_vllm_metrics("")


def test_only_comments_raises():
    text = (
        "# HELP vllm:prefix_cache_hits ...\n"
        "# TYPE vllm:prefix_cache_hits counter\n"
    )

    with pytest.raises(ValueError, match="not found"):
        parse_vllm_metrics(text)


def test_a_metric_named_similarly_but_not_exactly_is_ignored():
    # `vllm:prefix_cache_hits_total` (hypothetical) is not the same metric.
    # Exact name match only.
    text = (
        "vllm:prefix_cache_hits_total 100\n"
        "vllm:prefix_cache_queries 250\n"
    )

    with pytest.raises(ValueError, match="prefix_cache_hits.*not found"):
        parse_vllm_metrics(text)
