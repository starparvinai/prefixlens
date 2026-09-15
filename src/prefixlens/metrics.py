"""Parser for vLLM's Prometheus /metrics exposition.

We only extract the two counters we need for prefix-cache validate mode:
`vllm:prefix_cache_hits` and `vllm:prefix_cache_queries`. Everything else
in /metrics is ignored.

vLLM emits Prometheus text format, which looks like:

    # HELP vllm:prefix_cache_hits ...
    # TYPE vllm:prefix_cache_hits counter
    vllm:prefix_cache_hits{engine="0",model="llama-3"} 12345.0
    vllm:prefix_cache_hits{engine="1",model="llama-3"} 6789.0

    # HELP vllm:prefix_cache_queries ...
    # TYPE vllm:prefix_cache_queries counter
    vllm:prefix_cache_queries{engine="0",model="llama-3"} 30000.0
    vllm:prefix_cache_queries{engine="1",model="llama-3"} 17000.0

We sum across all label sets so that a multi-worker deployment reports
one number, matching how a user would think about their whole engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


PREFIX_CACHE_HITS = "vllm:prefix_cache_hits"
PREFIX_CACHE_QUERIES = "vllm:prefix_cache_queries"


# Matches a Prometheus sample line for a given metric name. Captures the
# numeric value only. Handles:
#   metric_name value
#   metric_name{k="v",k2="v2"} value
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{[^}]*\})?"
    r"\s+(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
    r"(?:\s+\d+)?$"  # optional timestamp — ignored
)


@dataclass(frozen=True)
class VllmMetrics:
    """The prefix-cache counters we compare our sim against.

    Both are counters (monotonically increasing) so a single snapshot at end
    of workload is enough for v0.1. Sum across all label sets in the /metrics
    scrape — a multi-worker deployment reports each worker separately.
    """
    prefix_cache_hits: int
    prefix_cache_queries: int

    @property
    def hit_rate(self) -> float:
        if self.prefix_cache_queries == 0:
            return 0.0
        return self.prefix_cache_hits / self.prefix_cache_queries


def _sum_metric(text: str, metric_name: str) -> tuple[int, bool]:
    """Sum every sample line matching `metric_name` (across label sets).

    Returns (total, present). `present=False` means no non-comment line
    named that metric appeared at all — the caller uses this to distinguish
    "found the metric, saw zero" from "engine never emitted this metric."
    """
    total = 0
    present = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _SAMPLE_RE.match(stripped)
        if m is None or m.group("name") != metric_name:
            continue
        present = True
        # Prometheus values are floats; the two counters we read are integer-
        # valued in vLLM (block counts) so `int(float(...))` is safe.
        total += int(float(m.group("value")))
    return total, present


def parse_vllm_metrics(text: str) -> VllmMetrics:
    """Parse a Prometheus /metrics scrape and return the prefix-cache counters.

    Raises ValueError if either counter is absent. Absence means either the
    engine wasn't configured with prefix caching, or the user scraped the
    wrong endpoint — both are user errors worth surfacing loudly.
    """
    hits, hits_present = _sum_metric(text, PREFIX_CACHE_HITS)
    queries, queries_present = _sum_metric(text, PREFIX_CACHE_QUERIES)

    if not hits_present:
        raise ValueError(
            f"metric {PREFIX_CACHE_HITS!r} not found in /metrics — is prefix "
            "caching enabled on the engine, and is this the right endpoint?"
        )
    if not queries_present:
        raise ValueError(
            f"metric {PREFIX_CACHE_QUERIES!r} not found in /metrics — is prefix "
            "caching enabled on the engine, and is this the right endpoint?"
        )

    return VllmMetrics(prefix_cache_hits=hits, prefix_cache_queries=queries)
