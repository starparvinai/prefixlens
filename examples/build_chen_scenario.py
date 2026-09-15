"""Build the Chen 2026-05 scenario as a JSONL corpus for demo + tests.

Reproduces the shape described in the DEV.to post (SPEC §1):

- Tenant 'acme': every request has an identical shared prefix (a system
  prompt / tool schemas / few-shot examples). Cache-friendly.
- Tenant 'widgets': every request has a UNIQUE first block — as if a
  session UUID or timestamp were injected at position 0. Cache-hostile:
  the very first block hash differs across requests, so the walk from
  root fails immediately and nothing downstream can hit.

The specific numbers below are deliberately small so a human can verify
the report by hand. For a bigger, more realistic scale, bump N_ACME and
N_WIDGETS and re-run.

Usage:
    python examples/build_chen_scenario.py > examples/chen_multitenant.jsonl
"""

import json
import sys

BLOCK_SIZE = 16
BLOCKS_PER_REQUEST = 4  # 64 tokens each -- 4 blocks per prompt
N_ACME = 5              # cache-friendly tenant
N_WIDGETS = 5           # cache-hostile tenant


def acme_tokens() -> list[int]:
    """All requests get the SAME 4-block prefix. Perfect cache reuse across
    identical repeats — everything after the first request should hit."""
    return list(range(1000, 1000 + BLOCK_SIZE * BLOCKS_PER_REQUEST))


def widgets_tokens(request_index: int) -> list[int]:
    """A unique token id in the FIRST block per request (simulating a session
    UUID injected at position 0 of the system prompt). The rest of the prompt
    is identical across requests — but because the first block hash differs,
    nothing chained after it can hit either."""
    unique = 9_000_000 + request_index  # arbitrary large distinct id per request
    common = list(range(2000, 2000 + BLOCK_SIZE * BLOCKS_PER_REQUEST - 1))
    return [unique] + common


def main() -> None:
    out = sys.stdout
    for i in range(N_ACME):
        record = {
            "request_id": f"acme-{i}",
            "token_ids": acme_tokens(),
            "tenant": "acme",
            "route": "/v1/chat/completions",
        }
        out.write(json.dumps(record) + "\n")

    for i in range(N_WIDGETS):
        record = {
            "request_id": f"widgets-{i}",
            "token_ids": widgets_tokens(i),
            "tenant": "widgets",
            "route": "/v1/chat/completions",
        }
        out.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
