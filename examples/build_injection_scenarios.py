"""Generate injection sub-workloads from an existing WildChat corpus.

Purpose: self-validate that prefixlens detects the patterns we say it
detects. We take a real corpus, apply a *known* transform to it (add a
shared prefix, inject a per-conversation UUID at some position), and
then check whether the analyze report shows exactly the shape the
transform was built to produce.

Five scenarios, one JSONL file each — analyzed separately so each gets
a clean cache (mixing them in one file would confound the results):

    baseline        unchanged corpus (reference point).

    shared_prompt   +32-token shared system prompt prepended to every
                    request. Expected: hit rate up; block 0 hits across
                    conversations because everyone starts identically.

    uuid_front      per-conversation UUID (16 tokens, deterministic
                    from conversation_id) BEFORE the shared prompt.
                    Expected: hit rate drops back near baseline; block
                    0 is the divergent position with 100% unique
                    content — the UUID poisons everything downstream.

    uuid_middle     shared prompt at position 0..31 + UUID at position
                    32..47 + conversation. Expected: blocks 0..1 hit
                    across all requests (shared prompt), block 2 is
                    the divergent position with 100% unique content.

    uuid_tail       shared prompt + conversation + UUID at the end.
                    Expected: near shared_prompt hit rate. UUID is
                    "in the safe zone" — everything before it hits,
                    only the last block(s) diverge but by then most
                    of the request has already scored.

If any of these reports show a different pattern than promised, the
bug is in prefixlens, not the transform.

Usage:
    python examples/build_injection_scenarios.py \\
        --base examples/wildchat_scenario.jsonl \\
        --out-dir examples/ \\
        --max-conversations 500
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


# 32-token shared "system prompt" — deterministic dummy IDs. Real text would
# tokenize similarly; the specific values don't matter to the simulator, only
# their identity across requests. 32 tokens = 2 blocks at block_size=16, so a
# clean pre-conversation prefix. Chosen well outside cl100k_base's typical
# range so collision with the base corpus is astronomically unlikely.
SHARED_SYSTEM_PROMPT_TOKENS = list(range(50_000, 50_032))

# UUID injection is 16 tokens = exactly 1 block.
UUID_TOKEN_COUNT = 16


def _per_conv_uuid_tokens(conv_id: str, n_tokens: int = UUID_TOKEN_COUNT) -> list[int]:
    """Deterministic-but-unique-per-conversation 16-token pseudo-UUID.

    blake2b(conv_id) → 16 bytes → 8 uint16s. Scale into token ID space
    starting at 60000 (again well outside base corpus range).
    """
    digest = hashlib.blake2b(conv_id.encode("utf-8"), digest_size=n_tokens * 2).digest()
    return [
        60_000 + int.from_bytes(digest[i * 2 : i * 2 + 2], "big") % 60_000
        for i in range(n_tokens)
    ]


SCENARIOS = ("baseline", "shared_prompt", "uuid_front", "uuid_middle", "uuid_tail")


def apply_scenario(record: dict, scenario: str, conv_id: str) -> dict:
    """Return a new record with the scenario transform applied to token_ids.

    Preserves request_id, model, language. Does not add a `scenario` tag —
    we intentionally analyze each scenario in isolation, not mixed.
    """
    tokens = list(record["token_ids"])
    shared = SHARED_SYSTEM_PROMPT_TOKENS
    uuid_toks = _per_conv_uuid_tokens(conv_id)

    if scenario == "baseline":
        new_tokens = tokens
    elif scenario == "shared_prompt":
        new_tokens = shared + tokens
    elif scenario == "uuid_front":
        new_tokens = uuid_toks + shared + tokens
    elif scenario == "uuid_middle":
        new_tokens = shared + uuid_toks + tokens
    elif scenario == "uuid_tail":
        new_tokens = shared + tokens + uuid_toks
    else:
        raise ValueError(f"unknown scenario: {scenario!r}")

    return {
        "request_id": record["request_id"],
        "token_ids": new_tokens,
        "model": record.get("model", "unknown"),
        "language": record.get("language", "English"),
    }


def _load_trimmed(base_path: Path, max_conversations: int | None) -> list[tuple[dict, str]]:
    """Load records from base, stopping once max_conversations distinct
    conversation_ids have been seen (in first-encounter order)."""
    records: list[tuple[dict, str]] = []
    seen_convs: set[str] = set()
    with base_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            conv_id = r["request_id"].rsplit("_turn_", 1)[0]
            if conv_id not in seen_convs:
                if max_conversations is not None and len(seen_convs) >= max_conversations:
                    # Skip any further requests from *new* conversations, but
                    # still include additional turns from already-seen ones
                    continue
                seen_convs.add(conv_id)
            records.append((r, conv_id))
    return records


def build(
    base_path: Path,
    out_dir: Path,
    scenarios: list[str],
    max_conversations: int | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_trimmed(base_path, max_conversations)
    unique_convs = {conv_id for _, conv_id in records}

    for scenario in scenarios:
        out_path = out_dir / f"injection_{scenario}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for record, conv_id in records:
                new_r = apply_scenario(record, scenario, conv_id)
                f.write(json.dumps(new_r) + "\n")
        print(
            f"wrote {out_path.name}: {len(records):,} requests, "
            f"{len(unique_convs):,} conversations"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else "",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("examples/wildchat_scenario.jsonl"),
        help="Base WildChat JSONL to derive scenarios from.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("examples/"),
        help="Directory to write injection_*.jsonl files into.",
    )
    parser.add_argument(
        "--max-conversations",
        type=int,
        default=500,
        help="Trim base corpus to this many distinct conversations "
        "(all turns kept). Keeps subsequent analyze runs fast.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(SCENARIOS),
        choices=SCENARIOS,
    )
    args = parser.parse_args()

    build(args.base, args.out_dir, args.scenarios, args.max_conversations)


if __name__ == "__main__":
    main()
