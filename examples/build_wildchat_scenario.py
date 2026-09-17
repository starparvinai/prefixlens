"""Adapter: turn WildChat-1M (allenai/WildChat-1M) into a prefixlens JSONL corpus.

WildChat-1M is a ~1M-conversation dataset of real users chatting with
LLM-backed products (public, MIT license, no HF auth needed for the
non-`toxic` subset). Each conversation has multiple turns.

For prefix-cache simulation, one *request* to a serving engine = one
prefill of a full prompt up through a user turn — so a K-turn
conversation generates K requests, where request k's prompt is the
concatenation of every message (user + assistant) preceding turn k
plus turn k's user message.

Output: one JSONL row per user turn, with:
    - request_id: "<conversation_id>_turn_<k>"
    - token_ids: cl100k_base tokens of the full running prompt
    - model: the assistant model name (a natural tag axis)
    - language: filter axis
    - conversation_id + turn: for later per-session analysis

Tokenizer: tiktoken cl100k_base (GPT-3.5/4). Not the *actual* per-model
tokenizer of every WildChat backend, but the *shape* of hit/miss
patterns is what matters for prefix-cache simulation. Real
production tests would use the exact serving tokenizer.

Usage:
    python examples/build_wildchat_scenario.py \\
        --max-conversations 5000 \\
        --out examples/wildchat_scenario.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import tiktoken
except ImportError:
    print(
        "This script needs `tiktoken`. Install into the venv with:\n"
        "    .venv/bin/pip install tiktoken",
        file=sys.stderr,
    )
    sys.exit(2)

try:
    from datasets import load_dataset
except ImportError:
    print(
        "This script needs `datasets`. Install into the venv with:\n"
        "    .venv/bin/pip install datasets",
        file=sys.stderr,
    )
    sys.exit(2)


def _format_message(role: str, content: str) -> str:
    """Render one message the way most chat templates roughly do —
    role tag, then content, then a blank line. Exact template varies
    per model, but the hit/miss shape is dominated by the shared
    prefix pattern, not by which delimiter you use.
    """
    return f"{role}: {content}\n\n"


def _tokenize_prompt(messages: list[dict], enc: "tiktoken.Encoding") -> list[int]:
    """Concatenate messages 0..N in order, format each, tokenize the whole
    resulting string as one prompt.
    """
    text_parts = [_format_message(m["role"], m["content"]) for m in messages]
    return enc.encode("".join(text_parts))


def _iter_wildchat_conversations(subset: str, streaming: bool):
    """Yield conversation dicts from the WildChat-1M `train` split.

    `subset` picks which split of the dataset to pull from — default
    'default' is the full (non-toxic-filtered) set. Streaming means we
    don't have to download the whole thing to disk before we start.
    """
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=streaming)
    yield from ds


def build(
    max_conversations: int,
    out_path: Path,
    language_filter: str = "English",
    min_turns: int = 2,
    streaming: bool = True,
) -> tuple[int, int, int]:
    """Build the corpus. Returns (conversations_used, requests_emitted, elapsed_seconds)."""
    enc = tiktoken.get_encoding("cl100k_base")

    n_convs = 0
    n_requests = 0
    n_seen = 0
    n_skipped_lang = 0
    n_skipped_short = 0
    t0 = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as fout:
        for conv in _iter_wildchat_conversations("default", streaming=streaming):
            n_seen += 1
            if n_convs >= max_conversations:
                break

            language = conv.get("language")
            if language_filter and language != language_filter:
                n_skipped_lang += 1
                continue

            conversation = conv.get("conversation") or []
            # Need at least one user turn AND enough total turns to be interesting.
            if len(conversation) < min_turns:
                n_skipped_short += 1
                continue

            conv_id = conv.get("conversation_hash") or conv.get("conversation_id") or f"conv_{n_convs}"
            model = conv.get("model") or "unknown"

            # Emit one request per user turn. The prompt at user turn k is the
            # concatenation of every message 0..2k inclusive (all history up
            # through this user message).
            turn_index = 0
            for i, msg in enumerate(conversation):
                if msg.get("role") != "user":
                    continue
                # Prompt = messages 0..i (inclusive)
                prompt_messages = conversation[: i + 1]
                token_ids = _tokenize_prompt(prompt_messages, enc)

                if not token_ids:
                    continue

                # conversation_id and turn stay embedded in request_id (which
                # already carries them as "<conv_hash>_turn_<n>"); putting them
                # in tags would create ~1 bucket per conversation and drown the
                # report. Only expose the axes worth aggregating on.
                record = {
                    "request_id": f"{conv_id}_turn_{turn_index}",
                    "token_ids": token_ids,
                    "model": model,
                    "language": language,
                }
                fout.write(json.dumps(record) + "\n")
                n_requests += 1
                turn_index += 1

            if turn_index > 0:
                n_convs += 1

            if n_convs and n_convs % 500 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  … {n_convs:,} conversations, {n_requests:,} requests "
                    f"({elapsed:.1f}s, seen {n_seen:,})",
                    file=sys.stderr,
                )

    elapsed = time.perf_counter() - t0
    print(
        f"done: {n_convs:,} conversations, {n_requests:,} requests emitted "
        f"in {elapsed:.1f}s.\n"
        f"  skipped {n_skipped_lang:,} for language != {language_filter!r}, "
        f"{n_skipped_short:,} for < {min_turns} turns.",
        file=sys.stderr,
    )
    return n_convs, n_requests, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--max-conversations", type=int, default=1000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/wildchat_scenario.jsonl"),
        help="Output path for the JSONL corpus (default: examples/wildchat_scenario.jsonl).",
    )
    parser.add_argument("--language", default="English", help="Filter to this language (default: English). Pass empty string to disable.")
    parser.add_argument("--min-turns", type=int, default=2, help="Skip conversations with fewer than this many messages (default: 2).")
    parser.add_argument("--no-streaming", action="store_true", help="Download the whole dataset first instead of streaming.")
    args = parser.parse_args()

    build(
        max_conversations=args.max_conversations,
        out_path=args.out,
        language_filter=args.language or None,
        min_turns=args.min_turns,
        streaming=not args.no_streaming,
    )


if __name__ == "__main__":
    main()
