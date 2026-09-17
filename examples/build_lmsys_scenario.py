"""Adapter: turn LMSys-Chat-1M (lmsys/lmsys-chat-1m) into a prefixlens JSONL corpus.

LMSys-Chat-1M is ~1M real conversations logged from Chatbot Arena and
LMSys Vicuna Demo. Gated dataset on HuggingFace — requires accepting
the license on the dataset page + `hf auth login` locally before use.

Same shape as the WildChat adapter (see examples/build_wildchat_scenario.py):
one JSONL row per user turn, prompt = concatenation of every prior
message plus this turn's user message. The value of running LMSys
alongside WildChat is model diversity — LMSys has ~25 distinct models
(Vicuna variants, GPT-3.5/4, Claude, Llama-2, PaLM-2, ...) vs
WildChat's small handful, so the per-model breakdown is richer.

Tokenizer: tiktoken cl100k_base. Not the exact per-model tokenizer for
every LMSys backend, but the hit/miss shape is dominated by
prefix-structure identity across requests, not by which tokenizer.

Usage:
    python examples/build_lmsys_scenario.py \\
        --max-conversations 2000 \\
        --out examples/lmsys_scenario.jsonl
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
    return f"{role}: {content}\n\n"


def _tokenize_prompt(messages: list[dict], enc: "tiktoken.Encoding") -> list[int]:
    text_parts = [_format_message(m["role"], m["content"]) for m in messages]
    return enc.encode("".join(text_parts))


def _iter_lmsys_conversations(streaming: bool):
    """Yield conversation dicts from lmsys/lmsys-chat-1m `train` split.

    Requires HF auth (`hf auth login`) since the dataset is gated.
    """
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=streaming)
    yield from ds


def build(
    max_conversations: int,
    out_path: Path,
    language_filter: str = "English",
    min_turns: int = 2,
    streaming: bool = True,
) -> tuple[int, int, float]:
    """Build the corpus. Returns (conversations_used, requests_emitted, elapsed_seconds)."""
    enc = tiktoken.get_encoding("cl100k_base")

    n_convs = 0
    n_requests = 0
    n_seen = 0
    n_skipped_lang = 0
    n_skipped_short = 0
    t0 = time.perf_counter()

    with out_path.open("w", encoding="utf-8") as fout:
        for conv in _iter_lmsys_conversations(streaming=streaming):
            n_seen += 1
            if n_convs >= max_conversations:
                break

            language = conv.get("language")
            if language_filter and language != language_filter:
                n_skipped_lang += 1
                continue

            conversation = conv.get("conversation") or []
            if len(conversation) < min_turns:
                n_skipped_short += 1
                continue

            conv_id = conv.get("conversation_id") or f"conv_{n_convs}"
            model = conv.get("model") or "unknown"

            turn_index = 0
            for i, msg in enumerate(conversation):
                if msg.get("role") != "user":
                    continue
                prompt_messages = conversation[: i + 1]
                token_ids = _tokenize_prompt(prompt_messages, enc)
                if not token_ids:
                    continue

                # Same tag policy as WildChat: only aggregation-worthy axes
                # (model, language). conversation_id + turn stay in the
                # request_id (which encodes them already).
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
    parser.add_argument("--max-conversations", type=int, default=2000)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/lmsys_scenario.jsonl"),
    )
    parser.add_argument("--language", default="English", help="Filter to this language (default: English). Pass empty string to disable.")
    parser.add_argument("--min-turns", type=int, default=2)
    parser.add_argument("--no-streaming", action="store_true")
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
