"""Command-line entry point for prefixlens.

The `prefixlens` command dispatches to subcommands. In v0.1 there is only
one — `analyze` — but the subparser scaffold is in place so `validate` and
`lint` (SPEC §4) can be added without restructuring.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Sequence

from prefixlens.loader import load_jsonl
from prefixlens.simulator import RadixCacheSimulator, Report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prefixlens",
        description="Cache-attribution analyzer for LLM prefix caches.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Simulate a prefix cache over a JSONL corpus and print a report.",
    )
    analyze.add_argument(
        "corpus",
        type=Path,
        help="Path to a JSONL corpus. Each line requires either 'token_ids' "
        "(list of ints) or 'prompt' (string; v0.1 CLI does not accept a "
        "tokenizer flag, so pre-tokenize your corpus).",
    )
    analyze.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size in tokens (default: 16, matching vLLM).",
    )
    analyze.add_argument(
        "--capacity-blocks",
        type=int,
        default=4096,
        help="Cache capacity in blocks (default: 4096).",
    )
    analyze.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON on stdout instead of the human-readable "
        "summary. Suitable for piping into jq or downstream tooling.",
    )
    return parser


def _format_percent(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _render_report_human(report: Report) -> str:
    """The README-style summary. Kept single-purpose (no I/O) so it is testable."""
    lines: list[str] = []
    lines.append(
        f"prefixlens analyze — {report.total_requests:,} requests, "
        f"{report.total_blocks:,} blocks total"
    )
    lines.append("")
    lines.append(f"  overall hit rate: {_format_percent(report.hit_rate)}  "
                 f"({report.cached_blocks:,} / {report.total_blocks:,} blocks)")

    if report.by_tag:
        lines.append("")
        # Sort tag keys alphabetically for deterministic output; within a key,
        # sort tag values by descending hit rate — the "who's fine, who's broken"
        # ordering the user is actually scanning for.
        for tag_key in sorted(report.by_tag.keys()):
            lines.append(f"  by {tag_key}:")
            values = sorted(
                report.by_tag[tag_key].items(),
                key=lambda kv: (-kv[1].hit_rate, kv[0]),
            )
            # Column width so the values align regardless of key length.
            max_name = max(len(v) for v, _ in values)
            for tag_value, stats in values:
                lines.append(
                    f"    {tag_value:<{max_name}}  "
                    f"{_format_percent(stats.hit_rate)}  "
                    f"({stats.total_requests:,} req, {stats.cached_blocks:,}/{stats.total_blocks:,} blk)"
                )
    return "\n".join(lines)


def _report_to_json(report: Report) -> str:
    """Serialize Report to a JSON string.

    dataclasses.asdict handles the nested TagStats and dicts; the whole
    structure is JSON-native (no tuples, no custom types) so no encoder
    hook is needed.
    """
    return json.dumps(dataclasses.asdict(report), indent=2)


def _run_analyze(args: argparse.Namespace) -> int:
    sim = RadixCacheSimulator(
        block_size=args.block_size,
        capacity_blocks=args.capacity_blocks,
    )
    for req in load_jsonl(args.corpus):
        sim.process(req)
    report = sim.report()

    if args.json:
        print(_report_to_json(report))
    else:
        print(_render_report_human(report))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return _run_analyze(args)
    # required=True on subparsers means argparse already errored; unreachable.
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
