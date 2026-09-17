"""Command-line entry point for prefixlens.

The `prefixlens` command dispatches to subcommands: `analyze` produces a
report from a corpus, and `validate` checks whether the sim's top-line hit
rate matches what a real vLLM engine reported for the same request stream —
the credibility anchor per SPEC §8. `lint` and `explain` are v0.2+.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Sequence

from prefixlens.explain import RequestExplanation, explain_request
from prefixlens.loader import load_jsonl
from prefixlens.metrics import parse_vllm_metrics
from prefixlens.simulator import RadixCacheSimulator, Report


# vs-real tolerance for the "trust score" verdict, in percentage points of
# absolute hit-rate difference. SPEC §8 target: within ±3pp.
DEFAULT_TRUST_TOLERANCE_PP = 3.0


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
        "--tag",
        action="append",
        dest="tags",
        default=None,
        metavar="NAME",
        help="Only render this tag axis in the human report. Repeatable. "
        "Without --tag, high-cardinality axes (many unique values, most of "
        "them size-1) are auto-suppressed with a hint so the report stays "
        "readable on real workloads. --json output is always unfiltered.",
    )
    analyze.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON on stdout instead of the human-readable "
        "summary. Suitable for piping into jq or downstream tooling. Unfiltered — "
        "every tag axis is included regardless of --tag.",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Compare simulated hit rate against a real vLLM /metrics scrape.",
        description="Differential check: run the same request stream through "
        "the sim and compare the top-line prefix-cache hit rate against what "
        "vLLM reported in its /metrics. Verdicts OK (within tolerance) or "
        "DIVERGED. This is the credibility anchor: if this passes, downstream "
        "attribution can be trusted; if it fails, the sim or the config is "
        "wrong before anything else.",
    )
    validate.add_argument(
        "corpus",
        type=Path,
        help="Path to the JSONL corpus of the requests actually sent to vLLM. "
        "Order and completeness matter — a skipped request drifts sim state.",
    )
    validate.add_argument(
        "--metrics-file",
        type=Path,
        required=True,
        help="Path to a text file containing a vLLM /metrics scrape "
        "(e.g. `curl http://vllm:8000/metrics > metrics.txt`) taken AFTER "
        "the workload completed.",
    )
    validate.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size in tokens. MUST match the engine's real config or "
        "the comparison is meaningless. Default: 16 (vLLM default).",
    )
    validate.add_argument(
        "--capacity-blocks",
        type=int,
        required=True,
        help="Cache capacity in blocks on the real engine. Required — no "
        "sensible default (depends on GPU memory and model config).",
    )
    validate.add_argument(
        "--tolerance-pp",
        type=float,
        default=DEFAULT_TRUST_TOLERANCE_PP,
        help=f"Absolute hit-rate difference (in percentage points) below which "
        f"the sim is considered calibrated. Default: {DEFAULT_TRUST_TOLERANCE_PP}. "
        "SPEC §8 target.",
    )
    validate.add_argument(
        "--json",
        action="store_true",
        help="Emit the comparison as JSON on stdout instead of the "
        "human-readable summary.",
    )

    explain_cmd = subparsers.add_parser(
        "explain",
        help="Block-by-block trace of one request through the simulated cache.",
        description="Per-request diagnostic. Reads the whole corpus (needed to "
        "warm the cache to the same state the target request saw), then prints "
        "a block-by-block HIT/MISS trace for the target request and the "
        "aggregate-divergence context for its tag buckets. Answers 'why did "
        "req_abc123 miss?' in one command.",
    )
    explain_cmd.add_argument(
        "corpus",
        type=Path,
        help="JSONL corpus containing the target request.",
    )
    explain_cmd.add_argument(
        "--request-id",
        required=True,
        help="request_id of the request to explain (must appear in the corpus).",
    )
    explain_cmd.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size in tokens (default: 16, matching vLLM).",
    )
    explain_cmd.add_argument(
        "--capacity-blocks",
        type=int,
        default=4096,
        help="Cache capacity in blocks (default: 4096).",
    )
    explain_cmd.add_argument(
        "--json",
        action="store_true",
        help="Emit the explanation as JSON on stdout instead of the "
        "human-readable trace.",
    )

    return parser


def _format_percent(x: float) -> str:
    return f"{x * 100:5.1f}%"


# Cardinality thresholds for the "which tag axes are worth aggregating on"
# heuristic. An axis is suppressed only if BOTH conditions hold — small
# workloads with a lot of tenants (say 30 tenants over 50 requests) legitimately
# have high unique-ratio, but are still useful to show. Only the pathological
# case (many unique values AND most requests get their own bucket) gets hidden.
_MAX_TAG_VALUES_SHOWN = 20  # per axis, in human output
_CARDINALITY_ABS_THRESHOLD = 20  # unique-value count above which we care
_CARDINALITY_RATIO_THRESHOLD = 0.25  # unique-values / total-requests


def _select_tag_axes(
    report: Report,
    user_include: list[str] | None,
) -> tuple[list[str], list[tuple[str, int, float]]]:
    """Decide which tag axes to render.

    Returns (kept_axes_in_order, suppressed_hints).

    - If `user_include` is given, use it as an explicit whitelist (order
      preserved for reproducible output). No suppression logic.
    - Otherwise, apply the cardinality heuristic: an axis is suppressed only
      when it exceeds BOTH the absolute threshold and the unique-ratio
      threshold — real workloads with a few dozen tenants stay visible.
    """
    if user_include:
        return [t for t in user_include if t in report.by_tag], []

    kept: list[str] = []
    suppressed: list[tuple[str, int, float]] = []
    total = report.total_requests
    for k in sorted(report.by_tag.keys()):
        n_unique = len(report.by_tag[k])
        ratio = n_unique / total if total else 0.0
        if n_unique > _CARDINALITY_ABS_THRESHOLD and ratio > _CARDINALITY_RATIO_THRESHOLD:
            suppressed.append((k, n_unique, ratio))
        else:
            kept.append(k)
    return kept, suppressed


def _render_report_human(report: Report, user_tags: list[str] | None = None) -> str:
    """The README-style summary. Kept single-purpose (no I/O) so it is testable."""
    lines: list[str] = []
    lines.append(
        f"prefixlens analyze — {report.total_requests:,} requests, "
        f"{report.total_blocks:,} blocks total"
    )
    lines.append("")
    lines.append(f"  overall hit rate: {_format_percent(report.hit_rate)}  "
                 f"({report.cached_blocks:,} / {report.total_blocks:,} blocks)")

    kept_axes, suppressed_hints = _select_tag_axes(report, user_tags)

    if kept_axes:
        lines.append("")
        # Sort tag keys alphabetically for deterministic output; within a key,
        # sort tag values by descending hit rate — the "who's fine, who's broken"
        # ordering the user is actually scanning for.
        for tag_key in kept_axes:
            lines.append(f"  by {tag_key}:")
            values = sorted(
                report.by_tag[tag_key].items(),
                key=lambda kv: (-kv[1].hit_rate, kv[0]),
            )
            # Column width so the values align regardless of key length.
            max_name = max(len(v) for v, _ in values)
            shown = values[:_MAX_TAG_VALUES_SHOWN]
            for tag_value, stats in shown:
                lines.append(
                    f"    {tag_value:<{max_name}}  "
                    f"{_format_percent(stats.hit_rate)}  "
                    f"({stats.total_requests:,} req, {stats.cached_blocks:,}/{stats.total_blocks:,} blk)"
                )
            if len(values) > _MAX_TAG_VALUES_SHOWN:
                lines.append(f"    … and {len(values) - _MAX_TAG_VALUES_SHOWN:,} more values (use --json for the full list)")

    if suppressed_hints:
        lines.append("")
        for tag_key, n_unique, ratio in suppressed_hints:
            lines.append(
                f"  by {tag_key}: suppressed ({n_unique:,} unique values, "
                f"{ratio * 100:.1f}% unique — pass --tag {tag_key} to include)"
            )

    # Divergent-position section: only render for tag buckets that actually
    # have misses. A UUID-case bucket (unique_content_ratio close to 1) prints
    # a "unique content" tag; a thrashing bucket (ratio close to 0) prints
    # "shared content" instead. This is the diagnosis, not just the number.
    divergence_lines = _render_divergence_section(report, kept_axes)
    if divergence_lines:
        lines.append("")
        lines.extend(divergence_lines)

    return "\n".join(lines)


def _classify_content(ratio: float) -> str:
    """Human-readable label for unique_content_ratio.

    The cutoffs are intentional but not sacred: 0.9+ is 'basically all unique'
    (UUID-shaped), 0.3 or less is 'basically all the same' (thrashing-shaped),
    the middle is 'mixed' and the caller should look closer.
    """
    if ratio >= 0.9:
        return "unique content per request"
    if ratio <= 0.3:
        return "shared content, thrashing"
    return "mixed content"


def _render_divergence_section(report, allowed_tag_keys: list[str] | None = None) -> list[str]:
    """Return the 'top divergent positions' block, empty if there's nothing
    to say. When allowed_tag_keys is given, only those axes appear — mirrors
    the by_tag filter so operators don't see divergence sections for axes
    the summary hid."""
    lines: list[str] = []
    tag_keys = allowed_tag_keys if allowed_tag_keys is not None else sorted(report.by_tag_divergence.keys())
    for tag_key in tag_keys:
        if tag_key not in report.by_tag_divergence:
            continue
        for tag_value in sorted(report.by_tag_divergence[tag_key].keys()):
            positions = report.by_tag_divergence[tag_key][tag_value]
            if not positions:
                continue
            top = positions[:3]  # top-3 by miss_count
            lines.append(f"  divergent positions for {tag_key}={tag_value}:")
            for p in top:
                diagnosis = _classify_content(p.unique_content_ratio)
                lines.append(
                    f"    block {p.block_position:>3}  "
                    f"{p.miss_count:>5,} miss  "
                    f"({p.unique_content_ratio * 100:5.1f}% unique — {diagnosis})"
                )
    return lines


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
        # JSON is always the full report — programmatic consumers get everything.
        print(_report_to_json(report))
    else:
        print(_render_report_human(report, user_tags=args.tags))
    return 0


def _render_validate_human(
    sim_hit_rate: float,
    sim_total_blocks: int,
    real_hits: int,
    real_queries: int,
    real_hit_rate: float,
    delta_pp: float,
    tolerance_pp: float,
    verdict: str,
) -> str:
    lines = [
        "prefixlens validate — sim vs real vLLM /metrics",
        "",
        f"  sim hit rate:   {sim_hit_rate * 100:5.1f}%   ({sim_total_blocks:,} blocks processed)",
        f"  real hit rate:  {real_hit_rate * 100:5.1f}%   "
        f"({real_hits:,} hits / {real_queries:,} queries)",
        f"  delta:          {delta_pp:5.2f} pp   (tolerance ±{tolerance_pp:.1f} pp)",
        "",
        f"  verdict: {verdict}",
    ]
    if verdict == "OK":
        lines.append("           Sim is calibrated on this workload; downstream")
        lines.append("           attribution is trustworthy.")
    else:
        lines.append("           Sim and engine disagree beyond tolerance. Check:")
        lines.append("           - --block-size and --capacity-blocks match the real engine")
        lines.append("           - the request log is complete and in order")
        lines.append("           - the /metrics scrape was taken AFTER the workload finished")
    return "\n".join(lines)


def _run_validate(args: argparse.Namespace) -> int:
    metrics_text = args.metrics_file.read_text(encoding="utf-8")
    real = parse_vllm_metrics(metrics_text)

    sim = RadixCacheSimulator(
        block_size=args.block_size,
        capacity_blocks=args.capacity_blocks,
    )
    for req in load_jsonl(args.corpus):
        sim.process(req)
    sim_report = sim.report()

    delta_pp = abs(sim_report.hit_rate - real.hit_rate) * 100
    verdict = "OK" if delta_pp <= args.tolerance_pp else "DIVERGED"

    if args.json:
        result = {
            "sim_hit_rate": sim_report.hit_rate,
            "sim_total_blocks": sim_report.total_blocks,
            "real_hit_rate": real.hit_rate,
            "real_prefix_cache_hits": real.prefix_cache_hits,
            "real_prefix_cache_queries": real.prefix_cache_queries,
            "delta_pp": delta_pp,
            "tolerance_pp": args.tolerance_pp,
            "verdict": verdict,
        }
        print(json.dumps(result, indent=2))
    else:
        print(_render_validate_human(
            sim_hit_rate=sim_report.hit_rate,
            sim_total_blocks=sim_report.total_blocks,
            real_hits=real.prefix_cache_hits,
            real_queries=real.prefix_cache_queries,
            real_hit_rate=real.hit_rate,
            delta_pp=delta_pp,
            tolerance_pp=args.tolerance_pp,
            verdict=verdict,
        ))

    # Exit code communicates the verdict for scripting: 0 = OK, 1 = DIVERGED.
    # Not a fatal error (the tool ran fine), just a signal.
    return 0 if verdict == "OK" else 1


def _format_block_tokens_truncated(tokens: tuple[int, ...]) -> str:
    """Truncate a 16-token block for human-readable display.

    Shows first 4 + last 4 with an ellipsis in between; blocks smaller than
    that render in full. Full arrays are still emitted through --json.
    """
    if len(tokens) <= 8:
        return "[" + ", ".join(str(t) for t in tokens) + "]"
    head = ", ".join(str(t) for t in tokens[:4])
    tail = ", ".join(str(t) for t in tokens[-4:])
    return f"[{head}, …, {tail}]"


def _render_explain_human(exp: RequestExplanation) -> str:
    lines: list[str] = []
    tag_str = ", ".join(f"{k}={v}" for k, v in exp.tags) if exp.tags else "(no tags)"
    lines.append(f"prefixlens explain — {exp.request_id}")
    lines.append("")
    lines.append(f"  tags:    {tag_str}")
    lines.append(f"  blocks:  {exp.cached_prefix_blocks} hit, "
                 f"{exp.total_prompt_blocks - exp.cached_prefix_blocks} miss, "
                 f"{exp.total_prompt_blocks} total")
    if exp.first_divergent_block is not None:
        lines.append(f"  first divergence: block {exp.first_divergent_block}")
    else:
        lines.append("  first divergence: none (full hit or empty prompt)")

    if exp.block_traces:
        lines.append("")
        lines.append("  block-by-block trace:")
        for t in exp.block_traces:
            lines.append(
                f"    block {t.position:>3}  {t.verdict:<4}  "
                f"{_format_block_tokens_truncated(t.tokens)}"
            )

    if exp.divergence_context:
        lines.append("")
        lines.append("  divergence context (aggregate signal at this position):")
        for ctx in exp.divergence_context:
            lines.append(
                f"    {ctx.tag_key}={ctx.tag_value}: "
                f"{ctx.position.miss_count:,} miss at block {ctx.position.block_position} "
                f"({ctx.position.unique_content_ratio * 100:5.1f}% unique)"
            )
    return "\n".join(lines)


def _run_explain(args: argparse.Namespace) -> int:
    sim = RadixCacheSimulator(
        block_size=args.block_size,
        capacity_blocks=args.capacity_blocks,
    )
    for req in load_jsonl(args.corpus):
        sim.process(req)
    report = sim.report()

    exp = explain_request(sim, args.request_id, report)
    if exp is None:
        print(
            f"prefixlens explain: request_id {args.request_id!r} not found in corpus",
            file=sys.stderr,
        )
        return 2

    if args.json:
        print(json.dumps(dataclasses.asdict(exp), indent=2))
    else:
        print(_render_explain_human(exp))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "validate":
        return _run_validate(args)
    if args.command == "explain":
        return _run_explain(args)
    # required=True on subparsers means argparse already errored; unreachable.
    parser.error(f"unknown command: {args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
