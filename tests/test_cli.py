"""Tests for the `prefixlens` command-line interface.

Split into two layers:

1. Pure renderers (`_render_report_human`, `_report_to_json`) are tested
   against hand-constructed Report objects — no I/O.
2. `main()` is tested end-to-end: it reads a real JSONL fixture from
   tmp_path, prints to captured stdout, and returns an exit code.
"""

import json

import pytest

from prefixlens.cli import _render_report_human, _report_to_json, main
from prefixlens.simulator import DivergentPosition, Report, TagStats


# ---- pure render layer ---------------------------------------------------


def test_human_render_shows_overall_and_per_tag_breakdown():
    report = Report(
        hit_rate=0.5,
        total_requests=10,
        cached_blocks=20,
        total_blocks=40,
        by_tag={
            "tenant": {
                "acme": TagStats(hit_rate=0.8, total_requests=5, cached_blocks=16, total_blocks=20),
                "widgets": TagStats(hit_rate=0.0, total_requests=5, cached_blocks=0, total_blocks=20),
            }
        },
    )

    out = _render_report_human(report)

    assert "10 requests" in out
    assert "40 blocks total" in out
    assert " 50.0%" in out    # overall
    assert " 80.0%" in out    # acme
    assert "  0.0%" in out    # widgets
    # Per-tenant lines both present
    assert "acme" in out
    assert "widgets" in out


def test_human_render_sorts_by_descending_hit_rate_within_a_tag_key():
    # The "who's fine, who's broken" scan works only if best-first ordering.
    # acme (80%) should render before widgets (0%).
    report = Report(
        hit_rate=0.4,
        total_requests=10,
        cached_blocks=16,
        total_blocks=40,
        by_tag={
            "tenant": {
                "widgets": TagStats(hit_rate=0.0, total_requests=5, cached_blocks=0, total_blocks=20),
                "acme": TagStats(hit_rate=0.8, total_requests=5, cached_blocks=16, total_blocks=20),
            }
        },
    )

    out = _render_report_human(report)
    acme_pos = out.index("acme")
    widgets_pos = out.index("widgets")
    assert acme_pos < widgets_pos


def test_human_render_omits_by_tag_section_when_empty():
    report = Report(
        hit_rate=0.0,
        total_requests=0,
        cached_blocks=0,
        total_blocks=0,
    )

    out = _render_report_human(report)

    assert "by " not in out


def test_json_render_is_valid_json_and_roundtrips():
    report = Report(
        hit_rate=0.5,
        total_requests=2,
        cached_blocks=2,
        total_blocks=4,
        by_tag={
            "tenant": {
                "acme": TagStats(hit_rate=0.5, total_requests=2, cached_blocks=2, total_blocks=4)
            }
        },
    )

    parsed = json.loads(_report_to_json(report))

    assert parsed["hit_rate"] == 0.5
    assert parsed["total_requests"] == 2
    assert parsed["by_tag"]["tenant"]["acme"]["hit_rate"] == 0.5
    assert parsed["by_tag"]["tenant"]["acme"]["total_blocks"] == 4


# ---- end-to-end via main() -----------------------------------------------


def _write_corpus(tmp_path, lines):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return path


def test_analyze_command_prints_human_report(tmp_path, capsys):
    corpus = _write_corpus(
        tmp_path,
        [
            {"request_id": "r1", "token_ids": list(range(32)), "tenant": "acme"},
            {"request_id": "r2", "token_ids": list(range(32)), "tenant": "acme"},
        ],
    )

    exit_code = main(["analyze", str(corpus), "--block-size", "16", "--capacity-blocks", "100"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "2 requests" in out
    assert " 50.0%" in out          # r1 misses, r2 hits, overall 50% (2/4 blocks)
    assert "acme" in out


def test_analyze_command_json_flag_emits_parseable_json(tmp_path, capsys):
    corpus = _write_corpus(
        tmp_path,
        [
            {"token_ids": list(range(32)), "tenant": "acme"},
            {"token_ids": list(range(32)), "tenant": "acme"},
        ],
    )

    exit_code = main([
        "analyze",
        str(corpus),
        "--block-size", "16",
        "--capacity-blocks", "100",
        "--json",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed["total_requests"] == 2
    assert parsed["hit_rate"] == 0.5
    assert parsed["by_tag"]["tenant"]["acme"]["cached_blocks"] == 2


def test_analyze_command_uses_defaults_when_flags_omitted(tmp_path, capsys):
    corpus = _write_corpus(tmp_path, [{"token_ids": list(range(32))}])

    exit_code = main(["analyze", str(corpus)])

    assert exit_code == 0
    # 1 request, both blocks are misses on an empty cache; 0/2 = 0% hit rate.
    out = capsys.readouterr().out
    assert "1 requests" in out
    assert "  0.0%" in out


def test_analyze_command_propagates_loader_errors(tmp_path):
    # Malformed corpus should surface as a ValueError (the loader's contract) —
    # for v0.1 we let it bubble; a future --skip-invalid flag would catch it.
    corpus = tmp_path / "bad.jsonl"
    corpus.write_text('{"token_ids": [1]}\n{not-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2.*invalid JSON"):
        main(["analyze", str(corpus)])


def test_analyze_command_fails_on_prompt_field_without_tokenizer(tmp_path):
    # The v0.1 CLI has no tokenizer flag — corpora with 'prompt' fields must
    # ship pre-tokenized. Any other input surfaces the loader's error.
    corpus = _write_corpus(tmp_path, [{"prompt": "hello world"}])

    with pytest.raises(ValueError, match="tokenizer"):
        main(["analyze", str(corpus)])


def test_missing_subcommand_exits_nonzero(capsys):
    # argparse's required=True enforcement: no subcommand → error to stderr,
    # exit code 2.
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


# ---- divergence rendering ------------------------------------------------


def test_human_render_includes_divergence_section_when_present():
    # A UUID-shaped bucket: high unique_content_ratio.
    report = Report(
        hit_rate=0.0,
        total_requests=5,
        cached_blocks=0,
        total_blocks=20,
        by_tag={
            "tenant": {
                "widgets": TagStats(hit_rate=0.0, total_requests=5, cached_blocks=0, total_blocks=20)
            }
        },
        by_tag_divergence={
            "tenant": {
                "widgets": (
                    DivergentPosition(block_position=0, miss_count=5, unique_content_ratio=1.0),
                )
            }
        },
    )

    out = _render_report_human(report)

    assert "divergent positions for tenant=widgets" in out
    assert "block   0" in out
    assert "5 miss" in out
    assert "unique content per request" in out


def test_human_render_labels_thrashing_bucket_differently():
    # Low unique_content_ratio → shared-content / thrashing diagnosis.
    report = Report(
        hit_rate=0.0,
        total_requests=6,
        cached_blocks=0,
        total_blocks=6,
        by_tag={
            "tenant": {
                "thrash": TagStats(hit_rate=0.0, total_requests=6, cached_blocks=0, total_blocks=6)
            }
        },
        by_tag_divergence={
            "tenant": {
                "thrash": (
                    DivergentPosition(block_position=0, miss_count=6, unique_content_ratio=0.2),
                )
            }
        },
    )

    out = _render_report_human(report)

    assert "shared content, thrashing" in out
    assert "unique content per request" not in out


def test_human_render_omits_divergence_for_all_hit_buckets():
    # by_tag_divergence has an entry but it's empty (all hits) → no section rendered.
    report = Report(
        hit_rate=1.0,
        total_requests=5,
        cached_blocks=20,
        total_blocks=20,
        by_tag={
            "tenant": {
                "acme": TagStats(hit_rate=1.0, total_requests=5, cached_blocks=20, total_blocks=20)
            }
        },
        by_tag_divergence={"tenant": {"acme": ()}},
    )

    out = _render_report_human(report)

    assert "divergent positions" not in out


def test_human_render_limits_to_top_three_positions():
    # Five positions in the bucket; only the top 3 (by miss count) should render.
    positions = tuple(
        DivergentPosition(block_position=i, miss_count=100 - i, unique_content_ratio=1.0)
        for i in range(5)
    )
    report = Report(
        hit_rate=0.0,
        total_requests=500,
        cached_blocks=0,
        total_blocks=500,
        by_tag={
            "tenant": {
                "t": TagStats(hit_rate=0.0, total_requests=500, cached_blocks=0, total_blocks=500)
            }
        },
        by_tag_divergence={"tenant": {"t": positions}},
    )

    out = _render_report_human(report)

    assert "block   0" in out
    assert "block   1" in out
    assert "block   2" in out
    assert "block   3" not in out
    assert "block   4" not in out


# ---- validate subcommand -------------------------------------------------


def _make_matching_metrics(hits: int, queries: int) -> str:
    return (
        f"vllm:prefix_cache_hits {hits}\n"
        f"vllm:prefix_cache_queries {queries}\n"
    )


def test_validate_reports_OK_when_sim_matches_real(tmp_path, capsys):
    # Two identical 2-block requests. Sim: r1 = 0/2, r2 = 2/2, overall 2/4 = 50%.
    # Hand-craft matching metrics.
    corpus = _write_corpus(
        tmp_path,
        [
            {"token_ids": list(range(32))},
            {"token_ids": list(range(32))},
        ],
    )
    metrics = tmp_path / "metrics.txt"
    metrics.write_text(_make_matching_metrics(hits=2, queries=4), encoding="utf-8")

    exit_code = main([
        "validate", str(corpus),
        "--metrics-file", str(metrics),
        "--block-size", "16",
        "--capacity-blocks", "100",
    ])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "verdict: OK" in out
    assert "sim is calibrated" in out.lower()


def test_validate_reports_DIVERGED_when_delta_exceeds_tolerance(tmp_path, capsys):
    # Sim will produce 50% hit rate; give real a very different one.
    corpus = _write_corpus(
        tmp_path,
        [
            {"token_ids": list(range(32))},
            {"token_ids": list(range(32))},
        ],
    )
    metrics = tmp_path / "metrics.txt"
    # real hit rate = 90%, sim = 50%, delta = 40pp >> tolerance
    metrics.write_text(_make_matching_metrics(hits=90, queries=100), encoding="utf-8")

    exit_code = main([
        "validate", str(corpus),
        "--metrics-file", str(metrics),
        "--block-size", "16",
        "--capacity-blocks", "100",
    ])

    assert exit_code == 1  # nonzero exit signals disagreement for scripts
    out = capsys.readouterr().out
    assert "verdict: DIVERGED" in out
    # Should print troubleshooting hints
    assert "block-size" in out


def test_validate_tolerance_flag_controls_verdict(tmp_path, capsys):
    # Sim = 50%, real = 55%, delta = 5pp.
    # Default tolerance 3pp → DIVERGED. Raise to 10pp → OK.
    corpus = _write_corpus(
        tmp_path,
        [
            {"token_ids": list(range(32))},
            {"token_ids": list(range(32))},
        ],
    )
    metrics = tmp_path / "metrics.txt"
    metrics.write_text(_make_matching_metrics(hits=55, queries=100), encoding="utf-8")

    exit_code = main([
        "validate", str(corpus),
        "--metrics-file", str(metrics),
        "--block-size", "16",
        "--capacity-blocks", "100",
        "--tolerance-pp", "10",
    ])

    assert exit_code == 0
    assert "verdict: OK" in capsys.readouterr().out


def test_validate_json_flag_emits_parseable_result(tmp_path, capsys):
    corpus = _write_corpus(tmp_path, [{"token_ids": list(range(32))}])
    metrics = tmp_path / "metrics.txt"
    metrics.write_text(_make_matching_metrics(hits=0, queries=2), encoding="utf-8")

    exit_code = main([
        "validate", str(corpus),
        "--metrics-file", str(metrics),
        "--block-size", "16",
        "--capacity-blocks", "100",
        "--json",
    ])

    assert exit_code == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["verdict"] == "OK"
    assert parsed["sim_hit_rate"] == 0.0
    assert parsed["real_hit_rate"] == 0.0
    assert parsed["real_prefix_cache_hits"] == 0
    assert parsed["real_prefix_cache_queries"] == 2
    assert parsed["delta_pp"] == 0.0


def test_validate_requires_capacity_blocks(tmp_path):
    # --capacity-blocks is required by design (no sensible default).
    corpus = _write_corpus(tmp_path, [{"token_ids": [1]}])
    metrics = tmp_path / "metrics.txt"
    metrics.write_text("vllm:prefix_cache_hits 0\nvllm:prefix_cache_queries 0\n")

    with pytest.raises(SystemExit) as exc:
        main(["validate", str(corpus), "--metrics-file", str(metrics)])
    assert exc.value.code == 2


def test_validate_surfaces_metrics_parse_error(tmp_path):
    corpus = _write_corpus(tmp_path, [{"token_ids": list(range(32))}])
    metrics = tmp_path / "empty.txt"
    metrics.write_text("# just comments\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        main([
            "validate", str(corpus),
            "--metrics-file", str(metrics),
            "--block-size", "16",
            "--capacity-blocks", "100",
        ])


def test_end_to_end_cli_prints_divergence_on_chen_scenario(tmp_path, capsys):
    """The bundled example should surface the UUID diagnosis in one command."""
    corpus = tmp_path / "chen.jsonl"
    lines = []
    for i in range(5):
        lines.append(json.dumps({
            "request_id": f"acme-{i}",
            "token_ids": list(range(1000, 1000 + 64)),  # shared 4-block prefix
            "tenant": "acme",
        }))
    for i in range(5):
        # Unique first token per widgets request
        tokens = [9_000_000 + i] + list(range(2001, 2001 + 63))  # 64 total
        lines.append(json.dumps({
            "request_id": f"widgets-{i}",
            "token_ids": tokens,
            "tenant": "widgets",
        }))
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")

    exit_code = main(["analyze", str(corpus), "--block-size", "16", "--capacity-blocks", "1000"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "divergent positions for tenant=widgets" in out
    assert "unique content per request" in out
    # Acme has one miss (r0), no diagnostic message we care to assert on here.
