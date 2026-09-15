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
from prefixlens.simulator import Report, TagStats


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
