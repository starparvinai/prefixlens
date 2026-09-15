"""JSONL corpus loader.

The loader is the boundary between an on-disk trace and the in-memory
Request stream the simulator consumes. Tests exercise real files on
disk (via pytest's tmp_path) — not mocks — because file-format bugs
show up at the boundary, not in unit isolation.
"""

import json

import pytest

from prefixlens import Request, load_jsonl


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


# ---- token_ids fast path -------------------------------------------------


def test_token_ids_path_yields_requests_in_order(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [
            {"request_id": "r1", "token_ids": [1, 2, 3]},
            {"request_id": "r2", "token_ids": [4, 5, 6]},
        ],
    )

    reqs = list(load_jsonl(corpus))

    assert [r.request_id for r in reqs] == ["r1", "r2"]
    assert reqs[0].token_ids == (1, 2, 3)
    assert reqs[1].token_ids == (4, 5, 6)


def test_token_ids_are_returned_as_tuples(tmp_path):
    # Request stores tuples; loader must not leak a mutable list into the frozen dataclass.
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"token_ids": [1, 2, 3]}])

    (req,) = load_jsonl(corpus)

    assert isinstance(req.token_ids, tuple)


def test_missing_request_id_defaults_to_line_number(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"token_ids": [1]}, {"token_ids": [2]}])

    reqs = list(load_jsonl(corpus))

    assert [r.request_id for r in reqs] == ["line-1", "line-2"]


# ---- prompt (text) path --------------------------------------------------


def test_prompt_path_uses_provided_tokenizer(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"request_id": "r1", "prompt": "hello world"}])

    # Trivial tokenizer for testing: one token per whitespace-separated word,
    # id = length of the word.
    tokenizer = lambda text: [len(w) for w in text.split()]

    (req,) = load_jsonl(corpus, tokenizer=tokenizer)

    assert req.token_ids == (5, 5)  # len("hello")=5, len("world")=5


def test_token_ids_wins_when_both_are_present(tmp_path):
    # If a corpus was pre-tokenized and *also* has the original text preserved,
    # we skip retokenizing — trust what the caller sent to the engine.
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [{"prompt": "hello", "token_ids": [999]}],
    )
    tokenizer_calls = []

    def tokenizer(text):
        tokenizer_calls.append(text)
        return [1, 2, 3]

    (req,) = load_jsonl(corpus, tokenizer=tokenizer)

    assert req.token_ids == (999,)
    assert tokenizer_calls == []  # tokenizer never invoked


def test_prompt_without_tokenizer_raises(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"request_id": "r1", "prompt": "hello"}])

    with pytest.raises(ValueError, match="line 1.*tokenizer"):
        list(load_jsonl(corpus))


# ---- tag flattening ------------------------------------------------------


def test_top_level_tag_fields_become_tags(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [{"token_ids": [1], "tenant": "acme", "route": "/chat", "model": "llama-3"}],
    )

    (req,) = load_jsonl(corpus)

    assert dict(req.tags) == {"tenant": "acme", "route": "/chat", "model": "llama-3"}


def test_nested_tags_dict_merges_with_top_level(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [
            {
                "token_ids": [1],
                "tenant": "acme",
                "tags": {"env": "prod", "prompt_family": "chat_v2"},
            }
        ],
    )

    (req,) = load_jsonl(corpus)

    assert dict(req.tags) == {
        "tenant": "acme",
        "env": "prod",
        "prompt_family": "chat_v2",
    }


def test_nested_tag_overrides_top_level_on_collision(tmp_path):
    # Edge case: same key at top level AND in nested tags. Nested wins.
    # (Last-write-wins is the least surprising rule; documented on _normalize_tags.)
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [{"token_ids": [1], "tenant": "top-level", "tags": {"tenant": "nested"}}],
    )

    (req,) = load_jsonl(corpus)

    assert dict(req.tags)["tenant"] == "nested"


def test_tag_values_are_coerced_to_strings(tmp_path):
    # Numeric tags in JSON are common (e.g. shard id). Coerce so downstream
    # aggregation isn't tripped by mixed types.
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"token_ids": [1], "tags": {"shard": 7, "region": "us"}}])

    (req,) = load_jsonl(corpus)

    assert dict(req.tags) == {"shard": "7", "region": "us"}


def test_tags_are_sorted_deterministically(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [{"token_ids": [1], "tenant": "z", "route": "/a", "model": "m"}],
    )

    (req,) = load_jsonl(corpus)

    keys = [k for k, _ in req.tags]
    assert keys == sorted(keys)


def test_no_tags_yields_empty_tuple(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"token_ids": [1]}])

    (req,) = load_jsonl(corpus)

    assert req.tags == ()


# ---- malformed input -----------------------------------------------------


def test_blank_lines_are_skipped(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(
        json.dumps({"token_ids": [1]}) + "\n"
        + "\n"
        + "   \n"
        + json.dumps({"token_ids": [2]}) + "\n",
        encoding="utf-8",
    )

    reqs = list(load_jsonl(corpus))

    assert [r.token_ids for r in reqs] == [(1,), (2,)]


def test_missing_both_token_ids_and_prompt_raises_with_line_number(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [{"token_ids": [1]}, {"tenant": "acme"}],  # line 2 has neither
    )

    with pytest.raises(ValueError, match="line 2.*token_ids.*prompt"):
        list(load_jsonl(corpus))


def test_invalid_json_raises_with_line_number(tmp_path):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('{"token_ids": [1]}\n{not-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 2.*invalid JSON"):
        list(load_jsonl(corpus))


def test_non_object_json_raises(tmp_path):
    # A bare array or string is valid JSON but not a record.
    corpus = tmp_path / "c.jsonl"
    corpus.write_text('[1, 2, 3]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 1.*expected a JSON object"):
        list(load_jsonl(corpus))


def test_non_integer_token_ids_raises(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"token_ids": [1, "two", 3]}])

    with pytest.raises(ValueError, match="line 1.*integers"):
        list(load_jsonl(corpus))


def test_non_string_prompt_raises(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"prompt": 42}])

    with pytest.raises(ValueError, match="line 1.*prompt.*string"):
        list(load_jsonl(corpus, tokenizer=lambda s: [1]))


def test_non_dict_tags_field_raises(tmp_path):
    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"token_ids": [1], "tags": ["not", "a", "dict"]}])

    with pytest.raises(ValueError, match="'tags' field must be a JSON object"):
        list(load_jsonl(corpus))


# ---- integration with the simulator --------------------------------------


def test_loaded_corpus_flows_through_the_simulator(tmp_path):
    """The whole point: loader output plugs directly into RadixCacheSimulator."""
    from prefixlens import RadixCacheSimulator

    corpus = tmp_path / "c.jsonl"
    _write_jsonl(
        corpus,
        [
            {"request_id": "r1", "token_ids": list(range(32)), "tenant": "acme"},
            {"request_id": "r2", "token_ids": list(range(32)), "tenant": "acme"},
        ],
    )

    sim = RadixCacheSimulator(block_size=16, capacity_blocks=100)
    for req in load_jsonl(corpus):
        sim.process(req)

    report = sim.report()
    assert report.total_requests == 2
    assert report.hit_rate == 0.5  # r1 misses both blocks, r2 hits both


def test_loader_is_streaming_not_load_all(tmp_path):
    """The loader must yield lazily — a 10M-line corpus can't be materialized."""
    import types

    corpus = tmp_path / "c.jsonl"
    _write_jsonl(corpus, [{"token_ids": [1]}])

    iterator = load_jsonl(corpus)

    # Not a list, not a tuple — a generator.
    assert isinstance(iterator, types.GeneratorType)
