"""JSONL corpus loader.

Reads a stream of prompt records from a JSONL file and yields Request objects.
No tokenizer library is bundled — the caller passes any callable that maps
str -> sequence of ints. Records that ship already-tokenized (`token_ids`
field present) bypass the tokenizer entirely.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator, Sequence

from prefixlens.request import Request


Tokenizer = Callable[[str], Sequence[int]]

# Top-level fields in the JSONL schema that become tags on the Request.
# Nested `tags: {...}` entries are merged in on top of these.
_TOP_LEVEL_TAG_FIELDS = ("tenant", "route", "model")


def _normalize_tags(record: dict) -> tuple[tuple[str, str], ...]:
    """Flatten top-level tag fields and any nested `tags` dict into sorted (k,v)
    pairs. Nested entries override top-level ones on key collision.
    """
    pairs: dict[str, str] = {}
    for k in _TOP_LEVEL_TAG_FIELDS:
        v = record.get(k)
        if v is not None:
            pairs[k] = str(v)
    nested = record.get("tags")
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError(
                f"'tags' field must be a JSON object, got {type(nested).__name__}"
            )
        for k, v in nested.items():
            pairs[str(k)] = str(v)
    return tuple(sorted(pairs.items()))


def _resolve_token_ids(
    record: dict,
    tokenizer: Tokenizer | None,
    lineno: int,
) -> tuple[int, ...]:
    """Return token_ids for a record, using the pre-tokenized fast path when
    available and falling back to the text path only if a tokenizer is given.
    """
    if "token_ids" in record:
        raw = record["token_ids"]
        if not isinstance(raw, list) or not all(isinstance(t, int) for t in raw):
            raise ValueError(
                f"line {lineno}: 'token_ids' must be a JSON array of integers"
            )
        return tuple(raw)

    if "prompt" in record:
        if tokenizer is None:
            raise ValueError(
                f"line {lineno}: record has 'prompt' but no tokenizer was provided; "
                "pass a tokenizer to load_jsonl or pre-tokenize the corpus into "
                "'token_ids' fields"
            )
        prompt = record["prompt"]
        if not isinstance(prompt, str):
            raise ValueError(
                f"line {lineno}: 'prompt' must be a string, got {type(prompt).__name__}"
            )
        return tuple(tokenizer(prompt))

    raise ValueError(
        f"line {lineno}: record must have either 'token_ids' or 'prompt'"
    )


def load_jsonl(
    path: str | Path,
    tokenizer: Tokenizer | None = None,
) -> Iterator[Request]:
    """Yield Request objects from a JSONL corpus, one per non-blank line.

    Schema per line (see SPEC §5): at minimum, either `token_ids` (list[int])
    or `prompt` (str) must be present. `token_ids` wins if both are given —
    the pre-tokenized fast path is preferred because the caller already knows
    exactly what the engine saw. Optional fields: `request_id`, `tenant`,
    `route`, `model`, `tags` (a nested object of extra key/value pairs).
    All tag-shaped fields are flattened into `Request.tags`.

    Blank lines are skipped. Missing `request_id` defaults to `"line-N"`.

    Raises ValueError with the line number for any malformed record. This is
    a developer-facing tool; failing loud on the first bad line is preferable
    to silently skipping and producing a wrong report.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {lineno}: invalid JSON ({e.msg})") from e
            if not isinstance(record, dict):
                raise ValueError(
                    f"line {lineno}: expected a JSON object, got "
                    f"{type(record).__name__}"
                )

            token_ids = _resolve_token_ids(record, tokenizer, lineno)
            request_id = str(record.get("request_id", f"line-{lineno}"))
            tags = _normalize_tags(record)

            yield Request(
                request_id=request_id,
                token_ids=token_ids,
                tags=tags,
            )
