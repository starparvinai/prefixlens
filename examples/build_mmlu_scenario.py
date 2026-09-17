"""Adapter: turn MMLU 5-shot evaluation into a prefixlens JSONL corpus.

MMLU (cais/mmlu) is the canonical multi-task benchmark for LLM evaluation.
Standard config is 5-shot: for each of 57 subjects, the `dev` split
provides 5 canonical few-shot examples, and the `test` split provides
the actual questions to answer.

The important shape property for prefix caching: within any subject,
every test question is prefixed by the SAME 5 few-shot examples plus
a subject-specific instruction. That shared preamble is a giant
identical prefix across every request tagged with that subject —
exactly what the prefix cache is designed to exploit.

This is the *opposite* shape from WildChat/LMSys chat corpora, where
different users share no common prefix and cache reuse is
within-conversation only.

Prediction (worth writing before we run):
    - overall hit rate: 85-95% (shared preamble dominates)
    - divergent position: a single dominant block position PER SUBJECT
      where the per-question stem begins (not block 0 like chat)
    - unique_content_ratio at that position: ~1.0 (each question unique)
    - hit rates roughly equal across all subjects (same template shape)

Usage:
    python examples/build_mmlu_scenario.py \\
        --questions-per-subject 50 \\
        --out examples/mmlu_scenario.jsonl
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


# Canonical MMLU 5-shot template. The subject-line placeholder is filled
# in per-subject (with underscores replaced by spaces). The five example
# blocks come from the `dev` split (which has exactly 5 canonical examples
# per subject).
INSTRUCTION_TEMPLATE = (
    "The following are multiple choice questions (with answers) about {subject}.\n\n"
)

CHOICE_LETTERS = ("A", "B", "C", "D")


def _format_example(question: str, choices: list[str], answer_idx: int | None) -> str:
    """Render one MMLU question. `answer_idx` None means the question is
    the query (no answer supplied — the model is meant to complete it).
    """
    lines = [question]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        lines.append(f"{letter}. {choice}")
    if answer_idx is None:
        lines.append("Answer:")
    else:
        lines.append(f"Answer: {CHOICE_LETTERS[answer_idx]}")
    return "\n".join(lines) + "\n\n"


def _build_preamble(subject: str, dev_examples: list[dict]) -> str:
    """Instruction + 5 few-shot examples. This is the shared prefix for
    every test question in the subject.
    """
    subject_pretty = subject.replace("_", " ")
    parts = [INSTRUCTION_TEMPLATE.format(subject=subject_pretty)]
    for ex in dev_examples[:5]:
        parts.append(_format_example(ex["question"], ex["choices"], ex["answer"]))
    return "".join(parts)


def build(
    subjects: list[str] | None,
    questions_per_subject: int,
    out_path: Path,
) -> tuple[int, int, float]:
    """Build the corpus. Returns (subjects_used, requests_emitted, elapsed_seconds)."""
    enc = tiktoken.get_encoding("cl100k_base")

    # Discover subjects from the `all` config. MMLU has one config per subject
    # (e.g. `abstract_algebra`, `philosophy`) — 57 total. We use the aggregated
    # `all` config's `dev` + `test` splits so we can iterate all subjects in
    # a single load.
    print("Loading MMLU dev + test splits...", file=sys.stderr)
    dev_ds = load_dataset("cais/mmlu", "all", split="dev")
    test_ds = load_dataset("cais/mmlu", "all", split="test")

    # Group dev examples by subject. Each subject has exactly 5 in the dev split.
    dev_by_subject: dict[str, list[dict]] = {}
    for ex in dev_ds:
        dev_by_subject.setdefault(ex["subject"], []).append(ex)

    all_subjects = sorted(dev_by_subject.keys())
    if subjects:
        # Filter to user-requested subjects
        target_subjects = [s for s in all_subjects if s in set(subjects)]
    else:
        target_subjects = all_subjects
    print(f"  {len(target_subjects)} subjects, {len(test_ds):,} total test questions available", file=sys.stderr)

    # Group test questions by subject and iterate
    test_by_subject: dict[str, list[dict]] = {}
    for ex in test_ds:
        subj = ex["subject"]
        if subj in set(target_subjects):
            test_by_subject.setdefault(subj, []).append(ex)

    t0 = time.perf_counter()
    n_subjects = 0
    n_requests = 0

    with out_path.open("w", encoding="utf-8") as fout:
        for subject in target_subjects:
            dev = dev_by_subject.get(subject, [])
            if len(dev) < 5:
                print(f"  skipping {subject}: only {len(dev)} dev examples (need 5)", file=sys.stderr)
                continue
            preamble = _build_preamble(subject, dev)

            test_questions = test_by_subject.get(subject, [])[:questions_per_subject]
            for i, q in enumerate(test_questions):
                query = _format_example(q["question"], q["choices"], answer_idx=None)
                full_prompt = preamble + query
                token_ids = enc.encode(full_prompt)
                if not token_ids:
                    continue

                # `subject` is a custom tag axis, so it goes under `tags`
                # (the loader only auto-recognizes tenant/route/model at
                # the top level). Also stash the answer index for future
                # per-request debugging — not used by analyze but harmless.
                record = {
                    "request_id": f"{subject}_q{i:03d}",
                    "token_ids": token_ids,
                    "tags": {
                        "subject": subject,
                    },
                }
                fout.write(json.dumps(record) + "\n")
                n_requests += 1

            n_subjects += 1
            if n_subjects % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  … {n_subjects}/{len(target_subjects)} subjects, "
                    f"{n_requests:,} requests ({elapsed:.1f}s)",
                    file=sys.stderr,
                )

    elapsed = time.perf_counter() - t0
    print(
        f"done: {n_subjects} subjects, {n_requests:,} requests emitted in {elapsed:.1f}s.",
        file=sys.stderr,
    )
    return n_subjects, n_requests, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/mmlu_scenario.jsonl"),
    )
    parser.add_argument(
        "--questions-per-subject",
        type=int,
        default=50,
        help="Max test questions to emit per subject (default: 50). "
        "MMLU test split has 100+ questions per subject; capping keeps the "
        "corpus size manageable for demo analyze runs.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Restrict to a subset of subjects (default: all 57).",
    )
    args = parser.parse_args()

    build(
        subjects=args.subjects,
        questions_per_subject=args.questions_per_subject,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
