# Reports

Saved output from `prefixlens analyze` on real corpora, so results are reproducible without re-running the whole pipeline. Each run has two files:

- `*.txt` — human-readable summary (what `prefixlens analyze` prints).
- `*.json` — full machine-readable report (what `--json` emits). Same content, everything included (no cardinality filtering).

## Real-workload comparison at a glance

| Corpus | Shape | Overall | Requests | Blocks | Runtime | Divergence pattern |
|---|---|---|---|---|---|---|
| **WildChat-1M** (2K conv) | ad-hoc chat | 70.2% | 5,832 | 404K | 3.05s | block 0 unique (~97%) |
| **LMSys-Chat-1M** (2K conv) | ad-hoc chat | 81.7% | 3,817 | 178K | 1.3s | block 0 unique (~97%) |
| **MMLU 5-shot** (57 subj × 50 q) | batch eval | 84.0% | 2,850 | 105K | 0.8s | subject-specific block (15–44) unique |

Same tool, same flags. **Three different workload shapes; three different diagnoses.** Both chat corpora produce block-0-unique patterns (users share no common prefix, only within-conversation history reuses). MMLU produces a completely different shape: dense shared preamble hits blocks 0..K−1 across every query in a subject, divergence lands at exactly block K where the per-question stem starts.

Consistent-with-mechanism findings:
- Hit rate in chat is driven by **turns per conversation** (Pearson r = 0.846 for LMSys — see [interpretation below](#lmsys-chat-1m-sample-2k-conversations-3817-requests)), not raw request count.
- MMLU's per-subject hit rates cluster in a **4.4pp range** (84.4%–88.8%) — all subjects use the same 5-shot template so they behave identically. LMSys's per-model range was 62pp because that spread was really about turns/conv, not model.

## WildChat-1M sample (2K conversations, 5,832 requests)

`wildchat_2k.{txt,json}` — the raw WildChat corpus (produced by `examples/build_wildchat_scenario.py`), analyzed at block-size 16, capacity 4096.

**Headline**: **70.2% overall hit rate**, 3.05s runtime on 404K blocks.

Per-model breakdown:

| Model | Hit rate | Requests |
|---|---|---|
| gpt-3.5-turbo-0301 | 75.5% | 3,426 |
| gpt-4-0314 | 65.0% | 2,406 |

Divergent-position pattern: block 0 dominates for both models (~95–98% unique content). This is **inherent** to the WildChat workload — conversations are from many different users with no shared system prompt across users. Not a bug to fix; a workload shape to know.

## LMSys-Chat-1M sample (2K conversations, 3,817 requests)

`lmsys_2k.{txt,json}` — English-only slice of `lmsys/lmsys-chat-1m` (gated dataset; requires `hf auth login`).

**Headline**: **81.7% overall hit rate** across 25+ distinct models. 62pp spread from best to worst:

| Model | Hit rate | Requests | Notes |
|---|---|---|---|
| vicuna-13b | 86.7% | 1,985 | Historical Chatbot Arena default → long conversations → strong within-conversation reuse. |
| stablelm-tuned-alpha-7b | 86.3% | 66 | |
| wizardlm-13b | 76.4% | 77 | |
| alpaca-13b | 75.0% | 222 | |
| koala-13b | 72.8% | 380 | |
| claude-1 | 69.0% | 66 | |
| gpt-3.5-turbo | 66.2% | 41 | |
| vicuna-33b | 65.7% | 110 | |
| … | … | … | |
| gpt-4 | 40.8% | 35 | Small sample; few conversations to accumulate reuse against. |
| palm-2 | 32.1% | 32 | Same. |
| RWKV-4-Raven-14B | 24.4% | 52 | Same. |

**The 62pp spread is real but it's a workload artifact, not a model-quality signal.** Models with lots of requests (Vicuna-13b at 1,985) have long accumulated cache history to hit against; models with tiny samples (GPT-4 at 35, PaLM-2 at 32) are dominated by fresh-conversation starts. The tool correctly surfaces the pattern; the operator has to interpret it as *volume, not model choice*.

Divergent-position pattern: block 0 dominates for every model, same as WildChat — no shared system prompt across LMSys users. This is a load-bearing property of ad-hoc multi-user traffic that both real-workload runs demonstrate.

## MMLU 5-shot sample (57 subjects × 50 questions = 2,850 requests)

`mmlu_5shot.{txt,json}` — canonical 5-shot MMLU evaluation format (produced by `examples/build_mmlu_scenario.py` from `cais/mmlu`, public, no auth).

**Headline**: **84.0% overall hit rate**, 0.8s runtime on 105K blocks. Every subject in a tight **4.4pp band** (84.4%–88.8%).

Why this workload is different from chat:

| Property | Chat (WildChat/LMSys) | MMLU |
|---|---|---|
| Where does cache reuse come from? | within-conversation history | shared 5-shot preamble across every question in a subject |
| Where does divergence land? | block 0 (each conversation starts unique) | block K per subject, where K = the length of the preamble |
| Cross-request sharing? | none | massive — same preamble used by every subject-B question |

Sample divergent-position report:

```
divergent positions for subject=abstract_algebra:
  block  18     46 miss  (100.0% unique — unique content per request)
divergent positions for subject=college_computer_science:
  block  44     49 miss  (100.0% unique — unique content per request)
divergent positions for subject=conceptual_physics:
  block  15     49 miss  (100.0% unique — unique content per request)
```

For **every subject**, ~49 of 50 requests first-diverge at the same specific block (the point where the shared 5-shot preamble ends and the per-question stem begins). The remaining 1 miss is at block 0 — that's the *first* request of the subject, which had nothing cached yet.

This is the canonical prefix-cache-friendly shape: shared preamble reused across users, unique tail. Prefixlens correctly identifies it, and would identify a regression instantly if e.g. someone injected a timestamp into the shared preamble template.

## Injection scenarios (500 conversations, 1,323 requests each)

Self-validation harness: take a slice of the WildChat corpus, apply a **known transform**, run analyze, verify the report matches ground truth. Generated by `examples/build_injection_scenarios.py`.

| File | Transform | Hit rate | Δ vs baseline | Divergent position |
|---|---|---|---|---|
| `injection_baseline.{txt,json}` | unchanged corpus | 69.0% | — | block 0 (~97% unique) |
| `injection_shared_prompt.{txt,json}` | +32-token shared system prompt at position 0 | 69.8% | **+0.8pp** | block 2 (~97%) |
| `injection_uuid_front.{txt,json}` | per-conversation UUID BEFORE the shared prompt | 63.8% | **−5.2pp** | block 0 (100%) |
| `injection_uuid_middle.{txt,json}` | shared prompt + UUID between prompt and conversation | 64.8% | **−4.2pp** | block 2 (100%) |
| `injection_uuid_tail.{txt,json}` | shared prompt + UUID appended at the end | 69.0% | 0.0pp | block 2 (98%) |

**Every prediction each transform encodes is exactly what the report shows**, arithmetic to two decimal places. Notable observations:

- **`uuid_front` is 1pp worse than `uuid_middle`**, even though the same UUID field is added in both. The parent-chained hash property means position-0 divergence orphans everything downstream — that 1pp asymmetry is the tool detecting *where* the injection sits.
- **`shared_prompt` only adds +0.8pp** on WildChat because within-conversation history already dominates cache reuse; a 32-token shared prefix is small next to 60–70 tokens/turn of conversation growth. Realistic operator finding.
- **`uuid_tail` matches baseline exactly** — the shared-prompt gain and the tail-UUID loss cancel. Divergent position is correctly attributed to within-conversation start (block 2), not the tail UUID — the tool doesn't chase every unique block, it locates where misses actually cluster.

## How to regenerate

```bash
# Corpus: streams WildChat-1M via HuggingFace datasets, tokenizes with tiktoken.
python examples/build_wildchat_scenario.py --max-conversations 2000 --out examples/wildchat_scenario.jsonl

# Injection scenarios: derived from the base corpus.
python examples/build_injection_scenarios.py --base examples/wildchat_scenario.jsonl --max-conversations 500

# Reports:
prefixlens analyze examples/wildchat_scenario.jsonl --block-size 16 --capacity-blocks 4096 > reports/wildchat_2k.txt
prefixlens analyze examples/wildchat_scenario.jsonl --block-size 16 --capacity-blocks 4096 --json > reports/wildchat_2k.json
for s in baseline shared_prompt uuid_front uuid_middle uuid_tail; do
    prefixlens analyze examples/injection_$s.jsonl --block-size 16 --capacity-blocks 4096 > reports/injection_$s.txt
    prefixlens analyze examples/injection_$s.jsonl --block-size 16 --capacity-blocks 4096 --json > reports/injection_$s.json
done
```

The corpora themselves (`examples/wildchat_scenario.jsonl`, `examples/injection_*.jsonl`) are gitignored — they regenerate from the scripts in seconds once tokenized shards are cached locally.
