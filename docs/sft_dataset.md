# Search Agent SFT Dataset — Milestone 3

## Scope and design

Data construction only; no training or model generation. The initial heuristic distribution
is 150 direct arithmetic, 400 clean search, 300 same-topic hard-negative search,
100 noisy search and 50 insufficient-evidence trajectories. Multi-search is omitted
(0%, within the <=10% limit) to prioritize single-search reliability. Two independent
30-row synthetic evaluation sets cover counterfactual grounding and OOD entities/templates.

## Sources and license

SQuAD 2.0 by Pranav Rajpurkar, Robin Jia and Percy Liang (2018), based on Wikipedia
and crowdworker question/answer annotations. Official source:
https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

The [official dataset website](https://rajpurkar.github.io/SQuAD-explorer/) distributes
the dataset under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Our derived JSONL data is distributed under the same license; credit the SQuAD authors
and Wikipedia contributors. Original article titles, paragraph IDs, QA IDs, answer spans,
URL and file SHA256 are retained for attribution and reproducibility. Modifications:
subsampling, title-prefixed questions, query transformation, evidence shuffling and
composition, protocol messages and local data splits. Synthetic arithmetic/registry
fixtures are also released under CC BY-SA 4.0. This data notice does not relicense code.
The 40 MB original source is cached outside the repository and is not committed.

Search-R1's `scripts/nq_hotpotqa/data_process.sh` uses NQ and HotpotQA for training,
and NQ/TriviaQA/PopQA/HotpotQA/2WikiMultihopQA/MuSiQue/Bamboogle for evaluation.
`qa_search_train_merge.py` reads FlashRAG `golden_answers` into
`reward_model.ground_truth.target`. These pipelines lack the per-document support
annotation needed here; SQuAD supplies actual source paragraphs and answer offsets.

Inspection of `search_r1/llm_agent/generation.py` and `verl/utils/reward_score/qa_em.py`
found search/answer actions and answer matching, but no dedicated insufficient-evidence
policy. `<answer>Insufficient information.</answer>` is an explicit project extension,
not claimed to be upstream behavior. It is used only for SQuAD's native `is_impossible`
items with their original paragraph and empty gold-answer list.

## Development gates

The ten-row gate covers two direct, two clean, two hard negative, one noisy, one
insufficient and two counterfactual records. It passes protocol/schema validation,
duplicate checks and statistics; 21 unit tests pass including all previous milestones.
The implementation now constructs only the requested tiny rows before scaling.
The subsequent gates expanded to 100 then 1000 core trajectories, with validation
and a Git checkpoint between stages.

The 100-row gate also passed: train/val/test = 80/10/10, 16 direct,
40 clean, 30 hard-negative, 10 noisy and 4 insufficient (per-split integer rounding).
Hard-negative support positions = 10/10/10. Duplicate IDs/questions and cross-split
question/source-group leakage are all zero. Generator, validator and statistics were
executed before expanding to 1000 rows.

## Canonical schema

Required top-level fields: `id`, `question`, `task_type`, `messages`, `answer`, `metadata`.
Messages are either user → assistant answer, or user → assistant search → tool information
→ assistant answer. No hidden reasoning is generated or saved. Assistant contents must
be a single fully closed protocol action with no prose/fences outside tags.

```json
{
  "id": "arithmetic-46",
  "question": "What is 57 minus 3?",
  "task_type": "direct_answer",
  "messages": [
    {"role": "user", "content": "What is 57 minus 3?"},
    {"role": "assistant", "content": "<answer>54</answer>"}
  ],
  "answer": "54",
  "metadata": {
    "requires_search": false,
    "search_count": 0,
    "correct_doc_position": null,
    "has_hard_negative": false,
    "source": "synthetic_arithmetic",
    "source_item": "arithmetic-46",
    "source_group": "arithmetic-46",
    "source_question": "What is 57 minus 3?",
    "license": "CC-BY-SA-4.0",
    "documents": [],
    "seed": 42,
    "counterfactual": false
  }
}
```

Search metadata additionally contains documents (`id`, `title`, `text`, `is_support`).
SQuAD rows preserve `source_question`, `source_paragraph`, `original_answers` with offsets,
`source_is_impossible` and `source_url`. Correct position is one-based; absent for direct
and insufficient rows. These labels and gold annotations never enter exported messages.

## Transformation and query quality

Filter contexts longer than 2200 characters, source answers longer than 8 words/80
characters, unsupported answer offsets, protocol-delimiter conflicts, questions containing
explicit anaphoric words, and questions containing their own gold answer. Use at most one
question per original paragraph. Keep the original context (not a synthesized answer sentence).
Keep the annotated short answer; do not generate explanatory completions.

When the original question already names the article topic, retain it unchanged (215/850).
Otherwise add an article-topic prefix to clarify the context. Display titles are URL-decoded;
raw source identities are preserved. Rule-based queries retain title/entity anchors and
question keywords after removing interrogative/auxiliary words. No LLM is used. Exact
query copy rate is tested against both the displayed and original source question, not
just the prefixed version. The query transformation is deterministic and sometimes stilted.

## Hard negatives and noisy retrieval

For 300 hard-negative samples, choose two other paragraphs from the same Wikipedia article,
ranked by question-token overlap. They refer to the same topic/entity but do not contain
the normalized gold-answer span. Keep the full annotated paragraph as the support document.
This is inexpensive lexical hard-negative mining, not proof of semantic contradiction.

For 100 noisy samples, include the full support paragraph, a partial sentence from it
without the answer, and a different same-topic paragraph with competing details/entities.
Negative order is seeded, and the correct document uses a shuffled balanced schedule.
The support position is not shown to the model. No artificial `[CORRECT]` markers appear.

Clean searches and insufficient examples each have one document; thus document count
alone cannot distinguish whether an answer is present. The native SQuAD unanswerable
paragraph is never replaced by unrelated text or by a claim that explicitly says no answer.

## Split strategy and leakage prevention

Union original article groups, source QA IDs and normalized exact questions before a
seeded article-level split. Sample each split to reach 800 train, 100 val and 100 test
trajectories. All same-article distractor paragraphs stay within the source article's split.
The validator additionally checks cross-split source document IDs, including partials.
Question variants sharing source items/groups remain together; a unit test verifies it.
Normalization uses Unicode NFKC, case folding, punctuation removal and collapsed words.

Counterfactual and OOD namespaces are independent from the main set. Reusing fixed
Shakespeare/Rowling background distractors between those two sets is intentional, not
a new question/source-item overlap. These names are not treated as unseen entities.
The four Milestone 2B questions have zero normalized original/displayed-question overlap
with the new splits, and are saved separately in `data/eval/milestone2b_baseline.jsonl`.
The original `artifacts/milestone2b/` files are unchanged.

## Counts and statistics (seed 42)

| Split | Count | Parameter updates |
|---|---:|---|
| train | 800 | Eligible, future milestone only |
| val | 100 | Never; model selection only |
| test | 100 | Never |
| test_ood | 30 | Never |
| counterfactual | 30 | Never |

Main-set distribution: direct 150 (15%), clean 400 (40%), hard negative 300 (30%),
noisy 100 (10%), insufficient 50 (5%). These are initial heuristic proportions, not
empirically optimal ratios. Main set is 85% real SQuAD questions and 15% synthetic arithmetic.
Including the two special sets, source counts are 850 SQuAD / 150 arithmetic / 60 registry.

Requires search 85%; mean searches 0.85; query copy 0%; multi-search 0%; hard negatives
30%; mean documents per information turn 1.9412. Mean question length 11.372 words;
mean answer length 2.252 words. IDs, normalized duplicate questions and split/group leakage
are all zero across the 1060 canonical rows.

| Supported-document positions | Doc 1 | Doc 2 | Doc 3 |
|---|---:|---:|---:|
| Main hard negatives | 100 | 100 | 100 |
| Main noisy | 32 | 33 | 35 |
| All main multi-document rows | 132 | 133 | 135 |
| Main clean (one document only) | 400 | 0 | 0 |
| Counterfactual | 10 | 10 | 10 |

OOD has five documents and six correct answers at each position. All main supported
rows together are 532/133/135; that aggregate is skewed by single-document clean rows,
not a fixed-first-document policy in multi-document tasks. Full per-split statistics
are committed in `data/sft/statistics.json`.

## Counterfactual and OOD generation

Counterfactual: 30 invented `FABLE-42-NNN` play IDs mapped to fictional author names,
with Shakespeare/Hamlet and Rowling/Harry Potter as competing documents. For example,
FABLE-42-018 → Liora Sen-018; FABLE-42-003 → Nira Voss-003. These are fixture facts,
not claims about real works or people. Questions, author mappings and document order
are deterministic under the seed. Ten examples place the support at Doc 2, reproducing
the position arrangement that exposed the Milestone 2B error.

OOD: 30 `ORBIT-42-NNN` fictional expeditions, a new registry/investigator question
template and five documents instead of three. Distractors include a near-matching
expedition ID and partial entity information. It is a small distribution-shift seed,
not a validated OOD benchmark or a claim that all possible names are novel to pretraining.

## MLX data export (no training)

Inspected local MLX-LM 0.31.3 `mlx_lm/tuner/datasets.py`: `ChatDataset` consumes
`{"messages": [...]}`, and local filenames are `train.jsonl`, `valid.jsonl`, `test.jsonl`.
The adapter maps canonical tool observations to user messages, matching Milestone 2B's
Qwen text-protocol interaction. It adds a concise system instruction, keeps message
contents/order intact and excludes support labels and all source/gold metadata.

Export emits one prefix ending in each assistant target, so search decisions are not
lost when future prompt masking supervises only the last assistant message. 800 train
trajectories become 1480 targets; 100 validation trajectories become 185 targets.
It never exports test/OOD/counterfactual rows into training inputs. `canonical_id` and
`target_message_index` provide alignment, not model text.

All 1665 exported prefixes were compared with canonical contents and processed by the
cached Qwen3 tokenizer at revision `73e3e38d981303bc594367cd910ea6eb48349da8`.
Only tokenization/data conversion was performed. Future training must explicitly mask
the context; using an unmasked objective would supervise user questions/observations too.
Future generation should preserve the reviewed non-thinking configuration. Canonical
labels have no CoT; any empty thinking wrappers originate in the official template.
Exports were verified in `/tmp`, not committed as duplicated training artifacts.

Final tokenizer audit: all 1665 context prefixes match their full tokenized conversations
at the masking boundary. Maximum exported length is 1481 tokens, mean 288.96 under the
cached template. No truncation was applied. Final regression suite: 23 tests passed;
final canonical validator: all 1060 records passed.

## Reproduction

Original source SHA256:
`68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002`.
The generator checks this reviewed release hash and rejects a different input.
The source was acquired from the official URL on 2026-09-11. No new dependencies are
required for generation/validation/statistics/export; Python standard library suffices.
Optional tokenizer verification used the existing offline environment.

```bash
# Download once outside the repository if the source is not already cached.
curl --fail --location -o /tmp/squad-train-v2.0.json \
  https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json

# Use fresh directories; the builder refuses to overwrite prior outputs.
python3 scripts/build_sft_dataset.py --source /tmp/squad-train-v2.0.json --size 10 --seed 42 --output /tmp/sft-tiny
python3 scripts/validate_sft_dataset.py /tmp/sft-tiny
python3 -m unittest discover -s tests -v
python3 scripts/analyze_sft_dataset.py /tmp/sft-tiny
python3 scripts/build_sft_dataset.py --source /tmp/squad-train-v2.0.json --size 100 --seed 42 --output /tmp/sft-100
python3 scripts/validate_sft_dataset.py /tmp/sft-100
python3 scripts/build_sft_dataset.py --source /tmp/squad-train-v2.0.json --size 1000 --seed 42 --output /tmp/sft-1000
python3 scripts/validate_sft_dataset.py /tmp/sft-1000
python3 scripts/analyze_sft_dataset.py /tmp/sft-1000
python3 scripts/export_mlx_sft.py /tmp/sft-1000 /tmp/sft-mlx
```

Rebuilding the final five canonical JSONL files with the same source/seed produced
byte-identical contents. The source group split and counterfactual determinism also
have small unit tests. Content spot checks are recorded in `sft_inspection.md`.

## Five design questions

**Q1 — Observed failures addressed:** strictly closed search/answer targets address
missing tags and plain prose; short source-span answers address verbose exact-match
failures; evidence discrimination addresses selecting Shakespeare despite contrary evidence.

**Q2 — Hard negatives:** retrieve same-article, different-paragraph candidates using
lexical overlap, reject candidates containing the answer, and keep original support spans.
Synthetic evaluation distractors deliberately contain famous competing author names.

**Q3 — Avoid always choosing Doc 1:** shuffle a balanced position schedule; hard-negative
training support positions are 80/80/80, and held-out hard negatives are also balanced.
Report positions separately for one-document and multi-document tasks.

**Q4 — Avoid searching everything:** 15% direct-answer targets have no tool turn;
future evaluation must measure both unnecessary search and missed search. Arithmetic-only
direct supervision is a limited first approximation, not a general search-necessity oracle.

**Q5 — Verify information use:** reserve fictional entity→answer tests with famous
distractors for future with/without-observation evaluation. Evaluate correct entity binding
and optionally swap evidence while holding the question fixed. This milestone validates
the fixtures; it does not claim improved model grounding before training/evaluation.

## Limitations and result

This is a gold-annotation-to-trajectory transformation, not expert model demonstrations
or recorded retrieval from a production index. Most search decisions are heuristic:
some real-world facts might be answerable from model memory. Direct examples are only
arithmetic and split-specific numeric ranges introduce a modest arithmetic distribution
shift. Many QA questions still rely on article context; title hints may become shortcuts.
No semantic deduplication or alias-aware negative verification is claimed. Same-topic
distractors are not always genuinely difficult; negative answer exclusion is lexical.
Original passages can contain multiple acceptable answers, old facts and Wikipedia noise.
Only ten final samples were manually read; independent human review is still appropriate.

No training, LoRA, RL, model-weight changes or Milestone 2B log edits occurred.
Milestone 3 dataset-construction result: **PASS**, subject to the documented spot-check
and semantic-quality limitations, ready for the user's dataset-design review.
