# Milestone 2B — Smoke Evaluation

Base: `d3a4e02`, branch: `master`. Result: **PASS** for the engineering loop; this is not a benchmark or evidence of reliable general tool use.

## Environment and architecture

Python 3.14.6, MLX 0.32.2, MLX-LM 0.31.3, Transformers 5.17.0.
Model: `mlx-community/Qwen3-0.6B-4bit`, revision `73e3e38d981303bc594367cd910ea6eb48349da8`.
All real runs used `HF_HUB_OFFLINE=1` and existing cached weights.

Question → Qwen3 → search action → existing Mock Retriever → information → Qwen3 → answer.

`QwenAgent` loads once, applies the cached tokenizer's official chat template with
`enable_thinking=False`, and streams up to 256 tokens with greedy sampling (`temp=0`).
The template supplies an empty thinking block; output is not stripped with regex.
Generation stops on EOS, token limit, or an emitted closing search/answer tag. Every
emitted text chunk is retained. Evidence is logged as role `tool` and passed into the
next chat turn as role `user`, wrapped in `<information>`; this is text protocol,
not Qwen native JSON tool calling. Answers are always model generated.

The parser follows `third_party/search-r1/search_r1/llm_agent/generation.py`:
first complete `<(search|answer)>...</...>` action, permitting preceding text.
Query extraction and document formatting reuse `scripts/minimal_search_demo.py`.
Three tool calls are permitted, followed by one final model opportunity to answer;
a further search terminates as `max_search_steps`. Invalid output terminates immediately.
Trajectory `success` means valid answer termination, not factual correctness.

## Zero-shot prompt

Adapted from `third_party/search-r1/infer.py`: retain search/information/answer
instructions, remove mandatory long thinking and unlimited searches, add one action per turn.

```text
Answer the given question. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Output one search or answer action per turn.
```

## Results

| Zero-shot question | Search | Protocol | Exact match | Calls | Termination |
|---|---|---|---|---:|---|
| XQZ-17 author | Yes | Yes | No | 1 | answer |
| Hamlet author | Yes | No | No | 1 | invalid_output |
| Eiffel Tower location | No | No | No | 0 | invalid_output |
| Relativity developer | No | No | No | 0 | invalid_output |

| Smoke metric | Zero-shot | 1-shot | 2-shot |
|---|---:|---:|---:|
| Search Trigger Rate | 50% | 75% | 75% |
| Protocol Success Rate | 25% | 75% | 75% |
| Answer Accuracy | 0% | 50% | 50% |
| Average Search Count | 0.5 | 0.75 | 0.75 |
| Max-step Failures | 0 | 0 | 0 |

Accuracy uses stripped, case-insensitive exact match of parsed final answer against
the reference name. XQZ-17 zero-shot contains the correct name in a sentence and thus
fails this deliberately strict metric but passes the separately defined counterfactual
criterion (answer contains `Alice Example`). Plain text answers without tags are not
parsed into final answers, even if factually correct.

1-shot demonstrates fictional Zeta-42 → Robin Sample. 2-shot adds Lumen-9 → Morgan Demo.
Both use user question → assistant search → user information → assistant answer.
Neither demonstration contains evaluation questions or their answers. The exact
examples are in `scripts/qwen_search_demo.py:examples`; zero-shot is preserved.

In both few-shot runs: Hamlet and relativity terminate correctly with exact reference
names; Eiffel Tower produces untagged `The Eiffel Tower is located in Paris, France.`;
XQZ-17 searches but answers `William Shakespeare` despite the correct first document.

## Counterfactual and complete real trajectory

Question: `Who wrote the fictional play XQZ-17?`

Without Search (raw output):

```text
The fictional play *XQZ-17* is not a real play in the real world. It is a fictional creation by the author **J.K. Rowling**. She wrote the story and the play in the form of *XQZ-17* as part of her *Harry Potter* series. The play is part of the *Harry Potter* universe and is set in the 1980s.
```

Zero-shot model output 1:

```text
<search> Who wrote the fictional play XQZ-17? </search>
```

Search query: `Who wrote the fictional play XQZ-17?`

Actual observation:

```text
<information>Doc 1(Title: XQZ-17) According to the fictional test knowledge base, the play XQZ-17 was written by Alice Example.
Doc 2(Title: William Shakespeare) William Shakespeare wrote Hamlet.
Doc 3(Title: Eiffel Tower) The Eiffel Tower is located in Paris.</information>
```

Zero-shot model output 2:

```text
<answer> Alice Example wrote the fictional play XQZ-17. </answer>
```

Final answer: `Alice Example wrote the fictional play XQZ-17.` Termination: `answer`.
Counterfactual: zero-shot **PASS**, 1-shot **FAIL**, 2-shot **FAIL**.
This one comparison supports observation use in this trajectory; it does not establish
robust grounding or isolate all prompt effects.

## Failure cases and main finding

Zero-shot Hamlet ends with raw `<answer>William Shakespeare wrote *Hamlet*.` (no closing tag).
Eiffel Tower emits `The Eiffel Tower is located in Chicago, United States.` without tags.
Relativity emits `The theory of relativity was developed by Albert Einstein in 1905.` without tags.

The primary observed weakness is inconsistent protocol adherence. Few-shot improves
formatting on these four questions but does not solve evidence selection: XQZ-17 picks
Shakespeare from a distractor. The inherited mock ranks substring overlaps and returns
three documents even when some are irrelevant; this is a limitation of the current
fixture, not a dense-retrieval experiment. No further prompt tuning was performed.

## Reproduction and artifacts

```bash
python3 -m unittest discover -s tests -v
HF_HUB_OFFLINE=1 .venv/bin/python scripts/qwen_search_demo.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/qwen_search_demo.py --shots 1
HF_HUB_OFFLINE=1 .venv/bin/python scripts/qwen_search_demo.py --shots 2
```

For repeats, add `--output-dir /tmp/agentrl-new-run` with a fresh directory. Existing
experiment files are never overwritten. Small raw outputs are explicitly committed:
`artifacts/milestone2b/zero_shot.jsonl`, `few_shot_1.jsonl`, `few_shot_2.jsonl`,
and `without_search.json`. No model weights are tracked.

Nine standard-library unit tests pass, including the two Milestone 1 tests; they load
no model. Shared loader smoke was also rerun offline successfully. All eleven
Milestone 2B acceptance criteria are satisfied; no SFT or RL functionality was added.
