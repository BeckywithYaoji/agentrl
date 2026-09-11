# License and split usage

Derived SQuAD 2.0 data: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Attribution: Pranav Rajpurkar, Robin Jia, Percy Liang, SQuAD annotators and Wikipedia contributors.
Original dataset: https://rajpurkar.github.io/SQuAD-explorer/ (2018).
Synthetic fixtures use the same license. This notice applies to data, not code.

Changes: filtered sampling, optional article context in questions, deterministic queries,
evidence composition/reordering, local splits and protocol formatting. Original question,
article title/paragraph IDs, QA IDs, answer spans, source URL and license are in metadata;
the original input SHA256 is in statistics.json. Original Wikipedia articles can be found
by the recorded article titles. These files retain the source's historical context.

Only train.jsonl is eligible for future parameter updates. val.jsonl is for model selection;
test.jsonl, test_ood.jsonl and counterfactual.jsonl are evaluation only. Local splits are
derived from SQuAD's public training release, not official SQuAD evaluation splits.
See ../../docs/sft_dataset.md for full design, provenance and reproduction instructions.
