# Milestone 3 stratified content inspection

Method: deterministic `random.Random(42)`, sampled in order from the combined train,
val and test lists: 2 direct, 2 clean, 2 hard negative; then 2 from counterfactual;
then 1 noisy and 1 insufficient. All ten questions, queries, full documents and final
answers were printed and read by the assistant. This is a manual content spot check,
not a claim of independent human annotation or user approval. The user can review
the complete committed canonical records by the IDs below.

| ID | Type | Answer / support | Inspection |
|---|---|---|---|
| arithmetic-46 | Direct | 57 minus 3 = 54 | Correct arithmetic; no search. |
| arithmetic-88 | Direct | 99 minus 9 = 90 | Correct arithmetic; no search. |
| squad-57301c9bb2c2fd140056889d | Clean | May 1, 2011 | Windows 8 paragraph explicitly distinguishes Build 7959 from other builds/dates. |
| squad-57277618dd62a815002e9d6f | Clean | Venetian | Carnival paragraph says established under Venetian rule around the 16th century. |
| squad-57328842b9988014000c7666 | Hard negative | Ezra Pound, Doc 3 | Doc 3 lists early critics; other same-topic documents describe history/ethics. There are other acceptable names in the support paragraph; the target retains the annotated one. |
| squad-5727d47eff5b5019007d9650 | Hard negative | Between 1346 and 1354, Doc 3 | Correct dynasty founding period; other docs discuss different dynasties/centuries. |
| FABLE-42-018 | Counterfactual | Liora Sen-018, Doc 1 | Exact fictional play ID maps to this author, not the Shakespeare/Rowling distractors. |
| FABLE-42-003 | Counterfactual | Nira Voss-003, Doc 1 | Same check; both randomly inspected rows happen to have position 1, but full set positions are 10/10/10. |
| squad-5725c08389a1e219009abde0 | Noisy | 784, Doc 2 | Support identifies first recorded theudisk text; partial sentence has no date, unrelated same-topic paragraph has competing dates. |
| squad-5a3997b52f14dd001ac7244b | Insufficient | Insufficient information. | Paragraph places Anglophones in Gustavia but does not identify the side of the island. Other language groups' locations cannot answer this question; original is_impossible=true. |

Queries retain the topic/entity and requested relation; all have closing tags and no
extra assistant prose. Evidence labels and position metadata are not exported into
model messages. No reasoning traces are present.

Findings acted on: percent-encoded article title `Saint Barth%C3%A9lemy` was decoded
to `Saint Barthélemy` in question/query/document display while raw IDs were preserved.
Questions already naming the article topic now retain original wording (215 of 850
source questions), instead of uniformly receiving a context prefix. The final rows
were regenerated and revalidated; factual content, targets and positions remain unchanged.

Limitations: this 10-row check is not exhaustive. Exact exclusion of the target string
does not prove a distractor cannot imply a synonym/alternative correct answer. The
humanism example demonstrates that annotated exact match is narrower than semantic
correctness. A full human adjudication has not been performed.
