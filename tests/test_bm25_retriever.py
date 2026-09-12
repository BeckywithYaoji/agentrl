"""Synthetic, CPU-only checks for query-sensitive retrieval."""
import json

import pytest

from agentrl.retrieval import TinyBM25Retriever, load_training_corpus


@pytest.fixture
def documents():
    return [
        {'title': 'Hamlet', 'text': 'Hamlet was written by William Shakespeare.'},
        {'title': 'Gravity', 'text': 'Isaac Newton studied gravity and motion.'},
        {'title': 'Plays', 'text': 'Shakespeare wrote plays including Hamlet.'},
    ]


def contents(response, index=0):
    return [item['document']['contents'] for item in response['result'][index]]


def test_query_changes_ranking(documents):
    retriever = TinyBM25Retriever(documents)
    assert contents(retriever.retrieve(['Hamlet author'], topk=1))[0].startswith('Hamlet\n')
    assert contents(retriever.retrieve(['gravity Newton'], topk=1))[0].startswith('Gravity\n')


def test_garbage_query_returns_empty(documents):
    assert TinyBM25Retriever(documents).retrieve(['asdfghjkl']) == {'result': [[]]}


def test_schema(documents):
    assert TinyBM25Retriever(documents).retrieve(['Newton'], topk=1) == {
        'result': [[{'document': {'contents': 'Gravity\nIsaac Newton studied gravity and motion.'}}]]
    }


def test_batch_queries_are_independent(documents):
    retriever = TinyBM25Retriever(documents)
    queries = ['Hamlet', 'asdfghjkl', 'Newton', 'Hamlet']
    assert retriever.retrieve(queries, topk=1)['result'] == [
        retriever.retrieve([q], topk=1)['result'][0] for q in queries
    ]


@pytest.mark.parametrize('topk, count', [(1, 1), (2, 2), (9, 2)])
def test_topk_and_positive_matches_only(documents, topk, count):
    assert len(contents(TinyBM25Retriever(documents).retrieve(['Hamlet'], topk))) == count


def test_deterministic(documents):
    expected = TinyBM25Retriever(documents).retrieve(['Hamlet Newton'])
    retriever = TinyBM25Retriever(documents)
    for _ in range(5):
        assert retriever.retrieve(['Hamlet Newton']) == expected


def test_stable_tie_order():
    docs = [{'title': title, 'text': 'shared'} for title in ['Z', 'A', 'M']]
    assert contents(TinyBM25Retriever(docs).retrieve(['shared'])) == ['Z\nshared', 'A\nshared', 'M\nshared']


def test_loader_deduplicates_by_title_and_text(tmp_path):
    a = {'title': 'A', 'text': 'shared', 'id': '1', 'is_support': False}
    b = {'title': 'B', 'text': 'shared', 'id': '1', 'is_support': True}
    rows = [{'metadata': {'documents': [a, b]}}, {'metadata': {'documents': []}},
            {'metadata': {'documents': [dict(a, id='2', is_support=True), {'title': 'A', 'text': 'different'}]}}]
    path = tmp_path / 'train.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows) + '\n\n', encoding='utf-8')
    assert load_training_corpus(path) == [
        {'title': 'A', 'text': 'shared'}, {'title': 'B', 'text': 'shared'},
        {'title': 'A', 'text': 'different'},
    ]


def test_loader_defaults_to_train_only(tmp_path, monkeypatch):
    directory = tmp_path / 'data' / 'sft'
    directory.mkdir(parents=True)
    (directory / 'train.jsonl').write_text(json.dumps({'metadata': {'documents': [{'title': 'A', 'text': 'alpha'}]}}))
    # Any accidental glob over adjacent splits would fail to parse this sentinel.
    (directory / 'val.jsonl').write_text('not JSON')
    monkeypatch.chdir(tmp_path)
    assert load_training_corpus() == [{'title': 'A', 'text': 'alpha'}]


def test_unicode_normalization():
    retriever = TinyBM25Retriever([{'title': 'ＣＡＦÉ', 'text': 'Straße'}])
    assert contents(retriever.retrieve(['cafe\u0301 STRASSE!'])) == ['ＣＡＦÉ\nStraße']


@pytest.mark.parametrize('query', ['', '   ', '!!!'])
def test_empty_query(documents, query):
    assert TinyBM25Retriever(documents).retrieve([query]) == {'result': [[]]}


def test_empty_corpus_and_batch(documents):
    assert TinyBM25Retriever([]).retrieve(['Hamlet', 'Newton']) == {'result': [[], []]}
    assert TinyBM25Retriever(documents).retrieve([]) == {'result': []}
    assert TinyBM25Retriever([{'title': '', 'text': ''}]).retrieve(['word']) == {'result': [[]]}


@pytest.mark.parametrize('topk', [0, -1])
def test_nonpositive_topk(documents, topk):
    assert TinyBM25Retriever(documents).retrieve(['Hamlet', 'Newton'], topk) == {'result': [[], []]}


def test_labels_do_not_affect_inclusion_or_ranking(tmp_path, documents):
    def load_variant(value):
        docs = [dict(d, id=str(i), is_support=value) for i, d in enumerate(documents)]
        row = {'answer': value, 'task_type': value, 'metadata': {
            'documents': docs, 'correct_doc_position': value, 'requires_search': value}}
        path = tmp_path / 'synthetic.jsonl'
        path.write_text(json.dumps(row), encoding='utf-8')
        return load_training_corpus(path), TinyBM25Retriever(docs).retrieve(['Hamlet Newton'])
    corpus_a, result_a = load_variant(False)
    corpus_b, result_b = load_variant(True)
    assert corpus_a == corpus_b == documents
    assert result_a == result_b
    assert TinyBM25Retriever(corpus_a).retrieve(['Hamlet Newton']) == result_a


def test_hand_checkable_bm25_length_normalization():
    # Lengths 10 and 2, avgdl=6, same IDF. Default TF factors:
    # long: 5/(2+2.25)=20/17; short: 2.5/(1+0.75)=10/7.
    # Removing length normalization reverses the ranking (10/7 > 1).
    docs = [{'title': 'long', 'text': 'apple apple x x x x x x x'},
            {'title': 'short', 'text': 'apple'}]
    assert contents(TinyBM25Retriever(docs).retrieve(['apple']))[0].startswith('short\n')
    assert contents(TinyBM25Retriever(docs, b=0).retrieve(['apple']))[0].startswith('long\n')


def test_hand_checkable_bm25_idf():
    # Equal lengths and TF: rare has df=1, common df=2, so rare wins.
    docs = [{'title': 'first', 'text': 'common'}, {'title': 'second', 'text': 'common'},
            {'title': 'third', 'text': 'rare'}]
    assert contents(TinyBM25Retriever(docs).retrieve(['common rare'])) == [
        'third\nrare', 'first\ncommon', 'second\ncommon']
