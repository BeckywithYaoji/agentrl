"""Canonical, model-independent dataset construction and validation (no training)."""

import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

from scripts.minimal_search_demo import format_information

SOURCE_URL = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json'
SOURCE_SHA256 = '68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002'
LICENSE = 'CC-BY-SA-4.0'
ABSTAIN = 'Insufficient information.'
TASK_RATIOS = {'direct_answer': .15, 'single_search_clean': .40,
               'single_search_hard_negative': .30, 'noisy_retrieval': .10,
               'insufficient_evidence': .05}
TASKS = set(TASK_RATIOS) | {'counterfactual', 'ood'}


def normalize(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold()))


def display_title(title):
    return unquote(title).replace('_', ' ')


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))


def group_split(rows, seed=42):
    """Union source groups AND normalized questions before seeded 80/10/10 split."""
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for key in [('group', row['metadata']['source_group']),
                    ('item', row['metadata']['source_item']),
                    ('question', normalize(row['question']))]:
            if key in seen:
                parents[root(i)] = root(seen[key])
            seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[root(i)].append(row)
    groups = list(groups.values())
    random.Random(seed).shuffle(groups)
    cut1, cut2 = int(.8 * len(groups)), int(.9 * len(groups))
    return {name: [row for group in part for row in group] for name, part in
            [('train', groups[:cut1]), ('val', groups[cut1:cut2]), ('test', groups[cut2:])]}


def observation(documents):
    return '<information>' + format_information([
        {'document': {'contents': d['title'] + '\n' + d['text']}} for d in documents
    ]) + '</information>'


def query_for(question, title):
    # Keep entity/title anchors, remove interrogatives and auxiliaries, never add gold answer.
    stop = {'who', 'what', 'where', 'when', 'which', 'how', 'why', 'is', 'was',
            'were', 'are', 'did', 'does', 'do', 'the', 'a', 'an', 'of', 'in', 'to'}
    words = re.findall(r"[\w'-]+", question)
    return display_title(title) + ' ' + ' '.join(w for w in words if w.casefold() not in stop)


def make_sample(uid, question, task, answer, docs, source, source_item, source_group,
                query=None, seed=42, source_question=None, counterfactual=False):
    messages = [{'role': 'user', 'content': question}]
    if task != 'direct_answer':
        messages += [{'role': 'assistant', 'content': f'<search>{query}</search>'},
                     {'role': 'tool', 'content': observation(docs)}]
    messages.append({'role': 'assistant', 'content': f'<answer>{answer}</answer>'})
    positions = [i + 1 for i, d in enumerate(docs) if d['is_support']]
    return {'id': uid, 'question': question, 'task_type': task, 'messages': messages,
            'answer': answer, 'metadata': {
                'requires_search': task != 'direct_answer',
                'search_count': int(task != 'direct_answer'),
                'correct_doc_position': positions[0] if positions else None,
                'has_hard_negative': task in {'single_search_hard_negative', 'counterfactual', 'ood'},
                'source': source, 'source_item': source_item, 'source_group': source_group,
                'source_question': source_question or question, 'license': LICENSE,
                'documents': docs, 'seed': seed, 'counterfactual': counterfactual}}


def counterfactual_set(count=30, seed=42, ood=False):
    rng = random.Random(seed)
    prefix = 'ORBIT' if ood else 'FABLE'
    rows = []
    for i in range(count):
        entity = f'{prefix}-{seed}-{i:03d}'
        name = f'{rng.choice(["Nira", "Liora", "Mira", "Tarin"])} {rng.choice(["Voss", "Vale", "Sen", "Quill"])}-{i:03d}'
        if ood:
            question = f'Consult the fictional registry: identify the credited investigator for expedition {entity}.'
            text = f'The fictional expedition {entity} credits {name} as its investigator.'
            query = f'{entity} credited investigator'
        else:
            question = f'Who wrote the fictional play {entity}?'
            text = f'In this fictional test registry, the play {entity} was written by {name}.'
            query = f'{entity} author'
        docs = [
            {'id': 'synthetic-hamlet', 'title': 'Hamlet', 'text': 'William Shakespeare wrote Hamlet.', 'is_support': False},
            {'id': 'synthetic-potter', 'title': 'Harry Potter', 'text': 'J.K. Rowling wrote Harry Potter.', 'is_support': False}]
        if ood:
            docs += [{'id': entity + '-noise', 'title': entity, 'text': f'{entity} used a blue flag; the departure date is not recorded.', 'is_support': False},
                     {'id': entity + '-near', 'title': entity + '-B', 'text': f'The different expedition {entity}-B credits Alex Other.', 'is_support': False}]
        position = i % (len(docs) + 1)
        rng.shuffle(docs)
        docs.insert(position, {'id': entity, 'title': entity, 'text': text, 'is_support': True})
        rows.append(make_sample(entity, question, 'ood' if ood else 'counterfactual', name,
                    docs, 'synthetic_registry', entity, entity, query, seed, counterfactual=True))
    rng.shuffle(rows)
    return rows


def load_source(path):
    raw = json.loads(Path(path).read_text())
    items = []
    for article in raw['data']:
        title = article['title']
        for pidx, paragraph in enumerate(article['paragraphs']):
            context = paragraph['context']
            if len(context) > 2200 or '<' in context or '```' in context:
                continue
            paragraph_id = f'{title}:{pidx}'
            for qa in paragraph['qas']:
                q = qa['question'].strip()
                if '<' in q or '```' in q or re.search(r'\b(he|she|his|her|they|their|this|these|those|it)\b', q, re.I):
                    continue
                answer = ABSTAIN if qa['is_impossible'] else qa['answers'][0]['text'].strip()
                if not answer or len(answer.split()) > 8 or len(answer) > 80 or '<' in answer:
                    continue
                if not qa['is_impossible']:
                    annotation = qa['answers'][0]
                    if context[annotation['answer_start']:annotation['answer_start'] + len(annotation['text'])] != annotation['text']:
                        continue
                    if normalize(answer) in normalize(q):
                        continue
                items.append({'id': qa['id'], 'question': q, 'answer': answer,
                              'title': title, 'context': context, 'paragraph_id': paragraph_id,
                              'impossible': qa['is_impossible'], 'answers': qa['answers'],
                              'metadata': {'source_group': title, 'source_item': qa['id']}})
    return items


def build_main(source, size=1000, seed=42):
    if size != 10 and (size % 100 or size < 100):
        raise ValueError('main size must be a multiple of 100; use --size 10 for the tiny gate')
    rng = random.Random(seed)
    partitions = {'tiny': source} if size == 10 else group_split(source, seed)
    output = {}
    used_questions = set()
    plan = [('tiny', 1)] if size == 10 else [('train', .8), ('val', .1), ('test', .1)]
    for split, proportion in plan:
        n = int(size * proportion)
        # Largest remainder preserves exact type totals for the small 100-row stage.
        targets = {task: int(n * ratio) for task, ratio in TASK_RATIOS.items()}
        for task in sorted(TASK_RATIOS, key=lambda t: (-(n * TASK_RATIOS[t] - targets[t]), t))[:n - sum(targets.values())]:
            targets[task] += 1
        if size == 10:
            targets = dict(zip(TASK_RATIOS, [2, 2, 2, 1, 1]))
        candidates = list(partitions[split])
        rng.shuffle(candidates)
        paragraphs = {}
        by_article = defaultdict(dict)
        for item in candidates:
            paragraphs[item['paragraph_id']] = item
            by_article[item['title']][item['paragraph_id']] = item
        selected_paragraphs = set()
        rows = []
        for task, needed in targets.items():
            if task == 'direct_answer':
                for i in range(needed):
                    offset = {'tiny': 0, 'train': 0, 'val': 10000, 'test': 20000}[split] + i
                    a, b = 11 + offset, 2 + (i % 9)
                    op = i % 3
                    question = [f'What is {a} + {b}?', f'What is {a} minus {b}?', f'What is {a} multiplied by {b}?'][op]
                    answer = str([a+b, a-b, a*b][op])
                    uid = f'arithmetic-{offset}'
                    rows.append(make_sample(uid, question, task, answer, [], 'synthetic_arithmetic', uid, uid, seed=seed))
                continue
            count = 0
            for item in candidates:
                if count == needed:
                    break
                if item['impossible'] != (task == 'insufficient_evidence'):
                    continue
                if item['paragraph_id'] in selected_paragraphs or normalize(item['question']) in used_questions:
                    continue
                q, answer = item['question'], item['answer']
                support = {'id': item['paragraph_id'], 'title': display_title(item['title']),
                           'text': item['context'], 'is_support': not item['impossible']}
                docs = [support]
                if task in {'single_search_hard_negative', 'noisy_retrieval'}:
                    # Same Wikipedia article, different paragraphs: topic-similar candidates.
                    negatives = [p for p in by_article[item['title']].values()
                                 if p['paragraph_id'] != item['paragraph_id']
                                 and normalize(answer) not in normalize(p['context'])]
                    qterms = set(normalize(q).split())
                    negatives.sort(key=lambda p: (-len(qterms & set(normalize(p['context']).split())), p['paragraph_id']))
                    if len(negatives) < 2:
                        continue
                    docs = [{'id': p['paragraph_id'], 'title': display_title(p['title']),
                             'text': p['context'], 'is_support': False} for p in negatives[:2]]
                    if task == 'noisy_retrieval':
                        # A partial relevant sentence without the answer plus another same-topic passage.
                        sentences = re.split(r'(?<=[.!?])\s+', item['context'])
                        partial = next((s for s in sentences if len(s.split()) >= 8 and normalize(answer) not in normalize(s)), None)
                        if partial is None:
                            continue
                        docs[0] = {'id': item['paragraph_id'] + ':partial', 'title': support['title'],
                                   'text': partial, 'is_support': False}
                    rng.shuffle(docs)
                    # Shuffled balanced schedule, independent of the question and answer.
                    if count % 3 == 0:
                        schedule = [0, 1, 2]
                        rng.shuffle(schedule)
                    docs.insert(schedule[count % 3], support)
                title = display_title(item['title'])
                question = q if normalize(title) in normalize(q) else f"Regarding {title}, {q[0].lower() + q[1:]}"
                row = make_sample('squad-' + item['id'], question, task, answer, docs,
                    'SQuAD-2.0-train', item['id'], item['title'], query_for(q, item['title']), seed, q)
                row['metadata']['source_url'] = SOURCE_URL
                row['metadata']['original_answers'] = item['answers']
                row['metadata']['source_is_impossible'] = item['impossible']
                row['metadata']['source_paragraph'] = item['paragraph_id']
                rows.append(row)
                selected_paragraphs.add(item['paragraph_id'])
                used_questions.add(normalize(q))
                count += 1
            if count < needed:
                raise ValueError(f'not enough eligible source items for {split}/{task}: {count}/{needed}')
        rng.shuffle(rows)
        output[split] = rows
    return output


def validate_sample(row):
    def require(condition, message):
        if not condition:
            raise ValueError(f"{row.get('id', '<missing id>')}: {message}")
    require(all(k in row for k in ['id', 'question', 'task_type', 'messages', 'answer', 'metadata']), 'required fields')
    require(all(isinstance(row[k], str) and row[k].strip() for k in ['id', 'question', 'answer']), 'nonempty strings')
    require(row['task_type'] in TASKS, 'task_type')
    meta = row['metadata']
    require(isinstance(meta, dict) and all(k in meta for k in ['requires_search', 'search_count', 'correct_doc_position',
            'source', 'source_item', 'source_group', 'has_hard_negative', 'documents', 'license']), 'metadata fields')
    require(all(isinstance(meta[k], str) and meta[k] for k in ['source', 'source_item', 'source_group', 'license']), 'source identifiers')
    msgs = row['messages']
    require(isinstance(msgs, list) and all(isinstance(m, dict) and isinstance(m.get('content'), str) for m in msgs), 'messages schema')
    direct = row['task_type'] == 'direct_answer'
    roles = ['user', 'assistant'] if direct else ['user', 'assistant', 'tool', 'assistant']
    require([m.get('role') for m in msgs] == roles, 'message order')
    require(msgs[0]['content'] == row['question'], 'question mismatch')
    require(type(meta['requires_search']) is bool and meta['requires_search'] == (not direct), 'requires_search')
    require(type(meta['search_count']) is int and meta['search_count'] == int(not direct), 'search_count')
    for message in msgs[1:]:
        tag = 'information' if message['role'] == 'tool' else ('answer' if message is msgs[-1] else 'search')
        match = re.fullmatch(fr'<{tag}>([^<>]+)</{tag}>', message['content'], re.DOTALL)
        require(match is not None and bool(match.group(1).strip()), f'{tag} protocol')
        require('```' not in message['content'], 'code fences')
    require(msgs[-1]['content'] == f"<answer>{row['answer']}</answer>", 'answer mismatch')
    docs = meta['documents']
    require(isinstance(docs, list), 'documents type')
    require(all(isinstance(d, dict) and all(k in d for k in ['id', 'title', 'text', 'is_support']) for d in docs), 'document schema')
    require(all(type(d['is_support']) is bool and all(isinstance(d[k], str) and d[k] for k in ['id', 'title', 'text']) for d in docs), 'document values')
    require(len({d['id'] for d in docs}) == len(docs), 'duplicate documents')
    if direct:
        require(not docs and meta['correct_doc_position'] is None, 'direct evidence')
    else:
        require(bool(docs) and msgs[2]['content'] == observation(docs), 'observation differs from documents')
        supports = [i+1 for i, d in enumerate(docs) if d['is_support']]
        if row['task_type'] == 'insufficient_evidence':
            require(not supports and meta['correct_doc_position'] is None and row['answer'] == ABSTAIN, 'insufficient evidence policy')
            require(meta.get('source_is_impossible') is True and meta.get('original_answers') == [], 'missing unanswerable source annotation')
        else:
            require(len(supports) == 1 and type(meta['correct_doc_position']) is int and supports[0] == meta['correct_doc_position'], 'correct_doc_position')
            require(normalize(row['answer']) in normalize(docs[supports[0]-1]['text']), 'answer absent from support')
            require(all(normalize(row['answer']) not in normalize(d['text']) for d in docs if not d['is_support']), 'distractor contains gold answer')
            if meta['source'] == 'SQuAD-2.0-train':
                annotations = meta.get('original_answers', [])
                require(bool(annotations) and meta.get('source_is_impossible') is False, 'missing source answer annotation')
                support = docs[supports[0]-1]
                require(support['id'] == meta.get('source_paragraph'), 'source paragraph mismatch')
                annotation = annotations[0]
                start, text = annotation['answer_start'], annotation['text']
                require(support['text'][start:start+len(text)] == text and text.strip() == row['answer'], 'source answer span mismatch')
    hard = row['task_type'] in {'single_search_hard_negative', 'counterfactual', 'ood'}
    require(type(meta['has_hard_negative']) is bool and meta['has_hard_negative'] == hard, 'hard-negative flag')
    if hard or row['task_type'] == 'noisy_retrieval':
        require(len(docs) >= 3 and sum(d['is_support'] for d in docs) == 1, 'hard/noisy evidence count')


def validate_splits(splits):
    ids, keys = set(), {}
    questions = Counter()
    for split, rows in splits.items():
        for row in rows:
            validate_sample(row)
            if row['id'] in ids:
                raise ValueError('duplicate ID: ' + row['id'])
            ids.add(row['id'])
            questions[normalize(row['question'])] += 1
            meta = row['metadata']
            for key in [('question', normalize(row['question'])),
                        ('original_question', normalize(meta.get('source_question', row['question']))),
                        ('group', meta['source_group']), ('source_item', meta['source_item'])]:
                if key in keys and keys[key] != split:
                    raise ValueError(f'cross-split leakage: {key}')
                keys[key] = split
            if meta['source'] == 'SQuAD-2.0-train':
                for document in meta['documents']:
                    key = ('source_document', document['id'].removesuffix(':partial'))
                    if key in keys and keys[key] != split:
                        raise ValueError(f'cross-split source document leakage: {key}')
                    keys[key] = split
    return {'sample_count': len(ids), 'duplicate_ids': 0,
            'duplicate_questions': sum(n-1 for n in questions.values()), 'cross_split_leakage': 0, 'group_leakage': 0}


def statistics(rows):
    n = len(rows)
    searches = [m['content'][8:-9] for r in rows for m in r['messages'] if m['role'] == 'assistant' and m['content'].startswith('<search>')]
    queries_copied = sum(normalize(m['content'][8:-9]) in {normalize(r['question']), normalize(r['metadata'].get('source_question', r['question']))}
                         for r in rows for m in r['messages'] if m['role'] == 'assistant' and m['content'].startswith('<search>'))
    distribution = Counter(r['task_type'] for r in rows)
    positions = Counter(str(r['metadata']['correct_doc_position']) for r in rows if r['metadata']['correct_doc_position'] is not None)
    by_task = {task: dict(Counter(str(r['metadata']['correct_doc_position']) for r in rows
                                if r['task_type'] == task and r['metadata']['correct_doc_position'] is not None)) for task in sorted(distribution)}
    return {'sample_count': n, 'task_distribution': dict(distribution),
            'requires_search_ratio': sum(r['metadata']['requires_search'] for r in rows)/n,
            'average_search_count': len(searches)/n,
            'hard_negative_ratio': sum(r['metadata']['has_hard_negative'] for r in rows)/n,
            'correct_doc_position_distribution': dict(positions), 'positions_by_task': by_task,
            'direct_answer_ratio': distribution['direct_answer']/n, 'multi_search_ratio': 0.0,
            'duplicate_question_count': n - len({normalize(r['question']) for r in rows}),
            'average_question_length_words': sum(len(r['question'].split()) for r in rows)/n,
            'average_answer_length_words': sum(len(r['answer'].split()) for r in rows)/n,
            'query_copy_rate': queries_copied / len(searches) if searches else 0.0,
            'average_information_document_count': sum(len(r['metadata']['documents']) for r in rows)/len(searches) if searches else 0.0}


def load_splits(directory):
    return {path.stem: read_jsonl(path) for path in sorted(Path(directory).glob('*.jsonl'))}
