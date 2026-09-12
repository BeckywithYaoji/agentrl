import pytest

from agentrl.hf_sft import encode_record, pad_batch, runtime_messages


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 33

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return ''.join(chr(i) for i in ids if i)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        text = ''.join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return self.encode(text) if tokenize else text


def search_row():
    return {'messages': [
        {'role': 'user', 'content': 'Question?'},
        {'role': 'assistant', 'content': '<search>needle</search>'},
        {'role': 'tool', 'content': '<information>evidence</information>'},
        {'role': 'assistant', 'content': '<answer>yes</answer>'},
    ]}


def test_runtime_messages_replace_tool_role_without_wrapper():
    messages = runtime_messages(search_row())
    assert [m['role'] for m in messages] == ['system', 'user', 'assistant', 'user', 'assistant']
    assert messages[3]['content'] == '<information>evidence</information>'
    assert '<tool_response>' not in str(messages)


def test_search_and_answer_targets_are_supervised_and_context_is_masked():
    tok = FakeTokenizer()
    search, answer = encode_record(search_row(), tok, max_length=1536)
    assert len(search['input_ids']) == len(search['labels'])
    assert len(answer['input_ids']) == len(answer['labels'])
    assert tok.decode([x for x in search['labels'] if x != -100]) == '<search>needle</search>!'
    assert tok.decode([x for x in answer['labels'] if x != -100]) == '<answer>yes</answer>!'
    assert '<information>evidence</information>' in tok.decode(answer['input_ids'])
    supervised_start = next(i for i, label in enumerate(answer['labels']) if label != -100)
    assert all(label == -100 for label in answer['labels'][:supervised_start])
    assert '<information>evidence</information>' in tok.decode(answer['input_ids'][:supervised_start])
    search_start = next(i for i, label in enumerate(search['labels']) if label != -100)
    assert '<information>evidence</information>' not in tok.decode(search['input_ids'][:search_start])
    assert 'Question?' in tok.decode(search['input_ids'][:next(i for i, x in enumerate(search['labels']) if x != -100)])
    assert 'Output one search' in tok.decode(search['input_ids'][:next(i for i, x in enumerate(search['labels']) if x != -100)])


def test_direct_answer_and_padding():
    tok = FakeTokenizer()
    row = {'messages': [{'role': 'user', 'content': '2+2?'}, {'role': 'assistant', 'content': '<answer>4</answer>'}]}
    example = encode_record(row, tok, max_length=1536)[0]
    assert tok.decode([x for x in example['labels'] if x != -100]) == '<answer>4</answer>!'
    batch = pad_batch([example, {'input_ids': [1], 'labels': [1]}], tok.pad_token_id)
    assert len(batch['input_ids'][0]) == len(batch['labels'][0])
    assert all(x == -100 for x in batch['labels'][1][1:])


def test_truncation_is_rejected():
    tok = FakeTokenizer()
    with pytest.raises(ValueError, match='max_length'):
        encode_record(search_row(), tok, max_length=30)


def test_real_qwen_tokenizer_roundtrip_preserves_runtime_encoding(tmp_path):
    from pathlib import Path
    from transformers import AutoTokenizer
    snapshot = Path('/root/autodl-tmp/cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca')
    if not snapshot.is_dir():
        pytest.skip('local Qwen tokenizer snapshot unavailable')
    original = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    original.save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    before = encode_record(search_row(), original, max_length=1536)
    after = encode_record(search_row(), reloaded, max_length=1536)
    assert before == after
    assert all(len(x['input_ids']) == len(x['labels']) for x in after)
    assert '<tool_response>' not in original.decode(after[-1]['input_ids'])
