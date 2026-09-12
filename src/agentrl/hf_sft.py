"""HF SFT examples matching the SearchXMLAgentLoop message boundary."""
from agentrl.agent_loop import SEARCH_PROMPT
from scripts.milestone5br3_concise_answer_probe import CONCISE_REMINDER

RUNTIME_SYSTEM_PROMPT = (SEARCH_PROMPT + '\n'
    'Use the search tool to verify facts before answering. Output only one '
    '<search>query</search> or <answer>answer</answer> action per turn. '
    'After receiving information, answer using the evidence if it is sufficient.\n'
    + CONCISE_REMINDER)


def runtime_messages(record):
    messages = [{'role': 'system', 'content': RUNTIME_SYSTEM_PROMPT}]
    for message in record['messages']:
        role = 'user' if message['role'] == 'tool' else message['role']
        if role not in {'user', 'assistant'}:
            raise ValueError(f'unsupported role: {role}')
        messages.append({'role': role, 'content': message['content']})
    return messages


def encode_record(record, tokenizer, *, max_length):
    """One target per assistant turn; all prior turns are untrained context."""
    messages = runtime_messages(record)
    examples = []
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        context = messages[:index]
        prompt = tokenizer.apply_chat_template(
            context, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        target = tokenizer.encode(message['content'], add_special_tokens=False)
        if not target:
            raise ValueError('empty assistant target')
        target = target + [tokenizer.eos_token_id]
        ids = prefix + target
        if len(ids) > max_length:
            raise ValueError(f'max_length {max_length} would truncate assistant target')
        examples.append({'input_ids': ids, 'labels': [-100] * len(prefix) + target})
    if not examples:
        raise ValueError('record has no assistant action')
    return examples


def pad_batch(examples, pad_token_id):
    length = max(len(x['input_ids']) for x in examples)
    return {'input_ids': [x['input_ids'] + [pad_token_id] * (length - len(x['input_ids'])) for x in examples],
            'attention_mask': [[1] * len(x['input_ids']) + [0] * (length - len(x['input_ids'])) for x in examples],
            'labels': [x['labels'] + [-100] * (length - len(x['labels'])) for x in examples]}
