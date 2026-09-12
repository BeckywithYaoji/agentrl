"""CPU-only adapter checks; no vLLM or CUDA imports are needed."""
import copy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from agentrl.agent_loop import SEARCH_PROMPT, run_agent
from agentrl.retrieval import TinyBM25Retriever
from agentrl.vllm_agent import VLLMSearchAgent, sampling_config
from scripts.vllm_search_smoke import completed_search_chain


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = copy.deepcopy(messages)
        self.kwargs = kwargs
        return repr(messages)

    def encode(self, prompt, **kwargs):
        return list(range(len(prompt.split())))


class Backend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def generate(self, prompts, **kwargs):
        self.calls.append((prompts, kwargs))
        return [SimpleNamespace(outputs=[SimpleNamespace(text=next(self.outputs))])]


def make_agent(outputs, **kwargs):
    return VLLMSearchAgent(Backend(outputs), Tokenizer(), object(), **kwargs)


def test_import_is_cpu_safe():
    subprocess.run([sys.executable, '-c',
        'import agentrl.vllm_agent; import scripts.vllm_search_smoke; import sys; '
        'assert "vllm" not in sys.modules; assert "torch" not in sys.modules'], check=True)


def test_sampling_config_preserves_action_closers():
    assert sampling_config() == dict(temperature=0.0, max_tokens=256,
        stop=['</search>', '</answer>'], include_stop_str_in_output=True)


def test_template_and_raw_output():
    agent = make_agent(['  <search>alpha</search>'])
    messages = [{'role': 'system', 'content': SEARCH_PROMPT}, {'role': 'user', 'content': 'Q'}]
    original = copy.deepcopy(messages)
    assert agent.generate(messages) == '  <search>alpha</search>'
    assert messages == original
    assert agent.tokenizer.messages == original
    assert agent.tokenizer.kwargs == dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert agent.llm.calls[0][1] == dict(sampling_params=agent.sampling_params, use_tqdm=False)
    messages[1]['content'] = 'mutated'
    assert agent.calls[0]['messages'] == original
    assert agent.calls[0]['raw_output'] == '  <search>alpha</search>'


def test_context_overflow_is_explicit():
    agent = make_agent(['unused'], max_model_len=256)
    with pytest.raises(ValueError, match='context'):
        agent.generate([{'role': 'user', 'content': 'question'}])
    assert not agent.llm.calls


def test_real_loop_and_bm25_observation():
    agent = make_agent(['<search>alpha</search>', '<answer>one</answer>'])
    retriever = TinyBM25Retriever([{'title': 'Alpha', 'text': 'alpha is one'}])
    result = run_agent('Q', agent, retriever)
    assert result['success'] and result['search_count'] == 1
    assert agent.calls[1]['messages'][-1] == {
        'role': 'user', 'content': '<information>Doc 1(Title: Alpha) alpha is one</information>'}
    assert completed_search_chain(result)


def test_zero_match_observation():
    agent = make_agent(['<search>asdfghjkl</search>', '<answer>Unknown</answer>'])
    result = run_agent('Q', agent, TinyBM25Retriever([{'title': 'Alpha', 'text': 'one'}]))
    assert agent.calls[1]['messages'][-1]['content'] == '<information></information>'
    assert result['success']
    assert not completed_search_chain(result)


@pytest.mark.parametrize('outputs', [
    ['<answer>direct</answer>'], ['malformed'],
    ['<search>alpha</search>'] * 4,
])
def test_incomplete_chains_do_not_pass(outputs):
    result = run_agent('Q', make_agent(outputs), TinyBM25Retriever([{'title': 'Alpha', 'text': 'one'}]))
    assert not completed_search_chain(result)


def test_reminder_is_generic_explicit_and_does_not_mutate_input():
    agent = make_agent(['<search>alpha</search>'], reminder='Use search before answering.')
    messages = [{'role': 'system', 'content': SEARCH_PROMPT}, {'role': 'user', 'content': 'Q'}]
    agent.generate(messages)
    assert messages[0]['content'] == SEARCH_PROMPT
    assert agent.calls[0]['messages'][0]['content'] == SEARCH_PROMPT + '\nUse search before answering.'
