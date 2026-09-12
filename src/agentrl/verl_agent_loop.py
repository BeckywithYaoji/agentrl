"""Native veRL AgentLoopBase adapter for the project's XML search protocol."""
from __future__ import annotations
import re
from typing import Any
from scripts.minimal_search_demo import format_information
from verl.experimental.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

_ACTION_RE = re.compile(r'<(search|answer)>(.*?)</\1>', re.DOTALL)

def parse_action(text: str):
    match = _ACTION_RE.search(text)
    if not match or not match.group(2).strip(): return None
    return match.group(1), match.group(2).strip()

def build_information(documents):
    return '<information>' + format_information(documents) + '</information>'

class SearchXMLAgentLoop(AgentLoopBase):
    """veRL-native multi-turn loop; observations are context-only tokens."""
    def __init__(self, *args, retriever=None, max_search_steps=3, **kwargs):
        super().__init__(*args, **kwargs)
        from agentrl.retrieval import TinyBM25Retriever, load_training_corpus
        self.retriever = retriever or TinyBM25Retriever(load_training_corpus())
        self.max_search_steps = max_search_steps
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        priority = int(priority)
        messages = [dict(m) for m in kwargs['raw_prompt']]
        mm_data = await self.process_multi_modal_info(messages)
        self._assert_mm_supported(bool(mm_data))
        prompt_ids = await self.ct_build_initial_tokens(messages, **{
            'images': mm_data.get('images'), 'videos': mm_data.get('videos'), 'audios': mm_data.get('audios')})
        runtime_ids = list(prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        search_queries: list[str] = []
        observation_token_count = 0
        assistant_turns = 0
        termination_reason = 'malformed'
        metrics = {}
        while assistant_turns <= self.max_search_steps:
            assistant_turns += 1
            with simple_timer('generate_sequences', metrics):
                output = await self.server_manager.generate(
                    request_id=f'verl-xml-{priority}-{assistant_turns}', prompt_ids=runtime_ids,
                    sampling_params=sampling_params, image_data=mm_data.get('images'),
                    video_data=mm_data.get('videos'), audio_data=mm_data.get('audios'),
                    mm_processor_kwargs=self._get_mm_processor_kwargs(mm_data.get('audios')), priority=priority)
            assistant_ids = list(output.token_ids)
            merged, assistant_mask, assistant_logprobs = await self.ct_merge_assistant_token(
                runtime_ids, assistant_ids, response_mask,
                None, assistant_logprobs=getattr(output, 'log_probs', None))
            runtime_ids = list(merged.token_ids)
            response_ids.extend(assistant_ids)
            response_mask.extend([1] * len(assistant_ids))
            assistant_text = self.tokenizer.decode(assistant_ids, skip_special_tokens=True)
            messages.append({'role': 'assistant', 'content': assistant_text})
            action = parse_action(assistant_text)
            if action is None:
                break
            kind, value = action
            if kind == 'answer':
                termination_reason = 'answer'; break
            if len(search_queries) >= self.max_search_steps:
                termination_reason = 'max_search_steps'; break
            search_queries.append(value)
            docs = self.retriever.retrieve([value], topk=3)['result'][0]
            observation = build_information(docs)
            messages.append({'role': 'user', 'content': observation})
            previous = messages[:-1]
            merged_context, _, _ = await self.ct_merge_context_msg(
                previous, messages, runtime_ids, response_mask, None)
            new_runtime = list(merged_context.token_ids)
            observation_token_count += len(new_runtime) - len(runtime_ids)
            obs_count = len(new_runtime) - len(runtime_ids)
            runtime_ids = new_runtime
            response_ids.extend(new_runtime[-obs_count:] if obs_count > 0 else [])
            response_mask.extend([0] * max(0, obs_count))
        output_obj = AgentLoopOutput(
            prompt_ids=prompt_ids, response_ids=response_ids[:self.response_length],
            response_mask=response_mask[:self.response_length],
            response_logprobs=None, num_turns=len(messages), metrics=metrics,
            extra_fields={'search_queries': search_queries,
                          'observation_token_count': observation_token_count,
                          'assistant_generated_token_count': sum(response_mask),
                          'termination_reason': termination_reason})
        return output_obj
