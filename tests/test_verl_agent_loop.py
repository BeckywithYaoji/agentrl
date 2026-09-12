import asyncio
from types import SimpleNamespace
import pytest
from agentrl.retrieval import TinyBM25Retriever
from agentrl.verl_agent_loop import SearchXMLAgentLoop, parse_action, build_information

class FakeServer:
    def __init__(self, outputs): self.outputs=iter(outputs); self.calls=[]
    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        value=next(self.outputs); value=[99] if isinstance(value,str) else list(value)
        return SimpleNamespace(token_ids=value, log_probs=None, num_preempted=0, routed_experts=None, extra_fields={})

def make_loop(outputs, max_search_steps=3):
    loop=SearchXMLAgentLoop.__new__(SearchXMLAgentLoop); loop.server_manager=FakeServer(outputs)
    texts={(10,11,12):'<search>Alpha query</search>',(20,21):'<answer>one</answer>',(1,):'<search>Alpha query</search>',(2,):'<search>Alpha query</search>',(3,):'<answer>done</answer>',(99,):'bad'}
    loop.tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: texts[tuple(ids)])
    loop.retriever=TinyBM25Retriever([{'title':'Alpha','text':'alpha is one'},{'title':'Beta','text':'beta is two'}])
    loop.max_search_steps=max_search_steps; loop.response_length=100; loop.prompt_length=100
    async def initial(messages, **kwargs): return [1,2]
    async def ma(prompt, ids, mask, response_logprobs=None, assistant_logprobs=None): return SimpleNamespace(token_ids=prompt+ids), [1]*len(ids), None
    async def mc(prev, updated, runtime, mask, response_logprobs=None, tools=None): return SimpleNamespace(token_ids=runtime+[100]), mask+[0], None
    loop.ct_build_initial_tokens=initial; loop.ct_merge_assistant_token=ma; loop.ct_merge_context_msg=mc
    loop.process_multi_modal_info=lambda messages: asyncio.sleep(0,result={}); loop._get_mm_processor_kwargs=lambda audios=None:{}; loop._assert_mm_supported=lambda value:None
    return loop

def test_parse_action_is_strict():
    assert parse_action('reason <search> Alpha query </search>') == ('search','Alpha query')
    assert parse_action('<answer>done</answer>') == ('answer','done'); assert parse_action('<search></search>') is None; assert parse_action('<search>unfinished') is None

def test_information_schema():
    assert build_information([])=='<information></information>'; assert 'Doc 1(Title: Alpha) alpha is one' in build_information([{'document':{'contents':'Alpha\nalpha is one'}}])

def test_search_information_answer_masks():
    loop=make_loop([[10,11,12],[20,21]])
    result=asyncio.run(loop.run({},raw_prompt=[{'role':'user','content':'Q'}],priority=0))
    assert result.num_turns==4; assert len(result.response_ids)==len(result.response_mask); assert result.response_mask==[1,1,1,0,1,1]
    assert len(loop.server_manager.calls)==2; assert loop.server_manager.calls[0]['prompt_ids']==[1,2]; assert loop.server_manager.calls[1]['prompt_ids'][-1]==100
    assert result.extra_fields['search_queries']==['Alpha query']; assert result.extra_fields['observation_token_count']==1

def test_two_searches_and_max_steps():
    loop=make_loop([[1],[2],[3]],max_search_steps=2); result=asyncio.run(loop.run({},raw_prompt=[{'role':'user','content':'Q'}],priority=0))
    assert result.num_turns==6; assert result.extra_fields['search_queries']==['Alpha query','Alpha query']; assert len(loop.server_manager.calls)==3; assert result.response_mask==[1,0,1,0,1]

def test_zero_match_and_malformed_terminate():
    loop=make_loop(['bad']); result=asyncio.run(loop.run({},raw_prompt=[{'role':'user','content':'Q'}],priority=0)); assert result.extra_fields['termination_reason']=='malformed'; assert len(loop.server_manager.calls)==1

def test_query_is_exactly_model_text():
    seen=[]
    class R:
        def retrieve(self,queries,topk=3): seen.extend(queries); return {'result':[[]]}
    loop=make_loop([[10,11,12],[20,21]]); loop.retriever=R(); asyncio.run(loop.run({},raw_prompt=[{'role':'user','content':'Q'}],priority=0)); assert seen==['Alpha query']
