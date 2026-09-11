import unittest
from src.agentrl.sft_training import tokenize_target, supervised_positions


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        text = ''.join(m['role']+':'+m['content']+'\n' for m in messages)
        if kwargs.get('add_generation_prompt'):
            text += 'assistant:'
        return list(map(ord, text))


class TargetTest(unittest.TestCase):
    def test_search_supervision_excludes_context_and_padding(self):
        self.check([{'role':'user','content':'question'}, {'role':'assistant','content':'<search>query</search>'}])

    def test_answer_supervision_excludes_tool_and_previous_search(self):
        self.check([{'role':'user','content':'question'}, {'role':'assistant','content':'<search>query</search>'},
                    {'role':'user','content':'<information>fact</information>'},
                    {'role':'assistant','content':'<answer>fact</answer>'}])

    def check(self, messages):
        tokens, offset=tokenize_target({'messages':messages},CharacterTokenizer())
        positions=supervised_positions(offset,len(tokens),len(tokens)+10)
        self.assertEqual(''.join(chr(tokens[i]) for i in positions), messages[-1]['content']+'\n')
        self.assertNotIn(len(tokens),positions)
