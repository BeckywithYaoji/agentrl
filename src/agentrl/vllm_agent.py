"""Dependency-injected vLLM adapter. Importing this module loads no GPU runtime."""
import copy


def sampling_config() -> dict:
    return dict(temperature=0.0, max_tokens=256,
                stop=['</search>', '</answer>'], include_stop_str_in_output=True)


class VLLMSearchAgent:
    def __init__(self, llm, tokenizer, sampling_params, *, max_model_len=2048,
                 reminder='', on_call=None):
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.max_model_len = max_model_len
        self.reminder = reminder
        self.on_call = on_call
        self.calls = []

    def generate(self, messages) -> str:
        messages = copy.deepcopy(messages)
        if self.reminder:
            messages[0]['content'] += '\n' + self.reminder
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) + sampling_config()['max_tokens'] > self.max_model_len:
            raise ValueError(f'context budget exceeded: {len(token_ids)} + 256 > {self.max_model_len}')
        output = self.llm.generate([{'prompt_token_ids': token_ids}],
                                  sampling_params=self.sampling_params, use_tqdm=False)
        text = output[0].outputs[0].text
        call = dict(messages=messages, formatted_prompt=prompt,
                    prompt_tokens=len(token_ids), raw_output=text)
        self.calls.append(call)
        if self.on_call:
            self.on_call(call)
        return text
