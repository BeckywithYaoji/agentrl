"""One cached model instance with the Qwen3 chat template."""

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL = 'mlx-community/Qwen3-0.6B-4bit'
REVISION = '73e3e38d981303bc594367cd910ea6eb48349da8'


def load_qwen(model_id=MODEL, adapter_path=None):
    return load(model_id, revision=REVISION if model_id == MODEL else None, adapter_path=adapter_path)


class QwenAgent:
    def __init__(self, max_tokens=256, adapter_path=None):
        self.model, self.tokenizer = load_qwen(adapter_path=adapter_path)
        self.max_tokens = max_tokens

    def generate(self, messages, stop_actions=True, temperature=0.0, top_p=1.0):
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        output = ''
        stream = stream_generate(self.model, self.tokenizer, prompt=prompt,
                                 max_tokens=self.max_tokens, sampler=make_sampler(temp=temperature, top_p=top_p))
        try:
            for chunk in stream:
                output += chunk.text
                # Preserve every emitted character, including the closing action tag.
                if stop_actions and ('</search>' in output or '</answer>' in output):
                    break
        finally:
            stream.close()
        return output
