"""Model-only MLX-LM smoke test for Qwen3."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlx_lm import generate
from src.agentrl.qwen_agent import load_qwen


DEFAULT_MODEL = "mlx-community/Qwen3-0.6B-4bit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    print(f"model: {args.model}")
    start = time.perf_counter()
    model, tokenizer = load_qwen(args.model)
    loaded = time.perf_counter()
    prompt = "Question: What is 2 + 2? Answer briefly."
    output = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        verbose=False,
    )
    finished = time.perf_counter()
    print(f"load_success: true")
    print(f"load_seconds: {loaded - start:.2f}")
    print(f"generation_seconds: {finished - loaded:.2f}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
