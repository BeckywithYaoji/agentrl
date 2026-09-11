"""Bounded Search-R1-style interaction, independent of model runtime."""

import re

from scripts.minimal_search_demo import extract_query, format_information

SEARCH_PROMPT = (
    "Answer the given question. After reasoning, if you find you lack some knowledge, "
    "you can call a search engine by <search> query </search> and it will return the "
    "top searched results between <information> and </information>. "
    "If you find no further external knowledge needed, you can directly provide the "
    "answer inside <answer> and </answer>, without detailed illustrations. "
    "Output one search or answer action per turn."
)


def run_agent(question, agent, retriever, max_search_steps=3, examples=()):
    if max_search_steps < 0:
        raise ValueError("max_search_steps must be nonnegative")
    messages = [{"role": "system", "content": SEARCH_PROMPT}]
    messages.extend(dict(message) for message in examples)
    messages.append({"role": "user", "content": question})
    result = dict(question=question, turns=[], final_answer=None, search_count=0,
                  success=False, termination_reason="invalid_output")
    for _ in range(max_search_steps + 1):
        output = agent.generate(messages)
        turn = {"role": "assistant", "content": output}
        result["turns"].append(turn)
        messages.append(dict(turn))
        # Upstream generation.py selects the first complete search/answer action.
        action = re.search(r'<(search|answer)>(.*?)</\1>', output, re.DOTALL)
        if action is None or not action.group(2).strip():
            break
        if action.group(1) == "answer":
            result.update(final_answer=action.group(2).strip(), success=True,
                          termination_reason="answer")
            break
        if result["search_count"] == max_search_steps:
            result["termination_reason"] = "max_search_steps"
            break
        query = extract_query(action.group(0))
        docs = retriever.retrieve([query], topk=3)["result"][0]
        observation = "<information>" + format_information(docs) + "</information>"
        result["search_count"] += 1
        result["turns"].append({"role": "tool", "content": observation, "query": query})
        # Qwen chat template receives evidence as a user message, not native tool JSON.
        messages.append({"role": "user", "content": observation})
    return result
