"""Deterministic Search-R1 protocol smoke test with an in-memory retriever."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    title: str
    text: str

    @property
    def contents(self) -> str:
        return f"{self.title}\n{self.text}"


class LocalSearchEnvironment:
    """Small local stand-in for Search-R1's /retrieve contract."""

    def __init__(self) -> None:
        self.documents = [
            Document("Hamlet", "Hamlet is a tragedy written by William Shakespeare."),
            Document("William Shakespeare", "William Shakespeare wrote Hamlet."),
            Document("Eiffel Tower", "The Eiffel Tower is located in Paris."),
            Document("Theory of relativity", "Albert Einstein developed the theory of relativity."),
            Document("XQZ-17", "According to the fictional test knowledge base, the play XQZ-17 was written by Alice Example."),
        ]

    def retrieve(self, queries: list[str], topk: int = 3) -> dict:
        results = []
        for query in queries:
            terms = set(query.lower().split())
            ranked = sorted(
                self.documents,
                key=lambda doc: sum(term in doc.contents.lower() for term in terms),
                reverse=True,
            )
            results.append(
                [{"document": {"contents": doc.contents}} for doc in ranked[:topk]]
            )
        return {"result": results}


def extract_query(text: str) -> str | None:
    matches = re.findall(r"<search>(.*?)</search>", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def format_information(documents: list[dict]) -> str:
    formatted = []
    for index, item in enumerate(documents, start=1):
        contents = item["document"]["contents"]
        title, _, text = contents.partition("\n")
        formatted.append(f"Doc {index}(Title: {title}) {text}")
    return "\n".join(formatted)


def run_demo() -> str:
    question = "Who wrote Hamlet?"
    agent_output = "I need to verify the author. <search>Hamlet author</search>"
    query = extract_query(agent_output)
    if query is None:
        raise RuntimeError("agent did not produce a Search-R1 search action")

    response = LocalSearchEnvironment().retrieve([query], topk=3)
    observation = format_information(response["result"][0])
    answer = "William Shakespeare"
    transcript = (
        f"Question: {question}\n"
        f"Agent: {agent_output}\n"
        f"Observation: <information>{observation}</information>\n"
        f"Answer: <answer>{answer}</answer>"
    )
    return transcript


if __name__ == "__main__":
    print(run_demo())
