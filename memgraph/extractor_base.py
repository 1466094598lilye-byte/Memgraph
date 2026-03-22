"""Extractor 抽象接口：extract(chunk) -> (nodes, edges, topic, usage)。"""

from typing import Protocol

from memgraph.models import Edge, Node


class ExtractorProtocol(Protocol):
    """抽取器接口：支持 LLM 与 NonLLM 实现切换。"""

    def extract(
        self,
        conversation_chunk: list[dict],
        existing_topics: list[str] | None = None,
    ) -> tuple[list[Node], list[Edge], str, dict]:
        """
        从对话片段抽取节点、边、话题。
        返回 (nodes, edges, topic_label, usage)
        usage: {"input_tokens": int, "output_tokens": int}，NonLLM 可全为 0。
        """
        ...
