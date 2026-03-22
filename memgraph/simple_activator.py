"""SimpleActivator: 纯 cosine similarity 检索，零额外机制。

假设：如果 embedding 质量足够好，纯语义距离就够了。
不需要 topic matching、boost、critic、slot allocation。
"""

import logging

import numpy as np

from memgraph.embedder import Embedder
from memgraph.graph import GraphStore
from memgraph.models import ActivateResult, NodeType

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SimpleActivator:
    """纯 cosine retrieval，零机制。

    activate 逻辑：
    1. Embed query
    2. 对所有 conclusion nodes 算 cosine similarity
    3. 取 top-k
    4. 如果 conclusion 不够，用 raw traces 补
    完毕。不做 topic matching，不做 boost，不做 critic。
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder

    def activate(
        self,
        graph: GraphStore,
        query: str,
        current_turn: int,
        top_k: int = 20,
        max_output_chars: int | None = None,
        l1_summary: str = "",
        **kwargs,  # 忽略 layered activator 的额外参数
    ) -> ActivateResult:
        query_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)

        # ── Step 1: Score ALL conclusion nodes by cosine similarity ──
        conclusions = [
            n for n in graph.nodes.values()
            if n.type == NodeType.CONCLUSION and n.embedding is not None
        ]

        scored = []
        for node in conclusions:
            node_vec = np.array(node.embedding, dtype=np.float32)
            sim = _cosine(query_vec, node_vec)
            scored.append((node, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        # ── Step 2: Take top-k conclusions ──
        lines = []

        # L1 summary: 只加一行，不占主要位置
        if l1_summary:
            lines.append(f"[global] {l1_summary}")

        conclusion_count = 0
        for node, sim in scored:
            if conclusion_count >= top_k:
                break
            if sim > 0.1:  # 只要不是完全不相关
                lines.append(f"[conclusion] {node.value}")
                conclusion_count += 1

        # ── Step 3: 如果 conclusion 不够，用 raw traces 补 ──
        remaining = top_k - conclusion_count
        if remaining > 0 and graph.raw_traces:
            raw_scored = graph.search_raw_traces(query_vec, top_k=remaining)
            for trace, sim in raw_scored:
                if sim > 0.1:
                    lines.append(f"[raw] {trace.content}")

        result = "\n".join(lines)

        # Truncate if needed
        if max_output_chars and len(result) > max_output_chars:
            result = result[:max_output_chars]
            last_nl = result.rfind("\n")
            if last_nl > 0:
                result = result[:last_nl]

        return ActivateResult(result)
