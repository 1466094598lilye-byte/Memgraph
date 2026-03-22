"""Embedding 层：sentence-transformers 向量化。"""

import os
from pathlib import Path

from memgraph.models import Node

# 缓存到项目目录，避免权限问题
_HF_CACHE = Path(__file__).resolve().parent.parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE))


class Embedder:
    """使用 paraphrase-multilingual-MiniLM-L12-v2 做节点向量化。"""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
        self.model = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            device=device,
        )

    def embed_node(self, node: Node) -> Node:
        """序列化时带上类型信息，提升结构感知。"""
        text = f"{node.type.value}: {node.value}"
        node.embedding = self.model.encode(text).tolist()
        return node

    def embed_query(self, query: str) -> list[float]:
        """查询向量化。"""
        return self.model.encode(query).tolist()
