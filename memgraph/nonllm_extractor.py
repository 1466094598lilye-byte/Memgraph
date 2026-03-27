"""NonLLM 抽取器：NER + 句子切分 + 共现边，零 API。"""

import hashlib
import re
from typing import Optional

from memgraph.embedder import Embedder
from memgraph.models import Edge, Node, NodeType

# NER 实体类型 → ENTITY 节点
NER_LABELS = {"PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT", "LOC"}

# 句子切分（通用标点，无域特定规则）
SENTENCE_PATTERN = re.compile(r"[.!?。！？\n]+")
MIN_SENTENCE_LEN = 5  # 过滤过短片段


def _node_id(value: str, turn: int, prefix: str = "n") -> str:
    """生成唯一节点 id。"""
    h = hashlib.md5(f"{value}|{turn}|{prefix}".encode()).hexdigest()[:8]
    return f"{prefix}_{h}"


def _load_spacy():
    """懒加载 spaCy，下载模型若不存在。"""
    try:
        import spacy
        try:
            return spacy.load("xx_ent_wiki_sm")
        except OSError:
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "spacy", "download", "xx_ent_wiki_sm"], check=True)
            return spacy.load("xx_ent_wiki_sm")
    except Exception as e:
        raise RuntimeError(f"spaCy NER 加载失败，请运行: python3 -m spacy download xx_ent_wiki_sm. 错误: {e}") from e


def _extract_ner(text: str, turn: int, nlp) -> list[Node]:
    """从文本抽取 NER 实体 → ENTITY 节点。"""
    doc = nlp(text)
    nodes = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ not in NER_LABELS:
            continue
        val = ent.text.strip()
        if not val or len(val) < 2 or val in seen:
            continue
        seen.add(val)
        nid = _node_id(val, turn, "ent")
        nodes.append(Node(id=nid, type=NodeType.ENTITY, value=val, turn=turn))
    return nodes


def _extract_sentences(text: str, turn: int) -> list[Node]:
    """按句子切分 → STATE 节点（完整信息单元，无损）。"""
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_PATTERN.split(text)
    nodes = []
    seen = set()
    for p in parts:
        p = p.strip()
        if not p or len(p) < MIN_SENTENCE_LEN:
            continue
        if p.lower() in seen:
            continue
        seen.add(p.lower())
        nid = _node_id(p, turn, "sent")
        nodes.append(Node(id=nid, type=NodeType.STATE, value=p, turn=turn))
    return nodes


def _make_edges(nodes: list[Node], turn_to_nodes: dict[int, list[Node]], adj_weight: float = 0.5) -> list[Edge]:
    """
    同轮共现 → 连边 weight=1.0
    相邻轮 ±1 → 连边 weight=adj_weight
    """
    seen: set[tuple[str, str]] = set()
    edges: list[Edge] = []
    turns = sorted(turn_to_nodes.keys())

    def _add(src: str, dst: str, rel: str, w: float) -> None:
        if src == dst:
            return
        key = (min(src, dst), max(src, dst))
        if key in seen:
            return
        seen.add(key)
        edges.append(Edge(src_id=src, dst_id=dst, relation=rel, weight=w))

    for t in turns:
        tnodes = turn_to_nodes[t]
        for i, a in enumerate(tnodes):
            for b in tnodes[i + 1:]:
                _add(a.id, b.id, "cooccur", 1.0)

    for i, t in enumerate(turns):
        for a in turn_to_nodes[t]:
            for j in (i - 1, i + 1):
                if 0 <= j < len(turns):
                    t2 = turns[j]
                    for b in turn_to_nodes[t2]:
                        _add(a.id, b.id, "adjacent", adj_weight)

    return edges


class NonLLMExtractor:
    """纯本地抽取：NER + 句子切分 + 共现边，零 API。"""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        adj_edge_weight: float = 0.5,
    ) -> None:
        self._embedder = embedder or Embedder()
        self._nlp = None
        self.adj_edge_weight = adj_edge_weight

    def _ensure_nlp(self):
        if self._nlp is None:
            self._nlp = _load_spacy()
        return self._nlp

    def extract(
        self,
        conversation_chunk: list[dict],
        existing_topics: list[str] | None = None,
    ) -> tuple[list[Node], list[Edge], str, dict]:
        """
        从对话片段抽取节点、边、话题。
        返回 (nodes, edges, topic_label, usage)，usage 全为 0（无 API）。
        """
        nlp = self._ensure_nlp()
        all_nodes: list[Node] = []
        turn_to_nodes: dict[int, list[Node]] = {}
        full_text_parts: list[str] = []

        for i, msg in enumerate(conversation_chunk):
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            content = content.strip()
            if not content:
                continue
            turn = i
            full_text_parts.append(content)

            # NER
            ner_nodes = _extract_ner(content, turn, nlp)
            # 句子切分（完整信息单元）
            sent_nodes = _extract_sentences(content, turn)

            turn_nodes = []
            for n in ner_nodes + sent_nodes:
                if n.id not in {x.id for x in all_nodes}:
                    all_nodes.append(n)
                    turn_nodes.append(n)
            turn_to_nodes[turn] = turn_nodes

        edges = _make_edges(all_nodes, turn_to_nodes, self.adj_edge_weight)

        # 话题：用首句前 30 字概括（通用，无 YAKE 依赖）
        topic_label = "general"
        for part in full_text_parts:
            first_sent = SENTENCE_PATTERN.split(part.strip())[0].strip()
            if len(first_sent) >= MIN_SENTENCE_LEN:
                topic_label = first_sent[:30] if len(first_sent) > 30 else first_sent
                break

        # Embed 节点
        embedded_nodes = [self._embedder.embed_node(n) for n in all_nodes]

        usage = {"input_tokens": 0, "output_tokens": 0}
        return embedded_nodes, edges, topic_label, usage
