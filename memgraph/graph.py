"""状态图数据结构封装：三层记忆 + 晋升/衰减。"""

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)

from memgraph.models import Edge, Layer, Node, NodeType, RawTrace, StateGraph, Topic

EMBEDDING_DEDUP_THRESHOLD = 0.85

# Metadata-driven similarity (SAGE-inspired)
ENTITY_EDGE_MIN_OVERLAP = 1       # 至少共享 1 个 entity 才建边
TOPIC_EDGE_JACCARD_THRESHOLD = 0.3  # topic 词级 Jaccard > 0.3 才建边


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class GraphStore:
    """状态图存储：nodes 为 dict[id->Node]，支持晋升、衰减、邻居查询。"""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.topics: dict[str, Topic] = {}
        self.raw_traces: list[RawTrace] = []
        self.raw_traces_by_id: dict[str, RawTrace] = {}
        self.turn_count: int = 0
        self._topic_counter: int = 0
        self._raw_trace_counter: int = 0
        self.current_session_id: str | None = None
        self._session_start_time: float = time.time()

    def set_session(self, session_id: str) -> None:
        """设置当前会话 ID，并重置会话起始时间。"""
        logger.info("[graph] set_session: %s", session_id)
        self.current_session_id = session_id
        self._session_start_time = time.time()

    def add_node(self, node: Node) -> None:
        """添加节点，embedding 相似度去重（cosine > 0.85 且类型相同则合并）。"""
        if node.session_id is None:
            node.session_id = self.current_session_id
        if node.timestamp is None:
            node.timestamp = time.time()
        if node.embedding is not None:
            node_vec = np.array(node.embedding, dtype=np.float32)
            for existing in self.nodes.values():
                if existing.embedding is None or existing.type != node.type:
                    continue
                ex_vec = np.array(existing.embedding, dtype=np.float32)
                sim = _cosine(node_vec, ex_vec)
                if sim > EMBEDDING_DEDUP_THRESHOLD:
                    existing.importance = max(existing.importance, node.importance)
                    existing.turn = max(existing.turn, node.turn)
                    existing.session_id = existing.session_id or node.session_id
                    existing.timestamp = max(
                        existing.timestamp or 0.0,
                        node.timestamp or 0.0,
                    ) or existing.timestamp or node.timestamp
                    existing.merge_count += 1
                    if len(node.value) > len(existing.value):
                        existing.value = node.value
                        existing.embedding = node.embedding
                    return
        # fallback: 精确文本去重
        for existing in self.nodes.values():
            if existing.value == node.value and existing.type == node.type:
                existing.importance = max(existing.importance, node.importance)
                existing.turn = max(existing.turn, node.turn)
                existing.session_id = existing.session_id or node.session_id
                existing.timestamp = max(
                    existing.timestamp or 0.0,
                    node.timestamp or 0.0,
                ) or existing.timestamp or node.timestamp
                existing.merge_count += 1
                return
        self.nodes[node.id] = node
        # ── metadata-driven edge building (SAGE-inspired) ──
        self._build_metadata_edges(node)

    def _build_metadata_edges(self, new_node: Node) -> None:
        """SAGE-inspired: 基于 entity overlap 和 topic Jaccard 自动建边。

        对每个已有节点检查：
        1. Entity overlap: 共享 entity >= ENTITY_EDGE_MIN_OVERLAP → 建 metadata 边
        2. Topic similarity: topic_id 相同 或 value 词级 Jaccard > 阈值 → 建 topic 边
        不重复建边（同一对节点只建一次）。
        """
        if not new_node.entities and not new_node.topic_id:
            return

        new_entities = set(e.lower() for e in new_node.entities)
        existing_edge_pairs = set()
        for e in self.edges:
            existing_edge_pairs.add((e.src_id, e.dst_id))
            existing_edge_pairs.add((e.dst_id, e.src_id))

        for existing in self.nodes.values():
            if existing.id == new_node.id:
                continue
            if (new_node.id, existing.id) in existing_edge_pairs:
                continue

            # 1. Entity overlap
            if new_entities and existing.entities:
                existing_entities = set(e.lower() for e in existing.entities)
                shared = new_entities & existing_entities
                if len(shared) >= ENTITY_EDGE_MIN_OVERLAP:
                    self.edges.append(Edge(
                        src_id=new_node.id,
                        dst_id=existing.id,
                        relation=f"shared_entities:{','.join(list(shared)[:3])}",
                    ))
                    existing_edge_pairs.add((new_node.id, existing.id))
                    existing_edge_pairs.add((existing.id, new_node.id))
                    continue  # 已建边，不重复

            # 2. Topic-based edge
            if new_node.topic_id and existing.topic_id:
                if new_node.topic_id == existing.topic_id:
                    # 同 topic 不一定需要额外边（topic 内已有结构），跳过
                    continue

            # 3. Value-level Jaccard (跨 topic 但内容相关)
            if new_node.value and existing.value:
                import re as _re
                tokens_a = set(_re.findall(r'\b\w{3,}\b', new_node.value.lower()))
                tokens_b = set(_re.findall(r'\b\w{3,}\b', existing.value.lower()))
                if tokens_a and tokens_b:
                    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                    if jaccard > TOPIC_EDGE_JACCARD_THRESHOLD:
                        self.edges.append(Edge(
                            src_id=new_node.id,
                            dst_id=existing.id,
                            relation=f"content_similarity:{jaccard:.2f}",
                        ))
                        existing_edge_pairs.add((new_node.id, existing.id))
                        existing_edge_pairs.add((existing.id, new_node.id))

    def add_nodes(self, nodes: list[dict] | list[Node]) -> None:
        """批量添加节点。"""
        for n in nodes:
            node = Node(**n) if isinstance(n, dict) else n
            self.add_node(node)

    def add_edge(self, edge: Edge) -> None:
        """添加边。"""
        self.edges.append(edge)

    def add_edges(self, edges: list[dict] | list[Edge]) -> None:
        """批量添加边。"""
        for e in edges:
            edge = Edge(**e) if isinstance(e, dict) else e
            self.edges.append(edge)

    def add_raw_trace(self, trace: RawTrace) -> str:
        """存储未抽取的原始消息痕迹，自动生成 id。返回 trace.id。"""
        if trace.id is None:
            self._raw_trace_counter += 1
            trace.id = f"raw_{self._raw_trace_counter}"
        if trace.session_id is None:
            trace.session_id = self.current_session_id
        if trace.timestamp is None:
            trace.timestamp = time.time()
        self.raw_traces.append(trace)
        self.raw_traces_by_id[trace.id] = trace
        return trace.id

    def search_raw_traces(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> list[tuple[RawTrace, float]]:
        """按 embedding 相似度搜索 raw traces，返回 (trace, sim) 列表。"""
        scored: list[tuple[RawTrace, float]] = []
        for trace in self.raw_traces:
            if not trace.embedding:
                continue
            t_vec = np.array(trace.embedding, dtype=np.float32)
            sim = _cosine(query_vec, t_vec)
            scored.append((trace, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_neighbors(self, node_id: str) -> list[str]:
        """获取节点的 1 跳邻居 id 列表。"""
        neighbor_ids = set()
        for e in self.edges:
            if e.src_id == node_id:
                neighbor_ids.add(e.dst_id)
            elif e.dst_id == node_id:
                neighbor_ids.add(e.src_id)
        return list(neighbor_ids)

    def get_chain(self, session_id: str) -> list[Node]:
        """返回某个会话内按 turn 排序的节点链。"""
        chain = [n for n in self.nodes.values() if n.session_id == session_id]
        chain = sorted(chain, key=lambda n: (n.turn, n.timestamp or self._session_start_time))
        logger.debug("[graph] get_chain: session=%s → %d nodes", session_id, len(chain))
        return chain

    def get_temporal_neighbors(self, node_id: str, window: int = 5) -> list[Node]:
        """返回目标节点前后 window 轮内的节点。"""
        node = self.nodes.get(node_id)
        if node is None:
            return []

        neighbors = []
        for candidate in self.nodes.values():
            if candidate.id == node_id:
                continue
            if candidate.session_id != node.session_id:
                continue
            if abs(candidate.turn - node.turn) <= window:
                neighbors.append(candidate)
        neighbors.sort(key=lambda n: (n.turn, n.timestamp or self._session_start_time))
        return neighbors

    def node_degree(self, node_id: str) -> int:
        """节点的边数（图中心度，主线节点 degree 高）。"""
        return sum(
            1 for e in self.edges if e.src_id == node_id or e.dst_id == node_id
        )

    def promote_nodes(self, current_turn: int) -> None:
        """层级晋升：根据激活频率提升节点层级。"""
        for node in list(self.nodes.values()):
            if node.layer == Layer.MEMORY:
                if node.activation_count >= 3:
                    node.layer = Layer.ANCHOR
            elif node.layer == Layer.ANCHOR:
                edge_count = len(
                    [e for e in self.edges if e.src_id == node.id or e.dst_id == node.id]
                )
                if node.activation_count >= 5 and edge_count >= 2:
                    node.layer = Layer.STABLE

    def decay_nodes(self, current_turn: int) -> None:
        """衰减：长期未激活的节点降级或降低 importance。"""
        for node in self.nodes.values():
            turns_since = current_turn - node.last_activated_turn
            if node.layer == Layer.MEMORY and turns_since > 5:
                node.importance *= 0.5
            elif node.layer == Layer.ANCHOR and turns_since > 20:
                node.layer = Layer.MEMORY

    def compute_importance(self, node: Node, current_turn: int) -> float:
        """计算节点综合重要性分。"""
        recency = 1.0 / (1 + current_turn - node.last_activated_turn)
        frequency = node.activation_count
        edge_count = len(
            [e for e in self.edges if e.src_id == node.id or e.dst_id == node.id]
        )
        return frequency * 0.5 + recency * 0.3 + edge_count * 0.2

    # ── 话题管理 ──

    def assign_topic(
        self,
        topic_label: str,
        node_ids: list[str],
        current_turn: int,
        embedder_fn=None,
    ) -> str:
        """
        将节点挂载到话题下。
        话题 embedding 使用节点质心（非 label），匹配时用质心相似度。
        若 topic 下无节点，fallback 到 label embedding。
        """
        topic_label = topic_label.strip()
        if not topic_label:
            topic_label = "未分类"

        # 新节点的 embedding 质心（用于匹配和更新）
        new_vecs = [
            self.nodes[nid].embedding
            for nid in node_ids
            if nid in self.nodes and self.nodes[nid].embedding is not None
        ]
        new_centroid = np.mean(new_vecs, axis=0) if new_vecs else None

        matched_topic: Topic | None = None

        if new_centroid is not None and self.topics:
            best_sim, best_topic = -1.0, None
            for t in self.topics.values():
                if t.embedding is None:
                    continue
                t_vec = np.array(t.embedding, dtype=np.float32)
                sim = _cosine(np.array(new_centroid, dtype=np.float32), t_vec)
                if sim > best_sim:
                    best_sim, best_topic = sim, t
            if best_sim > 0.7 and best_topic is not None:
                matched_topic = best_topic

        if matched_topic is None:
            self._topic_counter += 1
            tid = f"topic_{self._topic_counter}"
            matched_topic = Topic(id=tid, label=topic_label)
            self.topics[tid] = matched_topic

        matched_topic.node_count += len(node_ids)
        matched_topic.last_turn = current_turn
        if current_turn not in matched_topic.revisit_turns:
            matched_topic.revisit_turns.append(current_turn)

        for nid in node_ids:
            if nid in self.nodes:
                self.nodes[nid].topic_id = matched_topic.id

        # 用该 topic 下所有节点的质心更新 topic.embedding
        self._update_topic_centroid(matched_topic, embedder_fn, topic_label)

        return matched_topic.id

    def _update_topic_centroid(
        self, topic: Topic, embedder_fn=None, fallback_label: str = ""
    ) -> None:
        """用 topic 下所有节点的 embedding 质心更新 topic.embedding。"""
        nodes = self.get_nodes_by_topic(topic.id)
        vecs = [n.embedding for n in nodes if n.embedding is not None]
        if vecs:
            centroid = np.mean(vecs, axis=0)
            topic.embedding = centroid.tolist()
        elif embedder_fn and fallback_label:
            topic.embedding = embedder_fn(fallback_label)

    def get_topic_labels(self) -> list[str]:
        """返回所有已有话题的标签列表。"""
        return [t.label for t in self.topics.values()]

    def get_nodes_by_topic(self, topic_id: str) -> list[Node]:
        """获取某话题下的所有节点。"""
        return [n for n in self.nodes.values() if n.topic_id == topic_id]

    # ── 状态管理 ──

    def set_turn_count(self, count: int) -> None:
        """设置对话轮数。"""
        self.turn_count = count

    def get_graph(self) -> StateGraph:
        """获取 StateGraph 兼容视图（用于向后兼容）。"""
        return StateGraph(
            nodes=list(self.nodes.values()),
            edges=self.edges,
            topics=list(self.topics.values()),
            turn_count=self.turn_count,
        )

    def inspect(self) -> dict:
        """返回可直接序列化的状态图 dict。"""
        return {
            "nodes": [n.model_dump() for n in self.nodes.values()],
            "edges": [e.model_dump() for e in self.edges],
            "topics": [t.model_dump() for t in self.topics.values()],
            "turn_count": self.turn_count,
        }
