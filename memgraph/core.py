"""MemGraph 主入口：encode、activate、inspect。"""

import hashlib
import logging
import re
from typing import Callable

import numpy as np

from memgraph.activator import Activator
from memgraph.simple_activator import SimpleActivator
from memgraph.compressor import Compressor
from memgraph.config import MemGraphConfig
from memgraph.embedder import Embedder
from memgraph.graph import GraphStore
from memgraph.models import ActivateResult, Edge, Node, NodeType, RawTrace

logger = logging.getLogger(__name__)

# 宿主 agent 的 LLM 调用：(prompt, max_tokens) -> (response_text, usage_dict)
LLMFn = Callable[[str, int], tuple[str, dict]]

# set_working_memory 未传入 thread_id 时不改线程状态
_UNSET_THREAD = object()

L1_UPDATE_THRESHOLD = 5  # 每累积 N 条 L2 结论触发一次 L1 重压缩


def _strip_code_and_tools(text: str) -> str:
    """过滤代码块和 tool call，只保留自然语言部分。"""
    import json as _json

    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = _json.loads(stripped)
            if isinstance(parsed, list):
                text_parts = []
                for block in parsed:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    text = "\n".join(text_parts)
                else:
                    return ""
            elif isinstance(parsed, dict):
                if parsed.get("type") == "text":
                    text = parsed.get("text", "")
                elif parsed.get("type") in ("tool_use", "tool_result", "tool_call"):
                    return ""
        except (_json.JSONDecodeError, TypeError):
            pass

    text = re.sub(r"```[\s\S]*?```", "", text)
    for tag in ("tool_call", "tool_use", "function_calls", "antml:function_calls",
                "antml:invoke", "tool_result", "function_results"):
        pattern = rf"<{re.escape(tag)}[\s\S]*?</{re.escape(tag)}>"
        text = re.sub(pattern, "", text)
    text = re.sub(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:[\s\S]*?\}\s*\}', "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _conclusion_id(value: str, turn: int, index: int) -> str:
    h = hashlib.md5(f"{value}|{turn}|{index}".encode()).hexdigest()[:8]
    return f"concl_{h}"


def _create_extractor():
    """NonLLM 抽取器（压缩失败时的 fallback）。"""
    from memgraph.nonllm_extractor import NonLLMExtractor
    return NonLLMExtractor()


class MemGraph:
    """对话历史编码插件：分层压缩 + 抑制式检索。

    L0 = 工作记忆（当前目标、步骤、约束，始终置顶注入）
    L1 = 全局摘要（所有 L2 再压缩，1 条）
    L2 = 话题结论（每个聚类压缩为 1-3 条 CONCLUSION 节点）
    L3 = 原始消息 embedding（raw traces，兜底检索）
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4o-mini",
        llm_fn: LLMFn | None = None,
        *,
        activator_mode: str = "layered",  # "layered" or "simple"
        graph_expand_seed_k: int | None = None,
        graph_expand_neighbor_k: int | None = None,
        graph_expand_dense_weight: float | None = None,
    ) -> None:
        cfg_kw: dict = dict(llm_provider=llm_provider, model=model)
        if graph_expand_seed_k is not None:
            cfg_kw["graph_expand_seed_k"] = graph_expand_seed_k
        if graph_expand_neighbor_k is not None:
            cfg_kw["graph_expand_neighbor_k"] = graph_expand_neighbor_k
        if graph_expand_dense_weight is not None:
            cfg_kw["graph_expand_dense_weight"] = graph_expand_dense_weight
        self.config = MemGraphConfig(**cfg_kw)
        self._graph = GraphStore()
        self._compressor = Compressor(
            llm_provider=llm_provider,
            model=model,
            llm_fn=llm_fn,
        )
        self._extractor = _create_extractor()
        self._embedder = Embedder()
        if activator_mode == "simple":
            self._activator = SimpleActivator(self._embedder)
        else:
            self._activator = Activator(
                self._embedder,
                graph_expand_seed_k=self.config.graph_expand_seed_k,
                graph_expand_neighbor_k=self.config.graph_expand_neighbor_k,
                graph_expand_dense_weight=self.config.graph_expand_dense_weight,
                activation_seed_k=self.config.activation_seed_k,
                activation_max_hops=self.config.activation_max_hops,
                activation_decay=self.config.activation_decay,
                inhibition_top_n=self.config.inhibition_top_n,
                inhibition_min_ratio=self.config.inhibition_min_ratio,
                focus_recent_turn_window=self.config.focus_recent_turn_window,
                focus_out_decay=self.config.focus_out_decay,
                focus_broaden_threshold=self.config.focus_broaden_threshold,
                l1_only_when_broaden=self.config.l1_only_when_broaden,
            )

        self._cluster: list[dict] = []
        self._cluster_vecs: list[np.ndarray] = []
        self._total_turns: int = 0
        self._extract_count: int = 0
        self._encode_usage: dict = {"input_tokens": 0, "output_tokens": 0}

        # L0 工作记忆
        self._l0_goal: str = ""
        self._l0_step: str = ""
        self._l0_constraints: list[str] = []
        self._l0_thread_id: str | None = None
        self._l0_thread_anchor_turn: int = 0

        # 检索焦点（最近闭合 cluster 的结论与 raw；纯结构）
        self._focus_cluster_node_ids: list[str] = []
        self._focus_cluster_raw_ids: list[str] = []

        # L1 层
        self._l1_summary: str = ""
        self._l2_count_since_l1: int = 0

        # Profile fact pool（跨话题常驻用户状态，同时存为 PROFILE 图节点）
        self._profile_facts: list[dict] = []
        self._max_profile_facts: int = 7
        self._profile_counter: int = 0

        # 时序边跟踪
        self._last_cluster_node_ids: list[str] = []

    @property
    def extract_count(self) -> int:
        return self._extract_count

    @property
    def encode_usage(self) -> dict:
        return dict(self._encode_usage)

    # ── L0 工作记忆 ──

    def set_working_memory(
        self,
        goal: str,
        step: str | None = None,
        constraints: list[str] | None = None,
        *,
        thread_id: str | None | object = _UNSET_THREAD,
    ) -> None:
        """设置工作记忆：当前目标、步骤、约束。可选 thread_id 锚定「自何时起」的长时检索范围。

        仅当显式传入 thread_id=... 时更新线程；传入 None 表示清除线程锚点。
        thread_id 变更时，将 anchor_turn 设为当前总轮次（结构信号，非领域分类）。
        """
        if thread_id is not _UNSET_THREAD:
            if thread_id != self._l0_thread_id:
                self._l0_thread_anchor_turn = self._total_turns
            self._l0_thread_id = thread_id  # type: ignore[assignment]
        self._l0_goal = goal
        self._l0_step = step or ""
        self._l0_constraints = constraints or []

    def clear_working_memory(self) -> None:
        """清空工作记忆。"""
        self._l0_goal = ""
        self._l0_step = ""
        self._l0_constraints = []
        self._l0_thread_id = None
        self._l0_thread_anchor_turn = 0

    # ── 增量编码 ──

    def add_turn(self, message: dict) -> None:
        """增量编码：embedding 话题聚类，话题切换时关闭聚类并压缩。"""
        content = message.get("content", "")
        # 过滤代码块和 tool call
        content = _strip_code_and_tools(content)
        message = {**message, "content": content}
        msg_vec = np.array(
            self._embedder.embed_query(content),
            dtype=np.float32,
        )
        self._total_turns += 1

        should_close = False
        if self._cluster_vecs:
            centroid = np.mean(self._cluster_vecs, axis=0)
            sim = _cosine(centroid, msg_vec)
            if sim < self.config.topic_shift_threshold:
                should_close = True

        if len(self._cluster) >= self.config.max_cluster_size:
            should_close = True

        if should_close:
            self._close_cluster()

        self._cluster.append(message)
        self._cluster_vecs.append(msg_vec)

    def add_turns(self, messages: list[dict]) -> None:
        for msg in messages:
            self.add_turn(msg)

    # ── 聚类关闭 ──

    def _close_cluster(self) -> None:
        """关闭当前话题聚类：优先 LLM 压缩，失败则降级为 NonLLM 抽取。"""
        if not self._cluster:
            return

        current_node_ids: list[str] = []
        focus_raw_ids: list[str] = []

        if len(self._cluster) >= self.config.min_cluster_size:
            try:
                _, current_node_ids, focus_raw_ids = self._compress_cluster()
            except Exception as e:
                logger.warning("L2 compression failed (%s), falling back to NonLLM extractor", e)
                _, current_node_ids, focus_raw_ids = self._extract_cluster_fallback()
        else:
            focus_raw_ids = self._store_as_raw_traces()
            current_node_ids = []

        self._focus_cluster_node_ids = list(current_node_ids)
        self._focus_cluster_raw_ids = list(focus_raw_ids)

        # 时序边：相邻 cluster 间无条件连接
        if self._last_cluster_node_ids and current_node_ids:
            self._create_temporal_edges(
                self._last_cluster_node_ids,
                current_node_ids,
            )

        self._last_cluster_node_ids = current_node_ids

        self._cluster.clear()
        self._cluster_vecs.clear()

    def _create_temporal_edges(
        self,
        prev_node_ids: list[str],
        curr_node_ids: list[str],
    ) -> None:
        """相邻 cluster 间的时序边：纯结构信号，不看 topic label。"""
        if not prev_node_ids or not curr_node_ids:
            return

        for prev_id in prev_node_ids[-2:]:
            for curr_id in curr_node_ids[:2]:
                self._graph.add_edge(Edge(
                    src_id=prev_id,
                    dst_id=curr_id,
                    relation="temporal_sequence",
                    weight=0.4,
                ))

    def _compress_cluster(self) -> tuple[str, list[str], list[str]]:
        """L2 压缩：LLM 将聚类压缩为 1-3 条 CONCLUSION 节点。
        返回 (topic_label, node_ids, raw_trace_ids)。
        """
        existing_topics = self._graph.get_topic_labels()
        result = self._compressor.compress_cluster(self._cluster, existing_topics)

        self._extract_count += 1
        self._encode_usage["input_tokens"] += result.usage.get("input_tokens", 0)
        self._encode_usage["output_tokens"] += result.usage.get("output_tokens", 0)

        node_ids = []
        for i, conclusion in enumerate(result.conclusions):
            nid = _conclusion_id(conclusion, self._total_turns, i)
            node = Node(
                id=nid,
                type=NodeType.CONCLUSION,
                value=conclusion,
                turn=self._total_turns,
            )
            node = self._embedder.embed_node(node)
            self._graph.add_node(node)
            node_ids.append(nid)

        # 将边的占位符 ID (c0, c1, ...) 替换为实际节点 ID
        if result.edges and node_ids:
            for edge in result.edges:
                # edge.src_id 格式为 "c数字"，转换为实际节点 ID
                if edge.src_id.startswith("c"):
                    idx = int(edge.src_id[1:])
                    if 0 <= idx < len(node_ids):
                        edge.src_id = node_ids[idx]
                if edge.dst_id.startswith("c"):
                    idx = int(edge.dst_id[1:])
                    if 0 <= idx < len(node_ids):
                        edge.dst_id = node_ids[idx]
            self._graph.add_edges(result.edges)

        topic_id = None
        if node_ids:
            topic_id = self._graph.assign_topic(
                result.topic_label,
                node_ids,
                current_turn=self._total_turns,
                embedder_fn=self._embedder.embed_query,
            )

        raw_trace_ids = self._store_raw_traces_with_topic(topic_id)
        self._graph.set_turn_count(self._total_turns)

        # derived_from 边：结论 ↔ 它来源的原始消息
        for cid in node_ids:
            for rtid in raw_trace_ids:
                self._graph.add_edge(Edge(
                    src_id=cid, dst_id=rtid,
                    relation="derived_from", weight=0.7,
                ))

        # sibling 边：同 cluster 产出的结论互相连接
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                self._graph.add_edge(Edge(
                    src_id=node_ids[i],
                    dst_id=node_ids[j],
                    relation="sibling",
                    weight=0.8,
                ))

        self._l2_count_since_l1 += len(result.conclusions)
        if self._l2_count_since_l1 >= L1_UPDATE_THRESHOLD:
            self._update_l1()

        self._update_profile(result.conclusions, conclusion_node_ids=node_ids)

        return result.topic_label, node_ids, raw_trace_ids

    def _extract_cluster_fallback(self) -> tuple[str, list[str], list[str]]:
        """NonLLM fallback：NER + 句子切分（零 API）。"""
        existing_topics = self._graph.get_topic_labels()
        new_nodes, new_edges, topic_label, usage = self._extractor.extract(
            self._cluster, existing_topics
        )
        self._extract_count += 1
        self._encode_usage["input_tokens"] += usage.get("input_tokens", 0)
        self._encode_usage["output_tokens"] += usage.get("output_tokens", 0)

        node_ids = []
        for node in new_nodes:
            if node.embedding is None:
                node = self._embedder.embed_node(node)
            self._graph.add_node(node)
            node_ids.append(node.id)
        self._graph.add_edges(new_edges)

        # sibling 边：同 cluster 产出的节点互相连接
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                self._graph.add_edge(Edge(
                    src_id=node_ids[i],
                    dst_id=node_ids[j],
                    relation="sibling",
                    weight=0.8,
                ))

        if node_ids:
            self._graph.assign_topic(
                topic_label,
                node_ids,
                current_turn=self._total_turns,
                embedder_fn=self._embedder.embed_query,
            )

        raw_trace_ids = self._store_raw_traces_with_topic(None)
        self._graph.set_turn_count(self._total_turns)

        return topic_label, node_ids, raw_trace_ids

    def _store_raw_traces_with_topic(self, topic_id: str | None) -> list[str]:
        """将当前聚类消息存为 L3 raw traces。返回 trace ID 列表。"""
        trace_ids: list[str] = []
        for msg, vec in zip(self._cluster, self._cluster_vecs):
            tid = self._graph.add_raw_trace(
                RawTrace(
                    content=msg.get("content", ""),
                    embedding=vec.tolist(),
                    turn=self._total_turns,
                    density=0.0,
                    topic_id=topic_id,
                )
            )
            trace_ids.append(tid)
        return trace_ids

    def _store_as_raw_traces(self) -> list[str]:
        """小聚类：只存 raw traces，不压缩。返回 trace id 列表。"""
        return self._store_raw_traces_with_topic(None)

    # ── L1 层：全局摘要 ──

    def _update_l1(self) -> None:
        """将所有 L2 结论按话题分组，压缩为一条全局摘要。"""
        conclusion_nodes = [
            n for n in self._graph.nodes.values()
            if n.type == NodeType.CONCLUSION
        ]
        if not conclusion_nodes:
            return

        l2_by_topic: dict[str, list[str]] = {}
        for node in conclusion_nodes:
            topic = self._graph.topics.get(node.topic_id) if node.topic_id else None
            label = topic.label if topic else "general"
            l2_by_topic.setdefault(label, []).append(node.value)

        try:
            summary, usage = self._compressor.compress_l1(l2_by_topic)
            self._l1_summary = summary
            self._encode_usage["input_tokens"] += usage.get("input_tokens", 0)
            self._encode_usage["output_tokens"] += usage.get("output_tokens", 0)
            self._l2_count_since_l1 = 0
        except Exception as e:
            logger.warning("L1 compression failed (%s), keeping existing summary", e)

    # ── Profile fact pool ──

    def _update_profile(
        self,
        new_conclusions: list[str],
        conclusion_node_ids: list[str] | None = None,
    ) -> None:
        """用 L2 结论更新 profile fact pool，同时存为 PROFILE 图节点。"""
        if not new_conclusions:
            return
        try:
            condensed, condense_usage = self._compressor.condense_for_profile(new_conclusions)
            self._encode_usage["input_tokens"] += condense_usage.get("input_tokens", 0)
            self._encode_usage["output_tokens"] += condense_usage.get("output_tokens", 0)

            if not condensed:
                return

            DEDUP_THRESHOLD = 0.85
            condensed_vec = self._embedder.embed_query(condensed)
            best_sim = 0.0
            best_idx = -1
            for i, fact in enumerate(self._profile_facts):
                fact_vec = fact.get("_vec")
                if fact_vec is None:
                    fact_vec = self._embedder.embed_query(fact["text"])
                    fact["_vec"] = fact_vec
                sim = _cosine(
                    np.array(condensed_vec, dtype=np.float32),
                    np.array(fact_vec, dtype=np.float32),
                )
                if sim > best_sim:
                    best_sim = sim
                    best_idx = i

            if best_sim >= DEDUP_THRESHOLD and best_idx >= 0:
                old_text = self._profile_facts[best_idx]["text"]
                self._profile_facts[best_idx]["text"] = condensed
                self._profile_facts[best_idx]["last_updated_turn"] = self._total_turns
                self._profile_facts[best_idx]["_vec"] = condensed_vec
                # 更新对应的 PROFILE 图节点
                nid = self._profile_facts[best_idx].get("node_id")
                if nid and nid in self._graph.nodes:
                    node = self._graph.nodes[nid]
                    node.value = condensed
                    node.embedding = condensed_vec if isinstance(condensed_vec, list) else list(condensed_vec)
                    node.turn = self._total_turns
                logger.debug("Profile updated (sim=%.2f): '%s' → '%s'", best_sim, old_text[:50], condensed[:50])
            else:
                # 新 fact → 创建 PROFILE 图节点
                self._profile_counter += 1
                nid = f"prof_{self._profile_counter}"
                vec_list = condensed_vec if isinstance(condensed_vec, list) else list(condensed_vec)
                prof_node = Node(
                    id=nid, type=NodeType.PROFILE,
                    value=condensed, turn=self._total_turns,
                    embedding=vec_list,
                )
                self._graph.add_node(prof_node)
                self._profile_facts.append({
                    "text": condensed,
                    "last_updated_turn": self._total_turns,
                    "_vec": condensed_vec,
                    "node_id": nid,
                })
                # 连接到来源结论
                if conclusion_node_ids:
                    for cid in conclusion_node_ids:
                        self._graph.add_edge(Edge(
                            src_id=nid, dst_id=cid,
                            relation="derived_from", weight=0.6,
                        ))
                logger.debug("Profile added as node %s: '%s'", nid, condensed[:50])

            # 超出上限则淘汰最旧的
            while len(self._profile_facts) > self._max_profile_facts:
                oldest_idx = min(
                    range(len(self._profile_facts)),
                    key=lambda i: self._profile_facts[i]["last_updated_turn"],
                )
                evicted = self._profile_facts.pop(oldest_idx)
                evicted_nid = evicted.get("node_id")
                if evicted_nid and evicted_nid in self._graph.nodes:
                    del self._graph.nodes[evicted_nid]
                logger.debug("Profile evicted: %s (turn %d)", evicted["text"][:50], evicted["last_updated_turn"])

        except Exception as e:
            logger.warning("Profile update failed (%s), keeping existing profile", e)

    @property
    def profile_card(self) -> str:
        """当前 user profile card（渲染为文本，不按 query 过滤；调试/兼容用）。"""
        if not self._profile_facts:
            return ""
        return "\n".join(f"- {f['text']}" for f in self._profile_facts)

    # ── 批量编码 ──

    def encode(self, conversation: list[dict]) -> None:
        """批量编码（向后兼容）。"""
        self._cluster.clear()
        self._cluster_vecs.clear()
        self._total_turns = 0
        self._extract_count = 0
        self._encode_usage = {"input_tokens": 0, "output_tokens": 0}
        self._graph.nodes.clear()
        self._graph.edges.clear()
        self._graph.topics.clear()
        self._graph.raw_traces.clear()
        self._graph.raw_traces_by_id.clear()
        self._graph._topic_counter = 0
        self._graph._raw_trace_counter = 0
        self._graph.turn_count = 0
        self._l1_summary = ""
        self._l2_count_since_l1 = 0
        self._profile_facts = []
        self._profile_counter = 0
        self._focus_cluster_node_ids = []
        self._focus_cluster_raw_ids = []
        self._l0_thread_id = None
        self._l0_thread_anchor_turn = 0

        for msg in conversation:
            self.add_turn(msg)

        self._close_cluster()
        self._update_l1()

    # ── 推理时激活 ──

    def activate(
        self,
        query: str,
        top_k: int = 10,
        max_output_chars: int | None = None,
        *,
        graph_expand_seed_k: int | None = None,
        graph_expand_neighbor_k: int | None = None,
        graph_expand_dense_weight: float | None = None,
    ) -> ActivateResult:
        """推理时激活：扩散激活 + 竞争抑制。

        Profile facts 已作为 PROFILE 图节点参与竞争，不再单独注入。
        L0 工作记忆（若已设置）始终置顶注入。
        """
        self._close_cluster()
        result = self._activator.activate(
            self._graph,
            query,
            self._total_turns,
            top_k=top_k,
            max_output_chars=max_output_chars,
            l1_summary=self._l1_summary,
            graph_expand_seed_k=graph_expand_seed_k,
            graph_expand_neighbor_k=graph_expand_neighbor_k,
            graph_expand_dense_weight=graph_expand_dense_weight,
            focus_cluster_node_ids=self._focus_cluster_node_ids,
            focus_cluster_raw_ids=self._focus_cluster_raw_ids,
            l0_thread_id=self._l0_thread_id,
            l0_thread_anchor_turn=self._l0_thread_anchor_turn,
        )

        if self._l0_goal:
            l0_lines = [f"[goal] {self._l0_goal}"]
            if self._l0_thread_id:
                l0_lines.append(f"[thread] {self._l0_thread_id}")
            if self._l0_step:
                l0_lines.append(f"[step] {self._l0_step}")
            if self._l0_constraints:
                l0_lines.append(f"[constraints] {'; '.join(self._l0_constraints)}")
            l0_block = "\n".join(l0_lines)
            combined = l0_block + "\n" + str(result)
            return ActivateResult(
                combined,
                ambiguous=result.ambiguous,
                candidates=result.candidates,
                hint=result.hint,
            )
        return result

    def inspect(self) -> dict:
        return self._graph.inspect()
