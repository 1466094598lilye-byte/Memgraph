"""推理时激活：扩散激活 + 竞争抑制。

核心原则：无显式话题分类；焦点衰减与扩圈见下。
  1. 焦点 = 最近 cluster ∪ 同 session 近期 turn（或 L0 thread 锚）；焦点外乘 decay，弱匹配或全图更优则扩圈
  2. 沿图扩散 + 竞争抑制；L1 / L0 策略见 core

旧架构（向后兼容）：
  无 CONCLUSION 节点时走 _activate_legacy。
"""

import logging
import math
import re

import numpy as np

logger = logging.getLogger(__name__)

from memgraph.critic import CriticSignals, CriticVerdict, evaluate as critic_evaluate
from memgraph.embedder import Embedder
from memgraph.graph import GraphStore
from memgraph.models import ActivateResult, AmbiguityCandidate, Layer, NodeType, Topic

SEED_COUNT = 5

# 时间回溯意图检测
_TEMPORAL_STRONG_RE = re.compile(
    r"上次|当时|那时候|那时|那会儿|之前说过|之前提到过|之前讨论|以前说|以前讨论|"
    r"last\s+time|previously|when\s+did\s+(?:we|you|i)|back\s+then|at\s+that\s+time|"
    r"earlier\s+(?:today|this\s+week|we\s+(?:said|discussed|talked))",
    re.IGNORECASE,
)
_TEMPORAL_WEAK_RE = re.compile(
    r"之前|以前|最近|before|recently|earlier",
    re.IGNORECASE,
)
_TEMPORAL_SPATIAL_EXCLUDE_RE = re.compile(
    r"最近的.{0,4}(?:店|餐厅|地方|位置|站|路|距离|公里|米)|"
    r"之前的.{0,2}(?:版本|方案|方法|代码|实现)|"
    r"before\s+(?:you|we|i)\s+(?:start|begin|proceed|go)|"
    r"recently\s+(?:opened|built|created|released)\s",
    re.IGNORECASE,
)

# 旧路径阈值（向后兼容）
TOPIC_MATCH_THRESHOLD = 0.3
TOPIC_BROADEN_THRESHOLD = 0.2
TEMPORAL_BOOST = 1.4


def _is_temporal_query(query: str) -> bool:
    if _TEMPORAL_STRONG_RE.search(query):
        return True
    if _TEMPORAL_WEAK_RE.search(query):
        if _TEMPORAL_SPATIAL_EXCLUDE_RE.search(query):
            return False
        return True
    return False


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


_SPARSE_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)


def _tokens_sparse(text: str) -> set[str]:
    return {t for t in _SPARSE_TOKEN_RE.findall(text.lower()) if len(t) >= 2}


def _sparse_dice(query: str, text: str) -> float:
    q = _tokens_sparse(query)
    t = _tokens_sparse(text)
    if not q or not t:
        return 0.0
    inter = len(q & t)
    return 2.0 * inter / (len(q) + len(t))


def _hybrid_score(
    dense_sim: float, sparse_sim: float, dense_weight: float = 0.65,
) -> float:
    return dense_weight * dense_sim + (1.0 - dense_weight) * sparse_sim


def _topic_relative_centrality(
    graph: GraphStore,
    node_id: str,
    topic_max_cache: dict[str, float] | None = None,
) -> float:
    node = graph.nodes.get(node_id)
    if not node or not node.topic_id:
        return 1.0
    topic_nodes = [n for n in graph.nodes.values() if n.topic_id == node.topic_id]
    if len(topic_nodes) <= 1:
        return 1.0
    tid = node.topic_id
    if topic_max_cache is not None and tid in topic_max_cache:
        max_d = topic_max_cache[tid]
    else:
        degrees = [graph.node_degree(n.id) for n in topic_nodes]
        max_d = max(degrees) if degrees else 1
        max_d = max(max_d, 1)
        if topic_max_cache is not None:
            topic_max_cache[tid] = float(max_d)
    d = graph.node_degree(node_id)
    return (d + 1) / (max_d + 1)


def _build_focus_id_set(
    graph: GraphStore,
    all_nodes: list,
    current_turn: int,
    window: int,
    cluster_node_ids: list[str],
    cluster_raw_ids: list[str],
    l0_thread_id: str | None,
    l0_thread_anchor_turn: int,
) -> set[str]:
    """结构焦点 id 集合（无领域标签）。"""
    fs: set[str] = set(cluster_node_ids) | {r for r in cluster_raw_ids if r}
    if l0_thread_id:
        for n in all_nodes:
            if n.turn >= l0_thread_anchor_turn:
                fs.add(n.id)
        for t in graph.raw_traces:
            if t.id and t.turn >= l0_thread_anchor_turn:
                fs.add(t.id)
        return fs
    lo = max(0, current_turn - window)
    sid = graph.current_session_id
    for n in all_nodes:
        if n.turn < lo:
            continue
        if sid is None or n.session_id is None or n.session_id == sid:
            fs.add(n.id)
    for t in graph.raw_traces:
        if not t.id or t.turn < lo:
            continue
        if sid is None or t.session_id is None or t.session_id == sid:
            fs.add(t.id)
    return fs


def _truncate(result: str, max_output_chars: int | None) -> str:
    if max_output_chars and max_output_chars > 0 and len(result) > max_output_chars:
        truncated = result[:max_output_chars]
        last_nl = truncated.rfind("\n")
        new_text = truncated[: last_nl + 1] if last_nl >= 0 else truncated
        if isinstance(result, ActivateResult):
            return ActivateResult(
                new_text,
                ambiguous=result.ambiguous,
                candidates=result.candidates,
                hint=result.hint,
            )
        return new_text
    return result


class Activator:
    """扩散激活检索：CONCLUSION 存在时走扩散路径，否则走旧路径（向后兼容）。"""

    def __init__(
        self,
        embedder: Embedder,
        *,
        graph_expand_seed_k: int = 5,
        graph_expand_neighbor_k: int = 8,
        graph_expand_dense_weight: float = 0.65,
        activation_seed_k: int = 10,
        activation_max_hops: int = 2,
        activation_decay: float = 0.5,
        inhibition_top_n: int = 15,
        inhibition_min_ratio: float = 0.2,
        focus_recent_turn_window: int = 20,
        focus_out_decay: float = 0.22,
        focus_broaden_threshold: float = 0.18,
        l1_only_when_broaden: bool = False,
    ) -> None:
        self.embedder = embedder
        # 旧路径参数（向后兼容）
        self.graph_expand_seed_k = graph_expand_seed_k
        self.graph_expand_neighbor_k = graph_expand_neighbor_k
        self.graph_expand_dense_weight = graph_expand_dense_weight
        # 新路径参数
        self.activation_seed_k = activation_seed_k
        self.activation_max_hops = activation_max_hops
        self.activation_decay = activation_decay
        self.inhibition_top_n = inhibition_top_n
        self.inhibition_min_ratio = inhibition_min_ratio
        self.focus_recent_turn_window = focus_recent_turn_window
        self.focus_out_decay = focus_out_decay
        self.focus_broaden_threshold = focus_broaden_threshold
        self.l1_only_when_broaden = l1_only_when_broaden

    def activate(
        self,
        graph: GraphStore,
        query: str,
        current_turn: int,
        top_k: int = 20,
        max_output_chars: int | None = None,
        l1_summary: str = "",
        *,
        graph_expand_seed_k: int | None = None,
        graph_expand_neighbor_k: int | None = None,
        graph_expand_dense_weight: float | None = None,
        focus_cluster_node_ids: list[str] | None = None,
        focus_cluster_raw_ids: list[str] | None = None,
        l0_thread_id: str | None = None,
        l0_thread_anchor_turn: int = 0,
    ) -> ActivateResult:
        nodes = list(graph.nodes.values())
        query_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)

        has_conclusions = any(n.type == NodeType.CONCLUSION for n in nodes)

        if not nodes:
            if graph.raw_traces:
                raw_results = graph.search_raw_traces(query_vec, top_k=top_k)
                text = "\n".join(
                    f"[raw] {t.content}" for t, sim in raw_results if sim > 0.1
                )
                return _truncate(ActivateResult(text), max_output_chars)
            return ActivateResult("")

        if has_conclusions:
            result = self._activate_layered(
                graph,
                query,
                query_vec,
                nodes,
                current_turn,
                top_k,
                l1_summary,
                focus_cluster_node_ids=focus_cluster_node_ids or [],
                focus_cluster_raw_ids=focus_cluster_raw_ids or [],
                l0_thread_id=l0_thread_id,
                l0_thread_anchor_turn=l0_thread_anchor_turn,
            )
        else:
            result = self._activate_legacy(
                graph, query, query_vec, nodes, current_turn, top_k,
            )

        return _truncate(result, max_output_chars)

    # ── 新路径：扩散激活 + 竞争抑制 ──

    def _activate_layered(
        self,
        graph: GraphStore,
        query: str,
        query_vec: np.ndarray,
        all_nodes: list,
        current_turn: int,
        top_k: int,
        l1_summary: str,
        *,
        focus_cluster_node_ids: list[str],
        focus_cluster_raw_ids: list[str],
        l0_thread_id: str | None,
        l0_thread_anchor_turn: int,
    ) -> ActivateResult:
        lines: list[str] = []
        dense_w = self.graph_expand_dense_weight

        conclusion_nodes = [n for n in all_nodes if n.type == NodeType.CONCLUSION]
        is_temporal = _is_temporal_query(query)

        focus_ids = _build_focus_id_set(
            graph,
            all_nodes,
            current_turn,
            self.focus_recent_turn_window,
            focus_cluster_node_ids,
            focus_cluster_raw_ids,
            l0_thread_id,
            l0_thread_anchor_turn,
        )

        # (Bayesian surprise focus shift — disabled, no improvement over baseline)

        # 信息层级权重：结论是压缩后的高阶知识，原始消息信噪比较低
        CONCLUSION_W = 1.3
        PROFILE_W = 1.2
        RAW_W = 0.8

        # ── Phase 1: 从 query 相似度构建种子激活（先算 raw，再决定是否扩圈） ──
        raw_activation: dict[str, float] = {}

        for node in conclusion_nodes:
            if node.embedding is None:
                continue
            node_vec = np.array(node.embedding, dtype=np.float32)
            dense = _cosine(query_vec, node_vec)
            sparse = _sparse_dice(query, node.value)
            score = _hybrid_score(dense, sparse, dense_w) * CONCLUSION_W
            raw_activation[node.id] = score

        profile_nodes = [n for n in all_nodes if n.type == NodeType.PROFILE]
        for node in profile_nodes:
            if node.embedding is None:
                continue
            node_vec = np.array(node.embedding, dtype=np.float32)
            dense = _cosine(query_vec, node_vec)
            sparse = _sparse_dice(query, node.value)
            raw_activation[node.id] = _hybrid_score(dense, sparse, dense_w) * PROFILE_W

        for trace in graph.raw_traces:
            if not trace.id or not trace.embedding:
                continue
            t_vec = np.array(trace.embedding, dtype=np.float32)
            sim = _cosine(query_vec, t_vec)
            if sim > 0.05:
                raw_activation[trace.id] = sim * RAW_W

        # 时间回溯：近期节点的种子激活值获得加成（扩圈判定之前）
        if is_temporal:
            self._boost_recent_seeds(raw_activation, graph, all_nodes, current_turn)

        broaden = is_temporal or not focus_ids
        if not broaden and raw_activation:
            max_focus = max(
                (raw_activation[i] for i in focus_ids if i in raw_activation),
                default=0.0,
            )
            max_all = max(raw_activation.values())

            # 1) 焦点内整体很弱，但全图有更强命中 → 扩圈
            if max_focus < self.focus_broaden_threshold and max_all > max_focus + 0.02:
                broaden = True
            # 2) 焦点内「还行」但全图明显更高 → 扩圈（避免错误焦点霸占种子）
            elif (
                max_all > max_focus + 0.03
                and max_focus < max_all * 0.88
            ):
                broaden = True

        if self.l1_only_when_broaden:
            if broaden and l1_summary:
                lines.append(f"[global] {l1_summary}")
        elif l1_summary:
            lines.append(f"[global] {l1_summary}")

        activation = dict(raw_activation)
        if not broaden:
            decay = self.focus_out_decay
            for k in list(activation.keys()):
                node = graph.nodes.get(k)
                if node is not None and node.type == NodeType.PROFILE:
                    continue
                if k not in focus_ids:
                    activation[k] *= decay

        # 只保留 top-K 种子进入扩散阶段
        seed_k = self.activation_seed_k
        if len(activation) > seed_k * 2:
            sorted_acts = sorted(activation.items(), key=lambda x: -x[1])
            activation = dict(sorted_acts[: seed_k * 2])

        logger.info(
            "[activate-layered] query=%r seeds=%d conclusions=%d raw=%d temporal=%s "
            "broaden=%s focus_ids=%d",
            query[:60], len(activation), len(conclusion_nodes),
            len(graph.raw_traces), is_temporal, broaden, len(focus_ids),
        )

        # ── Phase 2: 沿图边扩散激活 ──
        activation = self._spread_activation(activation, graph)

        # ── Phase 3: 竞争抑制 ──
        winners = self._competitive_inhibition(activation, top_k)

        # ── Phase 4: 格式化输出 ──
        conclusion_items: list[tuple] = []
        profile_items: list[tuple] = []
        raw_items: list[tuple] = []

        for node_id, act_val in winners:
            if node_id in graph.nodes:
                node = graph.nodes[node_id]
                if node.type == NodeType.CONCLUSION:
                    conclusion_items.append((node, act_val))
                elif node.type == NodeType.PROFILE:
                    profile_items.append((node, act_val))
            elif node_id in graph.raw_traces_by_id:
                raw_items.append((graph.raw_traces_by_id[node_id], act_val))

        if profile_items:
            profile_text = "\n".join(f"- {n.value}" for n, _ in profile_items)
            lines.append(f"[profile] {profile_text}")

        lines.extend(self._format_node_lines(
            [n for n, _ in conclusion_items],
            lambda node: f"[conclusion] {node.value}",
        ))

        for trace, _ in raw_items:
            lines.append(f"[raw] {trace.content}")

        # ── 歧义检测 ──
        ambiguous, candidates, hint = self._detect_ambiguity(
            graph, query_vec, conclusion_items,
        )

        graph.promote_nodes(current_turn)
        graph.decay_nodes(current_turn)

        return ActivateResult(
            "\n".join(lines),
            ambiguous=ambiguous,
            candidates=candidates,
            hint=hint,
        )

    # ── 扩散激活 ──

    def _spread_activation(
        self,
        activation: dict[str, float],
        graph: GraphStore,
    ) -> dict[str, float]:
        """从种子节点沿图边扩散激活，每跳衰减 ×decay。取 max 防止密集区域爆炸。"""
        max_hops = self.activation_max_hops
        decay = self.activation_decay

        # 构建邻接表（含 derived_from 等所有边）
        adj: dict[str, list[tuple[str, float]]] = {}
        for e in graph.edges:
            adj.setdefault(e.src_id, []).append((e.dst_id, e.weight))
            adj.setdefault(e.dst_id, []).append((e.src_id, e.weight))

        known_ids = set(graph.nodes.keys()) | set(graph.raw_traces_by_id.keys())

        current = dict(activation)
        for _hop in range(max_hops):
            next_spread: dict[str, float] = {}
            for node_id, act_val in current.items():
                for neighbor_id, edge_weight in adj.get(node_id, []):
                    if neighbor_id not in known_ids:
                        continue
                    spread_val = act_val * decay * edge_weight
                    prev = activation.get(neighbor_id, 0.0)
                    if spread_val > prev:
                        activation[neighbor_id] = spread_val
                        next_spread[neighbor_id] = spread_val
            current = next_spread

        return activation

    # ── 竞争抑制 ──

    def _competitive_inhibition(
        self,
        activation: dict[str, float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """取 top-N + 相对阈值淘汰弱激活。"""
        top_n = min(self.inhibition_top_n, top_k)
        min_ratio = self.inhibition_min_ratio

        ranked = sorted(activation.items(), key=lambda x: -x[1])
        if not ranked:
            return []
        best = ranked[0][1]
        if best <= 0:
            return []
        threshold = best * min_ratio
        return [(nid, act) for nid, act in ranked[:top_n] if act >= threshold]

    # ── 时间回溯加成 ──

    def _boost_recent_seeds(
        self,
        activation: dict[str, float],
        graph: GraphStore,
        all_nodes: list,
        current_turn: int,
    ) -> None:
        """时间查询时给近期节点的种子激活值加成。"""
        for node in all_nodes:
            if node.id not in activation:
                continue
            turns_ago = current_turn - node.turn
            if turns_ago <= 10:
                boost = 1.0 + 0.3 * max(0.0, 1.0 - turns_ago / 10.0)
                activation[node.id] *= boost
        for trace in graph.raw_traces:
            if not trace.id or trace.id not in activation:
                continue
            turns_ago = current_turn - trace.turn
            if turns_ago <= 10:
                boost = 1.0 + 0.3 * max(0.0, 1.0 - turns_ago / 10.0)
                activation[trace.id] *= boost

    # ── 歧义检测（不依赖 topic） ──

    def _detect_ambiguity(
        self,
        graph: GraphStore,
        query_vec: np.ndarray,
        conclusion_items: list[tuple],
    ) -> tuple[bool, list[AmbiguityCandidate], str]:
        """通过 top 结论间的互相似度检测语义歧义。

        如果 top 结论分属两个低互相似的语义簇，说明 query 跨越了不同领域。
        """
        if len(conclusion_items) < 2:
            return False, [], ""

        top_nodes = [n for n, _ in conclusion_items[:6]]
        vecs = []
        for n in top_nodes:
            if n.embedding is not None:
                vecs.append((n, np.array(n.embedding, dtype=np.float32)))
        if len(vecs) < 2:
            return False, [], ""

        # 检查 top 结果间是否存在语义距离远的对
        INTER_SIM_THRESHOLD = 0.35
        distant_pairs = 0
        total_pairs = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                total_pairs += 1
                if _cosine(vecs[i][1], vecs[j][1]) < INTER_SIM_THRESHOLD:
                    distant_pairs += 1

        if distant_pairs < total_pairs * 0.3:
            return False, [], ""

        # 构建候选（用 topic label 做可读描述，若存在的话）
        seen_labels: set[str] = set()
        candidates: list[AmbiguityCandidate] = []
        for n, vec in vecs[:4]:
            topic = graph.topics.get(n.topic_id) if n.topic_id else None
            label = topic.label if topic else n.value[:30]
            if label in seen_labels:
                continue
            seen_labels.add(label)
            sim = float(_cosine(query_vec, vec))
            candidates.append(AmbiguityCandidate(
                topic_label=label, snippet=n.value[:200], confidence=round(sim, 3),
            ))

        if len(candidates) < 2:
            return False, [], ""

        labels = [c.topic_label for c in candidates]
        hint = f"query 匹配到 {len(candidates)} 个不同领域: {', '.join(labels)}"
        return True, candidates, hint

    # ── 格式化工具 ──

    def _format_node_lines(self, nodes: list, formatter) -> list[str]:
        lines: list[str] = []
        session_ids = {getattr(node, "session_id", None) for node in nodes}
        use_markers = len({sid for sid in session_ids if sid is not None}) > 1
        current_session = object()
        for node in nodes:
            session_id = getattr(node, "session_id", None)
            if use_markers and session_id != current_session:
                label = session_id or "unknown"
                lines.append(f"[session:{label}]")
                current_session = session_id
            lines.append(formatter(node))
        return lines

    # ── 旧路径：向后兼容（无 CONCLUSION 节点时） ──

    def _activate_legacy(
        self,
        graph: GraphStore,
        query: str,
        query_vec: np.ndarray,
        nodes: list,
        current_turn: int,
        top_k: int,
    ) -> ActivateResult:
        candidate_nodes, topic_scores = self._topic_filter(
            graph, query_vec, nodes, threshold=TOPIC_MATCH_THRESHOLD
        )

        scored: list[tuple] = []
        for node in candidate_nodes:
            if node.embedding is None:
                continue
            node_vec = np.array(node.embedding, dtype=np.float32)
            sim = _cosine(query_vec, node_vec)
            scored.append((node, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        seed_nodes = [n for n, _ in scored[:SEED_COUNT]]

        activated_ids = set(n.id for n in seed_nodes)
        for node in seed_nodes:
            activated_ids.update(graph.get_neighbors(node.id))

        activated_nodes = [
            graph.nodes[nid] for nid in activated_ids if nid in graph.nodes
        ]
        for node in activated_nodes:
            node.activation_count += 1
            node.last_activated_turn = current_turn

        layer_weight = {Layer.STABLE: 1.5, Layer.ANCHOR: 1.2, Layer.MEMORY: 1.0}
        scored_dict = {n.id: s for n, s in scored}
        topic_max_cache: dict[str, float] = {}
        final_scored = []
        for node in activated_nodes:
            sim = scored_dict.get(node.id, 0.0)
            lw = layer_weight.get(node.layer, 1.0)
            centrality = _topic_relative_centrality(graph, node.id, topic_max_cache)
            mention = 1.0 + math.log1p(getattr(node, "merge_count", 0))
            topic = graph.topics.get(node.topic_id) if node.topic_id else None
            revisit = len(getattr(topic, "revisit_turns", [])) if topic else 1
            topic_weight = 1.0 + math.log1p(max(0, revisit - 1))
            score = sim * lw * centrality * mention * topic_weight
            final_scored.append((node, score))

        final_scored.sort(key=lambda x: x[1], reverse=True)
        if _is_temporal_query(query):
            final_scored = self._boost_temporal_scored_legacy(graph, query_vec, final_scored, top_k)
        top_nodes = [n for n, _ in final_scored[:top_k]]

        topic_hit = topic_scores[0][1] if topic_scores else 1.0
        matched_count = (
            sum(1 for _, s in topic_scores if s > TOPIC_MATCH_THRESHOLD)
            if topic_scores
            else len(graph.topics)
        )
        signals = CriticSignals(
            top1_sim=scored[0][1] if scored else 0.0,
            top5_mean_sim=(
                sum(s for _, s in scored[:5]) / min(5, len(scored))
                if scored else 0.0
            ),
            topic_hit_sim=topic_hit,
            topic_hit_count=matched_count,
            candidate_ratio=len(candidate_nodes) / len(nodes) if nodes else 0.0,
            topic_count=len(graph.topics),
            result_count=len(top_nodes),
            unique_topics_in_result=len(
                {n.topic_id for n in top_nodes if n.topic_id}
            ),
        )
        verdict = critic_evaluate(signals)

        if not verdict.confident:
            l2_nodes = self._retrieve_l2(
                graph,
                query_vec,
                top_nodes,
                verdict,
                nodes,
                top_k,
            )
            seen_ids = {n.id for n in top_nodes}
            for node in l2_nodes:
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    top_nodes.append(node)
            top_nodes = top_nodes[: top_k * 2]
            scored_all = {n.id: scored_dict.get(n.id, 0.0) for n in top_nodes}

            def _score(n):
                sim = scored_all.get(n.id, 0.0)
                lw = layer_weight.get(n.layer, 1.0)
                cent = _topic_relative_centrality(graph, n.id, topic_max_cache)
                ment = 1.0 + math.log1p(getattr(n, "merge_count", 0))
                t = graph.topics.get(n.topic_id) if n.topic_id else None
                rv = len(getattr(t, "revisit_turns", [])) if t else 1
                tw = 1.0 + math.log1p(max(0, rv - 1))
                return sim * lw * cent * ment * tw

            top_nodes.sort(key=_score, reverse=True)
            top_nodes = top_nodes[:top_k]

        graph.promote_nodes(current_turn)
        graph.decay_nodes(current_turn)

        result_lines = self._format_node_lines(
            top_nodes,
            lambda node: f"[{node.layer.value}][{node.type.value}] {node.value}",
        )

        remaining_slots = max(0, top_k - len(result_lines))
        if remaining_slots > 0 and graph.raw_traces:
            raw_results = graph.search_raw_traces(query_vec, top_k=remaining_slots + 3)
            node_sims = {s for _, s in final_scored[:top_k]} if final_scored else set()
            min_node_sim = min(node_sims) if node_sims else 0.0
            for trace, sim in raw_results:
                if len(result_lines) >= top_k:
                    break
                if sim > max(min_node_sim * 0.8, TOPIC_MATCH_THRESHOLD):
                    result_lines.append(f"[raw] {trace.content}")

        return ActivateResult("\n".join(result_lines))

    # ── 旧路径工具方法 ──

    def _boost_temporal_scored_legacy(
        self,
        graph: GraphStore,
        query_vec: np.ndarray,
        scored: list[tuple],
        top_k: int,
    ) -> list[tuple]:
        if not scored:
            return scored
        seed_count = min(3, len(scored), top_k)
        extras: dict[str, tuple] = {}
        existing = {item[0].id: item for item in scored}

        for item in scored[:seed_count]:
            node = item[0]
            neighbors = graph.get_temporal_neighbors(node.id)
            for neighbor in neighbors:
                if neighbor.session_id != node.session_id:
                    continue
                base_sim = 0.0
                if neighbor.embedding is not None:
                    base_sim = _cosine(
                        query_vec,
                        np.array(neighbor.embedding, dtype=np.float32),
                    )
                boosted_score = base_sim * TEMPORAL_BOOST
                candidate = (neighbor, boosted_score)
                existing_item = existing.get(neighbor.id) or extras.get(neighbor.id)
                existing_score = existing_item[1] if existing_item else float("-inf")
                if boosted_score > existing_score:
                    extras[neighbor.id] = candidate

        merged = {item[0].id: item for item in scored}
        merged.update(extras)
        boosted = list(merged.values())
        boosted.sort(key=lambda x: x[1], reverse=True)
        return boosted

    def _retrieve_l2(
        self,
        graph: GraphStore,
        query_vec: np.ndarray,
        l1_nodes: list,
        verdict: CriticVerdict,
        all_nodes: list,
        top_k: int,
    ) -> list:
        layer_weight = {Layer.STABLE: 1.5, Layer.ANCHOR: 1.2, Layer.MEMORY: 1.0}
        l1_ids = {n.id for n in l1_nodes}

        if verdict.strategy == "global_search":
            scored: list[tuple] = []
            for node in all_nodes:
                if node.embedding is None:
                    continue
                sim = _cosine(query_vec, np.array(node.embedding, dtype=np.float32))
                scored.append((node, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            extra = [n for n, _ in scored[: top_k * 2] if n.id not in l1_ids]
            return extra[: top_k // 2]

        if verdict.strategy == "broaden_topic":
            candidate, _ = self._topic_filter(
                graph, query_vec, all_nodes, threshold=TOPIC_BROADEN_THRESHOLD
            )
            scored = []
            for node in candidate:
                if node.embedding is None or node.id in l1_ids:
                    continue
                sim = _cosine(query_vec, np.array(node.embedding, dtype=np.float32))
                scored.append((node, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [n for n, _ in scored[: top_k // 2]]

        if verdict.strategy == "cross_topic":
            l1_topic_ids = {n.topic_id for n in l1_nodes if n.topic_id}
            other_topics = [
                t.id for t in graph.topics.values()
                if t.id not in l1_topic_ids
            ]
            if not other_topics:
                return []
            candidates = [
                n for n in all_nodes
                if n.topic_id in other_topics and n.id not in l1_ids
            ]
            scored = []
            for node in candidates:
                if node.embedding is None:
                    continue
                sim = _cosine(query_vec, np.array(node.embedding, dtype=np.float32))
                scored.append((node, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [n for n, _ in scored[: top_k // 2]]

        return []

    def _topic_filter(
        self,
        graph: GraphStore,
        query_vec: np.ndarray,
        all_nodes: list,
        threshold: float = TOPIC_MATCH_THRESHOLD,
    ) -> tuple[list, list[tuple[Topic, float]]]:
        topics = list(graph.topics.values())
        topic_scores: list[tuple[Topic, float]] = []

        if len(topics) >= 2:
            for topic in topics:
                if topic.embedding is None:
                    continue
                t_vec = np.array(topic.embedding, dtype=np.float32)
                sim = _cosine(query_vec, t_vec)
                topic_scores.append((topic, sim))
            topic_scores.sort(key=lambda x: x[1], reverse=True)

        if len(topics) < 2 or not topic_scores or topic_scores[0][1] < threshold:
            return all_nodes, topic_scores

        matched = [t for t, sim in topic_scores if sim > threshold]
        matched_ids = {t.id for t in matched}
        candidate = [n for n in all_nodes if n.topic_id in matched_ids]

        if len(candidate) < SEED_COUNT:
            return all_nodes, topic_scores

        return candidate, topic_scores
