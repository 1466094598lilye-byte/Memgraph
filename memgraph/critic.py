"""Critic：判断 L1 检索是否足够，不足时给出 L2 升级策略。零 token，纯信号。"""

from dataclasses import dataclass


@dataclass
class CriticSignals:
    """L1 检索过程中的可观测信号。"""

    top1_sim: float  # 最佳节点与 query 的 cosine 相似度
    top5_mean_sim: float  # top-5 平均相似度
    topic_hit_sim: float  # 最佳话题匹配分（无话题时为 1.0）
    topic_hit_count: int  # 命中的话题数
    candidate_ratio: float  # 话题过滤后保留比例 (0~1)
    topic_count: int  # 图中话题总数
    result_count: int  # 最终返回的节点数
    unique_topics_in_result: int  # 结果来自几个不同话题


@dataclass
class CriticVerdict:
    """Critic 判断结果。"""

    confident: bool  # L1 够好吗？
    reason: str  # 触发的规则描述
    strategy: str  # "none" | "broaden_topic" | "global_search" | "cross_topic"
    fallback_topic_ids: list[str]  # cross_topic 时建议搜的话题 id


# 阈值（可调，无域特定规则）
SIM_LOW_THRESHOLD = 0.35
SIM_MEAN_LOW_THRESHOLD = 0.25
TOPIC_STRICT_RATIO = 0.15  # candidate_ratio < 此值视为话题过滤过严
MIN_TOPICS_FOR_DIVERSITY = 3  # 话题数 >= 此值才检查结果多样性


def evaluate(signals: CriticSignals) -> CriticVerdict:
    """
    根据 L1 信号判断是否足够，不足时返回升级策略。
    规则按优先级：低置信度 > 话题过严 > 结果过窄
    """
    # 规则 1：整体置信度低 → global_search
    if (
        signals.top1_sim < SIM_LOW_THRESHOLD
        and signals.top5_mean_sim < SIM_MEAN_LOW_THRESHOLD
    ):
        return CriticVerdict(
            confident=False,
            reason="low_confidence",
            strategy="global_search",
            fallback_topic_ids=[],
        )

    # 规则 2：话题命中但过滤过严 → broaden_topic
    if (
        signals.topic_hit_sim > 0.3
        and signals.candidate_ratio < TOPIC_STRICT_RATIO
        and signals.topic_count >= 2
    ):
        return CriticVerdict(
            confident=False,
            reason="topic_too_strict",
            strategy="broaden_topic",
            fallback_topic_ids=[],
        )

    # 规则 3：结果全来自单一话题，图中有多个话题 → cross_topic
    if (
        signals.unique_topics_in_result == 1
        and signals.topic_count >= MIN_TOPICS_FOR_DIVERSITY
        and signals.result_count >= 3
    ):
        return CriticVerdict(
            confident=False,
            reason="result_too_narrow",
            strategy="cross_topic",
            fallback_topic_ids=[],  # 由调用方根据 top_nodes 的 topic_id 反推
        )

    return CriticVerdict(
        confident=True,
        reason="ok",
        strategy="none",
        fallback_topic_ids=[],
    )
