"""Pydantic 数据模型：Node, Edge, Topic, StateGraph, NodeType, Layer, ActivateResult。"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Layer(str, Enum):
    """记忆层级。"""

    MEMORY = "memory"  # 最近上下文，快速衰减
    ANCHOR = "anchor"  # 高密度节点，慢速衰减
    STABLE = "stable"  # 提炼后的核心属性，极慢衰减


class NodeType(str, Enum):
    """节点类型枚举。"""

    INTENT = "intent"  # 目标/意图
    ENTITY = "entity"  # 实体/角色
    CONSTRAINT = "constraint"  # 约束条件
    STATE = "state"  # 当前状态
    COMPLETED = "completed"  # 已完成动作
    CONCLUSION = "conclusion"  # L2 压缩结论
    PROFILE = "profile"  # 跨话题用户状态


class Topic(BaseModel):
    """话题枢纽：对话中涌现的主题聚类。"""

    id: str
    label: str  # 话题名（2-6 字），如 "养猫日常"、"技术选型"
    embedding: Optional[list[float]] = None
    node_count: int = 0  # 挂载的节点数
    last_turn: int = 0  # 最近一次有新内容归入的轮次
    revisit_turns: list[int] = []  # 哪些 turn 有新内容归入（主线话题回访多）


class Node(BaseModel):
    """状态图节点。"""

    id: str
    type: NodeType
    value: str
    turn: int  # 出现在第几轮
    importance: float = 1.0  # 综合重要性分（动态计算）
    embedding: Optional[list[float]] = None  # 向量

    # Phase 2: 层级
    layer: Layer = Layer.MEMORY
    activation_count: int = 0
    last_activated_turn: int = 0

    # Phase 4: 话题归属
    topic_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[float] = None

    # 结构信号：检索时加权用
    merge_count: int = 0  # 被去重合并次数（= 被重复提及次数）

    # Metadata-driven similarity (SAGE-inspired)
    entities: list[str] = []  # 提取的实体（人名、概念、术语等），用于建 metadata 边


class RawTrace(BaseModel):
    """未经 LLM 抽取的原始消息痕迹，仅存 embedding + 元数据。"""

    id: Optional[str] = None
    content: str
    embedding: list[float]
    turn: int
    density: float  # 信息密度评分
    topic_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[float] = None


class Edge(BaseModel):
    """状态图边。"""

    src_id: str
    dst_id: str
    relation: str
    weight: float = 1.0
    temporal_order: Optional[int] = None


class StateGraph(BaseModel):
    """状态图数据结构。"""

    nodes: list[Node] = []
    edges: list[Edge] = []
    topics: list[Topic] = []
    turn_count: int = 0


# ── 歧义检测返回值 ──


class AmbiguityCandidate(BaseModel):
    """歧义检测中的竞争候选：来自不同话题的高置信匹配。"""

    topic_label: str
    snippet: str
    confidence: float


class ActivateResult(str):
    """activate() 的结构化返回值。

    继承 str 保持向后兼容：isinstance(r, str) == True，
    .split() / in / .strip() 等字符串操作正常工作。
    新代码可访问 .ambiguous / .candidates / .hint。
    """

    ambiguous: bool
    candidates: list[AmbiguityCandidate]
    hint: str

    def __new__(
        cls,
        context: str = "",
        *,
        ambiguous: bool = False,
        candidates: list[AmbiguityCandidate] | None = None,
        hint: str = "",
    ):
        instance = super().__new__(cls, context)
        instance.ambiguous = ambiguous
        instance.candidates = candidates or []
        instance.hint = hint
        return instance
