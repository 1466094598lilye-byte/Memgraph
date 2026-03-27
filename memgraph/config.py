"""MemGraph 配置：LLM provider、模型选择等。"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class MemGraphConfig:
    """MemGraph 运行时配置。"""

    llm_provider: Literal["openai", "anthropic"] = "openai"
    model: str = "deepseek-chat"  # default matches DeepSeek .env; override for other providers

    # 话题聚类（纯 embedding，领域无关）
    topic_shift_threshold: float = 0.55  # cosine < 此值 → 判定为新话题
    min_cluster_size: int = 3            # 聚类 >= 此大小才抽取，否则只存 raw trace
    min_content_length: int = 10         # 内容 < 此字符数 → 跳过（噪声过滤，不存入 cluster）
    max_cluster_size: int = 50           # 硬上限，防止单话题无限累积

    # 图邻居扩展（在线检索）：baseline seed 数、额外邻居槽、dense 在混合分中的权重
    graph_expand_seed_k: int = 5
    graph_expand_neighbor_k: int = 8
    graph_expand_dense_weight: float = 0.65

    # 扩散激活
    activation_seed_k: int = 10       # query 相似度取 top-K 种子（conclusion + raw）
    activation_max_hops: int = 2      # 沿图边扩散的跳数
    activation_decay: float = 0.5     # 每跳衰减系数

    # 竞争抑制
    inhibition_top_n: int = 15        # 最终保留的最大节点数
    inhibition_min_ratio: float = 0.2 # 激活值须达到最强的此比例

    # 焦点检索（结构信号：最近 cluster / 同 session 近期轮次；非领域规则）
    focus_recent_turn_window: int = 20   # 与当前 session 内、turn 在此窗口内的节点/raw 视为焦点
    focus_out_decay: float = 0.22        # 非扩圈时，焦点外种子的激活乘数（PROFILE 不参与衰减）
    # 扩圈：焦点内最高分过低、或全图最佳明显强于焦点内最佳时，取消焦点衰减（仍可用 l1_only_when_broaden 控 L1）
    focus_broaden_threshold: float = 0.18
    l1_only_when_broaden: bool = False   # False：始终带 L1（RealMem 等依赖全局摘要）；True 时仅扩圈注入，易掉 recall

    def __post_init__(self) -> None:
        if self.llm_provider not in ("openai", "anthropic"):
            raise ValueError(f"llm_provider 必须是 openai 或 anthropic，当前: {self.llm_provider}")
        if self.graph_expand_seed_k < 0:
            raise ValueError("graph_expand_seed_k 须 >= 0")
        if self.graph_expand_neighbor_k < 0:
            raise ValueError("graph_expand_neighbor_k 须 >= 0")
        w = self.graph_expand_dense_weight
        if not 0.0 <= w <= 1.0:
            raise ValueError("graph_expand_dense_weight 须在 [0, 1] 内")
        if not 0.0 <= self.focus_out_decay <= 1.0:
            raise ValueError("focus_out_decay 须在 [0, 1] 内")
        if self.focus_recent_turn_window < 0:
            raise ValueError("focus_recent_turn_window 须 >= 0")
