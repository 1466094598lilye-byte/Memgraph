"""Memgraph API 接口：摄入、召回、模型配置。

三个核心接口：
1. ingest  — 摄入一轮对话（纯文本）
2. recall  — 根据当前 query 召回相关记忆
3. config  — 配置 Memgraph 内部使用的 LLM（懒人模式 vs 自选便宜模型）

约定：只接收纯文本。调用方负责在传入前剥离代码块、tool call、function result 等非自然语言内容。
Memgraph 内部有兜底过滤，但不应依赖它。
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

from memgraph.attention_router import AttentionRouter
from memgraph.embedder import Embedder
from memgraph.models import ActivateResult
from memgraph.welcome import show_welcome

logger = logging.getLogger(__name__)


@dataclass
class MemgraphAPIConfig:
    """Memgraph API 配置。

    懒人模式：不配 llm_provider / llm_model，由调用方传入 llm_fn 复用主模型。
    省钱模式：配 llm_provider + llm_model + api_key，Memgraph 内部调便宜模型。
    """

    # 内部 LLM 配置（省钱模式）
    # 支持环境变量 fallback：
    #   MEMGRAPH_LLM_PROVIDER, MEMGRAPH_LLM_MODEL,
    #   MEMGRAPH_LLM_API_KEY, MEMGRAPH_LLM_BASE_URL
    llm_provider: Literal["openai", "anthropic", "deepseek"] | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_base_url: str | None = None  # DeepSeek 等需要自定义 base_url

    def __post_init__(self) -> None:
        """从环境变量读取未设置的配置项。"""
        if self.llm_provider is None:
            self.llm_provider = os.environ.get("MEMGRAPH_LLM_PROVIDER")
        if self.llm_model is None:
            self.llm_model = os.environ.get("MEMGRAPH_LLM_MODEL")
        if self.llm_api_key is None:
            self.llm_api_key = os.environ.get("MEMGRAPH_LLM_API_KEY")
        if self.llm_base_url is None:
            self.llm_base_url = os.environ.get("MEMGRAPH_LLM_BASE_URL")

    # 注意力路由参数
    top_k: int = 10          # 召回时返回的最相关对话轮数
    memo_k: int = 10         # 备忘录条目数上限
    max_output_chars: int | None = None  # 输出最大字符数


class MemgraphAPI:
    """Memgraph 对外 API。

    Usage (懒人模式 - 复用调用方主模型):
        api = MemgraphAPI()
        api.ingest("用户说的话", "助手回的话", llm_fn=my_llm_fn)

    Usage (省钱模式 - 自选便宜模型):
        api = MemgraphAPI(config=MemgraphAPIConfig(
            llm_provider="deepseek",
            llm_model="deepseek-chat",
            llm_api_key="sk-xxx",
            llm_base_url="https://api.deepseek.com",
        ))
        api.ingest("用户说的话", "助手回的话")

    接口约定:
        - user_text / assistant_text 只接收纯文本
        - 调用方应在传入前剥离：代码块、tool call、function result
        - Memgraph 内部有兜底过滤，但不应依赖
    """

    def __init__(
        self,
        config: MemgraphAPIConfig | None = None,
        embedder: Embedder | None = None,
        silent: bool = False,
    ) -> None:
        self.config = config or MemgraphAPIConfig()
        self._embedder = embedder or Embedder()
        self._router: AttentionRouter | None = None

        if not silent:
            show_welcome()

    def _get_router(self, llm_fn=None) -> AttentionRouter:
        """获取或初始化 AttentionRouter。"""
        if self._router is not None:
            return self._router

        cfg = self.config

        # 省钱模式：Memgraph 自己建 LLM client
        if cfg.llm_provider and cfg.llm_model:
            import os

            if cfg.llm_api_key:
                # 根据 provider 设置对应的环境变量
                if cfg.llm_provider == "deepseek":
                    os.environ.setdefault("OPENAI_API_KEY", cfg.llm_api_key)
                    os.environ.setdefault("OPENAI_BASE_URL", cfg.llm_base_url or "https://api.deepseek.com")
                elif cfg.llm_provider == "openai":
                    os.environ.setdefault("OPENAI_API_KEY", cfg.llm_api_key)
                elif cfg.llm_provider == "anthropic":
                    os.environ.setdefault("ANTHROPIC_API_KEY", cfg.llm_api_key)

            # DeepSeek 走 OpenAI 兼容接口
            provider = "openai" if cfg.llm_provider == "deepseek" else cfg.llm_provider

            self._router = AttentionRouter(
                embedder=self._embedder,
                llm_provider=provider,
                model=cfg.llm_model,
            )
        else:
            # 懒人模式：需要调用方传 llm_fn
            self._router = AttentionRouter(
                embedder=self._embedder,
                llm_fn=llm_fn,
            )

        return self._router

    # ── 接口 1: 摄入 ──

    def ingest(
        self,
        user_text: str,
        assistant_text: str,
        llm_fn=None,
    ) -> None:
        """摄入一轮对话。

        Args:
            user_text: 用户发言（纯文本，不含代码块/tool call）
            assistant_text: 助手回复（纯文本，不含代码块/tool call）
            llm_fn: 懒人模式下传入的 LLM 调用函数，签名 (prompt, max_tokens) -> (text, usage_dict)
        """
        router = self._get_router(llm_fn=llm_fn)
        router.encode(user_text, assistant_text)
        logger.info(
            "[memgraph-api] ingested turn, total_turns=%d, memo_keys=%d",
            len(router.turns), len(router.memo),
        )

    # ── 接口 2: 召回 ──

    def recall(self, query: str) -> ActivateResult:
        """根据当前 query 召回相关记忆。

        Args:
            query: 用户当前的问题/发言

        Returns:
            ActivateResult (str): 组装好的记忆文本，直接塞进 prompt 即可
        """
        if self._router is None:
            return ActivateResult("")

        cfg = self.config
        result = self._router.activate(
            query=query,
            top_k=cfg.top_k,
            memo_k=cfg.memo_k,
            max_output_chars=cfg.max_output_chars,
        )
        logger.info(
            "[memgraph-api] recall for query (len=%d), result_len=%d",
            len(query), len(result),
        )
        return result

    # ── 接口 3: 状态查询 ──

    def inspect(self) -> dict:
        """返回当前 Memgraph 状态（调试用）。"""
        if self._router is None:
            return {"status": "not_initialized", "total_turns": 0, "memo_keys": 0}
        info = self._router.inspect()
        info["encode_usage"] = self._router.encode_usage
        return info
