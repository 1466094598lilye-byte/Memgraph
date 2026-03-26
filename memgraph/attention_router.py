"""AttentionRouter: 注意力路由 + 备忘录 + unsatisfied 队列。

核心思路（Lily 2026-03-21）：
- 人类不怕灾难性遗忘，因为人类一直在遗忘
- 人类 context window 只有 3-5 句，但能无缝切 200 个话题
- 关键不是记住多少，是每一轮能多快跳到对的东西
- 注意力路由：每次只取相关的几轮原文注入 context
- 备忘录：精确事实（数字、日期、名字）单独存，全量注入
- unsatisfied 队列：跟踪每个活跃任务的当前进度，全量注入

三层架构：
1. 注意力路由层：对话原文全量存储 + cosine top-k 选择性加载
2. 备忘录层：LLM 提取精确事实 → key-value store → 全量注入
3. 任务队列层：跟踪活跃任务的 unsatisfied 项 → 全量注入（提供时序信息）
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable

import numpy as np

from memgraph.embedder import Embedder
from memgraph.models import ActivateResult

logger = logging.getLogger(__name__)

LLMFn = Callable[[str, int], tuple[str, dict]]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

_MEMO_EXTRACT_TEMPLATE = """Read this conversation turn. Extract ALL important facts into a flat JSON object.

Include:
- User's personal info (measurements, goals, preferences, schedule, constraints)
- User's decisions and agreements
- Specific details from any plan, proposal, or recommendation the user accepted (store the concrete details, not just "user accepted a plan")
- Current status or progress of ongoing tasks

Use descriptive key names. Reuse existing keys to update values when the same topic comes up again.

User: __USER__
Assistant: __ASSISTANT__

Existing memo keys (reuse to update): __KEYS__

JSON:"""


def _strip_code_and_tools(text: str) -> str:
    """过滤代码块和 tool call，只保留自然语言部分。"""

    # 0. 如果输入像是 JSON array（序列化的 content blocks），提取纯文本部分
    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                # content block array: 只保留 type=text 的 block
                text_parts = []
                for block in parsed:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    text = "\n".join(text_parts)
                else:
                    # 整个 array 没有 text block，返回空
                    return ""
            elif isinstance(parsed, dict):
                if parsed.get("type") == "text":
                    text = parsed.get("text", "")
                elif parsed.get("type") in ("tool_use", "tool_result", "tool_call"):
                    return ""
        except (json.JSONDecodeError, TypeError):
            pass  # 不是 JSON，走后续正则过滤

    # 1. 去掉 fenced code blocks: ```...```
    text = re.sub(r"```[\s\S]*?```", "", text)

    # 2. 去掉常见 tool call XML 标签块
    for tag in ("tool_call", "tool_use", "function_calls", "antml:function_calls",
                "antml:invoke", "tool_result", "function_results"):
        pattern = rf"<{re.escape(tag)}[\s\S]*?</{re.escape(tag)}>"
        text = re.sub(pattern, "", text)

    # 3. 去掉 JSON tool call 块 ({"name":..., "arguments":...} 格式)
    text = re.sub(r'\{"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:[\s\S]*?\}\s*\}', "", text)

    # 4. 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class MemoEntry:
    """备忘录条目：一条精确事实 + embedding。"""
    key: str
    value: str
    embedding: np.ndarray | None = None


@dataclass
class Turn:
    """一轮对话：用户说了什么 + 助手回了什么。"""
    turn_id: int
    user_text: str
    assistant_text: str
    embedding: np.ndarray | None = None


class AttentionRouter:
    """注意力路由 + 备忘录。

    encode: 存原文 + embedding + 提取精确事实到备忘录
    activate: 备忘录全量注入 + cosine top-k 选最相关原文
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        llm_fn: LLMFn | None = None,
        llm_provider: str = "openai",
        model: str = "gpt-4o-mini",
    ) -> None:
        self.embedder = embedder or Embedder()
        self.turns: list[Turn] = []
        self.memo: dict[str, MemoEntry] = {}  # 备忘录：key → MemoEntry

        self._llm_fn = llm_fn
        self._llm_provider = llm_provider
        self._model = model
        self._encode_usage: dict = {"input_tokens": 0, "output_tokens": 0}
        self._client = None

    def _call_llm(self, prompt: str, max_tokens: int = 300) -> tuple[str, dict]:
        """调用 LLM。"""
        if self._llm_fn is not None:
            text, usage = self._llm_fn(prompt, max_tokens)
            return text, usage or {"input_tokens": 0, "output_tokens": 0}

        if self._client is None:
            if self._llm_provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic()
            else:
                import openai
                self._client = openai.OpenAI()

        if self._llm_provider == "anthropic":
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        else:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
                "output_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
            }
        return text, usage

    # ── 存储 ──

    def encode(self, user_text: str, assistant_text: str) -> None:
        """存一轮对话 + 提取精确事实到备忘录。"""
        turn_id = len(self.turns)

        # 0. 过滤代码块和 tool call，只保留自然语言
        user_clean = _strip_code_and_tools(user_text)
        assistant_clean = _strip_code_and_tools(assistant_text)

        # 1. 存原文 + embedding（只 embed user_text，assistant 太长会稀释语义）
        vec = np.array(
            self.embedder.embed_query(user_clean),
            dtype=np.float32,
        )
        self.turns.append(Turn(
            turn_id=turn_id,
            user_text=user_clean,
            assistant_text=assistant_clean,
            embedding=vec,
        ))

        # 2. 提取精确事实到备忘录
        self._extract_memo(user_clean, assistant_clean)

        logger.debug(
            "[attention-router] stored turn %d, total=%d, memo_keys=%d",
            turn_id, len(self.turns), len(self.memo),
        )

    def _extract_memo(self, user_text: str, assistant_text: str) -> None:
        """LLM 提取精确事实，更新备忘录。每条 fact 带 embedding。"""
        prompt = (_MEMO_EXTRACT_TEMPLATE
            .replace("__USER__", user_text[:3000])
            .replace("__ASSISTANT__", assistant_text[:3000])
            .replace("__KEYS__", str(list(self.memo.keys())))
        )
        try:
            text, usage = self._call_llm(prompt, max_tokens=500)
            self._encode_usage["input_tokens"] += usage.get("input_tokens", 0)
            self._encode_usage["output_tokens"] += usage.get("output_tokens", 0)

            # 解析 JSON
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json.loads(text)
            # Flat dict: all keys are memo entries (facts + plans + progress)
            if isinstance(parsed, dict):
                # If LLM returned nested structure, flatten it
                flat = {}
                for k, v in parsed.items():
                    if isinstance(v, dict):
                        for sub_k, sub_v in v.items():
                            flat[sub_k] = str(sub_v)
                    else:
                        flat[k] = str(v)
                for k, v in flat.items():
                    fact_text = f"{k}: {v}"
                    vec = np.array(self.embedder.embed_query(fact_text), dtype=np.float32)
                    self.memo[k] = MemoEntry(key=k, value=v, embedding=vec)
                    logger.debug("[memo] %s = %s", k, v[:80])
        except (json.JSONDecodeError, Exception) as e:
            logger.debug("[memo-extract] failed to parse: %s", e)

        self._compact_memo()

    MAX_MEMO_KEYS = 80

    def _compact_memo(self) -> None:
        """如果 memo 超过 MAX_MEMO_KEYS，合并语义重复 + 淘汰低相关性条目。"""
        if len(self.memo) <= self.MAX_MEMO_KEYS:
            return

        original_count = len(self.memo)
        merged = 0
        evicted = 0

        # Phase 1: 合并语义重复的 key 对 (sim > 0.85)
        changed = True
        while changed and len(self.memo) > self.MAX_MEMO_KEYS:
            changed = False
            keys = list(self.memo.keys())
            best_pair = None
            best_sim = 0.85
            for i in range(len(keys)):
                ei = self.memo[keys[i]]
                if ei.embedding is None:
                    continue
                for j in range(i + 1, len(keys)):
                    ej = self.memo[keys[j]]
                    if ej.embedding is None:
                        continue
                    sim = _cosine(ei.embedding, ej.embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (keys[i], keys[j])
            if best_pair:
                k1, k2 = best_pair
                e1, e2 = self.memo[k1], self.memo[k2]
                # 保留 value 更长的那个
                if len(e1.value) >= len(e2.value):
                    del self.memo[k2]
                else:
                    del self.memo[k1]
                merged += 1
                changed = True

        # Phase 2: 按与最近 turns 的相关性淘汰
        if len(self.memo) > self.MAX_MEMO_KEYS:
            recent_turns = [t for t in self.turns[-10:] if t.embedding is not None]
            if recent_turns:
                relevance: dict[str, float] = {}
                for k, entry in self.memo.items():
                    if entry.embedding is None:
                        relevance[k] = 0.0
                        continue
                    max_sim = max(
                        _cosine(entry.embedding, t.embedding)
                        for t in recent_turns
                    )
                    relevance[k] = max_sim
                # 按相关性排序，淘汰最不相关的
                sorted_keys = sorted(relevance, key=relevance.get)
                to_remove = len(self.memo) - self.MAX_MEMO_KEYS
                for k in sorted_keys[:to_remove]:
                    del self.memo[k]
                    evicted += 1

        logger.info(
            "[memo-compact] %d -> %d keys (merged %d, evicted %d)",
            original_count, len(self.memo), merged, evicted,
        )

    # ── 检索 ──

    @property
    def encode_usage(self) -> dict:
        return dict(self._encode_usage)

    def activate(
        self,
        query: str,
        top_k: int = 10,
        memo_k: int = 10,
        max_output_chars: int | None = None,
        **kwargs,
    ) -> ActivateResult:
        """备忘录 cosine top-k + 原文 cosine top-k。"""
        lines: list[str] = []

        query_vec = np.array(
            self.embedder.embed_query(query),
            dtype=np.float32,
        )

        # 1. 备忘录：embedding top-k（只注入与 query 最相关的条目）
        if self.memo:
            scored_memo: list[tuple[MemoEntry, float]] = []
            for entry in self.memo.values():
                if entry.embedding is not None:
                    sim = _cosine(query_vec, entry.embedding)
                    scored_memo.append((entry, sim))
            scored_memo.sort(key=lambda x: x[1], reverse=True)
            top_memo = scored_memo[:memo_k]
            memo_lines = [f"  {e.key}: {e.value}" for e, s in top_memo if s > 0.1]
            if memo_lines:
                lines.append("[memo]\n" + "\n".join(memo_lines))

        # 2. 注意力路由：cosine top-k
        if self.turns:
            scored: list[tuple[Turn, float]] = []
            for turn in self.turns:
                if turn.embedding is not None:
                    sim = _cosine(query_vec, turn.embedding)
                    scored.append((turn, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            top_turns = scored[:top_k]

            # 按时序排列
            top_turns.sort(key=lambda x: x[0].turn_id)

            for turn, sim in top_turns:
                if sim > 0.1:
                    lines.append(f"[turn-{turn.turn_id}] User: {turn.user_text}")
                    lines.append(f"[turn-{turn.turn_id}] Assistant: {turn.assistant_text}")

        result = "\n".join(lines)

        if max_output_chars and len(result) > max_output_chars:
            result = result[:max_output_chars]
            last_nl = result.rfind("\n")
            if last_nl > 0:
                result = result[:last_nl]

        return ActivateResult(result)

    def inspect(self) -> dict:
        return {
            "total_turns": len(self.turns),
            "memo_keys": len(self.memo),
            "memo": {k: e.value for k, e in self.memo.items()},

            "type": "attention_router",
        }
