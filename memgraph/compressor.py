"""LLM 压缩器：将对话聚类压缩为高密度结论（L2），将所有 L2 压缩为全局摘要（L1）。"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from memgraph.models import Edge

logger = logging.getLogger(__name__)

# 宿主 agent 提供的 LLM 调用函数：(prompt, max_tokens) -> (response_text, usage_dict)
LLMFn = Callable[[str, int], tuple[str, dict]]

COMPRESS_CLUSTER_PROMPT = """\
Given the following conversation fragment, extract 1-3 high-density conclusions.

<conversation>
{messages}
</conversation>

Instructions:
- Each conclusion must be a complete, self-contained factual statement
- Focus on: decisions made, current states, goals, constraints, action items
- **Decisions are the highest priority**: when the user chose, selected, confirmed, or decided \
something specific, record the EXACT choice with full details (brand names, model numbers, \
dates, amounts, specific options). "Selected Britz Voyager 2-Berth RV" NOT "chose an RV".
- **Changes are strong signals**: if something was changed (A → B), always include \
it as "changed from A to B" with the reason. Changes override prior state.
- **Include specifics**: numbers, dates, names, percentages, brands, models — keep them in conclusions
- Discard: greetings, fillers, acknowledgments, repetition
- Use the conversation's own language
- Provide a short topic label (2-6 characters, in the conversation's language)

Respond ONLY with a JSON object (no markdown, no extra text):
{{"conclusions": ["conclusion 1", "conclusion 2"], "topic_label": "topic"}}"""

PROFILE_CONDENSE_PROMPT = """\
Summarize these conversation conclusions into ONE sentence about the user.
Focus on: who they are, what they want, what they decided, or what they prefer.
Keep specific values (numbers, dates, names, brands).
Do NOT include action items, to-do lists, or assistant suggestions.

Conclusions:
{conclusions}

Respond with ONE sentence only (no JSON, no markdown):"""

PROFILE_DIFF_PROMPT = """\
You are maintaining a user profile card — a list of facts about the user.
Given the current profile and new conclusions from a conversation, output what to add, update, or remove.

Current profile ({profile_count} items):
{current_profile}

New conclusions:
{new_conclusions}

Output a JSON diff. Rules:
- "add": new facts not already in profile. Keep specific values: numbers, dates, names, brands, models.
- "update": facts where the value CHANGED (e.g. target weight 60kg → 65kg). Key = old text, value = new text.
- Do NOT remove any facts. Only add new ones or update changed ones.
- Do NOT include unchanged facts.
- Exclude assistant behavior instructions (formatting, tone, etc.)
- Each fact: one concise line, preserve specifics (e.g. "Britz Voyager 2-Berth RV" not just "RV").

Respond ONLY with JSON (no markdown):
{{"add": ["new fact 1"], "update": {{"old fact": "updated fact"}}}}"""

COMPRESS_L1_PROMPT = """\
Given these topic-level conclusions from an ongoing conversation, create a single \
global summary that captures the overall direction and current state.

{conclusions_by_topic}

Instructions:
- Compress into 1-3 sentences
- Focus on the highest-level goals, key decisions, and current overall state
- This should give someone a quick understanding of "what is happening and where we are"
- Use the conclusions' own language

Respond ONLY with a JSON object (no markdown, no extra text):
{{"summary": "your global summary here"}}"""


@dataclass
class CompressResult:
    """L2 压缩结果。"""

    conclusions: list[str]
    topic_label: str
    usage: dict
    edges: list[Edge] = field(default_factory=list)


def _make_edges_from_messages(
    messages: list[dict],
    conclusion_count: int,
) -> list[Edge]:
    """
    基于原文对话消息的关系创建边。
    - 同一聚类内相邻结论 → 边连接
    - 边基于原文消息的相邻关系
    """
    edges: list[Edge] = []
    seen: set[tuple[str, str]] = set()

    if conclusion_count <= 1:
        return edges

    # 假设结论按原文顺序依次产出
    # 每个结论对应一段连续的消息
    msg_count = len(messages)
    if msg_count == 0:
        return edges

    # 计算每个结论对应消息范围，然后判断是否相邻
    msgs_per_conclusion = max(1, msg_count // conclusion_count)

    for i in range(conclusion_count):
        start_i = i * msgs_per_conclusion
        end_i = start_i + msgs_per_conclusion if i < conclusion_count - 1 else msg_count

        for j in range(i + 1, conclusion_count):
            start_j = j * msgs_per_conclusion
            end_j = start_j + msgs_per_conclusion if j < conclusion_count - 1 else msg_count

            # 判断两个结论对应的消息范围是否相邻（相交或相邻）
            ranges_overlap = (start_i < end_j) and (start_j < end_i)
            ranges_adjacent = (end_i == start_j) or (end_j == start_i)

            if ranges_overlap or ranges_adjacent:
                src, dst = f"c{i}", f"c{j}"
                key = (min(src, dst), max(src, dst))
                if key not in seen:
                    seen.add(key)
                    # 有重叠的边权重更高
                    weight = 1.0 if ranges_overlap else 0.8
                    edges.append(Edge(src_id=src, dst_id=dst, relation="sequential", weight=weight))

    return edges


def _parse_json(text: str) -> dict:
    """从 LLM 响应中提取 JSON，兼容 markdown 代码块包裹。"""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


class Compressor:
    """LLM 压缩器：支持 OpenAI / Anthropic，或复用宿主 agent 的 llm_fn。"""

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4o-mini",
        llm_fn: LLMFn | None = None,
    ) -> None:
        self.llm_provider = llm_provider
        self.model = model
        self.llm_fn = llm_fn
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.llm_provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic()
        else:
            import openai
            base_url = os.environ.get("OPENAI_BASE_URL")
            self._client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        return self._client

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> tuple[str, dict]:
        """调用 LLM，返回 (response_text, usage)。优先使用宿主 agent 的 llm_fn。"""
        if self.llm_fn is not None:
            text, usage = self.llm_fn(prompt, max_tokens)
            return text, usage or {"input_tokens": 0, "output_tokens": 0}

        client = self._get_client()
        if self.llm_provider == "anthropic":
            resp = client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        else:
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            usage = {
                "input_tokens": getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
                "output_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
            }
        return text, usage

    def compress_cluster(
        self,
        messages: list[dict],
        existing_topics: list[str] | None = None,
    ) -> CompressResult:
        """将一段对话压缩为 1-3 条 L2 结论 + 话题标签。"""
        formatted = "\n".join(
            f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
            for msg in messages
        )
        prompt = COMPRESS_CLUSTER_PROMPT.format(messages=formatted)

        text, usage = self._call_llm(prompt)
        try:
            data = _parse_json(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Compressor JSON parse failed, using raw text as conclusion")
            return CompressResult(
                conclusions=[text.strip()[:200]],
                topic_label="general",
                usage=usage,
            )

        conclusions = data.get("conclusions", [])
        if not conclusions:
            conclusions = [text.strip()[:200]]
        topic_label = str(data.get("topic_label", "general"))[:30]

        # 基于原文消息创建边
        edges = _make_edges_from_messages(messages, len(conclusions))

        return CompressResult(
            conclusions=conclusions,
            topic_label=topic_label,
            usage=usage,
            edges=edges,
        )

    def condense_for_profile(self, conclusions: list[str]) -> tuple[str, dict]:
        """将多条 conclusions 压缩为一条 profile 候选事实。"""
        if len(conclusions) <= 1:
            return (conclusions[0] if conclusions else ""), {"input_tokens": 0, "output_tokens": 0}

        conclusions_text = "\n".join(f"- {c}" for c in conclusions)
        prompt = PROFILE_CONDENSE_PROMPT.format(conclusions=conclusions_text)
        text, usage = self._call_llm(prompt, max_tokens=150)
        condensed = text.strip().strip('"').strip()
        return condensed, usage

    def compress_profile(
        self,
        current_facts: list[str],
        new_conclusions: list[str],
    ) -> tuple[list[str], list[tuple[int, str]], list[int], dict]:
        """Diff 模式更新 profile card：LLM 做 add/update，代码管上限淘汰。

        Returns:
            (to_add, to_update, to_remove, usage)
            to_add: 新增事实文本列表
            to_update: [(fact_index, new_text), ...] 更新列表
            to_remove: [] (always empty, removal is code-managed)
            usage: token 用量
        """
        if not new_conclusions:
            return [], [], [], {"input_tokens": 0, "output_tokens": 0}

        # Format current profile for prompt
        if current_facts:
            profile_text = "\n".join(f"- {f}" for f in current_facts)
        else:
            profile_text = "(empty)"

        conclusions_text = "\n".join(f"- {c}" for c in new_conclusions)
        prompt = PROFILE_DIFF_PROMPT.format(
            current_profile=profile_text,
            new_conclusions=conclusions_text,
            profile_count=len(current_facts),
        )

        text, usage = self._call_llm(prompt, max_tokens=400)
        try:
            data = _parse_json(text)
        except (json.JSONDecodeError, ValueError):
            try:
                import re
                add_m = re.search(r'"add"\s*:\s*\[(.*?)\]', text, re.DOTALL)
                adds = []
                if add_m:
                    adds = [s.strip().strip('"') for s in add_m.group(1).split('",') if s.strip().strip('"')]
                data = {"add": adds, "update": {}}
            except Exception:
                logger.warning("Profile diff parse failed, returning empty diff")
                return [], [], [], usage

        # Parse adds
        to_add = []
        for item in data.get("add", []):
            clean = item.strip().lstrip("- ").strip()
            if clean:
                if not any(clean[:30] in f for f in current_facts):
                    to_add.append(clean)

        # Parse updates: match old text to fact index
        to_update: list[tuple[int, str]] = []
        for old_text, new_text in data.get("update", {}).items():
            old_clean = old_text.strip().lstrip("- ").strip()
            new_clean = new_text.strip().lstrip("- ").strip()
            if not new_clean:
                continue
            for i, fact in enumerate(current_facts):
                if old_clean == fact or old_clean in fact:
                    to_update.append((i, new_clean))
                    break

        return to_add, to_update, [], usage

    def compress_l1(self, l2_by_topic: dict[str, list[str]]) -> tuple[str, dict]:
        """将所有 L2 结论压缩为一条 L1 全局摘要。"""
        if not l2_by_topic:
            return "", {"input_tokens": 0, "output_tokens": 0}

        parts = []
        for topic, conclusions in l2_by_topic.items():
            parts.append(f"[{topic}]")
            for c in conclusions:
                parts.append(f"  - {c}")
        formatted = "\n".join(parts)
        prompt = COMPRESS_L1_PROMPT.format(conclusions_by_topic=formatted)

        text, usage = self._call_llm(prompt, max_tokens=300)
        try:
            data = _parse_json(text)
            summary = str(data.get("summary", text.strip()))
        except (json.JSONDecodeError, ValueError):
            summary = text.strip()[:300]

        return summary, usage
