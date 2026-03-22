"""RealMem Benchmark：多 persona × 跨 session 记忆召回评估。

基于 RealMemBench 数据集，评估 MemGraph 在真实多 session 场景下的记忆检索能力。

评估协议：
1. 按时间顺序遍历 persona 的所有 session
2. 逐 turn 调用 mg.add_turn() 摄入
3. 遇到 is_query=true 时，先 mg.activate() 检索，再继续摄入
4. 对比检索结果与 ground-truth memory_used

用法:
    # dry-run（不调 LLM，只验证流程）
    python -m tests.benchmark.run_realmem_benchmark --dry-run

    # 跑单个 persona
    python -m tests.benchmark.run_realmem_benchmark --persona Lin_Wanyu

    # 限制 session 数（调试）
    python -m tests.benchmark.run_realmem_benchmark --persona Lin_Wanyu --limit-sessions 10

    # 完整运行
    python -m tests.benchmark.run_realmem_benchmark --output realmem_results.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# ── 加载 .env ──
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
if _ENV_PATH.exists():
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from tests.benchmark.realmem_loader import (
    extract_queries,
    iter_sessions,
    list_personas,
    load_persona,
)

EVAL_MODEL = os.environ.get("REALMEM_EVAL_MODEL", "deepseek-chat")
JUDGE_MODEL = os.environ.get("REALMEM_JUDGE_MODEL", "deepseek-chat")

def _get_openai_client():
    import openai

    base_url = os.environ.get("OPENAI_BASE_URL")
    return openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()


# Anthropic API key for judge (test key)
_ANTHROPIC_JUDGE_KEY = os.environ.get("ANTHROPIC_JUDGE_KEY", "")


def llm_judge_memory_recall(
    activate_result: str,
    ground_truth_memories: list[dict],
) -> tuple[dict, dict]:
    """用 LLM 判断 activate 结果是否语义覆盖了 ground-truth memory points。

    Args:
        activate_result: mg.activate() 返回的文本
        ground_truth_memories: [{session_uuid, content}, ...]

    Returns:
        (judge_result, usage)
        judge_result: {hits, total, details: [bool, ...]}
    """
    if not ground_truth_memories:
        return {"hits": 0, "total": 0, "details": []}, {"input_tokens": 0, "output_tokens": 0}

    memory_contents = [m.get("content", "") for m in ground_truth_memories]

    items_str = "\n".join(
        f"  {i + 1}. {content}" for i, content in enumerate(memory_contents)
    )
    prompt = f"""Please determine whether the following retrieved text semantically covers each of the expected memory items.

Retrieved text:
{activate_result}

Expected memory items:
{items_str}

For each memory item, determine whether the retrieved text contains the semantic meaning of that item (exact wording is not required, but the core meaning must be present).

Output strictly in this JSON format, nothing else:
{{"results": [true/false, true/false, ...]}}

Only output JSON, no explanation."""

    try:
        # Use Anthropic judge if configured
        if JUDGE_MODEL.startswith("claude") and _ANTHROPIC_JUDGE_KEY:
            import anthropic
            client = anthropic.Anthropic(api_key=_ANTHROPIC_JUDGE_KEY)
            response = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        else:
            client = _get_openai_client()
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = (response.choices[0].message.content or "").strip()
            u = getattr(response, "usage", None)
            usage = {
                "input_tokens": getattr(u, "prompt_tokens", 0) or 0 if u else 0,
                "output_tokens": getattr(u, "completion_tokens", 0) or 0 if u else 0,
            }

        if "{" in text:
            json_str = text[text.index("{") : text.rindex("}") + 1]
            parsed = json.loads(json_str)
            results = parsed.get("results", [])
            hits = sum(1 for r in results if r)
            return {"hits": hits, "total": len(memory_contents), "details": results}, usage
    except Exception as e:
        import traceback
        traceback.print_exc()
    # fallback: substring matching
    lower = activate_result.lower()
    hits = sum(1 for c in memory_contents if c.lower()[:50] in lower)
    return {
        "hits": hits,
        "total": len(memory_contents),
        "details": None,
    }, {"input_tokens": 0, "output_tokens": 0}



def run_persona_benchmark(
    persona_name: str,
    eval_model: str = EVAL_MODEL,
    judge_model: str = JUDGE_MODEL,
    limit_sessions: int | None = None,
    dry_run: bool = False,
    static: bool = False,
    activator_mode: str = "layered",
) -> list[dict]:
    """对单个 persona 运行 RealMem benchmark。

    Args:
        static: 如果 True，先全量摄入所有 session，再统一跑 query。
                GT memory 是最终累积 snapshot，静态模式与 GT 设计一致。
    """
    global JUDGE_MODEL
    JUDGE_MODEL = judge_model

    persona_data = load_persona(persona_name)
    meta = persona_data.get("_metadata", {})
    dialogues = persona_data.get("dialogues", [])

    if limit_sessions:
        dialogues = dialogues[:limit_sessions]
        # 覆盖以影响 iter_sessions
        persona_data = {**persona_data, "dialogues": dialogues}

    total_sessions = len(dialogues)
    query_info = extract_queries(persona_data)
    total_queries = len(query_info)

    mode_label = "STATIC" if static else "INCREMENTAL"
    print(f"\n{'=' * 60}")
    print(f"Persona: {meta.get('person_name', persona_name)}  [{mode_label}]")
    print(f"Sessions: {total_sessions}  |  Queries: {total_queries}")
    print(f"Eval model: {eval_model}  |  Judge model: {judge_model}")
    print(f"{'=' * 60}")

    if dry_run:
        results = []
        for q in query_info:
            results.append({
                "persona": persona_name,
                "query_id": q["query_id"],
                "query_content": q["query_content"][:100],
                "topic": q["topic"],
                "category": q["category_name"],
                "session_index": q["session_index"],
                "gt_memory_count": len(q["ground_truth_memories"]),
                "recall": -1.0,
                "dry_run": True,
            })
        print(f"  → {len(results)} queries found (dry-run)")
        return results

    # ── 实际运行 ──
    if activator_mode == "attention":
        from memgraph.attention_router import AttentionRouter
        ar = AttentionRouter(llm_provider="openai", model=eval_model)
        if static:
            return _run_static_attention(ar, persona_name, persona_data, query_info, total_sessions, total_queries)
        else:
            return _run_incremental_attention(ar, persona_name, persona_data, query_info, total_sessions, total_queries)

    if activator_mode in ("full_context", "h2o"):
        return _run_context_baseline(
            activator_mode, persona_name, persona_data, query_info,
            total_sessions, total_queries,
        )

    if activator_mode == "mem0":
        return _run_mem0_baseline(
            persona_name, persona_data, query_info,
            total_sessions, total_queries,
        )

    if activator_mode == "summary":
        return _run_summary_baseline(
            persona_name, persona_data, query_info,
            total_sessions, total_queries, eval_model,
        )

    from memgraph import MemGraph
    mg = MemGraph(llm_provider="openai", model=eval_model, activator_mode=activator_mode)

    if static:
        return _run_static(mg, persona_name, persona_data, query_info, total_sessions, total_queries)
    else:
        return _run_incremental(mg, persona_name, persona_data, query_info, total_sessions, total_queries)


def _run_static(mg, persona_name, persona_data, query_info, total_sessions, total_queries):
    """静态模式：先全量摄入，再统一 query。"""
    # Phase 1: 全量摄入
    print(f"\n  Phase 1: Ingesting all {total_sessions} sessions...")
    for si, session in iter_sessions(persona_data):
        turns = session.get("dialogue_turns", [])
        for ti, turn in enumerate(turns):
            speaker = turn.get("speaker", "").lower()
            content = turn.get("content", "")
            role = "user" if speaker == "user" else "assistant"
            mg.add_turn({"role": role, "content": content})

        if (si + 1) % 10 == 0 or si == total_sessions - 1:
            print(
                f"    [{si + 1}/{total_sessions}] "
                f"nodes={len(mg._graph.nodes)}  "
                f"topics={len(mg._graph.topics)}  "
                f"raw_traces={len(mg._graph.raw_traces)}"
            )

    final_encode_usage = mg.encode_usage
    graph_snapshot = {
        "node_count": len(mg._graph.nodes),
        "topic_count": len(mg._graph.topics),
        "raw_trace_count": len(mg._graph.raw_traces),
    }
    print(f"  Phase 1 done. Graph: {graph_snapshot}")

    # Phase 2: 统一 query
    print(f"\n  Phase 2: Running {total_queries} queries...")
    results = []
    for qi, q in enumerate(query_info):
        content = q["query_content"]
        gt_memories = q["ground_truth_memories"]

        t0 = time.time()
        result_text = mg.activate(content)
        retrieval_latency = time.time() - t0

        judge_result, judge_usage = llm_judge_memory_recall(result_text, gt_memories)
        recall = (
            judge_result["hits"] / judge_result["total"]
            if judge_result["total"] > 0
            else 0.0
        )

        results.append({
            "persona": persona_name,
            "query_id": q["query_id"],
            "query_content": content[:200],
            "topic": q["topic"],
            "category": q["category_name"],
            "session_index": q["session_index"],
            "session_identifier": q.get("session_identifier", ""),
            "turn_index": q["turn_index"],
            "result_text": result_text or "",
            "gt_memory_count": len(gt_memories),
            "gt_memories": [m.get("content", "") for m in gt_memories],
            "recall": recall,
            "hits": judge_result["hits"],
            "total": judge_result["total"],
            "judge_details": judge_result.get("details"),
            "retrieval_latency": retrieval_latency,
            "judge_usage": judge_usage,
            "encode_usage_snapshot": dict(final_encode_usage),
            "graph_snapshot": graph_snapshot,
            "final_encode_usage": dict(final_encode_usage),
            "mode": "static",
        })

        if (qi + 1) % 10 == 0 or qi == total_queries - 1:
            avg_recall = statistics.mean([r["recall"] for r in results])
            print(
                f"    [{qi + 1}/{total_queries}] "
                f"avg_recall: {avg_recall:.1%}  "
                f"latency: {retrieval_latency:.2f}s"
            )

    return results


def _run_static_attention(ar, persona_name, persona_data, query_info, total_sessions, total_queries):
    """AttentionRouter 静态模式：先全量存储原文，再统一 query。"""
    print(f"\n  Phase 1: Storing all {total_sessions} sessions (zero compression)...")
    for si, session in iter_sessions(persona_data):
        turns = session.get("dialogue_turns", [])
        # 按对话轮次配对存储
        i = 0
        while i < len(turns):
            user_text = ""
            assistant_text = ""
            if turns[i].get("speaker", "").lower() == "user":
                user_text = turns[i].get("content", "")
                if i + 1 < len(turns) and turns[i + 1].get("speaker", "").lower() == "assistant":
                    assistant_text = turns[i + 1].get("content", "")
                    i += 2
                else:
                    i += 1
            else:
                assistant_text = turns[i].get("content", "")
                i += 1
            if user_text or assistant_text:
                ar.encode(user_text, assistant_text)

        if (si + 1) % 10 == 0 or si == total_sessions - 1:
            print(f"    [{si + 1}/{total_sessions}] turns_stored={len(ar.turns)}")

    print(f"  Phase 1 done. Total turns: {len(ar.turns)}")

    # Phase 2: 统一 query
    print(f"\n  Phase 2: Running {total_queries} queries...")
    results = []
    for qi, q in enumerate(query_info):
        content = q["query_content"]
        gt_memories = q["ground_truth_memories"]

        t0 = time.time()
        result = ar.activate(content)
        result_text = str(result)
        retrieval_latency = time.time() - t0

        judge_result, judge_usage = llm_judge_memory_recall(result_text, gt_memories)
        recall = (
            judge_result["hits"] / judge_result["total"]
            if judge_result["total"] > 0
            else 0.0
        )
        results.append({
            "persona": persona_name,
            "query_id": q["query_id"],
            "query_content": content[:200],
            "topic": q["topic"],
            "category": q["category_name"],
            "session_index": q["session_index"],
            "session_identifier": q.get("session_identifier", ""),
            "turn_index": q["turn_index"],
            "result_text": result_text or "",
            "gt_memory_count": len(gt_memories),
            "gt_memories": [m.get("content", "") for m in gt_memories],
            "recall": recall,
            "hits": judge_result["hits"],
            "total": judge_result["total"],
            "judge_details": judge_result.get("details"),
            "retrieval_latency": retrieval_latency,
            "judge_usage": judge_usage,
            "encode_usage_snapshot": ar.encode_usage,
            "graph_snapshot": ar.inspect(),
            "final_encode_usage": ar.encode_usage,
            "mode": "static_attention",
        })
        if (qi + 1) % 10 == 0 or qi == total_queries - 1:
            avg_recall = statistics.mean([r["recall"] for r in results])
            print(f"    [{qi + 1}/{total_queries}] avg_recall: {avg_recall:.1%}  latency: {retrieval_latency:.2f}s")

    return results


def _run_context_baseline(mode, persona_name, persona_data, query_info, total_sessions, total_queries):
    """Full context / H2O baseline。

    full_context: 全量历史原文
    h2o: 模拟 H2O 的 KV cache eviction —— 保留 recent turns + heavy hitter turns。
         Heavy hitters = 累积 attention score 最高的 turns（用 cosine similarity 作为
         attention score 的代理：每来一个新 turn，跟所有历史 turns 算 cosine，累加到各
         turn 的 attention accumulator 上）。
         Budget: 20% heavy hitters + 最近 10% recent turns。
    """
    import numpy as np
    from memgraph.embedder import Embedder

    results = []
    queries_done = 0

    # 每个 turn 存：text, embedding, attention_accumulator
    all_turns_text: list[str] = []
    all_turns_emb: list[np.ndarray] = []
    all_turns_attn: list[float] = []  # accumulated attention score

    embedder = Embedder() if mode == "h2o" else None

    query_lookup: dict[tuple[int, int], dict] = {}
    for q in query_info:
        query_lookup[(q["session_index"], q["turn_index"])] = q

    for si, session in iter_sessions(persona_data):
        t_session_start = time.time()
        turns = session.get("dialogue_turns", [])
        session_id = session.get("session_identifier", f"S{si}")

        for ti, turn in enumerate(turns):
            speaker = turn.get("speaker", "").lower()
            content = turn.get("content", "")

            # 检查是否是 query 点
            if turn.get("is_query") and (si, ti) in query_lookup:
                q = query_lookup[(si, ti)]
                gt_memories = q["ground_truth_memories"]

                t0 = time.time()

                if mode == "h2o" and len(all_turns_text) > 0:
                    n = len(all_turns_text)
                    # Budget: 20% heavy hitters + 10% recent, 合计不超过 30%
                    n_recent = max(1, int(n * 0.10))
                    n_heavy = max(1, int(n * 0.20))

                    # Recent turns (最后 n_recent 个)
                    recent_indices = set(range(n - n_recent, n))

                    # Heavy hitter turns (attention accumulator 最高的，排除已在 recent 里的)
                    scored = [(i, all_turns_attn[i]) for i in range(n) if i not in recent_indices]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    heavy_indices = set(idx for idx, _ in scored[:n_heavy])

                    # 合并，按原始时序排列
                    keep_indices = sorted(recent_indices | heavy_indices)
                    context_turns = [all_turns_text[i] for i in keep_indices]
                elif mode == "full_context":
                    context_turns = all_turns_text
                    keep_indices = list(range(len(all_turns_text)))
                else:
                    context_turns = all_turns_text
                    keep_indices = list(range(len(all_turns_text)))

                result_text = "\n".join(context_turns)
                retrieval_latency = time.time() - t0

                judge_result, judge_usage = llm_judge_memory_recall(
                    result_text, gt_memories
                )
                recall = (
                    judge_result["hits"] / judge_result["total"]
                    if judge_result["total"] > 0
                    else 0.0
                )

                results.append({
                    "persona": persona_name,
                    "query_id": q["query_id"],
                    "query_content": content[:200],
                    "topic": q["topic"],
                    "category": q["category_name"],
                    "session_index": si,
                    "session_identifier": session_id,
                    "turn_index": ti,
                    "result_text": "",  # too large to store
                    "gt_memory_count": len(gt_memories),
                    "gt_memories": [m.get("content", "") for m in gt_memories],
                    "recall": recall,
                    "hits": judge_result["hits"],
                    "total": judge_result["total"],
                    "judge_details": judge_result.get("details"),
                    "retrieval_latency": retrieval_latency,
                    "judge_usage": judge_usage,
                    "encode_usage_snapshot": {"input_tokens": 0, "output_tokens": 0},
                    "graph_snapshot": {
                        "total_turns": len(all_turns_text),
                        "context_turns": len(context_turns),
                        "context_chars": len(result_text),
                        "keep_ratio": len(keep_indices) / max(1, len(all_turns_text)),
                        "type": mode,
                    },
                    "mode": mode,
                })

                queries_done += 1

            # 累积历史
            role_label = "User" if speaker == "user" else "Assistant"
            turn_text = f"[{role_label}]: {content}"
            all_turns_text.append(turn_text)

            if mode == "h2o":
                # Embed this turn
                vec = np.array(embedder.embed_query(content), dtype=np.float32)
                # Update attention accumulators: add cosine similarity to all existing turns
                for i, prev_emb in enumerate(all_turns_emb):
                    na, nb = np.linalg.norm(vec), np.linalg.norm(prev_emb)
                    if na > 1e-8 and nb > 1e-8:
                        sim = float(np.dot(vec, prev_emb) / (na * nb))
                        all_turns_attn[i] += max(0, sim)  # only accumulate positive attention
                all_turns_emb.append(vec)
                all_turns_attn.append(0.0)  # new turn starts with 0 accumulated attention

        session_time = time.time() - t_session_start

        if queries_done > 0 and ((si + 1) % 5 == 0 or si == total_sessions - 1):
            avg_recall = statistics.mean([r["recall"] for r in results])
            print(
                f"  [{si + 1}/{total_sessions}] queries: {queries_done}/{total_queries}  "
                f"avg_recall: {avg_recall:.1%}  "
                f"session_time: {session_time:.1f}s"
            )

    return results


def _run_mem0_baseline(persona_name, persona_data, query_info, total_sessions, total_queries):
    """Mem0 baseline: uses mem0 library for memory storage and retrieval.

    Static mode: ingest all sessions → then run all queries.
    """
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig

    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")

    config = MemoryConfig(
        llm={
            "provider": "openai",
            "config": {
                "model": "deepseek-chat",
                "api_key": api_key,
                "openai_base_url": base_url,
            }
        },
        embedder={
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small",
                "api_key": api_key,
                "openai_base_url": base_url,
            }
        },
    )

    try:
        mem = Memory(config)
    except Exception as e:
        # Fallback: if DeepSeek doesn't support embeddings, use huggingface
        print(f"  ⚠️ OpenAI embedder failed ({e}), falling back to huggingface embedder")
        config = MemoryConfig(
            llm={
                "provider": "openai",
                "config": {
                    "model": "deepseek-chat",
                    "api_key": api_key,
                    "openai_base_url": base_url,
                }
            },
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                }
            },
        )
        mem = Memory(config)

    user_id = f"bench_{persona_name}"

    # Phase 1: Ingest all sessions
    print("  [mem0] Phase 1: Ingesting all sessions...")
    for si, session in iter_sessions(persona_data):
        turns = session.get("dialogue_turns", [])
        session_id = session.get("session_identifier", f"S{si}")

        session_text = "\n".join(
            f"{t.get('speaker', 'user').capitalize()}: {t.get('content', '')}"
            for t in turns
            if not t.get("is_query")
        )

        if session_text.strip():
            try:
                mem.add(
                    session_text,
                    user_id=user_id,
                    metadata={"session_id": session_id},
                )
            except Exception as e:
                print(f"  ⚠️ mem0 add failed for session {si}: {e}")

        if (si + 1) % 10 == 0 or si == total_sessions - 1:
            print(f"  [mem0] Ingested {si + 1}/{total_sessions} sessions")

    # Phase 2: Run queries
    print("  [mem0] Phase 2: Running queries...")
    results = []
    for qi, q in enumerate(query_info):
        query_text = q["query_content"]
        gt_memories = q["ground_truth_memories"]

        t0 = time.time()

        try:
            search_results = mem.search(query=query_text, user_id=user_id, limit=10)
            if search_results and "results" in search_results:
                retrieved = [r.get("memory", "") for r in search_results["results"]]
            else:
                retrieved = []
        except Exception as e:
            print(f"  ⚠️ mem0 search failed for query {qi}: {e}")
            retrieved = []

        result_text = "\n".join(retrieved) if retrieved else "(no memories found)"
        retrieval_latency = time.time() - t0

        judge_result, judge_usage = llm_judge_memory_recall(result_text, gt_memories)
        recall = judge_result["hits"] / judge_result["total"] if judge_result["total"] > 0 else 0.0

        results.append({
            "persona": persona_name,
            "query_id": q["query_id"],
            "query_content": query_text[:200],
            "topic": q["topic"],
            "category": q["category_name"],
            "session_index": q["session_index"],
            "session_identifier": q.get("session_identifier", ""),
            "turn_index": q["turn_index"],
            "result_text": result_text[:500],
            "gt_memory_count": len(gt_memories),
            "gt_memories": [m.get("content", "") for m in gt_memories],
            "recall": recall,
            "hits": judge_result["hits"],
            "total": judge_result["total"],
            "judge_details": judge_result.get("details"),
            "retrieval_latency": retrieval_latency,
            "judge_usage": judge_usage,
            "encode_usage_snapshot": {},
            "graph_snapshot": {"type": "mem0", "retrieved_count": len(retrieved)},
            "mode": "mem0",
        })

        if (qi + 1) % 5 == 0 or qi == len(query_info) - 1:
            avg_recall = statistics.mean([r["recall"] for r in results])
            print(f"  [mem0] queries: {qi + 1}/{len(query_info)}  avg_recall: {avg_recall:.1%}")

    return results


def _run_summary_baseline(persona_name, persona_data, query_info, total_sessions, total_queries, eval_model):
    """Summary-based baseline: maintains a rolling LLM-generated summary.

    Similar to ChatGPT's conversation summary approach.
    Static mode: summarize all sessions → then run all queries against the summary.
    """
    client = _get_openai_client()
    summary = ""

    _SUMMARY_UPDATE_PROMPT = """Update this conversation summary with the new session content.
Keep ALL specific facts: numbers, dates, names, decisions, preferences, plans, constraints.
Do not lose any concrete details — they are more important than narrative flow.
Keep the summary under 4000 characters.

Current summary:
{summary}

New session content:
{session_text}

Updated summary:"""

    # Phase 1: Build rolling summary
    print("  [summary] Phase 1: Building rolling summary...")
    for si, session in iter_sessions(persona_data):
        turns = session.get("dialogue_turns", [])

        session_text = "\n".join(
            f"{t.get('speaker', 'user').capitalize()}: {t.get('content', '')}"
            for t in turns
            if not t.get("is_query")
        )

        if not session_text.strip():
            continue

        # Truncate session text if too long
        if len(session_text) > 6000:
            session_text = session_text[:6000] + "\n... (truncated)"

        prompt = _SUMMARY_UPDATE_PROMPT.format(
            summary=summary if summary else "(empty — first session)",
            session_text=session_text,
        )

        try:
            resp = client.chat.completions.create(
                model=eval_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = resp.choices[0].message.content or summary
        except Exception as e:
            print(f"  ⚠️ summary update failed for session {si}: {e}")

        if (si + 1) % 10 == 0 or si == total_sessions - 1:
            print(f"  [summary] Processed {si + 1}/{total_sessions} sessions  summary_len={len(summary)}")

    print(f"  [summary] Final summary length: {len(summary)} chars")

    # Phase 2: Run queries
    print("  [summary] Phase 2: Running queries...")
    results = []
    for qi, q in enumerate(query_info):
        query_text = q["query_content"]
        gt_memories = q["ground_truth_memories"]

        t0 = time.time()
        result_text = summary
        retrieval_latency = time.time() - t0

        judge_result, judge_usage = llm_judge_memory_recall(result_text, gt_memories)
        recall = judge_result["hits"] / judge_result["total"] if judge_result["total"] > 0 else 0.0

        results.append({
            "persona": persona_name,
            "query_id": q["query_id"],
            "query_content": query_text[:200],
            "topic": q["topic"],
            "category": q["category_name"],
            "session_index": q["session_index"],
            "session_identifier": q.get("session_identifier", ""),
            "turn_index": q["turn_index"],
            "result_text": "",
            "gt_memory_count": len(gt_memories),
            "gt_memories": [m.get("content", "") for m in gt_memories],
            "recall": recall,
            "hits": judge_result["hits"],
            "total": judge_result["total"],
            "judge_details": judge_result.get("details"),
            "retrieval_latency": retrieval_latency,
            "judge_usage": judge_usage,
            "encode_usage_snapshot": {},
            "graph_snapshot": {"type": "summary", "summary_length": len(summary)},
            "mode": "summary",
        })

        if (qi + 1) % 5 == 0 or qi == len(query_info) - 1:
            avg_recall = statistics.mean([r["recall"] for r in results])
            print(f"  [summary] queries: {qi + 1}/{len(query_info)}  avg_recall: {avg_recall:.1%}")

    return results


def _run_incremental_attention(ar, persona_name, persona_data, query_info, total_sessions, total_queries):
    """AttentionRouter 增量模式：边存原文边 query。"""
    results = []
    queries_done = 0
    session_times: list[float] = []

    query_lookup: dict[tuple[int, int], dict] = {}
    for q in query_info:
        query_lookup[(q["session_index"], q["turn_index"])] = q

    # 跟踪当前对话对
    pending_user_text = ""

    for si, session in iter_sessions(persona_data):
        t_session_start = time.time()
        turns = session.get("dialogue_turns", [])
        session_id = session.get("session_identifier", f"S{si}")

        for ti, turn in enumerate(turns):
            speaker = turn.get("speaker", "").lower()
            content = turn.get("content", "")

            # 检查是否是 query 点
            if turn.get("is_query") and (si, ti) in query_lookup:
                q = query_lookup[(si, ti)]
                gt_memories = q["ground_truth_memories"]

                t0 = time.time()
                result = ar.activate(content)
                result_text = str(result)
                retrieval_latency = time.time() - t0

                judge_result, judge_usage = llm_judge_memory_recall(
                    result_text, gt_memories
                )
                recall = (
                    judge_result["hits"] / judge_result["total"]
                    if judge_result["total"] > 0
                    else 0.0
                )

                results.append({
                    "persona": persona_name,
                    "query_id": q["query_id"],
                    "query_content": content[:200],
                    "topic": q["topic"],
                    "category": q["category_name"],
                    "session_index": si,
                    "session_identifier": session_id,
                    "turn_index": ti,
                    "result_text": result_text or "",
                    "gt_memory_count": len(gt_memories),
                    "gt_memories": [m.get("content", "") for m in gt_memories],
                    "recall": recall,
                    "hits": judge_result["hits"],
                    "total": judge_result["total"],
                    "judge_details": judge_result.get("details"),
                    "retrieval_latency": retrieval_latency,
                    "judge_usage": judge_usage,
                    "encode_usage_snapshot": ar.encode_usage,
                    "graph_snapshot": ar.inspect(),
                    "mode": "incremental_attention",
                })

                queries_done += 1

            # 存储对话
            if speaker == "user":
                pending_user_text = content
            elif speaker == "assistant" and pending_user_text:
                ar.encode(pending_user_text, content)
                pending_user_text = ""

        session_time = time.time() - t_session_start
        session_times.append(session_time)

        if queries_done > 0 and ((si + 1) % 5 == 0 or si == total_sessions - 1):
            avg_recall = statistics.mean([r["recall"] for r in results])
            print(
                f"  [{si + 1}/{total_sessions}] queries: {queries_done}/{total_queries}  "
                f"avg_recall: {avg_recall:.1%}  "
                f"session_time: {session_time:.1f}s"
            )

    return results


def _run_incremental(mg, persona_name, persona_data, query_info, total_sessions, total_queries):
    """增量模式：边摄入边 query（原逻辑）。"""
    results = []
    queries_done = 0
    session_times: list[float] = []

    # 构建 query 查找表：{(session_index, turn_index): query_info}
    query_lookup: dict[tuple[int, int], dict] = {}
    for q in query_info:
        query_lookup[(q["session_index"], q["turn_index"])] = q

    for si, session in iter_sessions(persona_data):
        t_session_start = time.time()
        turns = session.get("dialogue_turns", [])
        session_id = session.get("session_identifier", f"S{si}")

        for ti, turn in enumerate(turns):
            speaker = turn.get("speaker", "").lower()
            content = turn.get("content", "")
            role = "user" if speaker == "user" else "assistant"

            # 检查是否是 query 点
            if turn.get("is_query") and (si, ti) in query_lookup:
                q = query_lookup[(si, ti)]
                gt_memories = q["ground_truth_memories"]

                # 先 activate 再 add_turn
                t0 = time.time()
                result_text = mg.activate(content)
                retrieval_latency = time.time() - t0

                # 评估
                judge_result, judge_usage = llm_judge_memory_recall(
                    result_text, gt_memories
                )
                recall = (
                    judge_result["hits"] / judge_result["total"]
                    if judge_result["total"] > 0
                    else 0.0
                )

                # 记录当前 encode 用量快照
                current_encode_usage = mg.encode_usage

                results.append({
                    "persona": persona_name,
                    "query_id": q["query_id"],
                    "query_content": content[:200],
                    "topic": q["topic"],
                    "category": q["category_name"],
                    "session_index": si,
                    "session_identifier": session_id,
                    "turn_index": ti,
                    "result_text": result_text or "",
                    "gt_memory_count": len(gt_memories),
                    "gt_memories": [m.get("content", "") for m in gt_memories],
                    "recall": recall,
                    "hits": judge_result["hits"],
                    "total": judge_result["total"],
                    "judge_details": judge_result.get("details"),
                    "retrieval_latency": retrieval_latency,
                    "judge_usage": judge_usage,
                    "encode_usage_snapshot": current_encode_usage,
                    "graph_snapshot": {
                        "node_count": len(mg._graph.nodes),
                        "topic_count": len(mg._graph.topics),
                        "raw_trace_count": len(mg._graph.raw_traces),
                    },
                })
                queries_done += 1

            # 摄入 turn
            mg.add_turn({"role": role, "content": content})

        elapsed = time.time() - t_session_start
        session_times.append(elapsed)

        # 进度输出
        if (si + 1) % 10 == 0 or si == total_sessions - 1:
            done_q = queries_done
            avg_recall = (
                statistics.mean([r["recall"] for r in results])
                if results
                else 0.0
            )
            print(
                f"  [{si + 1}/{total_sessions}] "
                f"queries: {done_q}/{total_queries}  "
                f"avg_recall: {avg_recall:.1%}  "
                f"session_time: {elapsed:.1f}s"
            )

    # 最终 encode 用量
    final_encode_usage = mg.encode_usage
    for r in results:
        r["final_encode_usage"] = dict(final_encode_usage)

    return results


def print_realmem_report(results: list[dict]) -> None:
    """输出 RealMem benchmark 报告。"""
    if not results:
        print("No results to report.")
        return

    if results[0].get("dry_run"):
        _print_dry_run_report(results)
        return

    print(f"\n{'=' * 70}")
    print("RealMem Benchmark 报告")
    print(f"{'=' * 70}")

    # 总体 Recall
    recalls = [r["recall"] for r in results]
    avg = statistics.mean(recalls)
    std = statistics.stdev(recalls) if len(recalls) > 1 else 0
    print(f"\n总体 Memory Recall: {avg:.1%} ± {std:.1%}  (n={len(recalls)})")

    # 按 persona 分组
    by_persona: dict[str, list[float]] = {}
    for r in results:
        by_persona.setdefault(r["persona"], []).append(r["recall"])

    if len(by_persona) > 1:
        print(f"\n按 Persona:")
        for p, vals in sorted(by_persona.items()):
            a = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"  {p:<25} {a:>6.1%} ± {s:.1%}  (n={len(vals)})")

    # 按 category 分组
    by_category: dict[str, list[float]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, []).append(r["recall"])

    print(f"\n按 Category:")
    for cat, vals in sorted(by_category.items()):
        a = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  {cat:<25} {a:>6.1%} ± {s:.1%}  (n={len(vals)})")

    # 按 topic 分组
    by_topic: dict[str, list[float]] = {}
    for r in results:
        topic = r.get("topic", "unknown")
        by_topic.setdefault(topic, []).append(r["recall"])

    print(f"\n按 Topic:")
    for topic, vals in sorted(by_topic.items()):
        a = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0
        print(f"  {topic:<25} {a:>6.1%} ± {s:.1%}  (n={len(vals)})")

    # 延迟
    ret_lats = [r["retrieval_latency"] for r in results if "retrieval_latency" in r]
    if ret_lats:
        sorted_lats = sorted(ret_lats)
        p50 = sorted_lats[len(sorted_lats) // 2]
        p95 = sorted_lats[int(len(sorted_lats) * 0.95)]
        print(f"\n延迟:")
        print(f"  retrieval:  p50={p50 * 1000:.0f}ms  p95={p95 * 1000:.0f}ms")

    # Token 用量汇总
    # encode: MemGraph 的 extractor + compressor 消耗（按 persona 汇总）
    encode_by_persona: dict[str, dict] = {}
    for r in results:
        p = r["persona"]
        if p not in encode_by_persona:
            encode_by_persona[p] = r.get("final_encode_usage", {})
        else:
            # 取最后一条（最终累计值）
            eu = r.get("final_encode_usage", {})
            if eu.get("input_tokens", 0) > encode_by_persona[p].get("input_tokens", 0):
                encode_by_persona[p] = eu

    total_encode_input = sum(u.get("input_tokens", 0) for u in encode_by_persona.values())
    total_encode_output = sum(u.get("output_tokens", 0) for u in encode_by_persona.values())

    # judge: LLM 语义匹配消耗
    total_judge_input = sum(
        r.get("judge_usage", {}).get("input_tokens", 0) for r in results
    )
    total_judge_output = sum(
        r.get("judge_usage", {}).get("output_tokens", 0) for r in results
    )

    total_input = total_encode_input + total_judge_input
    total_output = total_encode_output + total_judge_output

    print(f"\n{'─' * 50}")
    print(f"Token 用量:")
    print(f"  {'':20} {'input':>10} {'output':>10} {'total':>10}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10}")
    print(
        f"  {'encode (MG)':20} {total_encode_input:>10} "
        f"{total_encode_output:>10} {total_encode_input + total_encode_output:>10}"
    )
    print(
        f"  {'judge (eval)':20} {total_judge_input:>10} "
        f"{total_judge_output:>10} {total_judge_input + total_judge_output:>10}"
    )
    print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10}")
    print(
        f"  {'TOTAL':20} {total_input:>10} "
        f"{total_output:>10} {total_input + total_output:>10}"
    )

    # 按 persona 细分 encode 用量
    if len(encode_by_persona) > 1:
        print(f"\n  Encode 按 Persona:")
        for p, u in sorted(encode_by_persona.items()):
            inp = u.get("input_tokens", 0)
            out = u.get("output_tokens", 0)
            print(f"    {p:<23} {inp:>10} {out:>10} {inp + out:>10}")

    # Graph 统计（最终状态）
    last_snap = results[-1].get("graph_snapshot", {}) if results else {}
    if last_snap:
        print(f"\nGraph 最终状态:")
        print(
            f"  nodes={last_snap.get('node_count', 0)}  "
            f"topics={last_snap.get('topic_count', 0)}  "
            f"raw_traces={last_snap.get('raw_trace_count', 0)}"
        )

    print(f"\n{'=' * 70}")


def _print_dry_run_report(results: list[dict]) -> None:
    """dry-run 报告。"""
    print(f"\n{'=' * 70}")
    print("RealMem Benchmark DRY-RUN 报告")
    print(f"{'=' * 70}")
    print(f"\n总 Query 数: {len(results)}")

    by_persona: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_topic: dict[str, int] = {}
    for r in results:
        by_persona[r["persona"]] = by_persona.get(r["persona"], 0) + 1
        by_category[r.get("category", "?")] = by_category.get(r.get("category", "?"), 0) + 1
        by_topic[r.get("topic", "?")] = by_topic.get(r.get("topic", "?"), 0) + 1

    print(f"\n按 Persona:")
    for p, c in sorted(by_persona.items()):
        print(f"  {p:<25} {c} queries")

    print(f"\n按 Category:")
    for cat, c in sorted(by_category.items()):
        print(f"  {cat:<25} {c} queries")

    print(f"\n按 Topic:")
    for t, c in sorted(by_topic.items()):
        print(f"  {t:<25} {c} queries")

    print(f"\n{'=' * 70}")


def save_realmem_results(results: list[dict], path: str | None = None) -> str:
    """保存结果到 JSON。"""
    if path is None:
        path = str(Path(__file__).parent / "realmem_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="RealMem Benchmark: 多 persona 跨 session 记忆召回评估"
    )
    parser.add_argument("--dry-run", action="store_true", help="不调 LLM，只验证流程")
    parser.add_argument(
        "--persona",
        type=str,
        default=None,
        help="Persona 名称，逗号分隔（默认全部）",
    )
    parser.add_argument(
        "--limit-sessions",
        type=int,
        default=None,
        help="限制每个 persona 的 session 数量（调试用）",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        default=EVAL_MODEL,
        help=f"MemGraph 使用的模型（默认: {EVAL_MODEL}）",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=JUDGE_MODEL,
        help=f"语义判断模型（默认: {JUDGE_MODEL}）",
    )
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    parser.add_argument(
        "--static", action="store_true",
        help="静态模式：先全量摄入所有 session，再统一跑 query（与 GT 设计一致）",
    )
    parser.add_argument(
        "--activator",
        type=str,
        default="layered",
        choices=["layered", "simple", "attention", "full_context", "h2o", "mem0", "summary"],
        help="Activator mode: attention / mem0 / summary / layered / simple / full_context / h2o",
    )
    parser.add_argument(
        "--list-personas", action="store_true", help="列出可用 persona 后退出"
    )
    args = parser.parse_args()

    if args.list_personas:
        personas = list_personas()
        print(f"\n可用 Personas ({len(personas)}):")
        print(f"  {'Name':<25} {'Sessions':>8} {'Tokens':>10} {'Queries':>8}")
        print(f"  {'-' * 25} {'-' * 8} {'-' * 10} {'-' * 8}")
        for p in personas:
            print(
                f"  {p['person_name']:<25} "
                f"{p['total_sessions']:>8} "
                f"{p['total_tokens']:>10} "
                f"{p['total_queries']:>8}"
            )
        return

    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print("请先设置 OPENAI_API_KEY（可放在项目根 .env 中）")
        sys.exit(1)

    # 确定要跑的 persona
    if args.persona:
        persona_names = [p.strip() for p in args.persona.split(",")]
    else:
        personas = list_personas()
        persona_names = [p["person_name"] for p in personas]

    all_results = []
    total_personas = len(persona_names)

    import gc

    for pi, pname in enumerate(persona_names):
        print(f"\n[{pi + 1}/{total_personas}] Running persona: {pname}")
        try:
            results = run_persona_benchmark(
                persona_name=pname,
                eval_model=args.eval_model,
                judge_model=args.judge_model,
                limit_sessions=args.limit_sessions,
                dry_run=args.dry_run,
                static=args.static,
                activator_mode=args.activator,
            )
            all_results.extend(results)
            # Incremental save: write after each persona so data is not lost on timeout
            if args.output:
                save_realmem_results(all_results, args.output)
                print(f"  → Incremental save: {len(all_results)} results to {args.output}")
        except Exception as e:
            print(f"  → ERROR: {e}")
        # 释放内存，避免多 persona 累积 OOM
        gc.collect()

    print_realmem_report(all_results)

    path = save_realmem_results(all_results, args.output)
    print(f"\n结果已保存到 {path}")


if __name__ == "__main__":
    main()
