"""RealMem benchmark 数据加载模块。

加载 RealMemBench 数据集（JSON 格式），提供统一接口供 benchmark 使用。
数据目录：tests/benchmark/realmem_data/dataset/
"""

import json
from pathlib import Path
from typing import Any

_DATASET_DIR = Path(__file__).parent / "realmem_data" / "dataset"


def list_personas() -> list[dict]:
    """列出所有可用 persona 及基本统计。"""
    results = []
    if not _DATASET_DIR.exists():
        return results
    for f in sorted(_DATASET_DIR.glob("*_dialogues_256k.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        meta = data.get("_metadata", {})
        dialogues = data.get("dialogues", [])
        total_queries = sum(
            1
            for s in dialogues
            for t in s.get("dialogue_turns", [])
            if t.get("is_query")
        )
        results.append({
            "person_name": meta.get("person_name", f.stem),
            "total_sessions": meta.get("total_sessions", len(dialogues)),
            "total_tokens": meta.get("total_tokens", 0),
            "total_queries": total_queries,
            "path": str(f),
        })
    return results


def load_persona(persona_name: str) -> dict[str, Any]:
    """加载指定 persona 的完整数据。

    Args:
        persona_name: persona 名称（如 'Lin_Wanyu'），不区分大小写下划线。

    Returns:
        包含 _metadata 和 dialogues 的完整数据 dict。

    Raises:
        FileNotFoundError: persona 数据文件不存在。
    """
    # 尝试精确匹配
    target = _DATASET_DIR / f"{persona_name}_dialogues_256k.json"
    if not target.exists():
        # 尝试模糊匹配
        candidates = list(_DATASET_DIR.glob("*_dialogues_256k.json"))
        matched = [
            c for c in candidates
            if persona_name.lower().replace(" ", "_") in c.stem.lower()
        ]
        if matched:
            target = matched[0]
        else:
            available = [c.stem.replace("_dialogues_256k", "") for c in candidates]
            raise FileNotFoundError(
                f"Persona '{persona_name}' not found. Available: {available}"
            )

    with open(target, encoding="utf-8") as f:
        return json.load(f)


def extract_queries(persona_data: dict) -> list[dict]:
    """从 persona 数据中提取所有 query 及其 ground truth。

    Returns:
        list of {
            session_identifier: str,
            session_uuid: str,
            session_index: int,
            turn_index: int,
            query_id: str,
            query_content: str,
            topic: str,
            category_name: str,
            ground_truth_memories: list[{session_uuid, content}],
            preceding_turns: int,  # 该 query 之前累计摄入的 turn 数
        }
    """
    dialogues = persona_data.get("dialogues", [])
    queries = []
    cumulative_turns = 0

    for si, session in enumerate(dialogues):
        turns = session.get("dialogue_turns", [])
        for ti, turn in enumerate(turns):
            if turn.get("is_query"):
                # ground truth 在下一个 assistant turn 的 memory_used 里
                gt_memories = []
                if ti + 1 < len(turns):
                    next_turn = turns[ti + 1]
                    gt_memories = next_turn.get("memory_used", [])

                queries.append({
                    "session_identifier": session.get("session_identifier", ""),
                    "session_uuid": session.get("session_uuid", ""),
                    "session_index": si,
                    "turn_index": ti,
                    "query_id": turn.get("query_id", ""),
                    "query_content": turn.get("content", ""),
                    "topic": turn.get("topic", ""),
                    "category_name": turn.get("category_name", ""),
                    "ground_truth_memories": gt_memories,
                    "preceding_turns": cumulative_turns + ti,
                })
            cumulative_turns += len(turns)
        # reset not needed: cumulative_turns accumulates across sessions

    return queries


def iter_sessions(persona_data: dict):
    """逐 session yield，方便 benchmark 按时间顺序遍历。

    Yields:
        (session_index, session_dict)
    """
    for i, session in enumerate(persona_data.get("dialogues", [])):
        yield i, session
