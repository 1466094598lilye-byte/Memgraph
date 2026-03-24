"""FastAPI server exposing MemGraph over HTTP.

Persistence: turns + memo saved to MEMGRAPH_DATA_DIR (default: /root/memgraph/data/).
On startup, loads existing data. On each ingest, saves to disk.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np  # noqa: E402 — used for cosine similarity in check_context
from fastapi import FastAPI
from pydantic import BaseModel, Field

from memgraph.attention_router import AttentionRouter, Turn, MemoEntry
from memgraph.embedder import Embedder

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-chat"
DEFAULT_PORT = 18820
DATA_DIR = Path(os.environ.get("MEMGRAPH_DATA_DIR", "/root/memgraph/data"))


# ── Persistence ──

def _save_state(router: AttentionRouter, data_dir: Path) -> None:
    """Save turns + memo to disk as JSON."""
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save turns
    turns_data = []
    for t in router.turns:
        entry = {
            "turn_id": t.turn_id,
            "user_text": t.user_text,
            "assistant_text": t.assistant_text,
        }
        if t.embedding is not None:
            entry["embedding"] = t.embedding.tolist()
        turns_data.append(entry)

    with open(data_dir / "turns.json", "w") as f:
        json.dump(turns_data, f, ensure_ascii=False)

    # Save memo
    memo_data = {}
    for key, me in router.memo.items():
        entry = {"key": me.key, "value": me.value}
        if me.embedding is not None:
            entry["embedding"] = me.embedding.tolist()
        memo_data[key] = entry

    with open(data_dir / "memo.json", "w") as f:
        json.dump(memo_data, f, ensure_ascii=False)

    logger.info("[memgraph] Saved %d turns, %d memo keys to %s", len(turns_data), len(memo_data), data_dir)


def _load_state(router: AttentionRouter, data_dir: Path) -> None:
    """Load turns + memo from disk."""
    turns_path = data_dir / "turns.json"
    memo_path = data_dir / "memo.json"

    if turns_path.exists():
        try:
            with open(turns_path) as f:
                turns_data = json.load(f)
            for td in turns_data:
                emb = np.array(td["embedding"], dtype=np.float32) if "embedding" in td else None
                router.turns.append(Turn(
                    turn_id=td["turn_id"],
                    user_text=td["user_text"],
                    assistant_text=td["assistant_text"],
                    embedding=emb,
                ))
            logger.info("[memgraph] Loaded %d turns from disk", len(router.turns))
        except Exception as e:
            logger.error("[memgraph] Failed to load turns: %s", e)

    if memo_path.exists():
        try:
            with open(memo_path) as f:
                memo_data = json.load(f)
            for key, md in memo_data.items():
                emb = np.array(md["embedding"], dtype=np.float32) if "embedding" in md else None
                router.memo[key] = MemoEntry(
                    key=md["key"],
                    value=md["value"],
                    embedding=emb,
                )
            logger.info("[memgraph] Loaded %d memo keys from disk", len(router.memo))
        except Exception as e:
            logger.error("[memgraph] Failed to load memo: %s", e)


# ── Router singleton ──

_router: AttentionRouter | None = None


def get_router() -> AttentionRouter:
    global _router
    if _router is not None:
        return _router

    provider, model = _configure_llm_env()
    _router = AttentionRouter(
        embedder=Embedder(),
        llm_provider="openai" if provider == "deepseek" else provider,
        model=model,
    )

    # Load persisted state
    _load_state(_router, DATA_DIR)
    return _router


def _configure_llm_env() -> tuple[str, str]:
    provider = os.environ.get("MEMGRAPH_LLM_PROVIDER")
    model = os.environ.get("MEMGRAPH_LLM_MODEL")
    api_key = os.environ.get("MEMGRAPH_LLM_API_KEY")
    base_url = os.environ.get("MEMGRAPH_LLM_BASE_URL")

    if provider or model or api_key or base_url:
        resolved_provider = provider or "openai"
        resolved_model = model or DEFAULT_MODEL
        if api_key:
            if resolved_provider == "anthropic":
                os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
            else:
                os.environ.setdefault("OPENAI_API_KEY", api_key)
        if base_url and resolved_provider != "anthropic":
            os.environ.setdefault("OPENAI_BASE_URL", base_url)
        return resolved_provider, resolved_model

    if os.environ.get("OPENAI_API_KEY"):
        return "openai", DEFAULT_MODEL

    return "openai", DEFAULT_MODEL


# ── FastAPI ──

app = FastAPI(title="MemGraph HTTP Server")


class IngestRequest(BaseModel):
    user_text: str
    assistant_text: str
    session_id: Optional[str] = None


class RecallRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


class CheckContextRequest(BaseModel):
    """Check if query points to turns not in the current context window."""
    query: str
    context_turn_ids: list[int] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=50)


class ResetContextRequest(BaseModel):
    """Reset context window tracking."""
    pass


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict[str, object]:
    router = get_router()
    router.encode(payload.user_text, payload.assistant_text)

    # Persist after each ingest
    _save_state(router, DATA_DIR)

    return {
        "status": "ok",
        "session_id": payload.session_id,
        "total_turns": len(router.turns),
        "memo_keys": len(router.memo),
    }


@app.post("/recall")
def recall(payload: RecallRequest) -> dict[str, object]:
    router = get_router()
    result = router.activate(payload.query, top_k=payload.top_k)
    result_text = getattr(result, "text", str(result))
    return {
        "status": "ok",
        "result_text": result_text,
        "result_length": len(result_text),
        "top_k": payload.top_k,
    }


@app.post("/check_context")
def check_context(payload: CheckContextRequest) -> dict[str, object]:
    """Check if query's top-k matches point to turns outside the current context.
    
    Returns needs_recall=True if any top-k turn is NOT in context_turn_ids.
    This enables the "recall only when needed" pattern:
    - embedding similarity runs every turn (free, local model)
    - LLM recall only fires when matches point to compressed/lost turns
    """
    router = get_router()
    if not router.turns:
        return {"needs_recall": False, "missing_turn_ids": [], "reason": "no turns stored"}

    # Get query embedding
    query_vec = router.embedder.embed_query(payload.query)
    if query_vec is None:
        return {"needs_recall": False, "missing_turn_ids": [], "reason": "embedding failed"}

    # Find top-k similar turns
    scored = []
    for turn in router.turns:
        if turn.embedding is not None:
            sim = float(np.dot(query_vec, turn.embedding) / (
                max(np.linalg.norm(query_vec), 1e-8) * max(np.linalg.norm(turn.embedding), 1e-8)
            ))
            scored.append((turn.turn_id, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = scored[:payload.top_k]

    # Check which top-k turns are NOT in the current context
    context_set = set(payload.context_turn_ids)
    missing = [tid for tid, sim in top_k if tid not in context_set]

    return {
        "needs_recall": len(missing) > 0,
        "missing_turn_ids": missing,
        "top_k_matches": [{"turn_id": tid, "similarity": round(sim, 4)} for tid, sim in top_k],
        "reason": f"{len(missing)} of {len(top_k)} top matches not in context" if missing else "all matches in context",
    }


@app.get("/inspect")
def inspect() -> dict[str, object]:
    router = get_router()
    return router.inspect()


def main() -> None:
    import uvicorn
    port = int(os.environ.get("MEMGRAPH_PORT", DEFAULT_PORT))
    uvicorn.run("memgraph.server:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
