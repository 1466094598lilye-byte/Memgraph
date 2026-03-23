"""FastAPI server exposing MemGraph over HTTP."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from memgraph.attention_router import AttentionRouter
from memgraph.embedder import Embedder


DEFAULT_MODEL = "deepseek-chat"
DEFAULT_PORT = 18820


class IngestRequest(BaseModel):
    user_text: str
    assistant_text: str
    session_id: Optional[str] = None


class RecallRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)


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


@lru_cache(maxsize=1)
def get_router() -> AttentionRouter:
    provider, model = _configure_llm_env()
    return AttentionRouter(
        embedder=Embedder(),
        llm_provider="openai" if provider == "deepseek" else provider,
        model=model,
    )


app = FastAPI(title="MemGraph HTTP Server")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(payload: IngestRequest) -> dict[str, object]:
    router = get_router()
    router.encode(payload.user_text, payload.assistant_text)
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
