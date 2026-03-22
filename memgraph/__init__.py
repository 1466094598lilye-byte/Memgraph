"""MemGraph - 对话历史编码插件，将多轮对话压缩为结构化状态图。"""

import os
from pathlib import Path

# 自动加载 .env（无需额外依赖）
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from memgraph.core import LLMFn, MemGraph

__all__ = ["LLMFn", "MemGraph"]
