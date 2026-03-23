"""Memgraph 首次加载欢迎提示。"""

WELCOME_MESSAGE = """
╔══════════════════════════════════════════════════════════════╗
║                    🧠 Memgraph 记忆模块                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Memgraph 需要调用大语言模型来提取和整理你的对话记忆。        ║
║  这部分调用比较频繁，但任务简单（信息提取/总结），             ║
║  不需要很强的模型。                                          ║
║                                                              ║
║  请选择模式：                                                ║
║                                                              ║
║  【1】默认模式（零配置）                                     ║
║      复用你 OpenClaw 的主模型。                              ║
║      无需额外配置，开箱即用。                                ║
║                                                              ║
║  【2】省钱模式（推荐）                                       ║
║      自己配一个便宜的模型，专门用于记忆提取。                ║
║      在 .env 文件中添加：                                    ║
║                                                              ║
║        MEMGRAPH_LLM_PROVIDER=openai                          ║
║        MEMGRAPH_LLM_MODEL=你选的模型                         ║
║        MEMGRAPH_LLM_API_KEY=你的key                          ║
║        MEMGRAPH_LLM_BASE_URL=对应的API地址                   ║
║                                                              ║
║      支持任何 OpenAI 兼容的 API。                            ║
║      找一个你觉得够便宜的模型就行。                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

_shown = False


def show_welcome(force: bool = False) -> None:
    """首次加载时显示欢迎提示。设置 force=True 强制显示。"""
    global _shown
    if _shown and not force:
        return
    _shown = True
    print(WELCOME_MESSAGE)
