"""Benchmark 主入口：RealMem 多 persona 跨 session 记忆召回评估。

基于 RealMemBench 数据集；入口与 `run_realmem_benchmark` 相同。

用法:
    # 列出可用 persona
    python -m tests.benchmark.run_benchmark --list-personas

    # dry-run（不调 LLM，只验证流程）
    python -m tests.benchmark.run_benchmark --dry-run

    # 跑单个 persona（调试）
    python -m tests.benchmark.run_benchmark --persona Lin_Wanyu --limit-sessions 20

    # 完整运行
    python -m tests.benchmark.run_benchmark --output benchmark_results.json
"""

# 直接转发到 RealMem benchmark
from tests.benchmark.run_realmem_benchmark import main

if __name__ == "__main__":
    main()
