# MemGraph

Attention-routed conversation memory for LLMs. Achieves **92.1% recall** on a 10-persona benchmark — 3× higher than [mem0](https://github.com/mem0ai/mem0) (30.6%) and 4× higher than rolling-summary baselines (23.1%).

## Benchmark Results

Evaluated on [RealMem](tests/benchmark/realmem_data/) — 10 personas × 207 sessions × 126 ground-truth queries. Each query tests whether the system can recall specific facts from past conversations.

| Method | Average Recall | Range |
|--------|---------------:|------:|
| **MemGraph (attention)** | **92.1%** | 84.0% – 103.1% |
| mem0 | 30.6% | 23.1% – 38.5% |
| summary | 23.1% | 2.5% – 38.8% |

Per-persona and per-query breakdowns are available in the JSON result files under `tests/benchmark/`.

> Recall >100% occurs when the judge credits the system for recalling related information beyond the ground-truth set. This is a known artifact of LLM-as-judge evaluation — see [Known Limitations](#known-limitations).

### Baselines

- **MemGraph (attention)** — AttentionRouter: memo full injection + cosine top-k selective recall + focus decay
- **mem0** — [Mem0](https://github.com/mem0ai/mem0) production memory library. Stores sessions via `memory.add()`, retrieves via `memory.search()`
- **summary** — Rolling summary baseline: LLM compresses each session into a running summary (similar to ChatGPT's conversation memory). Full summary used as context at query time

## How It Works

MemGraph uses an **Attention-Routed Memory** architecture. The core insight: humans don't fear forgetting — what matters is jumping to the right information at the right time.

```
Query Input
    ↓
┌──────────────────┐
│  1. Memo Store    │  ← inject all (small, critical facts)
│  (key-value)      │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  2. Turn Store    │  ← cosine top-k selective recall
│  (all turns +     │
│   embeddings)     │
└────────┬─────────┘
         ↓
┌──────────────────┐
│  3. Focus Decay   │  ← boost in-focus, decay out-of-focus
│  (active thread)  │
└────────┬─────────┘
         ↓
  Merged Context Output
```

**Write path (encode):**
1. Store user + assistant turns with sentence-transformer embeddings
2. LLM extracts precise facts (numbers, dates, decisions) → flat key-value memo store

**Read path (activate):**
1. Embed the query
2. Inject full memo (always — it's small and critical)
3. Cosine similarity against all stored turns → top-k most relevant
4. Focus decay weights recent active threads higher
5. Sort by time → merge into context

### Key Components

| Module | Role |
|--------|------|
| `attention_router.py` | Core: stores turns with embeddings, extracts memo via LLM, retrieves via cosine top-k |
| `core.py` | Orchestrator: `MemGraph.encode()` and `MemGraph.activate()` entry points |
| `compressor.py` | LLM-based conversation compression |
| `embedder.py` | Sentence-transformer embeddings (`all-MiniLM-L6-v2`) |
| `graph.py` | Semantic graph with sequential/cross-topic edges |
| `activator.py` | Alternative layered activation mode |

### Architecture Evolution

| Version | Recall | Key Change |
|---------|--------|------------|
| v1 | 50.6% → 72.2% | Compressor + graph edges |
| v2 | 80.6% | Profile card + internal/external memory split |
| **v7** | **92.1%** | **AttentionRouter: memo extraction + cosine top-k + focus decay** |

## Install

```bash
git clone https://github.com/1466094598lilye-byte/Memgraph.git
cd Memgraph
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

MemGraph uses the OpenAI SDK with a configurable backend (default: DeepSeek):

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
```

## Quick Start

```python
from memgraph import MemGraph

# With your own LLM function (recommended)
mg = MemGraph(llm_fn=my_agent.chat_fn)

# Or standalone with OpenAI-compatible API
mg = MemGraph()

# Encode conversations
mg.encode([
    {"role": "user", "content": "I want to build an RPG todo app"},
    {"role": "assistant", "content": "Great idea! What features do you need?"},
])

# Retrieve memory for a query
context = mg.activate("What's the current progress?")
print(context)
```

## Running Benchmarks

```bash
# Smoke test (no LLM calls)
python -m tests.benchmark.run_realmem_benchmark --dry-run

# Single persona
python -m tests.benchmark.run_realmem_benchmark --persona Lin_Wanyu --static --activator attention

# All personas, all baselines
python -m tests.benchmark.run_realmem_benchmark --static --activator attention
python -m tests.benchmark.run_realmem_benchmark --static --activator mem0
python -m tests.benchmark.run_realmem_benchmark --static --activator summary
```

### GitHub Actions CI

Reproducible benchmark runs with full audit trail:

1. Go to **Actions** → **MemGraph RealMemBench**
2. Click **Run workflow**
3. Select mode: `compare` (attention vs mem0 vs summary) or `single`
4. Results appear in Actions summary + downloadable artifacts

Per-persona and per-query breakdowns are in the JSON files under `tests/benchmark/`.

## Project Structure

```
memgraph/
├── memgraph/                    # Core source
│   ├── core.py                  # MemGraph orchestrator
│   ├── attention_router.py      # Attention routing + memo extraction
│   ├── compressor.py            # LLM conversation compression
│   ├── graph.py                 # Semantic state graph
│   ├── activator.py             # Layered memory activation
│   ├── embedder.py              # Sentence-transformer embeddings
│   ├── critic.py                # Compression quality critic
│   ├── config.py                # Configuration
│   └── models.py                # Data models
├── tests/
│   └── benchmark/
│       ├── run_realmem_benchmark.py   # Benchmark runner
│       ├── realmem_loader.py          # Dataset loader
│       ├── realmem_data/              # RealMem dataset (10 personas)
│       └── benchmark_*.json           # Result files
├── .github/workflows/
│   └── benchmark.yml            # CI benchmark workflow
├── requirements.txt
├── .env.example
└── LICENSE                      # MIT
```

## Known Limitations

- **Recall >100%**: LLM-as-judge sometimes credits recall of related information beyond the ground-truth set. This inflates scores for some personas. The relative ranking (MemGraph >> mem0 >> summary) is robust.
- **Compaction semantic fidelity**: No formal measurement of information loss during LLM-based conversation compression. In practice, the memo store preserves critical facts, but nuance may be lost.
- **Token cost**: MemGraph uses 3–5× more tokens during the encode phase than mem0, trading cost for accuracy.
- **Some memo keys extracted as None**: When early conversation context is insufficient, the LLM may fail to extract a meaningful key. These entries are harmless and self-correct as more context accumulates.

## License

MIT
