# MemGraph

**Conversation memory encoding plugin** — compresses multi-turn dialogues into a structured state graph (nodes + edges) for LLM context injection at inference time.

## Architecture

MemGraph uses a **three-layer memory architecture** with an attention-based routing system:

```
┌─────────────────────────────────────────────────────┐
│                   Query Input                        │
│                      ↓                               │
│            ┌─────────────────┐                       │
│            │ AttentionRouter  │ (semantic routing)    │
│            └────────┬────────┘                       │
│         ┌───────────┼───────────┐                    │
│         ↓           ↓           ↓                    │
│    ┌─────────┐ ┌─────────┐ ┌─────────┐              │
│    │   L1    │ │   L2    │ │   L3    │              │
│    │ Profile │ │Conclus. │ │Raw Trace│              │
│    │  Card   │ │ + Graph │ │  Store  │              │
│    └─────────┘ └─────────┘ └─────────┘              │
│         │           │           │                    │
│         └───────────┴───────────┘                    │
│                     ↓                                │
│            Merged Context Output                     │
└─────────────────────────────────────────────────────┘
```

- **L1 (Profile Card)**: Persistent user profile — preferences, background, key facts
- **L2 (Conclusions + Graph)**: Compressed conclusions organized as a semantic graph with edges (sequential, cross-topic, topic-shift)
- **L3 (Raw Traces)**: Full conversation traces for detail retrieval when L2 is insufficient

### Key Components

| Module | Role |
|--------|------|
| `core.py` | Main orchestrator — `MemGraph.encode()` and `MemGraph.activate()` |
| `attention_router.py` | Semantic attention routing — decides which layers to query and how |
| `compressor.py` | LLM-based conversation compression into structured conclusions |
| `graph.py` | State graph with semantic edges (sequential, cross-topic, topic-shift) |
| `activator.py` | Layered activation with focus-based retrieval |
| `embedder.py` | Sentence-transformer embeddings for semantic search |
| `critic.py` | Quality critic for compression output |

### Evolution

| Version | Recall | Key Changes |
|---------|--------|-------------|
| v1 (Compressor) | 50.6% → 72.2% | L1/L2/L3 + compressor + graph edges |
| v2 (Profile + IM/EM) | 80.6% | Profile card, IM/EM separation, L2.5 detail retrieval |
| **v3 (AttentionRouter)** | **84.2%** | Attention-based routing, focus mechanism, memo prompt rewrite |

## Setup

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
# Edit .env with your API keys
```

MemGraph uses the OpenAI SDK with configurable backend (default: DeepSeek):

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.deepseek.com
```

## Quick Start

```python
from memgraph import MemGraph

# Using external LLM function (recommended — zero config)
mg = MemGraph(llm_fn=my_agent.chat_fn)

# Or standalone with OpenAI-compatible API
mg = MemGraph()

# Encode conversations
mg.encode([
    {"role": "user", "content": "I want to build an RPG todo app"},
    {"role": "assistant", "content": "Great idea! What features do you need?"},
])

# Activate memory for a query
context = mg.activate("What's the current progress?")
print(context)
```

## Benchmark

MemGraph is evaluated on [RealMem](tests/benchmark/realmem_data/) — a multi-persona, multi-session dialogue benchmark with 10 personas × 207 sessions × 126 ground-truth queries.

### Run Locally

```bash
# Smoke test (no LLM calls)
python -m tests.benchmark.run_realmem_benchmark --dry-run

# Run full benchmark (single persona)
python -m tests.benchmark.run_realmem_benchmark --persona Lin_Wanyu --static --activator attention

# Run full benchmark (all personas)
python -m tests.benchmark.run_realmem_benchmark --static --activator attention
```

### Run via GitHub Actions CI

The repository includes a GitHub Actions workflow for reproducible benchmark runs:

1. Go to **Actions** → **MemGraph RealMemBench**
2. Click **Run workflow**
3. Select mode: `compare` (attention vs h2o) or `single`
4. Results appear in the Actions summary + downloadable artifacts

CI runs execute on GitHub's infrastructure with full audit trail (commit hash → workflow trigger → logs → artifacts).

### Latest Results (v7, AttentionRouter)

| Activator | Overall Recall | Sessions | Queries |
|-----------|---------------|----------|---------|
| **attention** | **84.2%** | 207 | 126 |
| h2o | — | — | — |

> Full per-persona and per-category breakdowns are available in the benchmark result JSON files.

## Project Structure

```
memgraph/
├── memgraph/              # Core source code
│   ├── core.py            # Main MemGraph class
│   ├── attention_router.py # Semantic attention routing
│   ├── compressor.py      # LLM conversation compression
│   ├── graph.py           # Structured state graph
│   ├── activator.py       # Layered memory activation
│   ├── embedder.py        # Sentence-transformer embeddings
│   ├── critic.py          # Compression quality critic
│   ├── config.py          # Configuration
│   ├── models.py          # Data models
│   └── ...
├── tests/
│   └── benchmark/
│       ├── run_realmem_benchmark.py  # Main benchmark script
│       ├── realmem_loader.py         # Data loader
│       └── realmem_data/             # RealMem dataset (10 personas)
├── .github/workflows/
│   └── benchmark.yml      # CI benchmark workflow
├── requirements.txt
└── .env.example
```

## License

Research use. See individual component licenses.
