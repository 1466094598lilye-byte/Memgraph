# MemGraph

**Conversation memory plugin** — encodes multi-turn dialogues into a structured memory system for LLM context injection at inference time.

## Architecture (v7 — AttentionRouter)

MemGraph v7 uses an **Attention-Routed Memory** architecture inspired by how human memory works: humans don't fear catastrophic forgetting because they're always forgetting — what matters is jumping to the right information at each turn.

```
┌──────────────────────────────────────────────────────────┐
│                      Query Input                          │
│                          ↓                                │
│                 ┌────────────────┐                        │
│                 │ AttentionRouter │                        │
│                 └───────┬────────┘                        │
│            ┌────────────┼────────────┐                    │
│            ↓            ↓            ↓                    │
│    ┌──────────────┐ ┌────────┐ ┌──────────┐              │
│    │  Memo Store   │ │ Focus  │ │ Turn     │              │
│    │  (全量注入)    │ │ Track  │ │ Retrieval│              │
│    │              │ │        │ │ (top-k)  │              │
│    │ key-value     │ │ active │ │ cosine   │              │
│    │ precise facts │ │ thread │ │ semantic │              │
│    └──────────────┘ └────────┘ └──────────┘              │
│            │            │            │                    │
│            └────────────┴────────────┘                    │
│                         ↓                                 │
│               Merged Context Output                       │
└──────────────────────────────────────────────────────────┘
```

### Core Design Principles

1. **Store everything, retrieve selectively** — All conversation turns are stored with embeddings. At query time, cosine top-k retrieves only the most relevant turns.
2. **Memo for precision** — LLM extracts precise facts (numbers, dates, names, decisions, accepted plans) into a flat key-value store. Memo is injected in full — it's small and critical.
3. **Focus tracking** — Tracks the active conversation thread to provide temporal coherence.

### Key Components

| Module | Role |
|--------|------|
| `attention_router.py` | Core engine — stores turns with embeddings, extracts memo facts via LLM, retrieves via cosine top-k |
| `core.py` | Orchestrator — `MemGraph.encode()` and `MemGraph.activate()` entry points |
| `compressor.py` | LLM-based conversation compression into structured conclusions |
| `graph.py` | Semantic graph with edges (sequential, cross-topic, topic-shift) |
| `embedder.py` | Sentence-transformer embeddings for semantic search |
| `activator.py` | Alternative activator (layered mode) |
| `critic.py` | Compression quality critic |

### How It Works

**Write path (encode):**
1. Store user + assistant text as a `Turn` with sentence-transformer embedding
2. LLM extracts precise facts → update memo key-value store (each fact also gets an embedding)

**Read path (activate):**
1. Embed the query
2. Inject full memo (all precise facts — small, always relevant)
3. Cosine similarity against all stored turns → select top-k most relevant
4. Sort selected turns by time order → merge into context output

### Evolution

| Version | Recall | Architecture | Key Change |
|---------|--------|-------------|------------|
| v1 | 50.6% → 72.2% | Compressor + Graph edges | L1/L2/L3 layers + sequential/cross-topic edges |
| v2 | 80.6% | + Profile + IM/EM | Profile card, internal/external memory split |
| **v7** | **84.2%** | **AttentionRouter** | **Attention routing + memo extraction + focus tracking** |

## Setup

```bash
pip install -r requirements.txt
```

### Configuration

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

# Single persona
python -m tests.benchmark.run_realmem_benchmark --persona Lin_Wanyu --static --activator attention

# All personas
python -m tests.benchmark.run_realmem_benchmark --static --activator attention
```

### GitHub Actions CI

Reproducible benchmark runs with full audit trail:

1. Go to **Actions** → **MemGraph RealMemBench**
2. Click **Run workflow**
3. Select mode: `compare` (attention vs h2o) or `single`
4. Results appear in Actions summary + downloadable artifacts

CI runs on GitHub infrastructure — commit hash, logs, and artifacts form a verifiable chain.

### Results (v7, AttentionRouter, 207 sessions)

| Activator | Overall Recall | Sessions | Queries |
|-----------|---------------|----------|---------|
| **attention** | **84.2%** | 207 | 126 |

> Per-persona and per-category breakdowns available in benchmark result JSON files under `tests/benchmark/`.

## Project Structure

```
memgraph/
├── memgraph/                    # Core source code
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
│       ├── run_realmem_benchmark.py   # Main benchmark runner
│       ├── realmem_loader.py          # Dataset loader
│       └── realmem_data/              # RealMem dataset (10 personas)
├── .github/workflows/
│   └── benchmark.yml            # CI benchmark workflow
├── requirements.txt
├── .env.example
└── LICENSE                      # MIT
```

## License

MIT
