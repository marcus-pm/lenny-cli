# Lenny CLI

A terminal chat app for exploring [Lenny's Podcast](https://www.lennyspodcast.com/) transcripts using the [Recursive Language Model (RLM)](https://github.com/alexzhang13/rlm) paradigm.

Ask natural-language questions across every episode. Get cited answers with guest names, episode titles, and YouTube links.

## Features

- **Automatic query routing** — fast mode for targeted lookups, research mode for cross-episode synthesis
- **Natural language queries** across all podcast transcripts
- **Episode citations** with guest names, titles, and YouTube links
- **Dual-model architecture** — Claude Opus 4.6 orchestrates via code, Claude Haiku 4.5 analyzes transcript chunks
- **BM25 search index** — paragraph-level full-text search with disk caching for fast startup
- **Real-time cost tracking** per query and per session
- **Verbose mode** to watch the research orchestration (code generation, sub-LM calls, iteration steps)
- **Slash commands** for session control (`/help`, `/episodes`, `/cost`, `/mode`, `/theme`, `/verbose`, `/quit`)
- **Visual themes** — warm editorial default, minimal low-decoration option (`/theme`)
- **Session memory** for follow-up questions

## Prerequisites

- **Python 3.11+**
- **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com/)

## Install

```bash
# Option A (recommended): pipx for global `lenny` command
pipx install "git+https://github.com/marcus-pm/lenny-cli.git"
lenny
```

```bash
# Option B: from source
git clone https://github.com/marcus-pm/lenny-cli.git
cd lenny-cli
python3 -m venv .venv
source .venv/bin/activate
pip install .
lenny
```

> **First run:** If `ANTHROPIC_API_KEY` is missing, `lenny` prompts you for it and can save it to `~/.config/lenny/config.env`.
>
> **Transcripts:** On first launch, `lenny` will offer to download the transcript corpus automatically (~50 MB). Alternatively, set `LENNY_TRANSCRIPTS=/path/to/episodes` to point to an existing copy.

## Usage

```bash
lenny
```

### Example Session

```
  Lenny
  Podcast transcript explorer

  152 episodes loaded  ·  mode: auto

  Try: "What frameworks do guests recommend for prioritization?"

  /help  /episodes  /cost  /mode  /theme  /quit

You: What did Brian Chesky say about founder mode?

   FAST   specific guest lookup
  Searching transcripts...

┌─ Lenny fast ──────────────────────────────────────────┐
│                                                         │
│  Brian Chesky discussed founder mode in his episode...  │
│  ...cited answer with YouTube links...                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

  Haiku: 1 calls, 5,200 in / 1,800 out tokens ($0.0114)
  Query total: $0.0114 in 4.2s

You: What frameworks do guests recommend for prioritization?

   RESEARCH   multi-guest synthesis
  Searching 152 episodes...

┌─ Lenny research ──────────────────────────────────────┐
│                                                         │
│  Several guests have shared prioritization frameworks:  │
│  ...cited answer with YouTube links...                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

  Opus: 2 calls, 12,340 in / 3,210 out tokens ($0.1418)
  Haiku: 8 calls, 24,560 in / 4,120 out tokens ($0.0361)
  Query total: $0.1779 in 42.3s
```

### Slash Commands

| Command           | Description                                     |
|-------------------|-------------------------------------------------|
| `/help`           | Show help message                               |
| `/episodes`       | List loaded episodes (count + sample)           |
| `/cost`           | Show session token usage and estimated cost     |
| `/mode`           | Show current routing mode                       |
| `/mode auto`      | Automatic routing based on query (default)      |
| `/mode fast`      | Force fast path for all queries                 |
| `/mode research`  | Force research path for all queries             |
| `/theme`          | Show current visual theme                       |
| `/theme warm`     | Warm editorial theme with colors (default)      |
| `/theme minimal`  | Low-decoration plain theme                      |
| `/verbose`        | Toggle verbose mode (see research orchestration)|
| `/quit`           | Exit (also `/exit`, `/q`)                       |

## How It Works

Lenny CLI uses two query paths, automatically selected based on the question:

### Fast Path (~5-15s, ~$0.01)

For targeted lookups like "What did Brian Chesky say about founder mode?":

1. A BM25 search index finds the most relevant transcript paragraphs
2. Top results are sent to Claude Haiku 4.5 for synthesis
3. Single API call returns a cited answer

### Research Path (~30-120s, ~$0.15)

For cross-episode synthesis like "What themes do guests disagree about?":

1. Claude Opus 4.6 writes and executes Python code in a sandboxed REPL
2. It searches transcripts, extracts sections, and dispatches analysis to Haiku 4.5
3. The root model iterates (up to 30 times) to build a comprehensive answer

### Query Routing

The router uses a three-tier classifier:

1. **Deterministic guardrails** for obvious fast/research cases
2. **LLM judge** (Haiku) for ambiguous queries
3. **Conservative fallback** to research when uncertain

Use `/mode fast` or `/mode research` to force a specific path.

> **Technical note:** The fast path uses RAG (retrieval-augmented generation) and the research path uses the [RLM](https://github.com/alexzhang13/rlm) framework.

## Cost

The app uses two Anthropic models with the following pricing (per million tokens):

| Model             | Input   | Output  |
|-------------------|---------|---------|
| Claude Opus 4.6   | $5.00   | $25.00  |
| Claude Haiku 4.5  | $0.80   | $4.00   |

Fast queries cost roughly $0.005-0.02. Research queries cost roughly $0.10-0.25. Use `/cost` during a session to see running totals.

## Configuration

| Variable            | Required | Description                              |
|---------------------|----------|------------------------------------------|
| `ANTHROPIC_API_KEY` | Yes      | Your Anthropic API key                   |
| `LENNY_TRANSCRIPTS` | No       | Override path to transcripts/episodes/   |

Set these in your `.env` file (see `.env.example`).

## Project Structure

```
lenny-cli/
├── src/lenny/
│   ├── cli.py          # Chat loop, slash commands, routing dispatch
│   ├── style.py        # UI tokens, themes, colors, ASCII art, microcopy
│   ├── engine.py       # Research engine orchestration, sandbox, system prompt
│   ├── rag.py          # Fast path engine (BM25 + single Haiku call)
│   ├── router.py       # Hybrid router (guardrails + LLM judge + fallback)
│   ├── search.py       # BM25 paragraph-level search index
│   ├── data.py         # Transcript loading and indexing
│   ├── costs.py        # Token usage and cost tracking
│   ├── persist.py      # Response saving and citation formatting
│   ├── progress.py     # Live progress display for research queries
│   ├── __main__.py     # python -m lenny entry point
│   └── __init__.py
├── tests/
│   ├── test_sandbox.py # REPL sandbox regression tests
│   └── test_router.py  # Router regression tests
├── .env.example        # Template for API key
├── pyproject.toml      # Package metadata and dependencies
├── Makefile            # Setup automation
└── LICENSE
```

## License

MIT License. See [LICENSE](LICENSE) for details.
