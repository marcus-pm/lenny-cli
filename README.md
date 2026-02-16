# Lenny CLI

A terminal chat app for exploring [Lenny's Podcast](https://www.lennyspodcast.com/) transcripts using the [Recursive Language Model (RLM)](https://github.com/alexzhang13/rlm) paradigm.

Ask natural-language questions across 303 episodes. Get cited answers with guest names, episode titles, and YouTube links.

## Features

- **Natural language queries** across 303 podcast transcripts
- **Episode citations** with guest names, titles, and YouTube links
- **Dual-model architecture** — Claude Opus 4.6 orchestrates via code, Claude Haiku 4.5 analyzes transcript chunks
- **Real-time cost tracking** per query and per session
- **Verbose mode** to watch the RLM orchestration (code generation, sub-LM calls, iteration steps)
- **Slash commands** for session control (`/help`, `/episodes`, `/cost`, `/verbose`, `/quit`)
- **Session memory** for follow-up questions

## Prerequisites

- **Python 3.11+**
- **Anthropic API key** — get one at [console.anthropic.com](https://console.anthropic.com/)
- **Git** (for cloning with submodules)

## Quick Start

```bash
# Clone with transcripts submodule
git clone --recursive https://github.com/marcus-pm/lenny-cli.git
cd lenny-cli

# Option A: one-command setup
make setup
# Then edit .env and add your Anthropic API key

# Option B: manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install .
cp .env.example .env
# Edit .env and add your Anthropic API key
```

> **Note:** The `--recursive` flag is required to pull the transcript submodule. If you already cloned without it, run `git submodule update --init`.

## Usage

```bash
source .venv/bin/activate
lenny
```

Or without activating the venv:

```bash
.venv/bin/lenny
```

### Example Session

```
Lenny CLI — Explore Lenny's Podcast with RLM

Ask questions about themes, patterns, and insights across 303 episodes.

You: What frameworks do guests recommend for prioritization?

  Searching 303 episodes...

┌─ Lenny ─────────────────────────────────────────────────┐
│                                                         │
│  Several guests have shared prioritization frameworks:  │
│                                                         │
│  ...cited answer with YouTube links...                  │
│                                                         │
└─────────────────────────────────────────────────────────┘

  Opus: 2 calls, 12,340 in / 3,210 out tokens ($0.1418)
  Haiku: 8 calls, 24,560 in / 4,120 out tokens ($0.0361)
  Query total: $0.1779 in 42.3s
```

### Slash Commands

| Command     | Description                                    |
|-------------|------------------------------------------------|
| `/help`     | Show help message                              |
| `/episodes` | List loaded episodes (count + sample)          |
| `/cost`     | Show session token usage and estimated cost    |
| `/verbose`  | Toggle verbose mode (see RLM orchestration)    |
| `/quit`     | Exit (also `/exit`, `/q`)                      |

## How It Works

Lenny CLI uses the [RLM (Recursive Language Model)](https://github.com/alexzhang13/rlm) framework. A root LLM (Claude Opus 4.6) acts as an orchestrator that writes and executes Python code in a sandboxed REPL. It searches transcripts, extracts relevant sections using string operations, and dispatches focused analysis prompts to a sub-LLM (Claude Haiku 4.5). The root model iterates — up to 30 times per query — refining its approach until it can synthesize a comprehensive, cited answer.

This is fundamentally different from stuffing transcripts into a single prompt. The root model *programs* its way to an answer, using code to filter 303 episodes down to the relevant passages before engaging the sub-LM.

## Cost

The app uses two Anthropic models with the following pricing (per million tokens):

| Model             | Input   | Output  |
|-------------------|---------|---------|
| Claude Opus 4.6   | $5.00   | $25.00  |
| Claude Haiku 4.5  | $0.80   | $4.00   |

A typical query costs a few cents. Use `/cost` during a session to see running totals.

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
│   ├── cli.py          # Chat loop and slash commands
│   ├── engine.py       # RLM orchestration and system prompt
│   ├── data.py         # Transcript loading and indexing
│   ├── costs.py        # Token usage and cost tracking
│   ├── __main__.py     # python -m lenny entry point
│   └── __init__.py
├── transcripts/        # Git submodule — 303 episode transcripts
├── .env.example        # Template for API key
├── pyproject.toml      # Package metadata and dependencies
├── Makefile            # Setup automation
└── LICENSE
```

## License

MIT License. See [LICENSE](LICENSE) for details.
