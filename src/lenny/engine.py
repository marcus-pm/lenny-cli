"""RLM engine for exploring Lenny's Podcast transcripts."""

from __future__ import annotations

import asyncio
import os
import textwrap
import time
from pathlib import Path

from dotenv import load_dotenv
from rlm import RLM
from rlm.core.lm_handler import LMRequestHandler, LMHandler
from rlm.core.comms_utils import LMRequest, LMResponse
from rlm.core.types import RLMChatCompletion, UsageSummary

from lenny.costs import QueryCost, SessionCosts
from lenny.data import TranscriptIndex


def _find_project_root() -> Path:
    """Find the project root by walking up from this file looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        if current.parent == current:
            break
        current = current.parent
    # Fallback
    return Path(__file__).resolve().parent.parent.parent


_PROJECT_ROOT = _find_project_root()
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# The model the root LLM (orchestrator) uses
ROOT_MODEL = "claude-opus-4-6"
# The model used for sub-LM chunk processing
SUB_MODEL = "claude-haiku-4-5-20251001"

# Rate limit config: max concurrent sub-LM calls and delay between them
MAX_CONCURRENT_SUB_CALLS = 2
DELAY_BETWEEN_CALLS = 1.0  # seconds
MAX_RETRIES = 3
RETRY_BASE_DELAY = 5.0  # seconds — exponential backoff base for 429s


# ---------------------------------------------------------------------------
# Monkeypatch: throttled batched handler to avoid Tier 1 rate limits
# ---------------------------------------------------------------------------
def _handle_batched_throttled(
    self: LMRequestHandler,
    request: LMRequest,
    handler: LMHandler,
) -> LMResponse:
    """Rate-limited version of _handle_batched with retry on 429.

    Uses a semaphore to limit concurrency, adds a small delay between
    calls, and retries with exponential backoff on rate limit errors.
    Uses a dedicated event loop to avoid conflicts with RLM's threaded server.
    """
    client = handler.get_client(request.model, request.depth)
    start_time = time.perf_counter()

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SUB_CALLS)

    async def throttled_call(prompt: str) -> str:
        async with semaphore:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    result = await client.acompletion(prompt)
                    await asyncio.sleep(DELAY_BETWEEN_CALLS)
                    return result
                except Exception as e:
                    err_str = str(e).lower()
                    if ("429" in str(e) or "rate" in err_str) and attempt < MAX_RETRIES:
                        wait = RETRY_BASE_DELAY * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue
                    raise

    async def run_all():
        tasks = [throttled_call(prompt) for prompt in request.prompts]
        return await asyncio.gather(*tasks)

    # Use a dedicated event loop — this runs in a thread spawned by
    # RLM's ThreadingTCPServer, so asyncio.run() can cause conflicts.
    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(run_all())
    finally:
        loop.close()

    end_time = time.perf_counter()

    total_time = end_time - start_time
    model_usage = client.get_last_usage()
    root_model = request.model or client.model_name
    usage_summary = UsageSummary(model_usage_summaries={root_model: model_usage})

    chat_completions = [
        RLMChatCompletion(
            root_model=root_model,
            prompt=prompt,
            response=content,
            usage_summary=usage_summary,
            execution_time=total_time / len(request.prompts),
        )
        for prompt, content in zip(request.prompts, results, strict=True)
    ]

    return LMResponse.batched_success_response(chat_completions=chat_completions)


# Apply the monkeypatch
LMRequestHandler._handle_batched = _handle_batched_throttled


SYSTEM_PROMPT = textwrap.dedent("""\
You are a research assistant specialized in analyzing Lenny's Podcast transcripts. \
You have access to a REPL environment with a catalog of podcast episodes and the ability \
to read full transcript files and query sub-LLMs for analysis.

## REPL Environment

Your REPL is initialized with:

1. A `context` variable — a JSON dict containing:
   - `"catalog"`: a list of episode dicts, each with: slug, guest, title, youtube_url, publish_date, duration, keywords
   - `"transcript_dir"`: the filesystem path to the episodes directory
   - `"conversation_history"`: list of prior Q&A pairs from this session (for follow-up context)

2. `llm_query(prompt: str) -> str` — call a sub-LLM (handles ~500K chars). Use this for analyzing transcript chunks.

3. `llm_query_batched(prompts: list[str]) -> list[str]` — concurrent sub-LM calls (rate-limited to avoid API throttling). Use for processing multiple chunks in parallel.

4. `SHOW_VARS()` — see all variables you've created.

5. `print()` — view intermediate results (outputs are truncated, so store important data in variables).

6. `open()` — read transcript files from disk.

## How to Load a Transcript

Each episode transcript is a markdown file at: `{transcript_dir}/{slug}/transcript.md`

To load a transcript:
```repl
slug = "brian-chesky"
with open(f"{{context['transcript_dir']}}/{{slug}}/transcript.md", "r") as f:
    transcript = f.read()
print(transcript[:500])  # Preview
```

## Rate Limit Awareness

The system automatically handles rate limiting with throttling and retries. Follow these guidelines for best results:
- **NEVER send full transcripts to sub-LLMs.** Transcripts are 30-80K chars each — always extract relevant sections first.
- **Extract relevant sections** using Python string operations (regex, find, split) before calling sub-LLMs.
- **Send focused, excerpt-based prompts** to sub-LLMs. A good prompt includes just the relevant paragraphs/sections plus a clear question.
- Focus on writing good extraction code — the system handles pacing automatically.

## Recommended Strategy

1. **Identify relevant episodes**: Search the catalog by guest name, title, or keywords. Use Python string/list operations on `context["catalog"]`.

2. **Search and extract relevant sections** (NO sub-LM needed for this step):
```repl
import re
relevant_excerpts = []
for ep in context["catalog"]:
    path = f"{{context['transcript_dir']}}/{{ep['slug']}}/transcript.md"
    with open(path, "r") as f:
        text = f.read()
    # Find paragraphs containing the topic
    paragraphs = text.split("\\n\\n")
    for p in paragraphs:
        if re.search(r"your_pattern", p, re.IGNORECASE):
            relevant_excerpts.append({{
                "guest": ep["guest"],
                "title": ep["title"],
                "youtube_url": ep["youtube_url"],
                "excerpt": p[:3000]  # Keep excerpts small
            }})
print(f"Found {{len(relevant_excerpts)}} relevant excerpts")
```

3. **Analyze excerpts with sub-LLMs** (send small, focused prompts):
```repl
# Group excerpts into small batches and send focused prompts
batch_size = 3
for i in range(0, len(relevant_excerpts), batch_size):
    batch = relevant_excerpts[i:i+batch_size]
    prompts = []
    for ex in batch:
        prompts.append(
            f"Guest: {{ex['guest']}} — {{ex['title']}}\\n\\n"
            f"Excerpt:\\n{{ex['excerpt']}}\\n\\n"
            f"Question: {{your_question}}\\n\\n"
            f"Extract the key insight or quote. Be concise."
        )
    answers = llm_query_batched(prompts)
    # Process answers...
```

4. **Synthesize**: Combine extracted insights into a final response using another `llm_query` call.

## Citation Requirements

ALWAYS cite specific episodes in your final answer:
- Include the guest name, episode title, and YouTube URL
- Format: "**Guest Name** in *Episode Title* ([link](youtube_url))"
- When quoting, attribute the quote to the speaker

## Follow-up Questions

Check `context["conversation_history"]` for prior Q&A pairs. When the user asks a follow-up \
(e.g., "narrow that down to..."), use the prior answers as context.

## Output Format

Your final answer should be well-structured markdown with:
- A clear answer to the question
- Cited episodes with YouTube links
- Key quotes or findings attributed to specific guests
- Organized with headers/bullets as appropriate

## REPL Code Format

Write Python code in triple-backtick blocks with 'repl' language tag:
```repl
# your code here
```

## Finishing

When done, provide your final answer:
- Use FINAL(your complete markdown answer) to return your answer directly
- Or create a variable with the answer and use FINAL_VAR(variable_name)

IMPORTANT: FINAL_VAR retrieves an EXISTING variable. Create and assign it in a ```repl``` block FIRST, then call FINAL_VAR in a SEPARATE step.

Think step by step. Plan your approach, execute it in the REPL, and synthesize a thorough answer. \
Do not provide a final answer without first examining the actual transcript data.
""")


class LennyEngine:
    """Wraps the RLM library for podcast transcript exploration."""

    def __init__(
        self,
        index: TranscriptIndex,
        verbose: bool = True,
        max_iterations: int = 30,
    ):
        self.index = index
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.session_costs = SessionCosts()
        self.conversation_history: list[dict[str, str]] = []

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not found.\n"
                "  Option 1: Add it to .env in the project root:\n"
                f"           echo 'ANTHROPIC_API_KEY=sk-ant-...' >> {_PROJECT_ROOT / '.env'}\n"
                "  Option 2: Export it in your shell:\n"
                "           export ANTHROPIC_API_KEY=sk-ant-..."
            )

        self.rlm = RLM(
            backend="anthropic",
            backend_kwargs={
                "api_key": api_key,
                "model_name": ROOT_MODEL,
                "max_tokens": 16384,
            },
            other_backends=["anthropic"],
            other_backend_kwargs=[{
                "api_key": api_key,
                "model_name": SUB_MODEL,
                "max_tokens": 8192,
            }],
            environment="local",
            max_iterations=max_iterations,
            max_depth=1,
            custom_system_prompt=SYSTEM_PROMPT,
            verbose=verbose,
            persistent=False,  # We manage context ourselves per query
        )

    def query(self, question: str) -> tuple[str, QueryCost]:
        """Run a question through the RLM engine.

        Returns (answer_text, query_cost).
        """
        # Build context payload with catalog + conversation history
        context_payload = {
            "catalog": self.index.get_catalog(),
            "transcript_dir": self.index.transcript_dir,
            "conversation_history": self.conversation_history[-10:],  # Last 10 exchanges
        }

        result = self.rlm.completion(
            prompt=context_payload,
            root_prompt=question,
        )

        # Track costs
        query_cost = self.session_costs.add_query(
            result.usage_summary,
            result.execution_time,
        )

        # Update conversation history for follow-up context
        self.conversation_history.append({
            "question": question,
            "answer": result.response[:2000],  # Truncate for context management
        })

        return result.response, query_cost

    def close(self):
        """Clean up resources."""
        if hasattr(self.rlm, 'close'):
            self.rlm.close()
