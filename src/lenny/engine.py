"""RLM engine for exploring Lenny's Podcast transcripts."""

from __future__ import annotations

import asyncio
import builtins
import os
import random
import textwrap
import time
from pathlib import Path
from typing import Callable

import anthropic
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

# Query-level retry config for root model 429s (TPM window resets).
# This is the single retry authority — SDK retries stay at default (2) for
# transient errors; this wrapper handles longer TPM-style waits.
_MAX_QUERY_RETRIES = 2          # 3 total attempts
_MAX_TOTAL_RETRY_SECS = 120.0   # wall-clock cap (outer wait only; excludes SDK internal waits)
_DEFAULT_RETRY_WAIT = 30.0      # roughly one TPM window


def is_rate_limit_error(e: Exception) -> bool:
    """Check if an exception is a rate limit error (429).

    Detection priority: isinstance → structured status → string fallback.
    """
    if isinstance(e, anthropic.RateLimitError):
        return True
    # Structured status code check (some SDK wrappers expose this)
    status = getattr(e, "status_code", None) or getattr(e, "status", None)
    if status == 429:
        return True
    # String fallback — intentionally conservative
    err_str = str(e)
    return "rate_limit_error" in err_str.lower()


def _compute_retry_wait(e: Exception, attempt: int) -> float:
    """Compute wait time: Retry-After header > exponential backoff, plus jitter.

    Returns seconds to sleep, capped at 90s per attempt.
    """
    retry_after = None
    response = getattr(e, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            raw = headers.get("retry-after")
            if raw is not None:
                try:
                    retry_after = float(raw)
                except (ValueError, TypeError):
                    pass

    if retry_after is not None and 0 < retry_after <= 120:
        base_wait = retry_after
    else:
        base_wait = _DEFAULT_RETRY_WAIT * (1.5 ** attempt)  # 30s, 45s

    jitter = random.uniform(0, 5)
    return min(base_wait + jitter, 90)


# ---------------------------------------------------------------------------
# Throttled batched handler — avoids Tier 1 rate limits
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


# ---------------------------------------------------------------------------
# Lazy throttle patch — applied on first LennyEngine instantiation, not at
# import time, so other code importing this module is unaffected.
# ---------------------------------------------------------------------------
_THROTTLE_APPLIED = False


def _apply_throttle_patch() -> None:
    """Apply the rate-limited batched handler to LMRequestHandler.

    This is intentionally a process-wide monkeypatch because the RLM library
    does not support dependency injection for the handler class.  We apply it
    lazily (not at import time) so that importing lenny.engine for other
    purposes does not trigger the side effect.
    """
    global _THROTTLE_APPLIED
    if not _THROTTLE_APPLIED:
        LMRequestHandler._handle_batched = _handle_batched_throttled
        _THROTTLE_APPLIED = True


# ---------------------------------------------------------------------------
# REPL sandbox — restricts file access and dangerous imports
# ---------------------------------------------------------------------------
_BLOCKED_MODULES = frozenset({
    "subprocess",       # shell command execution
    "socket",           # raw network access
    "http",             # HTTP client/server
    "urllib",           # URL handling
    "requests",         # popular HTTP library
    "ftplib",           # FTP
    "smtplib",          # SMTP
    "xmlrpc",           # XML-RPC
    "ctypes",           # C foreign function interface
    "multiprocessing",  # process spawning
    "signal",           # signal handling
    "webbrowser",       # browser launching
    "importlib",        # import machinery bypass
    "posix",            # low-level POSIX syscalls (os.open bypass on Linux/macOS)
    "nt",               # low-level Windows syscalls (os.open bypass on Windows)
    "posixpath",        # rarely imported directly, but block for completeness
    "ntpath",           # Windows path variant
    "_io",              # C-level io implementation
    "_posixsubprocess", # subprocess internals
})


def _make_restricted_open(allowed_dirs: list[str]):
    """Create a restricted open() that only allows reading from specified directories.

    This is the primary security control against prompt-injection attacks that
    attempt to read sensitive files (e.g., ~/.ssh/id_rsa, ~/.env).
    """
    _real_open = builtins.open
    resolved_dirs = [str(Path(d).resolve()) for d in allowed_dirs]

    def restricted_open(file, mode="r", *args, **kwargs):
        # Block all write modes
        if any(c in str(mode) for c in ("w", "a", "x", "+")):
            raise PermissionError(f"Write access is not allowed: {file}")

        # Resolve the target path (follows symlinks, normalises ..)
        target = str(Path(str(file)).resolve())

        # Check the resolved path falls under an allowed directory
        if not any(
            target == allowed_dir or target.startswith(allowed_dir + os.sep)
            for allowed_dir in resolved_dirs
        ):
            raise PermissionError(
                f"Access denied: {file} is outside allowed directories"
            )

        return _real_open(file, mode, *args, **kwargs)

    return restricted_open


def _make_restricted_import(blocked: frozenset[str], restricted_open_fn):
    """Create a restricted __import__ that blocks dangerous modules.

    Blocks modules that enable shell execution, network access, or
    system-level manipulation.  Returns sanitised proxies for modules
    that contain file-access bypasses:

    * ``os``      — strips os.open, os.read, os.system, os.popen, etc.
    * ``io``      — replaces io.open / io.FileIO with the restricted open
    * ``pathlib`` — replaces Path.read_text / read_bytes / write_* / open

    Args:
        blocked: Set of top-level module names to block entirely.
        restricted_open_fn: The restricted open() to inject into io/pathlib.
    """
    _real_import = builtins.__import__
    _proxy_cache: dict[str, object] = {}

    def restricted_import(name, *args, **kwargs):
        top_level = name.split(".")[0]
        if top_level in blocked:
            raise ImportError(
                f"Import of '{name}' is not allowed in this environment. "
                f"Module '{top_level}' is blocked for security."
            )

        module = _real_import(name, *args, **kwargs)

        # Sanitised os — strip dangerous file I/O and process execution
        if top_level == "os":
            if "os" not in _proxy_cache:
                real_os = module if name == "os" else _real_import("os")
                _proxy_cache["os"] = _build_safe_os(real_os)
            return _proxy_cache["os"]

        # Sanitised io — replace io.open / io.FileIO
        if top_level == "io":
            if "io" not in _proxy_cache:
                real_io = module if name == "io" else _real_import("io")
                _proxy_cache["io"] = _build_safe_io(real_io, restricted_open_fn)
            return _proxy_cache["io"]

        # Sanitised pathlib — strip Path.read_text, read_bytes, write_*, open
        if top_level == "pathlib":
            if "pathlib" not in _proxy_cache:
                real_pathlib = module if name == "pathlib" else _real_import("pathlib")
                _proxy_cache["pathlib"] = _build_safe_pathlib(
                    real_pathlib, restricted_open_fn,
                )
            return _proxy_cache["pathlib"]

        return module

    return restricted_import


# ---------------------------------------------------------------------------
# Safe os proxy
# ---------------------------------------------------------------------------
_OS_SAFE_ATTRS = frozenset({
    # path utilities
    "path", "sep", "altsep", "extsep", "pathsep", "curdir", "pardir",
    "devnull", "linesep", "fspath",
    # directory listing (read-only, non-sensitive when open() is restricted)
    "listdir", "scandir", "walk", "fwalk",
    "getcwd", "getcwdb",
    "stat", "lstat", "fstat",
    # environment (read-only access)
    "environ", "getenv", "get_exec_path",
    # constants & helpers used by standard library internals
    "name", "supports_dir_fd", "supports_effective_ids",
    "supports_fd", "supports_follow_symlinks",
    "cpu_count", "getpid", "getppid", "getlogin", "getuid", "getgid",
    "uname", "strerror", "urandom",
    # commonly used by pathlib / other safe libs
    "fsdecode", "fsencode",
    "DirEntry",
})


def _build_safe_os(real_os):
    """Return a proxy module that exposes only safe os attributes."""
    import types

    safe = types.ModuleType("os")
    safe.__package__ = real_os.__package__
    safe.__loader__ = getattr(real_os, "__loader__", None)
    safe.__spec__ = getattr(real_os, "__spec__", None)

    for attr in _OS_SAFE_ATTRS:
        val = getattr(real_os, attr, None)
        if val is not None:
            setattr(safe, attr, val)

    safe.path = real_os.path
    return safe


# ---------------------------------------------------------------------------
# Safe io proxy
# ---------------------------------------------------------------------------
def _build_safe_io(real_io, restricted_open_fn):
    """Return a copy of the io module with open/FileIO replaced."""
    import types

    safe = types.ModuleType("io")
    safe.__package__ = real_io.__package__
    safe.__loader__ = getattr(real_io, "__loader__", None)
    safe.__spec__ = getattr(real_io, "__spec__", None)

    # Copy everything first
    for attr in dir(real_io):
        if not attr.startswith("_"):
            try:
                setattr(safe, attr, getattr(real_io, attr))
            except (AttributeError, TypeError):
                pass

    # Replace open with the restricted version
    safe.open = restricted_open_fn

    # Replace FileIO with a blocked version (low-level file descriptor open)
    def _blocked_fileio(*args, **kwargs):
        raise PermissionError("io.FileIO is not allowed in this environment")

    safe.FileIO = _blocked_fileio

    return safe


# ---------------------------------------------------------------------------
# Safe pathlib proxy
# ---------------------------------------------------------------------------
def _build_safe_pathlib(real_pathlib, restricted_open_fn):
    """Return a copy of pathlib with Path file I/O methods restricted."""
    import copy
    import types

    safe = types.ModuleType("pathlib")
    safe.__package__ = real_pathlib.__package__
    safe.__loader__ = getattr(real_pathlib, "__loader__", None)
    safe.__spec__ = getattr(real_pathlib, "__spec__", None)

    # Copy all module-level attributes
    for attr in dir(real_pathlib):
        if not attr.startswith("_"):
            try:
                setattr(safe, attr, getattr(real_pathlib, attr))
            except (AttributeError, TypeError):
                pass

    # Create a restricted Path subclass
    class RestrictedPath(real_pathlib.Path):
        """Path subclass that routes file reads through the restricted open."""

        def open(self, mode="r", *args, **kwargs):
            return restricted_open_fn(str(self), mode, *args, **kwargs)

        def read_text(self, encoding=None, errors=None):
            kwargs = {}
            if encoding is not None:
                kwargs["encoding"] = encoding
            if errors is not None:
                kwargs["errors"] = errors
            with restricted_open_fn(str(self), "r", **kwargs) as f:
                return f.read()

        def read_bytes(self):
            with restricted_open_fn(str(self), "rb") as f:
                return f.read()

        def write_text(self, *args, **kwargs):
            raise PermissionError(f"Write access is not allowed: {self}")

        def write_bytes(self, *args, **kwargs):
            raise PermissionError(f"Write access is not allowed: {self}")

    # Also restrict PurePosixPath / PosixPath if they exist
    safe.Path = RestrictedPath
    if hasattr(real_pathlib, "PosixPath"):
        safe.PosixPath = RestrictedPath
    if hasattr(real_pathlib, "WindowsPath"):
        safe.WindowsPath = RestrictedPath

    return safe


_SANDBOX_APPLIED = False


def _apply_repl_sandbox(transcript_dir: str) -> None:
    """Monkeypatch LocalREPL.setup() to inject restricted open/import.

    After the original setup() builds the namespace we replace the builtins
    with restricted versions that:
      - Only allow reading files under the transcripts directory and the
        REPL's own temp directory (used by load_context for JSON context).
      - Block importing dangerous modules (subprocess, socket, http, …).

    Must be called BEFORE any RLM.completion() call.
    """
    global _SANDBOX_APPLIED
    if _SANDBOX_APPLIED:
        return

    from rlm.environments.local_repl import LocalREPL

    _original_setup = LocalREPL.setup

    def _sandboxed_setup(self):
        _original_setup(self)

        # Allowed dirs: transcripts + the REPL's temp dir (for context JSON)
        allowed_dirs = [transcript_dir]
        if hasattr(self, "temp_dir") and self.temp_dir:
            allowed_dirs.append(self.temp_dir)

        builtins_dict = self.globals["__builtins__"]
        restricted_open_fn = _make_restricted_open(allowed_dirs)
        builtins_dict["open"] = restricted_open_fn
        builtins_dict["__import__"] = _make_restricted_import(
            _BLOCKED_MODULES, restricted_open_fn,
        )

    LocalREPL.setup = _sandboxed_setup
    _SANDBOX_APPLIED = True


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""\
You are a research assistant specialized in analyzing Lenny's Podcast transcripts. \
You have access to a REPL environment with a catalog of podcast episodes and the ability \
to read full transcript files and query sub-LLMs for analysis.

## REPL Environment

Your REPL is initialized with:

1. A `context` variable — a JSON dict containing:
   - `"catalog"`: a list of episode dicts, each with: slug, guest, title, publish_date, duration, keywords
   - `"youtube_urls"`: a `{slug: url}` lookup dict for YouTube links (use this for citation URLs)
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

## Security Restrictions

File access is restricted to the transcripts directory only. You cannot read files \
outside of the transcripts path. Write access is not available. Some Python modules \
(subprocess, socket, http, urllib) are blocked. Use the available tools \
(open, re, json, os.path) for transcript analysis.

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
                "youtube_url": context["youtube_urls"].get(ep["slug"], ""),
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
        verbose: bool = False,
        max_iterations: int = 30,
    ):
        # Apply runtime patches lazily (not at import time)
        _apply_throttle_patch()
        _apply_repl_sandbox(index.transcript_dir)

        self.index = index
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.session_costs = SessionCosts()
        self.conversation_history: list[dict[str, str]] = []
        self._on_rate_limit: Callable[[float, int, int], None] | None = None

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

    @property
    def api_key(self) -> str:
        """Return the Anthropic API key (for shared use by RAGEngine)."""
        return os.environ.get("ANTHROPIC_API_KEY", "")

    def query(self, question: str) -> tuple[str, QueryCost]:
        """Run a question through the RLM engine.

        Includes adaptive payload trimming and retry with backoff on 429.
        Returns (answer_text, query_cost).
        """
        context_payload = self._build_context_payload()

        retry_start = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(_MAX_QUERY_RETRIES + 1):
            try:
                result = self.rlm.completion(
                    prompt=context_payload,
                    root_prompt=question,
                )
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt >= _MAX_QUERY_RETRIES:
                    raise
                wait = _compute_retry_wait(e, attempt)
                elapsed = time.perf_counter() - retry_start
                if elapsed + wait > _MAX_TOTAL_RETRY_SECS:
                    raise
                if self._on_rate_limit:
                    self._on_rate_limit(wait, attempt + 1, _MAX_QUERY_RETRIES)
                time.sleep(wait)
                continue
            except Exception as e:
                if not is_rate_limit_error(e):
                    raise
                last_error = e
                if attempt >= _MAX_QUERY_RETRIES:
                    raise
                wait = _compute_retry_wait(e, attempt)
                elapsed = time.perf_counter() - retry_start
                if elapsed + wait > _MAX_TOTAL_RETRY_SECS:
                    raise
                if self._on_rate_limit:
                    self._on_rate_limit(wait, attempt + 1, _MAX_QUERY_RETRIES)
                time.sleep(wait)
                continue

            # Success — track costs and return
            query_cost = self.session_costs.add_query(
                result.usage_summary,
                result.execution_time,
            )
            self.conversation_history.append({
                "question": question,
                "answer": result.response[:2000],
            })
            return result.response, query_cost

        # All retries exhausted (shouldn't reach here — raise above covers it)
        raise last_error  # type: ignore[misc]

    def _build_context_payload(self) -> dict:
        """Build a token-trimmed context payload for the RLM."""
        catalog = self.index.get_catalog()
        slim_catalog = [
            {
                k: (v[:3] if k == "keywords" and isinstance(v, list) else v)
                for k, v in entry.items()
                if k != "youtube_url"
            }
            for entry in catalog
        ]
        return {
            "catalog": slim_catalog,
            "transcript_dir": self.index.transcript_dir,
            "youtube_urls": {ep["slug"]: ep["youtube_url"] for ep in catalog},
            "conversation_history": self._trimmed_history(),
        }

    def _trimmed_history(self) -> list[dict]:
        """Adaptive trimming: recent turns full, older turns compressed."""
        recent = self.conversation_history[-3:]
        older = self.conversation_history[-7:-3]
        trimmed_older = [
            {"question": h.get("question", ""), "answer": h.get("answer", "")[:300]}
            for h in older
        ]
        trimmed_recent = [
            {"question": h.get("question", ""), "answer": h.get("answer", "")[:1500]}
            for h in recent
        ]
        return trimmed_older + trimmed_recent

    def close(self):
        """Clean up resources."""
        if hasattr(self.rlm, 'close'):
            self.rlm.close()
