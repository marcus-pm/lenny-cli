"""CLI chat loop for exploring Lenny's Podcast transcripts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.text import Text

from lenny.costs import format_query_cost, format_session_cost
from lenny.data import TranscriptIndex
from lenny.engine import LennyEngine
from lenny.progress import ProgressDisplay
from lenny.rag import RAGEngine
from lenny.router import QueryMode, RouteDecision, classify_query
from lenny.search import TranscriptSearchIndex

console = Console()

WELCOME = """\
[bold]Lenny CLI[/bold] — Explore Lenny's Podcast with RLM + RAG

Ask questions about themes, patterns, and insights across {count} episodes.
Queries are auto-routed: fast RAG for targeted lookups, deep RLM for synthesis.

Commands: /help /episodes /cost /mode /verbose /quit
"""

HELP_TEXT = """\
[bold]Commands[/bold]
  /help      Show this help message
  /episodes  List loaded episodes (count + sample)
  /cost      Show session token usage and cost
  /mode      Show or set routing mode (auto, rag, rlm)
  /verbose   Toggle verbose mode (see RLM orchestration)
  /quit      Exit

[bold]Routing modes[/bold]
  /mode auto   Automatic routing based on query (default)
  /mode rag    Force fast RAG path for all queries
  /mode rlm    Force deep RLM path for all queries

[bold]Example queries[/bold]
  What did Brian Chesky say about founder mode?          → [dim]RAG (fast)[/dim]
  What frameworks do guests recommend for prioritization? → [dim]RLM (deep)[/dim]
  Which guests disagree with each other on hiring?        → [dim]RLM (deep)[/dim]
  Find the quote about 'disagree and commit'              → [dim]RAG (fast)[/dim]
"""


def main():
    """Entry point for the lenny CLI."""
    console.print()

    config_path = _load_user_config_env()
    try:
        _ensure_api_key(config_path)
    except EnvironmentError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    # Load transcripts
    with console.status("[bold]Loading transcripts..."):
        try:
            index = TranscriptIndex.load()
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    console.print(f"  Loaded [bold]{len(index.episodes)}[/bold] episodes")

    # Build BM25 search index (cached to disk after first build)
    cache_path = os.path.join(
        os.path.dirname(index.transcript_dir), ".cache", "bm25_index.pkl",
    )
    with console.status("[bold]Loading search index..."):
        search_index = TranscriptSearchIndex.load_or_build(index, cache_path)

    console.print(f"  Search index: [bold]{len(search_index.chunks):,}[/bold] chunks")
    console.print()

    # Initialize engines
    try:
        engine = LennyEngine(index=index, verbose=False)
    except EnvironmentError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    rag_engine = RAGEngine(
        search_index=search_index,
        api_key=engine.api_key,
    )

    verbose = False
    forced_mode: QueryMode | None = None  # None = auto
    console.print(WELCOME.format(count=len(index.episodes)))

    # Chat loop
    while True:
        try:
            query = console.input("[bold green]You:[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue

        # Slash commands
        if query.startswith("/"):
            cmd_parts = query.lower().split()
            cmd = cmd_parts[0]
            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/help":
                console.print(HELP_TEXT)
                continue
            elif cmd == "/episodes":
                _show_episodes(index)
                continue
            elif cmd == "/cost":
                _show_cost(engine)
                continue
            elif cmd == "/mode":
                forced_mode = _handle_mode_command(cmd_parts, forced_mode)
                continue
            elif cmd == "/verbose":
                verbose = not verbose
                engine.verbose = verbose
                engine.rlm.verbose = verbose
                console.print(f"  Verbose mode: [bold]{'on' if verbose else 'off'}[/bold]")
                continue
            else:
                console.print(f"  Unknown command: {cmd}. Type /help for options.")
                continue

        # Route the query
        if forced_mode == QueryMode.RLM:
            route = RouteDecision(QueryMode.RLM, "forced")
        elif forced_mode == QueryMode.RAG:
            route = RouteDecision(QueryMode.RAG, "forced")
        else:
            route = classify_query(
                query, engine.conversation_history, client=rag_engine.client,
            )

        console.print()
        mode_label = "rag" if route.mode == QueryMode.RAG else "rlm"
        console.print(f"  [dim]→ {mode_label} ({route.reason})[/dim]")

        # Execute via the appropriate path
        try:
            if route.mode == QueryMode.RAG:
                with ProgressDisplay(
                    console, initial_status="Searching transcripts...",
                ):
                    answer, query_cost = rag_engine.query(
                        query, engine.conversation_history,
                    )
                engine.session_costs.add_raw_query_cost(query_cost)
                engine.conversation_history.append({
                    "question": query,
                    "answer": answer[:2000],
                    "mode": "rag",
                })
            else:
                # RLM path
                if verbose:
                    console.print(f"[dim]  Searching {len(index.episodes)} episodes...[/dim]\n")
                    answer, query_cost = engine.query(query)
                else:
                    progress = ProgressDisplay(
                        console,
                        initial_status=f"Searching {len(index.episodes)} episodes...",
                    )
                    engine.rlm.logger = progress
                    try:
                        with progress:
                            answer, query_cost = engine.query(query)
                    finally:
                        engine.rlm.logger = None
                # engine.query() already appends to conversation_history — add mode tag
                if engine.conversation_history:
                    engine.conversation_history[-1]["mode"] = "rlm"
        except KeyboardInterrupt:
            console.print("\n[dim]Query interrupted.[/dim]")
            continue
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            continue

        # Display answer
        console.print()
        border = "cyan" if route.mode == QueryMode.RAG else "blue"
        title_suffix = " [dim](RAG)[/dim]" if route.mode == QueryMode.RAG else " [dim](RLM)[/dim]"
        console.print(Panel(
            Markdown(answer),
            title=f"[bold]Lenny[/bold]{title_suffix}",
            border_style=border,
            padding=(1, 2),
        ))

        # Display cost
        console.print()
        console.print(Text(format_query_cost(query_cost), style="dim"))
        console.print()


def _handle_mode_command(
    cmd_parts: list[str],
    current_mode: QueryMode | None,
) -> QueryMode | None:
    """Handle the /mode slash command. Returns the new forced mode."""
    if len(cmd_parts) == 1:
        # Show current mode
        mode_name = current_mode.value if current_mode else "auto"
        console.print(f"  Routing mode: [bold]{mode_name}[/bold]")
        return current_mode

    arg = cmd_parts[1]
    if arg == "auto":
        console.print("  Routing mode: [bold]auto[/bold] (queries routed automatically)")
        return None
    elif arg == "rag":
        console.print("  Routing mode: [bold]rag[/bold] (all queries use fast RAG)")
        return QueryMode.RAG
    elif arg == "rlm":
        console.print("  Routing mode: [bold]rlm[/bold] (all queries use deep RLM)")
        return QueryMode.RLM
    else:
        console.print(f"  Unknown mode: {arg}. Options: auto, rag, rlm")
        return current_mode


def _show_episodes(index: TranscriptIndex):
    """Show episode count and a sample."""
    console.print(f"\n  [bold]{len(index.episodes)}[/bold] episodes loaded\n")
    sample = list(index.episodes.values())[:10]
    for ep in sample:
        console.print(f"  [dim]{ep.publish_date}[/dim]  {ep.guest} — {ep.title}")
    if len(index.episodes) > 10:
        console.print(f"  [dim]... and {len(index.episodes) - 10} more[/dim]")
    console.print()


def _show_cost(engine: LennyEngine):
    """Show session cost summary."""
    console.print()
    if not engine.session_costs.queries:
        console.print("  No queries yet.")
    else:
        console.print(Text(format_session_cost(engine.session_costs), style="dim"))
    console.print()


def _user_config_path() -> Path:
    """Return the per-user config file path."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "lenny" / "config.env"


def _load_user_config_env() -> Path:
    """Load user-level env config if present and return its path."""
    config_path = _user_config_path()
    if config_path.is_file():
        load_dotenv(config_path, override=True)
    return config_path


def _ensure_api_key(config_path: Path) -> str:
    """Ensure ANTHROPIC_API_KEY is present, prompting interactively if missing."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        return api_key

    if not sys.stdin.isatty():
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not found.\n"
            "Run `lenny` in an interactive terminal to set it up, or set it manually:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    console.print("\n[bold]First-time setup[/bold]")
    console.print("  An Anthropic API key is required to run queries.")
    while True:
        entered = Prompt.ask("  Enter your Anthropic API key", password=True).strip()
        if entered:
            api_key = entered
            break
        console.print("  [yellow]API key cannot be empty.[/yellow]")

    os.environ["ANTHROPIC_API_KEY"] = api_key

    if Confirm.ask(f"  Save this key for future runs at {config_path}?", default=True):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"ANTHROPIC_API_KEY={api_key}\n")
        try:
            os.chmod(config_path, 0o600)
        except OSError:
            pass
        console.print("  [dim]Saved.[/dim]\n")

    return api_key
