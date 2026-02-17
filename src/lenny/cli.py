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
from lenny.persist import format_terminal_citations, save_response_markdown
from lenny.progress import ProgressDisplay
from lenny.search import TranscriptSearchIndex
from lenny.style import (
    DEFAULT_THEME,
    GOODBYE_TEXT,
    HELP_TEXT,
    PROGRESS_LABELS,
    THEMES,
    LennyTheme,
    answer_panel_params,
    build_splash_card,
    format_cost_compact,
    format_route_badge,
    format_save_confirmation,
)

_initial_theme = THEMES[DEFAULT_THEME]
console = Console(theme=_initial_theme.to_rich_theme())


def main():
    """Entry point for the lenny CLI."""
    # Handle --version before any heavy imports or loading
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        from importlib.metadata import version
        print(f"lenny {version('lenny-cli')}")
        sys.exit(0)

    # Deferred heavy imports — keeps --version fast (no anthropic SDK or rlms loaded
    # until the user actually runs the app).
    from lenny.engine import LennyEngine
    from lenny.rag import RAGEngine
    from lenny.router import QueryMode, RouteDecision, classify_query

    # ---------------------------------------------------------------------------
    # Helpers that reference QueryMode — nested here so the deferred import above
    # is in scope at runtime (they are only ever called from within main()).
    # ---------------------------------------------------------------------------

    def _handle_mode_command(
        cmd_parts: list[str],
        current_mode: QueryMode | None,
    ) -> QueryMode | None:
        """Handle the /mode slash command. Returns the new forced mode."""
        if len(cmd_parts) == 1:
            mode_name = current_mode.value if current_mode else "auto"
            console.print(f"  Routing mode: [accent]{mode_name}[/accent]")
            return current_mode

        arg = cmd_parts[1]
        if arg == "auto":
            console.print("  Routing mode: [accent]auto[/accent] (queries routed automatically)")
            return None
        elif arg in ("fast", "rag"):
            console.print("  Routing mode: [accent]fast[/accent] (all queries use fast path)")
            return QueryMode.FAST
        elif arg in ("research", "rlm"):
            console.print("  Routing mode: [accent]research[/accent] (all queries use research path)")
            return QueryMode.RESEARCH
        else:
            console.print(f"  Unknown mode: {arg}. Options: auto, fast, research")
            return current_mode

    def _handle_theme_command(
        cmd_parts: list[str],
        current_theme: LennyTheme,
        theme_pushed: bool,
    ) -> tuple[LennyTheme, bool]:
        """Handle /theme [name]. Returns (new_theme, theme_pushed)."""
        if len(cmd_parts) == 1:
            names = ", ".join(THEMES.keys())
            console.print(f"  Theme: [accent]{current_theme.name}[/accent] (options: {names})")
            return current_theme, theme_pushed

        name = cmd_parts[1].lower()
        if name not in THEMES:
            names = ", ".join(THEMES.keys())
            console.print(f"  Unknown theme: {name}. Options: {names}")
            return current_theme, theme_pushed

        new_theme = THEMES[name]

        # Pop the previous pushed theme to avoid stacking
        if theme_pushed:
            console.pop_theme()

        console.push_theme(new_theme.to_rich_theme())
        console.print(f"  Theme: [accent]{new_theme.name}[/accent]")
        return new_theme, True

    # ---------------------------------------------------------------------------

    console.print()

    current_theme = THEMES[DEFAULT_THEME]
    _theme_pushed = False  # tracks whether we've pushed a theme on top

    config_path = _load_user_config_env()
    try:
        _ensure_api_key(config_path)
    except EnvironmentError as e:
        console.print(f"[error]Error:[/error] {e}")
        sys.exit(1)

    # Find or download transcripts, then load index
    from lenny.transcripts import ensure_transcripts  # noqa: E402

    try:
        transcript_dir = ensure_transcripts(console)
    except FileNotFoundError as e:
        console.print(f"[error]Error:[/error] {e}")
        sys.exit(1)

    with console.status("[accent]Loading transcripts...[/accent]"):
        try:
            index = TranscriptIndex.load(transcript_dir)
        except FileNotFoundError as e:
            console.print(f"[error]Error:[/error] {e}")
            sys.exit(1)

    console.print(f"  [success]\u2713[/success] {len(index.episodes)} episodes loaded")

    # Build BM25 search index (cached to disk after first build)
    cache_path = os.path.join(
        os.path.dirname(index.transcript_dir), ".cache", "bm25_index.json",
    )
    with console.status("[accent]Loading search index...[/accent]"):
        search_index = TranscriptSearchIndex.load_or_build(index, cache_path)

    console.print(f"  [success]\u2713[/success] {len(search_index.chunks):,} chunks indexed")
    console.print()

    # Initialize engines
    try:
        engine = LennyEngine(index=index, verbose=False)
    except EnvironmentError as e:
        console.print(f"[error]Error:[/error] {e}")
        sys.exit(1)

    rag_engine = RAGEngine(
        search_index=search_index,
        api_key=engine.api_key,
    )

    verbose = False
    forced_mode: QueryMode | None = None  # None = auto

    # Splash card (replaces plain WELCOME text)
    active_mode_label = "auto"
    splash = build_splash_card(
        episode_count=len(index.episodes),
        active_mode=active_mode_label,
        theme=current_theme,
        console=console,
    )
    console.print(splash)

    # Chat loop
    while True:
        try:
            query = console.input("[prompt]You:[/prompt] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print(f"\n[faint]{GOODBYE_TEXT}[/faint]")
            break

        if not query:
            continue

        # Slash commands
        if query.startswith("/"):
            cmd_parts = query.lower().split()
            cmd = cmd_parts[0]
            if cmd in ("/quit", "/exit", "/q"):
                console.print(f"[faint]{GOODBYE_TEXT}[/faint]")
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
            elif cmd == "/theme":
                current_theme, _theme_pushed = _handle_theme_command(
                    cmd_parts, current_theme, _theme_pushed,
                )
                continue
            elif cmd == "/verbose":
                verbose = not verbose
                engine.verbose = verbose
                engine.rlm.verbose = verbose
                console.print(f"  Verbose mode: [accent]{'on' if verbose else 'off'}[/accent]")
                continue
            else:
                console.print(f"  Unknown command: {cmd}. Type /help for options.")
                continue

        # Route the query
        if forced_mode == QueryMode.RESEARCH:
            route = RouteDecision(QueryMode.RESEARCH, "forced")
        elif forced_mode == QueryMode.FAST:
            route = RouteDecision(QueryMode.FAST, "forced")
        else:
            route = classify_query(
                query, engine.conversation_history, client=rag_engine.client,
            )

        console.print()
        mode_label = "fast" if route.mode == QueryMode.FAST else "research"
        console.print(format_route_badge(mode_label, route.reason, current_theme))

        # Execute via the appropriate path
        try:
            if route.mode == QueryMode.FAST:
                with ProgressDisplay(
                    console,
                    initial_status=PROGRESS_LABELS["searching_transcripts"],
                    theme=current_theme,
                ):
                    answer, query_cost = rag_engine.query(
                        query, engine.conversation_history,
                    )
                engine.session_costs.add_raw_query_cost(query_cost)
                engine.conversation_history.append({
                    "question": query,
                    "answer": answer[:2000],
                    "mode": "fast",
                })
            else:
                # Research path
                if verbose:
                    console.print(f"[faint]  Searching {len(index.episodes)} episodes...[/faint]\n")
                    answer, query_cost = engine.query(query)
                else:
                    progress = ProgressDisplay(
                        console,
                        initial_status=PROGRESS_LABELS["searching_episodes"].format(
                            n=len(index.episodes),
                        ),
                        theme=current_theme,
                    )
                    engine.rlm.logger = progress
                    try:
                        with progress:
                            answer, query_cost = engine.query(query)
                    finally:
                        engine.rlm.logger = None
                # engine.query() already appends to conversation_history — add mode tag
                if engine.conversation_history:
                    engine.conversation_history[-1]["mode"] = "research"
        except KeyboardInterrupt:
            console.print("\n[faint]Query interrupted.[/faint]")
            continue
        except Exception as e:
            console.print(f"\n[error]Error:[/error] {e}")
            continue

        # Display answer (with terminal-friendly citation URLs)
        terminal_answer = format_terminal_citations(answer)
        console.print()
        panel_kw = answer_panel_params(mode_label, current_theme)
        console.print(Panel(
            Markdown(terminal_answer),
            **panel_kw,
        ))

        # Display cost
        cost_str = format_query_cost(query_cost)
        console.print()
        console.print(format_cost_compact(cost_str))

        # Save response to timestamped Markdown file
        try:
            saved = save_response_markdown(
                query=query,
                answer=answer,
                mode=mode_label,
                cost_summary=cost_str,
            )
            console.print(format_save_confirmation(saved.name))
        except Exception:
            console.print("  [warning]Could not save response file.[/warning]")

        console.print()


def _show_episodes(index: TranscriptIndex):
    """Show episode count and a sample."""
    console.print(f"\n  [accent]{len(index.episodes)}[/accent] episodes loaded\n")
    sample = list(index.episodes.values())[:10]
    for ep in sample:
        console.print(f"  [faint]{ep.publish_date}[/faint]  {ep.guest} \u2014 [dim]{ep.title}[/dim]")
    if len(index.episodes) > 10:
        console.print(f"  [faint]... and {len(index.episodes) - 10} more[/faint]")
    console.print()


def _show_cost(engine: object):
    """Show session cost summary."""
    console.print()
    if not engine.session_costs.queries:
        console.print("  No queries yet.")
    else:
        console.print(format_cost_compact(format_session_cost(engine.session_costs)))
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

    console.print("\n[accent]First-time setup[/accent]")
    console.print("  An Anthropic API key is required to run queries.")
    while True:
        entered = Prompt.ask("  Enter your Anthropic API key", password=True).strip()
        if entered:
            api_key = entered
            break
        console.print("  [warning]API key cannot be empty.[/warning]")

    os.environ["ANTHROPIC_API_KEY"] = api_key

    if Confirm.ask(f"  Save this key for future runs at {config_path}?", default=True):
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(f"ANTHROPIC_API_KEY={api_key}\n")
        try:
            os.chmod(config_path, 0o600)
        except OSError:
            pass
        console.print("  [faint]Saved.[/faint]\n")

    return api_key
