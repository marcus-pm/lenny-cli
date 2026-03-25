"""CLI chat loop for exploring Lenny's Podcast transcripts."""

from __future__ import annotations

import json
import os
import sys
import tempfile
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

    # Check for env var override BEFORE loading config files
    _env_api_key_before_config = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    _load_user_config_env()
    try:
        current_auth = _ensure_auth(env_override=_env_api_key_before_config)
    except EnvironmentError as e:
        console.print(f"[error]Error:[/error] {e}")
        sys.exit(1)

    api_key = current_auth["api_key"]

    # Initialize MCP client and transcript cache
    from lenny.cache import TranscriptCache
    from lenny.mcp_client import MCPClient, MCPError

    mcp_client = None
    transcript_cache = TranscriptCache()
    index = None

    # MCP connection is required
    mcp_token = os.environ.get("LENNY_MCP_TOKEN", "").strip()
    if mcp_token:
        mcp_client = MCPClient(token=mcp_token)
        with console.status("[accent]Connecting to Lenny's Data...[/accent]"):
            try:
                if mcp_client.health_check():
                    index = TranscriptIndex.load_from_mcp(mcp_client, transcript_cache)
                    console.print(
                        f"  [success]\u2713[/success] {len(index.episodes)} episodes connected via MCP"
                    )
                else:
                    console.print()
                    console.print("[error]MCP server unreachable.[/error]")
                    console.print("  LENNY_MCP_TOKEN is set but the server did not respond.")
                    console.print("  Check your internet connection or try again later.")
                    console.print()
                    sys.exit(1)
            except MCPError as e:
                console.print()
                console.print(f"[error]MCP connection failed:[/error] {e}")
                console.print("  Check your internet connection or try again later.")
                console.print()
                sys.exit(1)
    else:
        console.print()
        console.print("[error]MCP server token required.[/error]")
        console.print()
        console.print("  Lenny requires a connection to the MCP data server.")
        console.print("  Get your access token at:")
        console.print()
        console.print("    [accent]https://www.lennysdata.com/access/mcp?tab=claude-code[/accent]")
        console.print()
        console.print("  Then set it in your environment:")
        console.print()
        console.print("    [faint]export LENNY_MCP_TOKEN=your-token-here[/faint]")
        console.print()
        sys.exit(1)

    console.print()

    # Initialize engines — wrapped in helper for re-init on /auth switch
    def _init_engines(key: str):
        eng = LennyEngine(
            index=index,
            api_key=key,
            verbose=False,
            mcp_client=mcp_client,
            cache=transcript_cache,
        )
        rag = RAGEngine(
            api_key=key,
            mcp_client=mcp_client,
        )
        return eng, rag

    try:
        engine, rag_engine = _init_engines(api_key)
    except EnvironmentError as e:
        console.print(f"[error]Error:[/error] {e}")
        sys.exit(1)

    verbose = False
    forced_mode: QueryMode | None = None  # None = auto

    # Splash card (replaces plain WELCOME text)
    active_mode_label = "auto"
    splash = build_splash_card(
        episode_count=len(index.episodes),
        active_mode=active_mode_label,
        theme=current_theme,
        console=console,
        auth_label=_auth_mode_label(current_auth),
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
            elif cmd == "/auth":
                subcmd = cmd_parts[1] if len(cmd_parts) > 1 else ""
                if subcmd == "switch":
                    try:
                        new_auth = _run_auth_setup()
                    except (EOFError, KeyboardInterrupt):
                        console.print("\n  [faint]Cancelled.[/faint]")
                        continue
                    api_key = new_auth["api_key"]
                    os.environ["ANTHROPIC_API_KEY"] = api_key
                    current_auth = new_auth
                    try:
                        engine, rag_engine = _init_engines(api_key)
                        engine.verbose = verbose
                        engine.rlm.verbose = verbose
                        console.print("  [success]Authentication updated. Session restarted.[/success]")
                    except Exception as e:
                        console.print(f"  [error]Failed to reinitialize: {e}[/error]")
                elif subcmd == "unlink":
                    auth_path = _auth_config_path()
                    if not auth_path.is_file():
                        console.print("  No saved credentials to remove.")
                    elif Confirm.ask(
                        "  Remove saved credentials? You'll need to reconfigure on next launch",
                        default=False,
                    ):
                        auth_path.unlink()
                        console.print(f"  [faint]Credentials removed. Exiting...[/faint]")
                        break
                    else:
                        console.print("  [faint]Cancelled.[/faint]")
                else:
                    # Show auth status
                    console.print()
                    console.print(f"  Auth: [accent]{_auth_mode_label(current_auth)}[/accent]"
                                  f" ({_mask_key(current_auth['api_key'])})")
                    source = current_auth.get("source", "")
                    if source == "env":
                        console.print("  Source: environment variable")
                    elif _auth_config_path().is_file():
                        console.print(f"  Saved to: {_auth_config_path()}")
                    console.print()
                    console.print("  [faint]/auth switch  — change API key or auth mode[/faint]")
                    console.print("  [faint]/auth unlink  — remove saved credentials and exit[/faint]")
                    console.print()
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

                    # Wire rate-limit callback to update progress spinner
                    def _on_rate_limit(wait_secs, attempt, max_attempts):
                        progress.set_status(
                            PROGRESS_LABELS["rate_limited"].format(wait=int(wait_secs))
                        )

                    engine._on_rate_limit = _on_rate_limit
                    try:
                        with progress:
                            answer, query_cost = engine.query(query)
                    finally:
                        engine.rlm.logger = None
                        engine._on_rate_limit = None
                # engine.query() already appends to conversation_history — add mode tag
                if engine.conversation_history:
                    engine.conversation_history[-1]["mode"] = "research"
        except KeyboardInterrupt:
            console.print("\n[faint]Query interrupted.[/faint]")
            continue
        except Exception as e:
            import anthropic as _anthropic
            from lenny.engine import is_rate_limit_error, _MAX_QUERY_RETRIES
            if isinstance(e, _anthropic.AuthenticationError):
                console.print(
                    "\n[error]Authentication failed.[/error] "
                    "Your API key may be invalid or expired.\n"
                    "  Run [accent]/auth switch[/accent] to reconfigure."
                )
            elif is_rate_limit_error(e):
                console.print(
                    f"\n[warning]Rate limit reached after "
                    f"{_MAX_QUERY_RETRIES + 1} attempts.[/warning] "
                    "The API's per-minute token quota was exceeded.\n"
                    "  Wait a minute and try again, or use "
                    "[accent]/mode fast[/accent] for a lighter query."
                )
            else:
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
    console.print(f"\n  [accent]{len(index.episodes)}[/accent] episodes available\n")
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


def _config_dir() -> Path:
    """Return the per-user config directory (XDG-aware)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "lenny"


def _auth_config_path() -> Path:
    """Return the path to auth.json."""
    return _config_dir() / "auth.json"


def _legacy_config_path() -> Path:
    """Return the legacy config.env path (for migration)."""
    return _config_dir() / "config.env"


def _load_auth_config() -> dict | None:
    """Read auth.json and return its contents, or None if missing/corrupt."""
    path = _auth_config_path()
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("api_key"):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return None


def _save_auth_config(auth_mode: str, api_key: str, label: str = "") -> Path:
    """Atomically write auth.json with restricted permissions. Returns path."""
    path = _auth_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"auth_mode": auth_mode, "api_key": api_key, "label": label}
    # Atomic write: write to temp file in same dir, then rename
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def _mask_key(api_key: str) -> str:
    """Mask an API key for display, showing only last 4 characters."""
    if len(api_key) <= 8:
        return "****"
    return api_key[:7] + "..." + api_key[-4:]


def _validate_api_key(api_key: str) -> bool:
    """Validate an API key by making a lightweight API call.

    Returns True on success or non-auth errors (key might be valid).
    Returns False only on authentication errors.
    """
    import anthropic

    try:
        client = anthropic.Anthropic(api_key=api_key)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except anthropic.AuthenticationError:
        return False
    except Exception:
        # Network errors, rate limits, etc. — key format might be valid
        return True


def _load_user_config_env():
    """Load user-level env config if present (for non-auth vars like MCP token)."""
    config_path = _legacy_config_path()
    if config_path.is_file():
        load_dotenv(config_path, override=True)


def _run_auth_setup() -> dict:
    """Run the interactive auth selection flow. Returns auth config dict."""
    console.print("\n[accent]Authentication required[/accent]")
    console.print("  An Anthropic API key is required to run queries.\n")
    console.print("  How would you like to connect to Anthropic?\n")
    console.print("    [accent][1][/accent] Personal API Key  — your own Anthropic account")
    console.print("    [accent][2][/accent] Organization Key  — shared team or company key")
    console.print()

    while True:
        choice = Prompt.ask("  Choose", choices=["1", "2"], default="1")
        if choice in ("1", "2"):
            break

    auth_mode = "personal" if choice == "1" else "organization"
    label = ""

    if auth_mode == "organization":
        label = Prompt.ask("  Organization name (optional)", default="").strip()

    while True:
        api_key = Prompt.ask("  Enter your Anthropic API key", password=True).strip()
        if not api_key:
            console.print("  [warning]API key cannot be empty.[/warning]")
            continue

        with console.status("  [accent]Validating API key...[/accent]"):
            if _validate_api_key(api_key):
                break
            else:
                console.print("  [warning]Invalid API key. Please check and try again.[/warning]")

    auth = {"auth_mode": auth_mode, "api_key": api_key, "label": label}

    if Confirm.ask(f"  Save credentials to {_auth_config_path()}?", default=True):
        _save_auth_config(auth_mode, api_key, label)
        console.print("  [faint]Saved.[/faint]")

    console.print()
    return auth


def _ensure_auth(env_override: str = "") -> dict:
    """Ensure authentication is configured. Returns auth config dict.

    Resolution order:
    1. ANTHROPIC_API_KEY env var set before config loading (direct override)
    2. auth.json config file
    3. Migration from legacy config.env
    4. Interactive setup (TTY only)

    Args:
        env_override: The ANTHROPIC_API_KEY value from the environment *before*
            any config files were loaded. If set, takes priority over all other sources.
    """
    # 1. Environment variable override (set by user, not loaded from config)
    if env_override:
        return {"auth_mode": "personal", "api_key": env_override, "label": "", "source": "env"}

    # 2. Read auth.json
    auth = _load_auth_config()
    if auth:
        os.environ["ANTHROPIC_API_KEY"] = auth["api_key"]
        return auth

    # 3. Migrate from legacy config.env
    legacy = _legacy_config_path()
    if legacy.is_file():
        load_dotenv(legacy, override=True)
        migrated_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if migrated_key:
            auth = {"auth_mode": "personal", "api_key": migrated_key, "label": ""}
            _save_auth_config("personal", migrated_key)
            console.print("  [faint]Migrated API key to auth.json[/faint]")
            return auth

    # 4. Interactive setup
    if not sys.stdin.isatty():
        raise EnvironmentError(
            "Authentication not configured.\n"
            "Run `lenny` in an interactive terminal to set up your API key, or set it manually:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    auth = _run_auth_setup()
    os.environ["ANTHROPIC_API_KEY"] = auth["api_key"]
    return auth


def _auth_mode_label(auth: dict) -> str:
    """Return a human-readable label for the current auth mode."""
    mode = auth.get("auth_mode", "personal")
    label = auth.get("label", "")
    if mode == "organization" and label:
        return f"Organization Key ({label})"
    elif mode == "organization":
        return "Organization Key"
    return "Personal API Key"
