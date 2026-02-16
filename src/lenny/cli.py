"""CLI chat loop for exploring Lenny's Podcast transcripts."""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from lenny.costs import format_query_cost, format_session_cost
from lenny.data import TranscriptIndex
from lenny.engine import LennyEngine

console = Console()

WELCOME = """\
[bold]Lenny CLI[/bold] — Explore Lenny's Podcast with RLM

Ask questions about themes, patterns, and insights across {count} episodes.
The AI will search transcripts, analyze content, and cite specific episodes.

Commands: /help /episodes /cost /verbose /quit
"""

HELP_TEXT = """\
[bold]Commands[/bold]
  /help      Show this help message
  /episodes  List loaded episodes (count + sample)
  /cost      Show session token usage and cost
  /verbose   Toggle verbose mode (see RLM orchestration)
  /quit      Exit

[bold]Example queries[/bold]
  What frameworks do guests recommend for prioritization?
  Which guests disagree with each other on hiring?
  Find all mentions of 'founder mode' and summarize the perspectives
  What do PMs say about working with engineers?
"""


def main():
    """Entry point for the lenny CLI."""
    console.print()

    # Load transcripts
    with console.status("[bold]Loading transcripts..."):
        try:
            index = TranscriptIndex.load()
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

    console.print(f"  Loaded [bold]{len(index.episodes)}[/bold] episodes")
    console.print()

    # Initialize engine
    try:
        engine = LennyEngine(index=index, verbose=True)
    except EnvironmentError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    verbose = True
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
            cmd = query.lower().split()[0]
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
            elif cmd == "/verbose":
                verbose = not verbose
                engine.verbose = verbose
                engine.rlm.verbose = verbose
                console.print(f"  Verbose mode: [bold]{'on' if verbose else 'off'}[/bold]")
                continue
            else:
                console.print(f"  Unknown command: {cmd}. Type /help for options.")
                continue

        # Run query through RLM
        console.print()
        try:
            if verbose:
                # Verbose: RLM's VerbosePrinter streams iteration details to stdout
                console.print(f"[dim]  Searching {len(index.episodes)} episodes...[/dim]\n")
                answer, query_cost = engine.query(query)
            else:
                # Quiet: show an animated spinner while blocking
                with console.status(
                    f"[bold]  Searching {len(index.episodes)} episodes...",
                    spinner="dots",
                ):
                    answer, query_cost = engine.query(query)
        except KeyboardInterrupt:
            console.print("\n[dim]Query interrupted.[/dim]")
            continue
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            continue

        # Display answer
        console.print()
        console.print(Panel(
            Markdown(answer),
            title="[bold]Lenny[/bold]",
            border_style="blue",
            padding=(1, 2),
        ))

        # Display cost
        console.print()
        console.print(Text(format_query_cost(query_cost), style="dim"))
        console.print()


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
