"""Centralized UI tokens, themes, and rendering helpers for Lenny CLI.

All visual constants — colors, styles, ASCII art, microcopy strings,
and theme definitions — live in this single module.  Nothing in cli.py
or progress.py should hard-code a color or style string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date

from rich.console import Console
from rich.style import Style
from rich.text import Text
from rich.theme import Theme


# -------------------------------------------------------------------
# Color palette — warm amber / editorial
# -------------------------------------------------------------------
# Rich auto-downgrades hex colors to nearest match on 256-color
# terminals.  The hex values here were chosen to degrade gracefully.

# Primary accent — warm amber/orange
AMBER         = "#D4894A"
AMBER_BRIGHT  = "#E8A45E"
AMBER_DIM     = "#A66A32"

# Text hierarchy
CREAM         = "#F5E6D0"   # Primary body text (warm white)
CREAM_DIM     = "#C9B89E"   # Secondary text
GRAY_MUTED    = "#7A7A7A"   # Tertiary / metadata
GRAY_DARK     = "#555555"   # Faint / decorative
LOG_BROWN     = "#8B5E3C"   # Warm brown for campfire logs

# Route colors
FAST_COLOR     = "#6BBF6B"   # Soft green
RESEARCH_COLOR = "#7A9EC9"   # Soft blue

# Feedback
SUCCESS_COLOR  = "#6BBF6B"
ERROR_COLOR    = "#D45B5B"
WARNING_COLOR  = "#D4A94A"


# -------------------------------------------------------------------
# Theme definitions
# -------------------------------------------------------------------

@dataclass
class LennyTheme:
    """Bundled theme settings — Rich styles + behavioural flags."""

    name: str

    # Rich style strings (used as values in a Rich Theme dict)
    accent: str
    text: str
    text_dim: str
    text_faint: str
    fast: str
    research: str
    success: str
    error: str
    warning: str
    prompt: str
    border_fast: str
    border_research: str

    # Behavioural flags
    show_splash_art: bool = True
    show_daily_prompt: bool = True
    spinner_style: str = "dots"

    def to_rich_theme(self) -> Theme:
        """Convert to a ``rich.theme.Theme`` for ``Console(theme=...)``."""
        return Theme({
            "accent":           self.accent,
            "body":             self.text,
            "dim":              self.text_dim,
            "faint":            self.text_faint,
            "fast":             self.fast,
            "research":         self.research,
            "success":          self.success,
            "error":            self.error,
            "warning":          self.warning,
            "prompt":           self.prompt,
            "border.fast":      self.border_fast,
            "border.research":  self.border_research,
        })


THEME_WARM = LennyTheme(
    name="warm",
    accent=f"bold {AMBER}",
    text=CREAM,
    text_dim=CREAM_DIM,
    text_faint=GRAY_MUTED,
    fast=f"bold {FAST_COLOR}",
    research=f"bold {RESEARCH_COLOR}",
    success=SUCCESS_COLOR,
    error=ERROR_COLOR,
    warning=WARNING_COLOR,
    prompt=f"bold {AMBER}",
    border_fast=FAST_COLOR,
    border_research=RESEARCH_COLOR,
    show_splash_art=True,
    show_daily_prompt=True,
    spinner_style="dots",
)

THEME_MINIMAL = LennyTheme(
    name="minimal",
    accent="bold",
    text="white",
    text_dim="dim",
    text_faint="dim",
    fast="bold green",
    research="bold blue",
    success="green",
    error="red",
    warning="yellow",
    prompt="bold",
    border_fast="green",
    border_research="blue",
    show_splash_art=False,
    show_daily_prompt=False,
    spinner_style="dots",
)

THEMES: dict[str, LennyTheme] = {
    "warm": THEME_WARM,
    "minimal": THEME_MINIMAL,
}

DEFAULT_THEME = "warm"


# -------------------------------------------------------------------
# Terminal capability detection
# -------------------------------------------------------------------

def detect_color_depth(console: Console) -> str:
    """Detect terminal color depth.

    Returns ``'truecolor'``, ``'256'``, ``'standard'``, or ``'none'``.
    """
    colorterm = os.environ.get("COLORTERM", "").lower()
    if colorterm in ("truecolor", "24bit"):
        return "truecolor"

    term = os.environ.get("TERM", "")
    if "256color" in term:
        return "256"

    system = console.color_system
    if system is None:
        return "none"
    return system


def is_wide_terminal(console: Console, threshold: int = 80) -> bool:
    """Return *True* if the terminal is at least *threshold* columns wide."""
    return console.width >= threshold


# -------------------------------------------------------------------
# Brand mark — campfire (flame teardrop + crossed logs)
# -------------------------------------------------------------------

# Full mark (~100 chars wide).  Requires width >= 100 to display.
_BRAND_MARK_FLAME = [
    "                                           @@@@@@@@",
    "                                            @@@@@@@@@%+",
    "                                              @@@@@@@@@*+:",
    "                                              @@@@@=+*@@%*==",
    "                                               @@@@@===%@%#==",
    "                                   @@@          @@@@====#@%+==",
    "                                 @@@@@          @@@@====+%@%===",
    "                                @@@@@#===       @@@@=====#@@===",
    "                               @@@@@@#===       @@@@===-=#@@===",
    "                               @@@@%@%*===     @@@@*===-*%@#===",
    "                              @@@@ =*%@#+++ @@@@@@======#@%+=-",
    "                              @@@@===+*@@@@@@@%#=======*%%#-=",
    "                              @@@ -=====+++++=========+@@*=--",
    "                              @@@ ===================+@@*=-",
    "                              @@@@ =================+#@%+-=   @@@@@=-",
    "                               @@@@ ================*@@+==    @@@@@#+--",
    "                                @@@@*===============#@@===    @@@@@@%*=--",
    "                                 @@@@%==============*@@===    @@@@-+*@%+==",
]
_BRAND_MARK_LOGS = [
    "                      @@@@@@@@#+=  @@@@-============+#@#++*@@@@@@+===*@%*==",
    "                       @@@@@@@@*==  @@@@+=============*@@@@@@@#+=====-*%%*-=",
    "                        @@@@-=%@*-== @@@@================+++===-=======#@%===",
    "                        @@@@--+%#==+ @@@@======+*@%#+==================+#@#=-=",
    "                        @@@@=--#%*-=+@@@ ======--#@@@%#=================*@@===",
    "                        @@@@===+#%##%%#=========-=@@##@%#+==============*%@+==",
    "                       @@@@@=======+============--#@*-=%@%*-============+#@*==",
    "                       @@@@@============+##+-====-+%#=:-+%@#============+#@#==+",
    "                       @@@@---=======--*@@#+=====-=%%+:-=+#@#+==========+#@#+=+",
    "                      @@@@@=========-=*@*%%+-====-=%#=----+%@*==========+#@*==*",
    "                      @@@@@========-=#%*-*%#=====:*@*-----=+%%+========-+%@*==",
    "                      @@@@@=======--*%#--=*%#*+++*%#=:-----=%@*=========*@@===",
    "                      @@@@@=======-=##+----=+#%%%*=--------=%@*-========#@%===",
    "                      @@@@@=======-+%#=--------------------=%@*========*%%+==",
    "                       @@@@=======-=%#+:-------------------=@%*=======+%%#-==",
    "                       @@@@@======--#%*:------------------=#@*========%@*=== @@@@@",
    "                         @@@%======-+%%+-----------------=#@#+======+%@*=-+@@@@@@@@",
    "                          @@@ -=====-+%@#-:-------------+%@%=======*%@+=-@@@@ %@@@",
    "                         @@@@@@*-======*@@#*+=-----==+*%@#*======+%%#+-*@@@==#%%++",
    "                   @@@@@@@@@@@@@@--======*%@@@@@@@@@@@@*=======+%@#+=    ==+%@@#*",
    "            @@@@@@@@@@@@%+===-=*%@@%+========+******+=======+#%%#+==========+#@@@@@@@@@",
    "           @@@@@@@@@@%+==========++#@@%%*+===--====-===+*#%%%*+==============+@@@@@@@@@@@",
    "          @@@%=---=+%@%*===========-=++##%%%%%%%%%%%%%%%#*+=================*%@#+=--=-@@@@",
    "          @@#--+++===#@%+==================================================*%%*===+*+-=#@@@",
    "         @@%+-=@*=====#@%=====+*#%@@#+========================*#@@%#*+=====@@*====+#%=-*@@@",
    "         @@%+=+@*=*%+=*@@==*#%#**++=+**###+==============*###*+++++**#%#+=+@%+=+%++#@+=*@@@",
    "          @@*-=@#+=@#=+%@==---=+#%@@@@%#+====+#%@@@@#*+====+#%@@@@%*+=--==+@#+=%%=*@@==#@@@",
    "          @@#+-*%@%@*=*@%==*#%%%#**+==++*#%@@%#**++**#%@%##*+===+**#%%%#+==@%+=*@@@%+-+%@@",
    "           @@#-===+===#%+===--:--=*%@@@@@#+=--==+  ==---=*#@@@@@#+=-:--====*%*===+====%@@",
    "           *%%%+====-+++==+**#%@@%#**++==--=+          =---==+**##%@%%#**++=++======*%%%*",
    "            =+%@@#*+**%@@@@@%*+==-====                        =======+#%@@@@%#*+**%@@#+-",
    "             -=+*#######*+=----=+                                  ==---==+*####%##*=--",
    "               -========--=                                             ==-=========-",
]


def build_splash_art(theme: LennyTheme, console: Console) -> Text | None:
    """Build the styled brand mark, or *None* if too narrow / minimal theme.

    Returns a ``rich.text.Text`` object with amber flame and gray logs.
    The art is ~100 chars wide, so the terminal must be at least 100 columns.
    """
    if not theme.show_splash_art:
        return None

    if console.width < 100:
        return None

    flame_style = Style(color=AMBER)
    log_style = Style(color=LOG_BROWN)

    text = Text()
    for line in _BRAND_MARK_FLAME:
        text.append(line + "\n", style=flame_style)
    for line in _BRAND_MARK_LOGS:
        text.append(line + "\n", style=log_style)

    return text


# -------------------------------------------------------------------
# Rotating daily prompt examples
# -------------------------------------------------------------------

_DAILY_PROMPTS = [
    "What frameworks do guests recommend for prioritization?",
    "What did Brian Chesky say about founder mode?",
    "Which guests disagree on when to hire your first PM?",
    "What's the best advice about running user interviews?",
    "How do top PMs think about saying no to feature requests?",
    "What do guests say about the difference between growth and product?",
    "Find the quote about 'disagree and commit'",
    "What patterns emerge in how guests think about strategy?",
    "What do founders say about finding product-market fit?",
    "How do guests recommend structuring a product team?",
    "What advice do guests give about career transitions into product?",
    "What do guests think makes a great product leader?",
]


def _pick_daily_prompt() -> str:
    """Pick a deterministic daily prompt.  Changes once per calendar day."""
    day_index = date.today().toordinal() % len(_DAILY_PROMPTS)
    return _DAILY_PROMPTS[day_index]


# -------------------------------------------------------------------
# Startup splash card
# -------------------------------------------------------------------

def build_splash_card(
    episode_count: int,
    active_mode: str,
    theme: LennyTheme,
    console: Console,
) -> Text:
    """Build the full startup splash card as a ``rich.text.Text`` object.

    Layout adapts to terminal width and theme (art / daily prompt may
    be suppressed).
    """
    text = Text()

    # Brand mark (warm theme only, wide enough terminals)
    art = build_splash_art(theme, console)
    if art is not None:
        text.append("\n")
        text.append_text(art)
        text.append("\n")

    # App name + tagline
    text.append("  Lenny\n", style=Style(color=AMBER, bold=True))
    text.append("  Podcast transcript explorer\n\n", style=Style(color=CREAM_DIM))

    # Status line
    text.append(f"  {episode_count} episodes loaded", style=Style(color=CREAM))
    text.append("  \u00b7  ", style=Style(color=GRAY_DARK))
    text.append(f"mode: {active_mode}\n", style=Style(color=CREAM_DIM))

    # Daily prompt example (warm theme only)
    if theme.show_daily_prompt:
        prompt_example = _pick_daily_prompt()
        text.append(f'\n  Try: "{prompt_example}"\n', style=Style(color=GRAY_MUTED))

    # Command bar
    text.append(
        "\n  /help  /episodes  /cost  /mode  /theme  /quit\n",
        style=Style(color=GRAY_DARK),
    )

    return text


# -------------------------------------------------------------------
# Route badge rendering
# -------------------------------------------------------------------

def format_route_badge(mode: str, reason: str, theme: LennyTheme) -> Text:
    """Render the route badge displayed before an answer.

    Uses the theme's body text color (``theme.text``) for the label so
    it's readable on dark terminals without a colored background.

    Example output::

        FAST  specific guest lookup
        RESEARCH  cross-episode synthesis
    """
    text = Text("  ")
    if mode == "fast":
        text.append("FAST", style=Style(color=theme.text, bold=True))
    else:
        text.append("RESEARCH", style=Style(color=theme.text, bold=True))
    text.append(f"  {reason}", style=Style(color=GRAY_MUTED))
    return text


# -------------------------------------------------------------------
# Answer panel formatting
# -------------------------------------------------------------------

def answer_panel_params(mode: str, theme: LennyTheme) -> dict:
    """Return ``**kwargs`` for ``Panel()`` to render an answer card.

    Usage::

        console.print(Panel(content, **answer_panel_params("fast", theme)))
    """
    if mode == "fast":
        return {
            "title": f"[{theme.accent}]Lenny[/{theme.accent}] [{theme.text_faint}]fast[/{theme.text_faint}]",
            "border_style": theme.border_fast,
            "padding": (1, 2),
        }
    return {
        "title": f"[{theme.accent}]Lenny[/{theme.accent}] [{theme.text_faint}]research[/{theme.text_faint}]",
        "border_style": theme.border_research,
        "padding": (1, 2),
    }


# -------------------------------------------------------------------
# Cost display
# -------------------------------------------------------------------

def format_cost_compact(cost_str: str) -> Text:
    """Wrap the cost string in muted styling for inline display."""
    return Text(cost_str, style=Style(color=GRAY_MUTED))


# -------------------------------------------------------------------
# Save confirmation
# -------------------------------------------------------------------

def format_save_confirmation(filename: str) -> Text:
    """Render the save confirmation line."""
    text = Text("  ")
    text.append("\u2713", style=Style(color=SUCCESS_COLOR))
    text.append(f" Saved: {filename}", style=Style(color=GRAY_MUTED))
    return text


# -------------------------------------------------------------------
# Help text (uses custom theme tags)
# -------------------------------------------------------------------

HELP_TEXT = """\
[accent]Commands[/accent]
  /help      Show this help message
  /episodes  List loaded episodes (count + sample)
  /cost      Show session token usage and cost
  /mode      Show or set routing mode (auto, fast, research)
  /theme     Switch visual theme (warm, minimal)
  /verbose   Toggle verbose mode (see research orchestration)
  /quit      Exit

[accent]Routing modes[/accent]
  /mode auto      Automatic routing based on query (default)
  /mode fast      Force fast path for all queries
  /mode research  Force research path for all queries

[accent]Example queries[/accent]
  What did Brian Chesky say about founder mode?          [faint]fast[/faint]
  What frameworks do guests recommend for prioritization? [faint]research[/faint]
  Which guests disagree with each other on hiring?        [faint]research[/faint]
  Find the quote about 'disagree and commit'              [faint]fast[/faint]
"""


# -------------------------------------------------------------------
# Goodbye
# -------------------------------------------------------------------

GOODBYE_TEXT = "Until next time."


# -------------------------------------------------------------------
# Progress / status microcopy
# -------------------------------------------------------------------

PROGRESS_LABELS: dict[str, str] = {
    # Code-inferred statuses
    "reading_transcripts":   "Pulling relevant transcripts...",
    "analyzing_parallel":    "Analyzing excerpts in parallel...",
    "analyzing_ai":          "Weighing the evidence...",
    "searching_text":        "Searching transcript text...",
    "scanning_catalog":      "Scanning episode catalog...",

    # Lifecycle statuses
    "preparing_answer":      "Synthesizing final answer...",
    "thinking":              "Thinking...",

    # Count-based (use .format(n=..., s=...))
    "analyzing_excerpts":    "Analyzing {n} excerpt{s}...",

    # Fast-path specific
    "searching_fast":        "Looking across episodes...",

    # Initial statuses
    "searching_episodes":    "Searching {n} episodes...",
    "searching_transcripts": "Searching transcripts...",
}
