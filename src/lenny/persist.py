"""Response persistence and citation formatting utilities.

Saves each query response as a timestamped Markdown file and transforms
Markdown-style citations into terminal-friendly plain-URL format.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

# Words stripped from query when building the filename slug
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "has", "have", "had", "i", "me", "my", "you",
    "your", "we", "our", "they", "their", "it", "its", "this", "that",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "about", "all", "any", "can", "could", "would", "should", "will",
    "just", "not", "no", "so", "if", "then", "than", "too", "very",
    "some", "most", "more", "much", "many", "each", "every",
    "tell", "say", "said", "think", "know",
})

_MAX_SLUG_LEN = 48
_MIN_SLUG_TOKENS = 2
_MAX_SLUG_TOKENS = 5


def build_query_slug(query: str) -> str:
    """Derive a short, filesystem-safe slug from a user query.

    Rules:
    - 2-5 informative tokens, lowercase, hyphen-separated
    - Stopwords and filler removed
    - Sanitised to ``[a-z0-9-]``, repeated hyphens collapsed
    - Max 48 characters
    - Deterministic for the same input
    - Falls back to ``"response"`` if no valid tokens remain
    """
    # Lowercase and strip punctuation (keep alphanumeric + spaces)
    clean = re.sub(r"[^a-z0-9\s]", " ", query.lower())
    tokens = clean.split()

    # Remove stopwords, keep informative tokens
    informative = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    if len(informative) < _MIN_SLUG_TOKENS:
        # Fall back: use all non-trivial tokens
        informative = [t for t in tokens if len(t) > 1]

    if not informative:
        return "response"

    # Take up to _MAX_SLUG_TOKENS tokens
    selected = informative[:_MAX_SLUG_TOKENS]
    slug = "-".join(selected)

    # Collapse repeated hyphens, strip leading/trailing hyphens
    slug = re.sub(r"-{2,}", "-", slug).strip("-")

    # Truncate to max length (break at hyphen boundary if possible)
    if len(slug) > _MAX_SLUG_LEN:
        truncated = slug[:_MAX_SLUG_LEN]
        last_hyphen = truncated.rfind("-")
        if last_hyphen > 10:
            slug = truncated[:last_hyphen]
        else:
            slug = truncated.rstrip("-")

    return slug or "response"


def _resolve_collision(path: Path) -> Path:
    """Append -1, -2, ... before .md if the file already exists."""
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}-{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_response_markdown(
    *,
    query: str,
    answer: str,
    mode: str,
    cost_summary: str,
    output_dir: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Write the query response to a timestamped Markdown file.

    Parameters
    ----------
    query:
        The original user question.
    answer:
        Full answer body (Markdown).
    mode:
        ``"fast"`` or ``"research"``.
    cost_summary:
        Pre-formatted cost string (from ``format_query_cost``).
    output_dir:
        Directory to write to; defaults to ``Path.cwd()``.
    now:
        Override timestamp for testing; defaults to ``datetime.now()``.

    Returns
    -------
    Path
        The file that was written.
    """
    if now is None:
        now = datetime.now()
    if output_dir is None:
        output_dir = Path.cwd()

    timestamp = now.strftime("%Y%m%d-%H%M%S")
    slug = build_query_slug(query)
    filename = f"{timestamp}-{slug}.md"
    path = _resolve_collision(output_dir / filename)

    # Build file content
    header = (
        f"---\n"
        f"timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"query: \"{query}\"\n"
        f"route: {mode}\n"
        f"cost: |\n"
    )
    # Indent each cost line under the YAML block scalar
    for line in cost_summary.strip().splitlines():
        header += f"  {line}\n"
    header += "---\n\n"

    content = header + answer + "\n"

    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Citation formatting for terminal display
# ---------------------------------------------------------------------------

# Matches Markdown links: [text](url)
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\((https?://[^)]+)\)")


def format_terminal_citations(answer: str) -> str:
    """Transform Markdown links into terminal-friendly plain-URL citations.

    Converts patterns like:
        **Guest Name** in *Episode Title* ([link](https://youtube.com/...))
    Into:
        **Guest Name** in *Episode Title*
        https://youtube.com/...

    Also handles inline ``[text](url)`` by appending the URL as visible text:
        text: url

    The goal is to ensure every URL is visible as plain text in the terminal,
    making it clickable in iTerm/macOS Terminal (which auto-detect URLs).
    """
    lines = answer.split("\n")
    result: list[str] = []

    for line in lines:
        # Check for the specific citation pattern:
        #   (**Guest** in *Title* ([link](url)))
        #   or: ([link](url))  at end of line
        citation_match = re.search(
            r"\(\[(?:link|Link|watch|Watch|YouTube|youtube|video|listen)\]\((https?://[^)]+)\)\)",
            line,
        )
        if citation_match:
            url = citation_match.group(1)
            # Remove the ([link](url)) part and append URL on same line
            cleaned = line[:citation_match.start()].rstrip()
            # Remove trailing colon or dash left behind
            cleaned = cleaned.rstrip(":").rstrip("-").rstrip()
            result.append(f"{cleaned}")
            result.append(f"  {url}")
            continue

        # General case: replace [text](url) with text (url)
        if _MD_LINK_RE.search(line):
            transformed = _MD_LINK_RE.sub(r"\1 (\2)", line)
            result.append(transformed)
        else:
            result.append(line)

    return "\n".join(result)
