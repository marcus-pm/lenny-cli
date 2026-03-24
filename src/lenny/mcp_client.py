"""MCP client for Lenny's Data archive server.

Wraps the four MCP tools (list_content, search_content, read_content,
read_excerpt) via JSON-RPC over Streamable HTTP transport.
"""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from typing import Any

MCP_ENDPOINT = "https://mcp.lennysdata.com/mcp"
MCP_PROTOCOL_VERSION = "2024-11-05"

# Timeouts (seconds)
_TIMEOUT_DEFAULT = 10
_TIMEOUT_READ_CONTENT = 30


class MCPError(Exception):
    """Raised when an MCP server call fails."""


class MCPClient:
    """HTTP client for the Lenny's Data MCP server.

    Manages session lifecycle (initialize → Mcp-Session-Id → tool calls)
    and strips the JSON-RPC + SSE envelope to return plain Python dicts.
    """

    def __init__(self, token: str | None = None, endpoint: str = MCP_ENDPOINT):
        self.token = token or os.environ.get("LENNY_MCP_TOKEN", "")
        self.endpoint = endpoint
        self._session_id: str | None = None
        self._request_id = 0

    # ------------------------------------------------------------------
    # Public API — mirrors the four MCP tools
    # ------------------------------------------------------------------

    def list_content(
        self,
        content_type: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """List available newsletter posts and podcast transcripts."""
        return self._call_tool(
            "list_content",
            {"content_type": content_type, "limit": limit, "offset": offset},
        )

    def search_content(
        self,
        query: str,
        content_type: str = "",
        limit: int = 20,
    ) -> dict:
        """Search across the archive for topics or keywords."""
        return self._call_tool(
            "search_content",
            {"query": query, "content_type": content_type, "limit": limit},
        )

    def read_content(self, filename: str) -> str:
        """Read the full markdown content of a specific post or transcript."""
        result = self._call_tool(
            "read_content",
            {"filename": filename},
            timeout=_TIMEOUT_READ_CONTENT,
        )
        # read_content returns raw text, not JSON
        if isinstance(result, str):
            return result
        return result.get("text", str(result))

    def read_excerpt(
        self,
        filename: str,
        query: str = "",
        match_index: int = 0,
        radius: int = 280,
    ) -> dict:
        """Read a focused excerpt from a specific file."""
        return self._call_tool(
            "read_excerpt",
            {
                "filename": filename,
                "query": query,
                "match_index": match_index,
                "radius": radius,
            },
        )

    def health_check(self) -> bool:
        """Test connectivity to the MCP server. Returns True if reachable."""
        try:
            self._ensure_session()
            return True
        except MCPError:
            return False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _ensure_session(self) -> None:
        """Initialize an MCP session if we don't have one."""
        if self._session_id is not None:
            return
        self._initialize()

    def _initialize(self) -> None:
        """Send the MCP initialize handshake and capture the session ID."""
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "lenny-cli", "version": "1.0"},
            },
        }
        req = self._build_request(payload, include_session=False)
        try:
            resp = urllib.request.urlopen(req, timeout=_TIMEOUT_DEFAULT)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            raise MCPError(f"Failed to connect to MCP server: {e}") from e

        # Extract session ID from response headers
        session_id = resp.headers.get("Mcp-Session-Id")
        if not session_id:
            # Try lowercase
            session_id = resp.headers.get("mcp-session-id")
        if not session_id:
            raise MCPError("MCP server did not return a session ID")

        self._session_id = session_id

    # ------------------------------------------------------------------
    # Tool call transport
    # ------------------------------------------------------------------

    def _call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: int = _TIMEOUT_DEFAULT,
    ) -> Any:
        """Call an MCP tool and return the parsed result.

        Handles session initialization, SSE envelope stripping, and
        auto-reinitializes the session on expiry.
        """
        self._ensure_session()

        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        }

        for attempt in range(2):  # retry once on session expiry
            req = self._build_request(payload, include_session=True)
            try:
                resp = urllib.request.urlopen(req, timeout=timeout)
                body = resp.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                err_body = ""
                try:
                    err_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                # Session expired — reinitialize and retry
                if e.code == 400 and "session" in err_body.lower() and attempt == 0:
                    self._session_id = None
                    self._initialize()
                    continue
                raise MCPError(
                    f"MCP tool '{tool_name}' failed (HTTP {e.code}): {err_body}"
                ) from e
            except (urllib.error.URLError, OSError) as e:
                raise MCPError(
                    f"MCP tool '{tool_name}' network error: {e}"
                ) from e

            return self._parse_sse_response(body, tool_name)

        raise MCPError(f"MCP tool '{tool_name}' failed after session reinit")

    def _parse_sse_response(self, body: str, tool_name: str) -> Any:
        """Parse an SSE-wrapped JSON-RPC response.

        The MCP Streamable HTTP transport wraps responses as:
            event: message
            data: {"jsonrpc": "2.0", "id": ..., "result": {"content": [...]}}
        """
        # Strip SSE envelope — find the JSON data line
        json_str = body
        for line in body.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                json_str = line[6:]
                break

        try:
            envelope = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise MCPError(
                f"MCP tool '{tool_name}' returned invalid JSON: {e}"
            ) from e

        # Check for JSON-RPC error
        if "error" in envelope:
            err = envelope["error"]
            raise MCPError(
                f"MCP tool '{tool_name}' error: {err.get('message', err)}"
            )

        result = envelope.get("result", {})

        # Extract the content text from the MCP response envelope
        content_list = result.get("content", [])
        if content_list and isinstance(content_list, list):
            text = content_list[0].get("text", "")
            # Try to parse as JSON (most tools return JSON strings)
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text

        # Fallback: try structuredContent
        structured = result.get("structuredContent", {})
        if structured:
            raw = structured.get("result", "")
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return raw

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_request(
        self, payload: dict, include_session: bool,
    ) -> urllib.request.Request:
        """Build an HTTP request with the correct headers."""
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if include_session and self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return urllib.request.Request(
            self.endpoint, data=data, headers=headers, method="POST",
        )

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id
