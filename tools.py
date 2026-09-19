"""Things the assistant can actually do, as opposed to talk about.

Design rules, because this is a voice-driven agent with shell access nearby:

* **Allowlist, not shell.** Launching resolves against an index of software that
  is already installed -- Steam games and Start Menu entries -- so a garbled
  transcript can at worst start the wrong program, never invent a command.
* **Shell is opt-in.** `run_command` stays disabled unless the config turns it
  on, because "open steam" and "format the disk" are one speech-recognition
  slip apart.
* **Everything is logged**, so there is a record of what was run and why.
"""

from __future__ import annotations

import difflib
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from html import unescape
from pathlib import Path
from urllib.parse import quote_plus

log = logging.getLogger("g2-bridge.tools")

IS_WINDOWS = platform.system() == "Windows"


# --------------------------------------------------------------------------
# Discovering what's installed
# --------------------------------------------------------------------------


def _steam_root() -> Path | None:
    if not IS_WINDOWS:
        mac = Path.home() / "Library/Application Support/Steam"
        linux = Path.home() / ".steam/steam"
        for candidate in (mac, linux):
            if candidate.is_dir():
                return candidate
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            return Path(winreg.QueryValueEx(key, "SteamPath")[0])
    except Exception:
        return None


def _steam_libraries(root: Path) -> list[Path]:
    """Steam spreads games over several drives; the vdf lists them."""
    libraries = [root]
    vdf = root / "steamapps" / "libraryfolders.vdf"
    if vdf.exists():
        try:
            text = vdf.read_text(encoding="utf-8", errors="ignore")
            for match in re.finditer(r'"path"\s+"([^"]+)"', text):
                path = Path(match.group(1).replace("\\\\", "\\"))
                if path.is_dir():
                    libraries.append(path)
        except OSError:
            pass
    return libraries


def steam_games() -> dict[str, str]:
    """Installed Steam games as {name: steam://rungameid/<appid>}."""
    root = _steam_root()
    if not root:
        return {}

    found: dict[str, str] = {}
    for library in _steam_libraries(root):
        steamapps = library / "steamapps"
        if not steamapps.is_dir():
            continue
        for manifest in steamapps.glob("appmanifest_*.acf"):
            try:
                text = manifest.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            appid = re.search(r'"appid"\s+"(\d+)"', text)
            name = re.search(r'"name"\s+"([^"]+)"', text)
            if appid and name:
                found[name.group(1).strip()] = f"steam://rungameid/{appid.group(1)}"
    return found


def start_menu_apps() -> dict[str, str]:
    """Start Menu shortcuts as {name: path to .lnk}.

    This is what makes the feature work with no configuration: anything the
    user could launch from the Start Menu, the assistant can launch too.
    """
    if not IS_WINDOWS:
        return {}

    roots = [
        Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
        Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
    ]
    found: dict[str, str] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for link in root.rglob("*.lnk"):
            name = link.stem.strip()
            # Installers and uninstallers are never what someone means.
            if re.search(r"\b(uninstall|readme|help|setup|repair)\b", name, re.I):
                continue
            found.setdefault(name, str(link))
    return found


class AppIndex:
    """Everything launchable, refreshed on demand."""

    def __init__(self, extra: dict[str, str] | None = None) -> None:
        self.extra = extra or {}
        self.entries: dict[str, str] = {}
        self.refresh()

    def refresh(self) -> None:
        entries: dict[str, str] = {}
        entries.update(start_menu_apps())
        # Games and explicit config win over Start Menu clutter.
        entries.update(steam_games())
        entries.update(self.extra)
        self.entries = entries
        log.info("App index: %d launchable entries", len(entries))

    def resolve(self, query: str) -> tuple[str, str] | None:
        """Best match for a spoken name, or None.

        Speech recognition mangles product names constantly ("counter strike
        two", "steam dot exe"), so exact matching alone would fail most of the
        time. Substring beats fuzzy, and fuzzy is a last resort.
        """
        if not query.strip():
            return None
        names = list(self.entries)
        wanted = _normalize(query)

        for name in names:
            if _normalize(name) == wanted:
                return name, self.entries[name]

        # Prefer the shortest containing match: "Counter-Strike 2" over
        # "Counter-Strike 2 Workshop Tools".
        contains = [n for n in names if wanted in _normalize(n) or _normalize(n) in wanted]
        if contains:
            best = min(contains, key=lambda n: len(n))
            return best, self.entries[best]

        close = difflib.get_close_matches(wanted, [_normalize(n) for n in names], n=1, cutoff=0.72)
        if close:
            for name in names:
                if _normalize(name) == close[0]:
                    return name, self.entries[name]
        return None


def _normalize(text: str) -> str:
    """Folds the differences speech recognition introduces."""
    text = text.lower().replace("&", " and ")
    text = re.sub(r"\b(\d+)\b", lambda m: m.group(1), text)
    for word, digit in (("two", "2"), ("three", "3"), ("four", "4"), ("one", "1")):
        text = re.sub(rf"\b{word}\b", digit, text)
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


def launch(target: str) -> str:
    """Starts a URI, shortcut or executable, detached from this process."""
    if target.startswith(("steam://", "http://", "https://")):
        return open_url(target)

    path = Path(target)
    if IS_WINDOWS:
        os.startfile(str(path))  # noqa: S606 - resolved from the installed-app index
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])
    return f"Launched {path.stem}"


def open_url(url: str) -> str:
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.I):
        url = "https://" + url
    if IS_WINDOWS:
        os.startfile(url)  # noqa: S606
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", url])
    else:
        subprocess.Popen(["xdg-open", url])
    return f"Opened {url}"


_TAG = re.compile(r"<[^>]+>")


def web_search(query: str, limit: int = 5) -> str:
    """Top results from DuckDuckGo's HTML endpoint -- no API key needed."""
    import urllib.request

    # A browser User-Agent is load-bearing: given a bot-looking one, DuckDuckGo
    # serves a JavaScript page with no results markup in it at all.
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        html = response.read().decode("utf-8", errors="ignore")

    results: list[str] = []
    for block in re.finditer(
        r'result__a[^>]*>(?P<title>.*?)</a>.*?result__snippet[^>]*>(?P<snippet>.*?)</a>',
        html,
        re.S,
    ):
        title = unescape(_TAG.sub("", block.group("title"))).strip()
        snippet = unescape(_TAG.sub("", block.group("snippet"))).strip()
        if title:
            results.append(f"{title}: {snippet}")
        if len(results) >= limit:
            break

    if not results:
        return "No results found."
    return "\n".join(results)


DEFAULT_NEWS_FEEDS = [
    ("BBC", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("NPR", "https://feeds.npr.org/1001/rss.xml"),
    ("Guardian", "https://www.theguardian.com/world/rss"),
]


def news_headlines(feeds: list | None = None, topic: str = "", limit: int = 8) -> str:
    """Current headlines from RSS.

    A plain web search cannot answer "what's the news": searching for it returns
    the *homepages* of news sites, so the summary comes back as "check CNN or
    Fox" -- which is exactly the unhelpful answer this replaces. RSS gives real
    headlines with summaries, no API key, no scraping.
    """
    import urllib.request
    import xml.etree.ElementTree as ET

    sources = feeds or DEFAULT_NEWS_FEEDS
    collected: list[list[str]] = []

    for entry in sources:
        name, url = (entry if isinstance(entry, (list, tuple)) else (entry, entry))
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=10) as response:
                root = ET.fromstring(response.read())
        except Exception as exc:
            log.warning("news feed %s failed: %s", name, exc)
            continue

        items: list[str] = []
        for item in root.findall(".//item")[:limit]:
            title = (item.findtext("title") or "").strip()
            summary = unescape(_TAG.sub("", item.findtext("description") or "")).strip()
            if not title:
                continue
            line = f"[{name}] {title}"
            if summary:
                line += f" - {summary[:200]}"
            items.append(line)
        if items:
            collected.append(items)

    if not collected:
        return "Could not reach any news feed."

    # Interleave sources so one outlet doesn't fill the whole list.
    merged: list[str] = []
    for row in range(max(len(items) for items in collected)):
        for items in collected:
            if row < len(items):
                merged.append(items[row])

    if topic:
        wanted = topic.lower()
        matches = [line for line in merged if wanted in line.lower()]
        if matches:
            merged = matches
        else:
            return f"Nothing in today's headlines about {topic}."

    return "\n".join(merged[:limit])


def run_command(command: str, timeout: int = 30) -> str:
    """Raw shell. Only reachable when the config explicitly enables it."""
    log.warning("Running shell command: %s", command)
    shell = shutil.which("powershell") if IS_WINDOWS else None
    args = [shell, "-NoProfile", "-Command", command] if shell else command
    try:
        completed = subprocess.run(
            args,
            shell=not shell,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s."
    output = (completed.stdout or "") + (completed.stderr or "")
    output = output.strip() or f"(no output, exit code {completed.returncode})"
    return output[:2000]


# --------------------------------------------------------------------------
# Schemas handed to the model
# --------------------------------------------------------------------------


def schemas(allow_shell: bool) -> list[dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "launch_app",
                "description": (
                    "Launch an installed application or Steam game on the user's computer. "
                    "Use for requests like 'open Steam', 'launch Counter-Strike 2', 'start Spotify'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the app or game, as the user said it"}
                    },
                    "required": ["name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_url",
                "description": "Open a web page in the user's default browser.",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string", "description": "The URL to open"}},
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Search the web and read the top results. Use for anything current, factual or "
                    "outside your knowledge, such as news, prices, scores or release dates."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The search query"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_news",
                "description": (
                    "Read today's news headlines. Use for any request about the news, headlines, "
                    "current events or what is happening today. Never tell the user to check a "
                    "news website instead."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": "Optional subject to filter headlines by, e.g. 'technology'",
                        }
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_apps",
                "description": "List installed apps and games matching a word, to check what is available.",
                "parameters": {
                    "type": "object",
                    "properties": {"contains": {"type": "string", "description": "Word to filter on"}},
                    "required": [],
                },
            },
        },
    ]
    if allow_shell:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": (
                        "Run a PowerShell/shell command on the user's computer and return its output. "
                        "Use only when no other tool fits."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string", "description": "The command to run"}},
                        "required": ["command"],
                    },
                },
            }
        )
    return tools
