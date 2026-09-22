"""One service behind the G2 glasses app.

Serves the built app and proxies everything it needs, so the glasses talk to
exactly one host:

    G2 mic --PCM--> /stt      -> faster-whisper (VoiceFlow's engine + model)
    question ------> /api/ask -> Ollama (local)

Why a single front door: the app is served from here too, so every request is
same-origin and no CORS configuration is involved. It also means Ollama can
stay bound to 127.0.0.1 -- only this process is exposed to the tailnet, and it
is the only thing that needs a firewall path.

Run:  python g2_bridge.py
Config: bridge/config.json (see config.example.json)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import locale
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import captions
import numpy as np
import pc
import personal
import tools
import vad
import wake
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from aiohttp.client_exceptions import ClientConnectionResetError

log = logging.getLogger("g2-bridge")

HERE = Path(__file__).resolve().parent
DIST = HERE.parent / "dist"

# The G2 microphone emits PCM s16le, mono, at this rate -- the same rate
# VoiceFlow records at, so the audio needs no resampling before Whisper.
SAMPLE_RATE = 16000

# How often to push an interim transcript while the user is still speaking.
PARTIAL_INTERVAL_SECONDS = 1.2
# How often to check whether the wake phrase was said. Shorter, because this
# delay is what the user experiences as the app being slow to notice them.
WAKE_CHECK_INTERVAL_SECONDS = 0.7

DEFAULTS = {
    # Listen on everything by default: a fresh install has no idea whether the
    # phone will arrive over Tailscale or the LAN. Pin a single address in
    # config.json (a tailnet IP, say) to narrow it.
    "host": "0.0.0.0",
    "port": 8770,
    # Publish g2-bridge.local over mDNS so a phone on the same network can
    # reach this machine without anyone typing an IP address.
    "mdns": True,
    "ollama_url": "http://127.0.0.1:11434",
    "ollama_model": "qwen3:4b",
    # VoiceFlow's Whisper setup, reused as-is so this is the same STT the
    # user already dictates with rather than a second, divergent model.
    "whisper_model": "small.en",
    "whisper_device": "auto",
    # Downloaded on first run. Point this at an existing faster-whisper cache
    # to reuse models you already have.
    "whisper_models_dir": str(HERE / "models"),
    "language": "en",
    # Let the model act on the machine: launch apps, open pages, search the web.
    "tools": True,
    # RSS sources for get_news. Replace with whatever you actually read.
    "news_feeds": [
        ["BBC", "https://feeds.bbci.co.uk/news/world/rss.xml"],
        ["NPR", "https://feeds.npr.org/1001/rss.xml"],
        ["Guardian", "https://www.theguardian.com/world/rss"],
    ],
    # Raw shell. Off by default -- a misheard word should not be able to run an
    # arbitrary command.
    "allow_shell": False,
    # Ignore blips of stray noise that decode into hallucinated words.
    "min_utterance_seconds": 0.4,
    # Say this instead of tapping, while the app is open.
    "wake_phrase": "hey ollama",
    # With auto-send on, this much quiet after speaking sends the question.
    "end_silence_seconds": 1.0,
    # Model for "what's on my screen" and photo questions. Empty picks the first
    # installed model that reports vision support.
    "vision_model": "",
    # "imperial", "metric", or "auto" (imperial if this machine's locale is US).
    "units": "auto",
    # Where "remember that..." is kept. Stays on this machine.
    "memory_file": str(HERE / "memory.json"),
    # Tools to switch off, by name -- e.g. ["lock_computer", "read_clipboard"].
    "disabled_tools": [],
    # Weather location when the phone doesn't share one, e.g. "Seattle".
    "home_location": "",
}

SYSTEM_PROMPT = (
    "You are a heads-up display assistant. The user reads your replies on a small "
    "pair of glasses, so answer in at most a few short sentences. No markdown, no "
    "bullet points, no code blocks -- plain prose only."
)

TOOL_PROMPT = (
    " You can act on the user's computer with the tools provided. Use launch_app to "
    "open programs and games, close_app to close them, open_url for web pages, "
    "control_media for music and volume, look_at_screen to see the screen, "
    "read_clipboard for what they copied, remember for facts they want kept, "
    "set_timer for timers and reminders, get_weather for weather, get_news for "
    "headlines and current events, and web_search for anything else you are unsure about. "
    "Never tell the user to go and look somewhere themselves -- do not suggest "
    "visiting a news site, searching Google, or checking a website. If you do not "
    "know something, call a tool and find out. If live information is provided to "
    "you below, summarise it directly. After a tool runs, say what happened in one "
    "or two short sentences."
)

# Questions whose answer changes daily. A 4B model will happily answer these
# from stale training data, so the lookup is forced rather than left to its
# judgement -- it decides how to summarise, not whether to check.
NEWS_RE = re.compile(r"\b(news|headlines|current events)\b", re.I)
FRESH_RE = re.compile(
    r"\b(today|tonight|latest|recent|right now|currently|this (week|morning|afternoon)"
    r"|score|stock|price of)\b",
    re.I,
)
# ...unless it's plainly an instruction to do something local.
ACTION_RE = re.compile(
    r"^\s*(please\s+)?(launch|open|start|run|play|close|quit|pause|skip|turn|set|cancel|stop"
    r"|lock|remember|forget|remind|copy)\b",
    re.I,
)
WEATHER_RE = re.compile(r"\b(weather|forecast|temperature outside|rain(ing)?|snow(ing)?|umbrella|jacket)\b", re.I)
# Questions about what is in front of them on the computer. Seeing is forced for
# the same reason as the news: a small model will guess rather than look.
SCREEN_RE = re.compile(r"\b(my|the|this|on) (screen|monitor|display)\b|\bscreen ?shot\b", re.I)


def load_config(path: Path) -> dict:
    cfg = dict(DEFAULTS)
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    return cfg


# --------------------------------------------------------------------------
# Whisper
# --------------------------------------------------------------------------


def register_cuda_dlls() -> list[str]:
    """Puts the pip-installed CUDA libraries on the DLL search path.

    nvidia-cublas-cu12 and friends drop their DLLs under
    site-packages/nvidia/*/bin, which Windows does not search. Without this,
    CTranslate2 reports "cublas64_12.dll is not found or cannot be loaded" and
    faster-whisper silently falls back to CPU even though the GPU is fine.

    PATH is what actually matters here: CTranslate2 resolves cuBLAS with a
    plain LoadLibrary, which consults PATH but ignores the add_dll_directory
    list (that only applies to loads using LOAD_LIBRARY_SEARCH_USER_DIRS).
    add_dll_directory is kept as well for anything that does honour it.

    Must run before anything imports ctranslate2 -- the DLLs resolve at import.
    """
    import site

    roots = [Path(p) / "nvidia" for p in site.getsitepackages()]
    roots.append(Path(sys.prefix) / "Lib" / "site-packages" / "nvidia")

    added: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for bin_dir in sorted(root.glob("*/bin")):
            if str(bin_dir) in added:
                continue
            added.append(str(bin_dir))
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(bin_dir))
                except OSError:
                    pass

    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
    return added


class Transcriber:
    """Wraps VoiceFlow's faster-whisper model. Loaded once, used from threads."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.model = None
        self._lock = asyncio.Lock()

    def load(self) -> None:
        added = register_cuda_dlls()
        log.info("CUDA DLL dirs registered: %d", len(added))
        for path in added:
            log.debug("  %s", path)

        from faster_whisper import WhisperModel

        name = self.cfg["whisper_model"]
        root = self.cfg["whisper_models_dir"]
        device = self.cfg["whisper_device"]

        attempts = []
        if device in ("auto", "cuda"):
            attempts.append(("cuda", "float16"))
        if device in ("auto", "cpu"):
            attempts.append(("cpu", "int8"))

        for dev, compute in attempts:
            try:
                log.info("Loading Whisper %s on %s (%s)...", name, dev, compute)
                model = WhisperModel(name, device=dev, compute_type=compute, download_root=root)
                # Force a real decode now so a broken CUDA install fails here,
                # at startup, instead of on the user's first spoken question.
                list(model.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))[0])
                self.model = model
                log.info("Whisper ready on %s.", dev)
                return
            except Exception as exc:
                log.warning("Could not use %s: %s", dev, exc)

        raise RuntimeError("No usable device for the Whisper model.")

    @property
    def can_translate(self) -> bool:
        # The ".en" models only know English, so they have nothing to translate from.
        return not str(self.cfg["whisper_model"]).endswith(".en")

    def _transcribe_sync(self, pcm: bytes, task: str = "transcribe") -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if task == "translate" and self.can_translate:
            # Detect the spoken language rather than forcing the configured one.
            segments, _info = self.model.transcribe(audio, task="translate")
        else:
            segments, _info = self.model.transcribe(audio, language=self.cfg["language"])
        return "".join(seg.text for seg in segments).strip()

    async def transcribe(self, pcm: bytes, task: str = "transcribe") -> str:
        if self.model is None:
            raise RuntimeError("Whisper model is not loaded")
        seconds = len(pcm) / 2 / SAMPLE_RATE
        if seconds < self.cfg["min_utterance_seconds"]:
            return ""
        # One decode at a time: concurrent calls would contend for the same
        # CTranslate2 model and the GPU it sits on.
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, pcm, task)


# --------------------------------------------------------------------------
# Answer backends
# --------------------------------------------------------------------------


MAX_TOOL_ROUNDS = 4


def use_imperial(cfg: dict) -> bool:
    units = str(cfg.get("units", "auto")).lower()
    if units in ("imperial", "fahrenheit", "us"):
        return True
    if units in ("metric", "celsius"):
        return False
    name = (locale.getlocale()[0] or "").lower()
    return "united states" in name or name.startswith("en_us")


def build_system_prompt(app: web.Application, use_tools: bool) -> str:
    # A model has no clock. Without this, "what time is it" and "set a timer
    # for 5pm" are answered from nowhere.
    now = datetime.now().strftime("%A, %B %d, %Y, %I:%M %p").replace(" 0", " ")
    prompt = SYSTEM_PROMPT + f" It is currently {now}."
    if use_tools:
        prompt += TOOL_PROMPT
    return prompt + app["memory"].prompt_block()


async def find_vision_model(app: web.Application, preferred: str = "") -> str:
    """First installed model that can see, preferring the one in use."""
    cfg = app["cfg"]
    if cfg.get("vision_model"):
        return cfg["vision_model"]
    cached = app.get("vision_model_cache")
    if cached:
        return cached
    try:
        async with app["http"].get(f"{cfg['ollama_url']}/api/tags") as resp:
            names = [m["name"] for m in (await resp.json()).get("models", [])]
    except Exception:
        return ""
    ordered = ([preferred] if preferred in names else []) + [n for n in names if n != preferred]
    for name in ordered:
        try:
            async with app["http"].post(f"{cfg['ollama_url']}/api/show", json={"model": name}) as resp:
                caps = (await resp.json()).get("capabilities") or []
        except Exception:
            continue
        if "vision" in caps:
            app["vision_model_cache"] = name
            return name
    return ""


NO_VISION = (
    "No vision model is installed, so I can't see images yet. On the computer, run: "
    "ollama pull qwen2.5vl:3b"
)
VISION_PROMPT = (
    "You are looking at an image for someone wearing smart glasses who will read your "
    "answer on a tiny display. Answer their question in two or three short plain "
    "sentences. If there is text that matters, such as an error message, quote the "
    "important part."
)


async def ask_vision_stream(app: web.Application, question: str, image_b64: str, preferred: str = ""):
    """Streams a vision model's answer about one image."""
    model = await find_vision_model(app, preferred)
    if not model:
        yield NO_VISION
        return
    messages = [
        {"role": "system", "content": VISION_PROMPT},
        {"role": "user", "content": question or "What is this?", "images": [image_b64]},
    ]
    async for delta in ask_ollama_stream(app, messages, model):
        yield delta


async def _run_tool(app: web.Application, name: str, args: dict, ctx: dict | None = None) -> tuple[str, str]:
    """Executes one tool call. Returns (status for the lens, result for the model).

    Status lines start with a tag in brackets -- [launch], [search] -- which the
    app turns into an icon on the lens.
    """
    ctx = ctx or {}
    index = app["apps"]
    cfg = app["cfg"]

    def num(key: str) -> float | None:
        try:
            return float(args[key]) if args.get(key) not in (None, "") else None
        except (TypeError, ValueError):
            return None

    if name in set(cfg.get("disabled_tools") or []):
        # Checked here too, not just left out of the schema list: the forced
        # lookups call tools directly, and a model can name a tool it wasn't given.
        return (f"[error] {name} is turned off", f"The {name} tool is disabled on this computer.")

    if name == "control_media":
        action = str(args.get("action", "play_pause"))
        amount = num("amount")
        result = await asyncio.to_thread(pc.media, action, int(amount) if amount is not None else None)
        return (f"[media] {result}", result)

    if name == "close_app":
        wanted = str(args.get("name", ""))
        result = await asyncio.to_thread(pc.close_app, wanted)
        return (f"[close] {result}", result)

    if name == "list_windows":
        return ("[windows] Checking open windows", await asyncio.to_thread(pc.open_windows))

    if name == "lock_computer":
        return ("[lock] Locking the computer", await asyncio.to_thread(pc.lock_computer))

    if name == "look_at_screen":
        question = str(args.get("question", "")) or "What is on the screen?"
        try:
            image = await asyncio.to_thread(pc.screenshot_jpeg_b64)
        except Exception as exc:
            return ("[screen] Couldn't capture the screen", f"Screen capture failed: {exc}")
        answer = "".join([d async for d in ask_vision_stream(app, question, image)])
        return ("[screen] Looking at your screen", answer)

    if name == "read_clipboard":
        return ("[clipboard] Reading your clipboard", await asyncio.to_thread(pc.read_clipboard))

    if name == "copy_to_clipboard":
        result = await asyncio.to_thread(pc.write_clipboard, str(args.get("text", "")))
        return (f"[clipboard] {result}", result)

    if name == "system_status":
        return ("[status] Checking the computer", await asyncio.to_thread(pc.system_status))

    if name == "remember":
        result = app["memory"].add(str(args.get("fact", "")))
        return (f"[memory] {result}", result)

    if name == "forget":
        result = app["memory"].forget(str(args.get("about", "")))
        return (f"[memory] {result}", result)

    if name == "set_timer":
        seconds = (num("minutes") or 0) * 60 + (num("seconds") or 0)
        result = app["timers"].add(seconds or None, str(args.get("at") or "") or None, str(args.get("label") or ""))
        return (f"[timer] {result}", result)

    if name == "cancel_timer":
        result = app["timers"].cancel(str(args.get("label") or ""))
        return (f"[timer] {result}", result)

    if name == "list_timers":
        result = app["timers"].describe_all()
        return (f"[timer] {result}", result)

    if name == "get_weather":
        place = str(args.get("place") or "").strip()
        imperial = use_imperial(cfg)
        if place:
            hit = await asyncio.to_thread(personal.geocode, place)
            if not hit:
                return (f"[weather] Couldn't find {place}", f"No place called {place} was found.")
            lat, lon, label = hit
        elif ctx.get("location"):
            lat, lon, label = ctx["location"]["lat"], ctx["location"]["lon"], "your location"
        elif cfg.get("home_location"):
            hit = await asyncio.to_thread(personal.geocode, cfg["home_location"])
            if not hit:
                return ("[weather] Home location not found", "The configured home location was not found.")
            lat, lon, label = hit
        else:
            return (
                "[weather] Where are you?",
                "Location is unavailable. Ask the user which city, or tell them to allow location "
                "for the app, or to set home_location in the bridge config.",
            )
        result = await asyncio.to_thread(personal.weather, lat, lon, label, imperial)
        return (f"[weather] Weather for {label}", result)

    if name == "launch_app":
        wanted = str(args.get("name", ""))
        hit = index.resolve(wanted)
        if not hit:
            index.refresh()  # might have been installed since startup
            hit = index.resolve(wanted)
        if not hit:
            return (f"[launch] No match for {wanted}", f"No installed app or game matches '{wanted}'.")
        label, target = hit
        await asyncio.to_thread(tools.launch, target)
        return (f"[launch] Launching {label}", f"Launched {label}.")

    if name == "open_url":
        url = str(args.get("url", ""))
        result = await asyncio.to_thread(tools.open_url, url)
        return (f"[web] Opening {url}", result)

    if name == "web_search":
        query = str(args.get("query", ""))
        result = await asyncio.to_thread(tools.web_search, query)
        return (f"[search] Searching: {query}", result)

    if name == "get_news":
        topic = str(args.get("topic", ""))
        result = await asyncio.to_thread(
            tools.news_headlines, app["cfg"].get("news_feeds"), topic, 8
        )
        return (f"[news] Reading the news{' about ' + topic if topic else ''}", result)

    if name == "list_apps":
        contains = str(args.get("contains", "")).lower()
        names = [n for n in index.entries if contains in n.lower()] if contains else list(index.entries)
        names = sorted(names)[:30]
        return ("[launch] Checking installed apps", ", ".join(names) if names else "Nothing matches.")

    if name == "run_command":
        if not app["cfg"].get("allow_shell", False):
            return ("[shell] Shell disabled", "Shell access is disabled in this bridge's config.")
        command = str(args.get("command", ""))
        result = await asyncio.to_thread(tools.run_command, command)
        return (f"[shell] Running: {command[:40]}", result)

    return (f"[error] Unknown tool {name}", f"No such tool: {name}")


async def run_agent(app: web.Application, prompt: str, model: str, history: list, ctx: dict | None = None):
    """Answers a question, using tools when the model asks for them.

    Yields dicts: {'status': ...} for something happening on the machine, and
    {'delta': ...} for answer text. Tool rounds are non-streaming because a
    tool call has to arrive complete before it can run; the final answer is
    streamed as usual.
    """
    cfg = app["cfg"]
    use_tools = cfg.get("tools", True)
    disabled = set(cfg.get("disabled_tools") or [])
    schemas = (
        [s for s in tools.schemas(cfg.get("allow_shell", False)) if s["function"]["name"] not in disabled]
        if use_tools
        else None
    )

    messages = [
        {"role": "system", "content": build_system_prompt(app, use_tools)},
        *history,
        {"role": "user", "content": prompt},
    ]

    # "What's on my screen?" goes straight to the vision model and its answer is
    # the reply -- routing it back through the chat model would only add a
    # second model load and a paraphrase.
    if use_tools and SCREEN_RE.search(prompt) and not ACTION_RE.search(prompt):
        log.info("forced lookup: look_at_screen")
        yield {"status": "[screen] Looking at your screen"}
        try:
            image = await asyncio.to_thread(pc.screenshot_jpeg_b64)
        except Exception as exc:
            yield {"delta": f"I couldn't capture the screen: {exc}"}
            return
        async for delta in ask_vision_stream(app, prompt, image, model):
            yield {"delta": delta}
        return

    # Fetch live data up front for questions whose answer changed since the
    # model was trained. Left to its own judgement a small model answers these
    # from memory, or worse, tells the user to go and check a news site --
    # which is the one thing an assistant on your face should never do.
    if use_tools and not ACTION_RE.search(prompt):
        forced: tuple[str, dict] | None = None
        if WEATHER_RE.search(prompt):
            # Only when no place is named: "weather in Paris" needs the model to
            # pull out the city, so that one is left to a normal tool call.
            if not re.search(r"\b(in|at|for)\s+[A-Z]", prompt):
                forced = ("get_weather", {})
        elif NEWS_RE.search(prompt):
            forced = ("get_news", {})
        elif FRESH_RE.search(prompt):
            forced = ("web_search", {"query": prompt})

        if forced:
            name, args = forced
            log.info("forced lookup: %s(%s)", name, args)
            try:
                status, result = await _run_tool(app, name, args, ctx)
                yield {"status": status}
                # Injected as system context rather than a tool result: there is
                # no assistant tool_call for it to answer, and every model
                # understands a plain instruction.
                messages.insert(
                    -1,
                    {
                        "role": "system",
                        "content": (
                            "Live information retrieved just now. Use it to answer the "
                            f"user's question directly:\n{result}"
                        ),
                    },
                )
            except Exception:
                log.exception("forced lookup failed")

    for _ in range(MAX_TOOL_ROUNDS if use_tools else 1):
        if not use_tools:
            break

        payload = {"model": model or cfg["ollama_model"], "messages": messages, "stream": False, "tools": schemas}
        async with app["http"].post(f"{cfg['ollama_url']}/api/chat", json=payload) as resp:
            if resp.status != 200:
                raise web.HTTPBadGateway(text=f"Ollama HTTP {resp.status}: {(await resp.text())[:300]}")
            message = (await resp.json()).get("message", {})

        calls = message.get("tool_calls") or []
        if not calls:
            # Nothing to do on the machine -- fall through and stream a reply.
            break

        # Keep the assistant's tool-call turn, or the follow-up loses context.
        messages.append({"role": "assistant", "content": message.get("content", ""), "tool_calls": calls})

        for call in calls:
            function = call.get("function", {})
            name = function.get("name", "")
            args = function.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if ctx and ctx.get("cancelled", lambda: False)():
                # The user tapped cancel while the model was deciding. Launching
                # the game anyway would be the one unforgivable outcome.
                log.info("skipped %s: the client cancelled", name)
                return
            log.info("tool call: %s(%s)", name, args)
            try:
                status, result = await _run_tool(app, name, args, ctx)
            except Exception as exc:
                log.exception("tool %s failed", name)
                status, result = (f"[error] {name} failed", f"Error running {name}: {exc}")
            yield {"status": status}
            messages.append({"role": "tool", "name": name, "content": result})

    # Final pass: no tools, so the model has to answer rather than call again.
    async for delta in ask_ollama_stream(app, messages, model):
        yield {"delta": delta}


async def ask_ollama_stream(app: web.Application, messages: list, model: str):
    """Streams a reply for an already-built message list."""
    cfg = app["cfg"]
    payload = {"model": model or cfg["ollama_model"], "messages": messages, "stream": True}

    async with app["http"].post(f"{cfg['ollama_url']}/api/chat", json=payload) as resp:
        if resp.status != 200:
            raise web.HTTPBadGateway(text=f"Ollama HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise web.HTTPBadGateway(text=str(chunk["error"]))
            content = chunk.get("message", {}).get("content")
            if content:
                yield content


async def ask_ollama(app: web.Application, prompt: str, model: str, history: list):
    """Yields reply text as Ollama produces it."""
    cfg = app["cfg"]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, {"role": "user", "content": prompt}]
    payload = {"model": model or cfg["ollama_model"], "messages": messages, "stream": True}

    async with app["http"].post(f"{cfg['ollama_url']}/api/chat", json=payload) as resp:
        if resp.status != 200:
            raise web.HTTPBadGateway(text=f"Ollama HTTP {resp.status}: {(await resp.text())[:300]}")
        async for raw in resp.content:
            line = raw.strip()
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise web.HTTPBadGateway(text=str(chunk["error"]))
            content = chunk.get("message", {}).get("content")
            if content:
                yield content


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------


async def handle_health(request: web.Request) -> web.Response:
    app = request.app
    cfg = app["cfg"]
    ollama_ok = False
    try:
        async with app["http"].get(f"{cfg['ollama_url']}/api/tags") as resp:
            ollama_ok = resp.status == 200
    except Exception:
        pass
    return web.json_response(
        {
            "whisper": app["stt"].model is not None,
            "whisper_model": cfg["whisper_model"],
            "ollama": ollama_ok,
            "app_built": (DIST / "index.html").exists(),
            "vision_model": await find_vision_model(app) if ollama_ok else "",
            # Lets the app hide features an older bridge doesn't have.
            "features": ["tools", "wake", "auto_send", "captions", "timers", "memory", "vision", "weather"],
        }
    )


async def handle_inbox(request: web.Request) -> web.Response:
    """What the glasses should know about without being asked: timers that went
    off, and the ones still counting down. Polled by the app while it is open."""
    try:
        since = float(request.query.get("since", "0"))
    except ValueError:
        since = 0.0
    timers = request.app["timers"]
    now = time.time()
    return web.json_response(
        {
            "now": now,
            "notifications": timers.drain_inbox(since),
            "timers": [{"label": t.label, "remaining": round(t.due - now)} for t in timers.active()],
        }
    )


async def handle_models(request: web.Request) -> web.Response:
    cfg = request.app["cfg"]
    try:
        async with request.app["http"].get(f"{cfg['ollama_url']}/api/tags") as resp:
            body = await resp.json()
    except Exception as exc:
        raise web.HTTPBadGateway(text=f"Ollama unreachable at {cfg['ollama_url']}: {exc}")
    names = [m["name"] for m in body.get("models", [])]
    return web.json_response({"models": names, "default": cfg["ollama_model"]})


async def handle_ask(request: web.Request) -> web.StreamResponse:
    """Streams the answer back as newline-delimited JSON."""
    body = await request.json()
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        raise web.HTTPBadRequest(text="prompt is required")

    history = body.get("history") or []
    ctx: dict = {}
    location = body.get("location") or {}
    try:
        ctx["location"] = {"lat": float(location["lat"]), "lon": float(location["lon"])}
    except (KeyError, TypeError, ValueError):
        pass
    image = body.get("image") or ""
    # A tap-to-cancel closes the request; tools check this before acting.
    ctx["cancelled"] = lambda: request.transport is None or request.transport.is_closing()

    # CORS headers go on before prepare(): prepare() flushes them to the wire,
    # so the middleware would be too late to add them to a streamed response.
    resp = web.StreamResponse(
        headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-store", **CORS_HEADERS}
    )
    await resp.prepare(request)

    try:
        # run_agent yields {"delta": ...} for answer text and {"status": ...}
        # when it is doing something on the machine, so both go straight out.
        if image:
            # A photo from the phone's camera: the vision model answers directly.
            await resp.write(json.dumps({"status": "[photo] Looking at your photo"}).encode() + b"\n")
            async for delta in ask_vision_stream(request.app, prompt, image, body.get("model", "")):
                await resp.write(json.dumps({"delta": delta}).encode() + b"\n")
        else:
            async for event in run_agent(request.app, prompt, body.get("model", ""), history, ctx):
                await resp.write(json.dumps(event).encode() + b"\n")
        await resp.write(json.dumps({"done": True}).encode() + b"\n")
    except (ConnectionResetError, ClientConnectionResetError):
        # Cancelled on the glasses. Nothing to report to a client that left.
        log.info("ask cancelled by the client")
    except web.HTTPException as exc:
        await resp.write(json.dumps({"error": exc.text}).encode() + b"\n")
    except Exception as exc:
        log.exception("ask failed")
        try:
            await resp.write(json.dumps({"error": str(exc)}).encode() + b"\n")
        except (ConnectionResetError, ClientConnectionResetError):
            pass

    return resp


async def handle_stt_http(request: web.Request) -> web.Response:
    """Transcribes one complete utterance posted as raw PCM.

    The WebSocket path is nicer -- audio streams while you talk, so the decode
    starts the moment you stop. But the Even WebView does not let a plugin open
    a WebSocket to the bridge (the request never arrives), while plain HTTP goes
    through, so this is the transport that actually works on the glasses.

    Body is raw PCM s16le @ 16 kHz mono, exactly what the mic emits.
    """
    pcm = await request.read()
    seconds = len(pcm) / 2 / SAMPLE_RATE
    # Partials arrive every second or so while the user talks; logging each at
    # INFO would bury everything else.
    partial = request.query.get("partial") == "1"
    log.log(
        logging.DEBUG if partial else logging.INFO,
        "STT over HTTP%s: %.2fs of audio",
        " (partial)" if partial else "",
        seconds,
    )
    # Scored first: Whisper decodes silence into "You" or "Thank you.", so a
    # buffer with no voice in it is answered with nothing rather than decoded.
    had_speech, trailing = await asyncio.to_thread(vad.analyze, pcm)
    try:
        text = await request.app["stt"].transcribe(pcm) if had_speech else ""
    except Exception as exc:
        log.exception("transcription failed")
        raise web.HTTPInternalServerError(text=str(exc))
    body = {"text": text, "seconds": round(seconds, 2)}

    # Auto-send over HTTP: the client can't hear silence itself, so each partial
    # also says whether the speaker has finished. Stateless -- the client posts
    # the whole utterance every time -- so the full buffer is scored.
    if partial and request.query.get("auto") == "1":
        cfg = request.app["cfg"]
        body["ended"] = had_speech and trailing >= cfg["end_silence_seconds"]
        body["no_speech"] = not had_speech and seconds >= 8.0
    return web.json_response(body)


async def handle_stt(request: web.Request) -> web.WebSocketResponse:
    """Collects PCM from the glasses and transcribes on stop.

    Binary frames are raw PCM s16le @ 16 kHz mono. Text frames are control:
    {"type":"start"} clears the buffer, {"type":"stop"} triggers a decode.
    """
    ws = web.WebSocketResponse(heartbeat=30, max_msg_size=8 * 1024 * 1024)
    await ws.prepare(request)
    stt: Transcriber = request.app["stt"]
    loop = asyncio.get_running_loop()

    chunks: list[bytes] = []
    log.info("STT client connected")

    # Re-decode what's been said so far every so often and push it, so the lens
    # fills in while the user is still talking instead of staying blank until
    # they stop. The audio is already here, so this costs one extra decode --
    # about 0.1s on a GPU -- and nothing on the wire.
    partial_task: asyncio.Task | None = None
    last_partial = 0.0

    async def emit_partial(snapshot: bytes) -> None:
        try:
            text = await stt.transcribe(snapshot)
            if text and not ws.closed:
                await ws.send_json({"type": "partial", "text": text})
        except Exception:
            # Partials are cosmetic; a failure here must not kill the capture.
            log.debug("partial transcription failed", exc_info=True)

    # Tap mode. The endpointer always watches the stream -- it is how partials
    # and the final know whether anyone actually spoke -- but only ends the
    # question by itself when the client asked for auto-send.
    tap_active = False
    auto_stop = False
    endpointer: vad.Endpointer | None = None
    since_vad = 0.0

    async def finish_tap(reason: str | None = None) -> None:
        """Transcribes the tap-mode buffer. Used by 'stop' and by auto-send."""
        nonlocal chunks, tap_active, endpointer
        tap_active = False
        heard = endpointer is not None and endpointer.had_speech
        endpointer = None
        if partial_task and not partial_task.done():
            # A partial resolving after the final would overwrite a good
            # transcript with a worse guess.
            partial_task.cancel()
        pcm = b"".join(chunks)
        chunks = []
        seconds = len(pcm) / 2 / SAMPLE_RATE
        if reason:
            log.info("auto-send: %s after %.1fs", reason, seconds)
            await ws.send_json({"type": "auto_stop", "reason": reason})
        await ws.send_json({"type": "transcribing", "seconds": round(seconds, 2)})
        try:
            if not heard and reason != "no_speech":
                # The periodic check may simply not have run since they spoke.
                heard, _trailing = await asyncio.to_thread(vad.analyze, pcm)
            # Whisper decodes silence into "You" or "Thank you." Nothing is a
            # better answer to nothing.
            text = await stt.transcribe(pcm) if heard else ""
            await ws.send_json({"type": "final", "text": text, "seconds": round(seconds, 2)})
        except Exception as exc:
            log.exception("transcription failed")
            await ws.send_json({"type": "error", "error": str(exc)})

    # Set when the client asks to listen for a wake phrase instead of taps.
    session: wake.WakeSession | None = None
    last_decode = 0.0

    # Live captions. While on, the wake session is parked rather than dropped,
    # so turning captions off goes straight back to listening for the phrase.
    caption: captions.CaptionSession | None = None
    parked_session: wake.WakeSession | None = None
    caption_line_task: asyncio.Task | None = None
    last_caption_partial = 0.0

    async def decode_line(pcm: bytes, task: str) -> None:
        try:
            text = await stt.transcribe(pcm, task)
            if text and not ws.closed:
                await ws.send_json({"type": "caption", "text": text, "final": True})
        except Exception:
            log.exception("caption decode failed")

    async def handle_caption_frame(data: bytes) -> None:
        nonlocal caption_line_task, last_caption_partial
        step = caption.add(data)
        task = "translate" if caption.translate else "transcribe"
        if step["action"] == "line":
            # Decoded in the background so audio keeps flowing into the next
            # line; the transcriber's lock keeps the decodes in order.
            caption_line_task = asyncio.create_task(decode_line(step["pcm"], task))
            last_caption_partial = loop.time()
            return
        if step["action"] != "progress":
            return
        now = loop.time()
        busy = caption_line_task is not None and not caption_line_task.done()
        if busy or now - last_caption_partial < PARTIAL_INTERVAL_SECONDS:
            return
        last_caption_partial = now
        text = await stt.transcribe(step["pcm"], task)
        if text:
            await ws.send_json({"type": "caption", "text": text, "final": False})
    last_level_log = 0.0

    async def handle_wake_frame(data: bytes) -> None:
        """One frame of always-on audio. Decodes only when it's worth it."""
        nonlocal last_decode
        nonlocal last_level_log
        step = session.add(data)
        action = step["action"]

        # Periodic levels, so a wake phrase that never fires can be diagnosed
        # from the log instead of guessed at.
        if loop.time() - last_level_log >= 5.0:
            last_level_log = loop.time()
            gate = session.gate
            log.info(
                "mic levels: now=%.0f peak=%.0f floor=%.0f threshold=%.0f voice=%s",
                gate.last_level,
                gate.peak_level,
                gate.noise_floor,
                max(gate.noise_floor * 2.5, wake.ABSOLUTE_SILENCE_RMS),
                session.since_speech <= wake.WAKE_TAIL_SECONDS or session.capturing,
            )
            gate.peak_level = 0.0

        if action == "none":
            return

        if action == "finish":
            # Never rate-limit this one. `add` has already reset the session, so
            # dropping it here would throw away the whole question and leave the
            # user waiting for an answer that never comes.
            last_decode = loop.time()
            text = "" if step.get("reason") == "no_speech" else await stt.transcribe(step["pcm"])
            stripped = wake.find_wake(text, session.phrase)
            text = stripped if stripped is not None else text
            seconds = len(step["pcm"]) / 2 / SAMPLE_RATE
            await ws.send_json({"type": "final", "text": text, "seconds": round(seconds, 2)})
            return

        now = loop.time()
        # Every decode competes for the same GPU, so rate-limit regardless of
        # how fast frames arrive. Spotting the phrase gets a shorter interval
        # than interim transcripts: that latency is the whole feel of it.
        interval = WAKE_CHECK_INTERVAL_SECONDS if action == "check_wake" else PARTIAL_INTERVAL_SECONDS
        if now - last_decode < interval:
            return
        last_decode = now

        if action == "check_wake":
            heard = await stt.transcribe(step["pcm"])
            if not heard:
                return
            rest = wake.find_wake(heard, session.phrase)
            if rest is None:
                return
            log.info("wake phrase heard")
            session.start_capture()
            await ws.send_json({"type": "wake"})
            if rest:
                # They said the phrase and the question in one breath; show it
                # now rather than waiting for the next decode.
                await ws.send_json({"type": "partial", "text": rest})
            return

        text = await stt.transcribe(step["pcm"])
        # Strip the phrase if it's still in frame, so "hey ollama launch steam"
        # doesn't get asked as a question containing its own wake word.
        stripped = wake.find_wake(text, session.phrase)
        text = stripped if stripped is not None else text

        if text:
            await ws.send_json({"type": "partial", "text": text})

    async for msg in ws:
        if msg.type is WSMsgType.BINARY:
            if caption is not None:
                try:
                    await handle_caption_frame(msg.data)
                except Exception:
                    log.exception("caption handling failed")
                continue
            if session is not None:
                try:
                    await handle_wake_frame(msg.data)
                except Exception:
                    log.exception("wake handling failed")
                continue

            if not tap_active:
                # Audio still arriving after an auto-send, before the client
                # has closed the mic. Keeping it would start the next question
                # with the tail of this one.
                continue
            chunks.append(msg.data)
            if endpointer is not None:
                endpointer.feed(msg.data)
                since_vad += len(msg.data) / 2 / SAMPLE_RATE
                if since_vad >= wake.VAD_INTERVAL_SECONDS:
                    since_vad = 0.0
                    verdict = endpointer.check()
                    if auto_stop and verdict != "continue":
                        await finish_tap(verdict)
                        continue
            if endpointer is None or not endpointer.had_speech:
                # Nothing said yet; a partial now would only be a hallucination.
                continue
            now = loop.time()
            # Skip while one is still running: a slow decode must not queue up
            # work the user would never see.
            if now - last_partial >= PARTIAL_INTERVAL_SECONDS and (
                partial_task is None or partial_task.done()
            ):
                last_partial = now
                partial_task = asyncio.create_task(emit_partial(b"".join(chunks)))
            continue
        if msg.type is not WSMsgType.TEXT:
            continue

        try:
            control = json.loads(msg.data)
        except json.JSONDecodeError:
            continue

        kind = control.get("type")
        if kind == "captions":
            if control.get("enabled"):
                translate = bool(control.get("translate")) and stt.can_translate
                caption = captions.CaptionSession(translate)
                if session is not None:
                    parked_session, session = session, None
                log.info("captions on%s", " (translating)" if translate else "")
                await ws.send_json({
                    "type": "captions",
                    "enabled": True,
                    "translate": translate,
                    "translate_available": stt.can_translate,
                })
            else:
                caption = None
                if parked_session is not None:
                    session, parked_session = parked_session, None
                    session.reset()
                log.info("captions off")
                await ws.send_json({"type": "captions", "enabled": False})
                if session is not None:
                    await ws.send_json({"type": "waiting"})
            continue
        if kind == "wake_mode":
            if control.get("enabled"):
                phrase = control.get("phrase") or request.app["cfg"].get("wake_phrase", "hey ollama")
                fresh = wake.WakeSession(phrase, request.app["cfg"]["end_silence_seconds"])
                log.info("wake mode on, listening for %r", phrase)
                if caption is not None:
                    # Captions own the mic for now; resume the phrase after.
                    parked_session = fresh
                else:
                    session = fresh
                    await ws.send_json({"type": "waiting"})
            else:
                session = None
                parked_session = None
                log.info("wake mode off")
            continue
        if kind == "force_wake" and session is not None:
            # Tap still works in wake mode: it just skips the phrase.
            session.start_capture()
            await ws.send_json({"type": "wake"})
            continue
        if kind == "force_stop" and session is not None:
            # Tap again to finish early rather than waiting out the silence.
            if session.capturing:
                pcm = bytes(session.capture_buffer)
                phrase = session.phrase
                session.reset()
                text = await stt.transcribe(pcm)
                stripped = wake.find_wake(text, phrase)
                text = stripped if stripped is not None else text
                seconds = len(pcm) / 2 / SAMPLE_RATE
                await ws.send_json({"type": "final", "text": text, "seconds": round(seconds, 2)})
            continue
        if kind == "start":
            chunks = []
            tap_active = True
            since_vad = 0.0
            auto_stop = bool(control.get("auto_stop"))
            endpointer = vad.Endpointer(end_silence=request.app["cfg"]["end_silence_seconds"])
            last_partial = loop.time()
            await ws.send_json({"type": "listening", "auto_stop": auto_stop})
        elif kind == "stop":
            # A tap after auto-send already finished has nothing left to send.
            if tap_active:
                await finish_tap()

    log.info("STT client disconnected")
    return ws


CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
}


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Allows the packed app to call the bridge cross-origin.

    While the bridge serves the app itself, every call is same-origin and this
    is a no-op. An .ehpk installed from the Hub runs from its own origin, so
    without these headers the browser blocks /api/* before it leaves the app.

    Open to any origin on purpose: the bridge only listens on the tailnet, it
    holds no credentials, and the set of origins a packed Even Hub app can
    present is not something we can enumerate ahead of time.
    """
    if request.method == "OPTIONS":
        resp: web.StreamResponse = web.Response(status=204)
    else:
        resp = await handler(request)

    # A streamed response has already flushed its headers by the time it gets
    # back here, so /api/ask sets these itself before prepare(). Assigning
    # again would raise, hence the guard.
    if not resp.prepared:
        resp.headers.update(CORS_HEADERS)
    return resp


async def handle_index(request: web.Request) -> web.StreamResponse:
    index = DIST / "index.html"
    if not index.exists():
        return web.Response(
            status=503,
            content_type="text/plain",
            text="The glasses app has not been built yet. Run `npm run build` in the project root.",
        )
    return web.FileResponse(index)


# --------------------------------------------------------------------------


async def resolve_ollama_url(app: web.Application) -> None:
    """Pins the address Ollama actually answers on.

    OLLAMA_HOST binds exactly one address, so whether Ollama is on 127.0.0.1 or
    on the tailnet IP depends on how it was last configured. Probing both here
    means the bridge keeps working across that change instead of failing with a
    connection refusal that looks like Ollama being down.
    """
    cfg = app["cfg"]
    candidates = [cfg["ollama_url"], "http://127.0.0.1:11434"]
    for extra in cfg.get("ollama_fallback_urls", []):
        candidates.append(extra)

    seen: list[str] = []
    for url in candidates:
        if url in seen:
            continue
        seen.append(url)
        try:
            async with app["http"].get(f"{url}/api/tags", timeout=ClientTimeout(total=4)) as resp:
                if resp.status == 200:
                    if url != cfg["ollama_url"]:
                        log.warning("Ollama not at %s; using %s", cfg["ollama_url"], url)
                    cfg["ollama_url"] = url
                    return
        except Exception:
            continue

    log.error("Ollama did not answer on any of: %s", ", ".join(seen))


def start_mdns(cfg: dict):
    """Publishes g2-bridge.local so phones on the LAN need no IP address.

    The app's manifest whitelist can only name fixed hosts, and every user's
    machine has a different address -- so a fixed mDNS name is what makes one
    published build work for everyone on a local network. Tailscale users reach
    the same bridge by their MagicDNS name instead.

    Best effort: no zeroconf, or a network that blocks multicast, just means
    falling back to typing an address.
    """
    if not cfg.get("mdns", True):
        return None
    try:
        import socket

        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        log.info("zeroconf not installed; skipping g2-bridge.local advertisement")
        return None

    try:
        host = str(cfg.get("host", "")).strip()
        if host in ("", "0.0.0.0", "127.0.0.1", "localhost"):
            # Bound to everything, so advertise whichever interface the machine
            # would actually route out of.
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(("8.8.8.8", 80))
            local_ip = probe.getsockname()[0]
            probe.close()
        else:
            # Bound to one specific address -- advertise THAT. Publishing any
            # other interface points g2-bridge.local at a port nothing is
            # listening on, which fails in a way that looks like mDNS is broken.
            local_ip = host

        zc = Zeroconf()
        info = ServiceInfo(
            "_http._tcp.local.",
            "G2 Bridge._http._tcp.local.",
            addresses=[socket.inet_aton(local_ip)],
            port=int(cfg["port"]),
            properties={"path": "/"},
            # This is the part that makes the name resolve, not just the service.
            server="g2-bridge.local.",
        )
        zc.register_service(info)
        log.info("Advertising g2-bridge.local -> %s:%s", local_ip, cfg["port"])
        return zc, info
    except Exception as exc:
        log.warning("Could not advertise over mDNS: %s", exc)
        return None


async def timer_loop(app: web.Application) -> None:
    while True:
        await asyncio.sleep(1)
        try:
            for timer in app["timers"].tick():
                asyncio.create_task(asyncio.to_thread(personal.ring, timer.label))
        except Exception:
            log.exception("timer tick failed")


async def on_startup(app: web.Application) -> None:
    app["http"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))
    app["mdns"] = await asyncio.to_thread(start_mdns, app["cfg"])
    app["apps"] = await asyncio.to_thread(tools.AppIndex, app["cfg"].get("apps", {}))
    await resolve_ollama_url(app)
    # Blocking model load, deliberately before the first request rather than
    # lazily: better a slow start than a 30s stall on the first question.
    await asyncio.to_thread(app["stt"].load)
    # Load the voice detector now too, so the first auto-send doesn't pay for it.
    await asyncio.to_thread(vad.scorer)
    app["timer_task"] = asyncio.create_task(timer_loop(app))


async def on_cleanup(app: web.Application) -> None:
    if app.get("timer_task"):
        app["timer_task"].cancel()
    await app["http"].close()
    if app.get("mdns"):
        zc, info = app["mdns"]
        # Withdraw the record instead of leaving a stale g2-bridge.local
        # pointing at a port nothing is listening on.
        await asyncio.to_thread(zc.unregister_service, info)
        await asyncio.to_thread(zc.close)


def build_app(cfg: dict, cfg_path: Path) -> web.Application:
    app = web.Application(client_max_size=16 * 1024 * 1024, middlewares=[cors_middleware])
    app["cfg"] = cfg
    app["cfg_path"] = cfg_path
    app["stt"] = Transcriber(cfg)
    app["memory"] = personal.Memory(Path(cfg["memory_file"]))
    app["timers"] = personal.Timers()

    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/models", handle_models)
    app.router.add_get("/api/inbox", handle_inbox)
    app.router.add_post("/api/ask", handle_ask)
    app.router.add_post("/api/stt", handle_stt_http)
    app.router.add_get("/stt", handle_stt)
    app.router.add_get("/", handle_index)
    if DIST.exists():
        app.router.add_static("/", DIST)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def wait_for_address(host: str, timeout: float = 180.0) -> bool:
    """Blocks until `host` is bindable on this machine.

    At logon this process can easily win the race against Tailscale bringing
    its interface up, and binding an address that doesn't exist yet fails
    outright. Retrying turns a hard startup failure into a short wait.
    """
    import socket
    import time

    if host in ("0.0.0.0", "127.0.0.1", "localhost", ""):
        return True

    deadline = time.monotonic() + timeout
    warned = False
    while time.monotonic() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.bind((host, 0))
            return True
        except OSError:
            if not warned:
                log.info("Waiting for %s to become available (is Tailscale up?)...", host)
                warned = True
            time.sleep(3)
        finally:
            probe.close()

    log.error("Address %s never became available after %ss", host, timeout)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--host", help="Override bind host")
    parser.add_argument("--port", type=int, help="Override bind port")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", help="Append logs here (required under pythonw, which has no console)")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        **({"filename": args.log_file, "filemode": "a"} if args.log_file else {}),
    )

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    if args.host:
        cfg["host"] = args.host
    if args.port:
        cfg["port"] = args.port

    if not wait_for_address(cfg["host"]):
        return 1

    app = build_app(cfg, cfg_path)
    log.info("Serving app from %s", DIST)
    log.info("Listening on http://%s:%s", cfg["host"], cfg["port"])
    web.run_app(app, host=cfg["host"], port=cfg["port"], print=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
