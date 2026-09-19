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
import re
import sys
from pathlib import Path

import numpy as np
import tools
import wake
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

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
}

SYSTEM_PROMPT = (
    "You are a heads-up display assistant. The user reads your replies on a small "
    "pair of glasses, so answer in at most a few short sentences. No markdown, no "
    "bullet points, no code blocks -- plain prose only."
)

TOOL_PROMPT = (
    " You can act on the user's computer with the tools provided. Use launch_app to "
    "open programs and games, open_url for web pages, get_news for headlines and "
    "current events, and web_search for anything else you are unsure about. "
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
    r"|score|weather|forecast|stock|price of)\b",
    re.I,
)
# ...unless it's plainly an instruction to do something local.
ACTION_RE = re.compile(r"^\s*(launch|open|start|run|play|close|quit)\b", re.I)


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

    def _transcribe_sync(self, pcm: bytes) -> str:
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _info = self.model.transcribe(audio, language=self.cfg["language"])
        return "".join(seg.text for seg in segments).strip()

    async def transcribe(self, pcm: bytes) -> str:
        if self.model is None:
            raise RuntimeError("Whisper model is not loaded")
        seconds = len(pcm) / 2 / SAMPLE_RATE
        if seconds < self.cfg["min_utterance_seconds"]:
            return ""
        # One decode at a time: concurrent calls would contend for the same
        # CTranslate2 model and the GPU it sits on.
        async with self._lock:
            return await asyncio.to_thread(self._transcribe_sync, pcm)


# --------------------------------------------------------------------------
# Answer backends
# --------------------------------------------------------------------------


MAX_TOOL_ROUNDS = 4


async def _run_tool(app: web.Application, name: str, args: dict) -> tuple[str, str]:
    """Executes one tool call. Returns (status for the lens, result for the model)."""
    index = app["apps"]

    if name == "launch_app":
        wanted = str(args.get("name", ""))
        hit = index.resolve(wanted)
        if not hit:
            index.refresh()  # might have been installed since startup
            hit = index.resolve(wanted)
        if not hit:
            return (f"No match for {wanted}", f"No installed app or game matches '{wanted}'.")
        label, target = hit
        await asyncio.to_thread(tools.launch, target)
        return (f"Launching {label}...", f"Launched {label}.")

    if name == "open_url":
        url = str(args.get("url", ""))
        result = await asyncio.to_thread(tools.open_url, url)
        return (f"Opening {url}", result)

    if name == "web_search":
        query = str(args.get("query", ""))
        result = await asyncio.to_thread(tools.web_search, query)
        return (f"Searching: {query}", result)

    if name == "get_news":
        topic = str(args.get("topic", ""))
        result = await asyncio.to_thread(
            tools.news_headlines, app["cfg"].get("news_feeds"), topic, 8
        )
        return (f"Reading the news{' about ' + topic if topic else ''}", result)

    if name == "list_apps":
        contains = str(args.get("contains", "")).lower()
        names = [n for n in index.entries if contains in n.lower()] if contains else list(index.entries)
        names = sorted(names)[:30]
        return ("Checking installed apps", ", ".join(names) if names else "Nothing matches.")

    if name == "run_command":
        if not app["cfg"].get("allow_shell", False):
            return ("Shell disabled", "Shell access is disabled in this bridge's config.")
        command = str(args.get("command", ""))
        result = await asyncio.to_thread(tools.run_command, command)
        return (f"Running: {command[:40]}", result)

    return (f"Unknown tool {name}", f"No such tool: {name}")


async def run_agent(app: web.Application, prompt: str, model: str, history: list):
    """Answers a question, using tools when the model asks for them.

    Yields dicts: {'status': ...} for something happening on the machine, and
    {'delta': ...} for answer text. Tool rounds are non-streaming because a
    tool call has to arrive complete before it can run; the final answer is
    streamed as usual.
    """
    cfg = app["cfg"]
    use_tools = cfg.get("tools", True)
    schemas = tools.schemas(cfg.get("allow_shell", False)) if use_tools else None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + (TOOL_PROMPT if use_tools else "")},
        *history,
        {"role": "user", "content": prompt},
    ]

    # Fetch live data up front for questions whose answer changed since the
    # model was trained. Left to its own judgement a small model answers these
    # from memory, or worse, tells the user to go and check a news site --
    # which is the one thing an assistant on your face should never do.
    if use_tools and not ACTION_RE.search(prompt):
        forced: tuple[str, dict] | None = None
        if NEWS_RE.search(prompt):
            forced = ("get_news", {})
        elif FRESH_RE.search(prompt):
            forced = ("web_search", {"query": prompt})

        if forced:
            name, args = forced
            log.info("forced lookup: %s(%s)", name, args)
            try:
                status, result = await _run_tool(app, name, args)
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
            log.info("tool call: %s(%s)", name, args)
            try:
                status, result = await _run_tool(app, name, args)
            except Exception as exc:
                log.exception("tool %s failed", name)
                status, result = (f"{name} failed", f"Error running {name}: {exc}")
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

    # CORS headers go on before prepare(): prepare() flushes them to the wire,
    # so the middleware would be too late to add them to a streamed response.
    resp = web.StreamResponse(
        headers={"Content-Type": "application/x-ndjson", "Cache-Control": "no-store", **CORS_HEADERS}
    )
    await resp.prepare(request)

    try:
        # run_agent yields {"delta": ...} for answer text and {"status": ...}
        # when it is doing something on the machine, so both go straight out.
        async for event in run_agent(request.app, prompt, body.get("model", ""), history):
            await resp.write(json.dumps(event).encode() + b"\n")
        await resp.write(json.dumps({"done": True}).encode() + b"\n")
    except web.HTTPException as exc:
        await resp.write(json.dumps({"error": exc.text}).encode() + b"\n")
    except Exception as exc:
        log.exception("ask failed")
        await resp.write(json.dumps({"error": str(exc)}).encode() + b"\n")

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
    try:
        text = await request.app["stt"].transcribe(pcm)
    except Exception as exc:
        log.exception("transcription failed")
        raise web.HTTPInternalServerError(text=str(exc))
    return web.json_response({"text": text, "seconds": round(seconds, 2)})


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

    # Set when the client asks to listen for a wake phrase instead of taps.
    session: wake.WakeSession | None = None
    last_decode = 0.0
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
                "mic levels: now=%.0f peak=%.0f floor=%.0f threshold=%.0f speaking=%s",
                gate.last_level,
                gate.peak_level,
                gate.noise_floor,
                max(gate.noise_floor * 2.5, wake.ABSOLUTE_SILENCE_RMS),
                gate.speaking,
            )
            gate.peak_level = 0.0

        if action == "none":
            return

        if action == "finish":
            # Never rate-limit this one. `add` has already reset the session, so
            # dropping it here would throw away the whole question and leave the
            # user waiting for an answer that never comes.
            last_decode = loop.time()
            text = await stt.transcribe(step["pcm"])
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
            if session is not None:
                try:
                    await handle_wake_frame(msg.data)
                except Exception:
                    log.exception("wake handling failed")
                continue

            chunks.append(msg.data)
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
        if kind == "wake_mode":
            if control.get("enabled"):
                phrase = control.get("phrase") or request.app["cfg"].get("wake_phrase", "hey ollama")
                session = wake.WakeSession(phrase)
                log.info("wake mode on, listening for %r", phrase)
                await ws.send_json({"type": "waiting"})
            else:
                session = None
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
            last_partial = loop.time()
            await ws.send_json({"type": "listening"})
        elif kind == "stop":
            if partial_task and not partial_task.done():
                # A partial resolving after the final would overwrite a good
                # transcript with a worse guess.
                partial_task.cancel()
            pcm = b"".join(chunks)
            chunks = []
            seconds = len(pcm) / 2 / SAMPLE_RATE
            await ws.send_json({"type": "transcribing", "seconds": round(seconds, 2)})
            try:
                text = await stt.transcribe(pcm)
                await ws.send_json({"type": "final", "text": text, "seconds": round(seconds, 2)})
            except Exception as exc:
                log.exception("transcription failed")
                await ws.send_json({"type": "error", "error": str(exc)})

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


async def on_startup(app: web.Application) -> None:
    app["http"] = ClientSession(timeout=ClientTimeout(total=None, sock_connect=10))
    app["mdns"] = await asyncio.to_thread(start_mdns, app["cfg"])
    app["apps"] = await asyncio.to_thread(tools.AppIndex, app["cfg"].get("apps", {}))
    await resolve_ollama_url(app)
    # Blocking model load, deliberately before the first request rather than
    # lazily: better a slow start than a 30s stall on the first question.
    await asyncio.to_thread(app["stt"].load)


async def on_cleanup(app: web.Application) -> None:
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

    app.router.add_get("/api/health", handle_health)
    app.router.add_get("/api/models", handle_models)
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
