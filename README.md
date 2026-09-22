# Ollama on G2 — bridge server

The companion server for the **Ollama on G2** app for Even Realities G2 smart
glasses. Ask a question out loud, read the answer on your lens — with the speech
recognition and the language model both running on your own machine.

Nothing leaves your network. No cloud, no API keys, no accounts.

> This is the server half. You also need the **Ollama on G2** app, installed on
> your glasses from Even Hub. Neither half is useful without the other.

## What it does

- **Speech to text** with [faster-whisper](https://github.com/SYSTRAN/faster-whisper),
  locally. The G2 microphone streams 16 kHz PCM straight in.
- **Knows when you've finished talking.** Silero voice detection (bundled with
  faster-whisper) sends the question about a second after you stop, so there is
  no second tap. It tells a voice from a fan or a TV by what the sound is, not
  how loud it is.
- **Answers** via [Ollama](https://ollama.com), running whatever model you like.
- **Acts on your computer:**
  - launches and closes apps and Steam games
  - controls music and volume (play, pause, skip, "set volume to 30")
  - **looks at your screen** — "what does this error say?" (needs a vision model)
  - reads your clipboard — "summarise what I copied"
  - locks the computer, reports CPU / GPU / memory / battery
- **Remembers things** you ask it to ("remember I parked on level 3"), in a JSON
  file on your machine.
- **Timers and reminders** that ring on the computer and pop up on the lens.
- **Weather** for where your phone is, from Open-Meteo (no key, no account).
- **News** from RSS, **web search** from DuckDuckGo.
- **Live captions** — continuous transcription of the people around you, on your
  lens. With a multilingual Whisper model it translates to English as it goes.
- **Wake phrase** — say "hey ollama" instead of tapping.
- **Photos** — take one on your phone and ask about it.

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) with a model pulled
- Windows, macOS or Linux. App launching is richest on Windows (Steam library
  and Start Menu are detected automatically).
- An NVIDIA GPU is optional but makes transcription roughly ten times faster.
- For the screen and photo features, a vision model: `ollama pull qwen2.5vl:3b`
  (or any model Ollama lists with vision capability — it is picked up
  automatically).

## Install

```bash
git clone https://github.com/Z-Gamez/ollama-on-g2-bridge.git
cd ollama-on-g2-bridge
pip install -r requirements.txt
ollama pull qwen3:4b
python g2_bridge.py
```

The first run downloads the Whisper model (~500 MB for `small.en`). It is ready
when the log says `Whisper ready`. Check it:

```bash
curl http://localhost:8770/api/health
```

```json
{"whisper": true, "whisper_model": "small.en", "ollama": true, "app_built": false}
```

`app_built: false` is normal — that only matters if you are also serving the web
app from here during development.

### NVIDIA: getting onto the GPU

If the log says `Whisper ready on cpu`, install the CUDA runtime libraries into
the same environment:

```bash
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12
```

Restart, and it should say `Whisper ready on cuda`.

On Windows those DLLs land somewhere Windows does not search. The bridge fixes
that itself by prepending `site-packages/nvidia/*/bin` to `PATH` before
importing faster-whisper — `os.add_dll_directory()` is *not* enough, because
CTranslate2 resolves cuBLAS with a plain `LoadLibrary`, which consults `PATH`
and ignores the added-directory list.

## Connecting your glasses

Enter your computer's address in the app on your phone.

**Same Wi-Fi** — the bridge advertises itself over mDNS, so enter:

```
g2-bridge.local
```

**Anywhere, including mobile data** — install [Tailscale](https://tailscale.com)
on both your computer and your phone, sign both into the same account, and enter
your computer's MagicDNS name:

```
my-pc.tailnet-name.ts.net
```

Tailscale is the better option: it works away from home, and nothing is exposed
to your local network.

**Use a hostname, not an IP address.** The app's manifest whitelists
`*.ts.net` and `g2-bridge.local`; a raw IP is not covered, and the WebSocket the
wake phrase depends on will be blocked.

## Configuration

Copy `config.example.json` to `config.json` and edit. Every key is optional —
defaults are in `DEFAULTS` at the top of `g2_bridge.py`.

| key | default | what it does |
| --- | --- | --- |
| `host` | `0.0.0.0` | Bind address. Pin it to one interface (a tailnet IP, say) to narrow exposure. |
| `port` | `8770` | HTTP and WebSocket port. |
| `mdns` | `true` | Advertise `g2-bridge.local` on the local network. |
| `ollama_url` | `http://127.0.0.1:11434` | Where Ollama is. |
| `ollama_model` | `qwen3:4b` | Default model. |
| `whisper_model` | `small.en` | Any faster-whisper model name. |
| `whisper_device` | `auto` | `auto`, `cuda` or `cpu`. |
| `whisper_models_dir` | `models` | Point at an existing cache to reuse models. |
| `wake_phrase` | `hey ollama` | What wakes it. |
| `end_silence_seconds` | `1.0` | How long a pause sends the question when auto-send is on. |
| `vision_model` | *(auto)* | Model for screen and photo questions. Empty picks the first one with vision. |
| `units` | `auto` | `imperial`, `metric`, or `auto` (imperial on a US locale). |
| `home_location` | *(empty)* | Weather location when the phone doesn't share one, e.g. `"Seattle"`. |
| `memory_file` | `memory.json` | Where "remember that..." is kept. |
| `tools` | `true` | Let the model act on the computer at all. |
| `disabled_tools` | `[]` | Switch off individual tools, e.g. `["lock_computer", "read_clipboard"]`. |
| `allow_shell` | `false` | **Raw shell access. See below.** |
| `apps` | `{}` | Extra launch targets: `"spoken name": "path or URI"`. |
| `news_feeds` | BBC, NPR, Guardian | RSS sources for headlines. |

## Security

This is a voice-driven agent with access to your machine, so the defaults are
deliberately conservative.

**Launching is an allowlist, not a shell.** `launch_app` resolves what you said
against software that is *already installed* — Steam games from
`libraryfolders.vdf`, plus Start Menu shortcuts — so a misheard word can at
worst start the wrong program. It cannot invent a command.

**`run_command` is off by default.** Speech recognition plus a language model
plus a shell is a bad combination; "open Steam" and something destructive are
one misheard word apart. Turn it on with `"allow_shell": true` only if you want
that, and know that every command it runs is logged.

**Closing an app closes its window**, exactly as clicking X would — the app can
still ask to save. Nothing is force-killed.

**Cancel means cancel.** Tapping during "thinking" closes the request, and the
bridge checks for that before running any tool, so a cancelled "launch
Counter-Strike" never launches.

**Screen, clipboard and photos stay local.** Screenshots go to your Ollama and
nowhere else; nothing is saved to disk. Turn any of it off with
`disabled_tools`.

**Bind deliberately.** The default `0.0.0.0` accepts connections from anywhere
that can reach the port. On a laptop that joins untrusted networks, set `host`
to your tailnet address so only your own devices can reach it.

**There is no authentication.** Anyone who can reach the port can ask it things
and launch apps. Keep it on a trusted network or a tailnet.

## API

| endpoint | method | purpose |
| --- | --- | --- |
| `/api/health` | GET | Readiness of Whisper and Ollama. |
| `/api/models` | GET | Models available from Ollama. |
| `/api/ask` | POST | Ask a question. Streams newline-delimited JSON. |
| `/api/stt` | POST | Transcribe raw PCM (s16le, 16 kHz, mono). `?partial=1&auto=1` also reports end of speech. |
| `/api/inbox` | GET | Timers that went off, and those still running. |
| `/stt` | WebSocket | Streaming speech, live partials, auto-send, wake phrase, captions. |

## Running it at startup

**Windows** — Task Scheduler, triggered at log on:

```powershell
$py  = "C:\path\to\pythonw.exe"
$dir = "C:\path\to\ollama-on-g2-bridge"
$action  = New-ScheduledTaskAction -Execute $py -Argument "g2_bridge.py --log-file bridge.log" -WorkingDirectory $dir
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
Register-ScheduledTask -TaskName "G2 Bridge" -Action $action -Trigger $trigger
```

`pythonw.exe` runs it without a console window, so `--log-file` is how you see
what it is doing.

**macOS / Linux** — a launchd plist or systemd user unit running
`python g2_bridge.py`.

If the bridge starts before your network is up, it waits for the configured
address rather than failing — handy when Tailscale is still connecting.

## Troubleshooting

**`Whisper ready on cpu`** — see the NVIDIA section above.

**The app says "Bridge unreachable"** — check `/api/health` on the machine
itself first, then that the phone can reach the address you typed.

**`g2-bridge.local` doesn't resolve** — some networks block mDNS, and phones
need local-network permission granted to the Even Realities app. Use Tailscale.

**The wake phrase never triggers** — it needs the WebSocket, so use a hostname
rather than an IP. If it still doesn't fire, the bridge logs microphone levels
every five seconds in wake mode:

```
mic levels: now=412 peak=1180 floor=190 threshold=475 speaking=True
```

If `peak` never rises above `threshold` when you speak, the voice-activity
thresholds in `wake.py` are too aggressive for your microphone.

**It sends before I've finished** — raise `end_silence_seconds` (1.5 suits people
who pause a lot), or hold the temple instead of tapping: hold-to-talk never
auto-sends.

**"What's on my screen" says no vision model** — `ollama pull qwen2.5vl:3b`.

**It won't launch something** — ask it "what apps do I have with <word> in the
name" to see what was found, and add anything missing under `apps` in your
config.

## Licence

MIT.

Not affiliated with Even Realities or Ollama.
