"""Controls for the computer itself: media, volume, windows, clipboard, status.

The point of running the assistant on your own machine is that it can reach
things no cloud assistant can -- the song playing on your PC, the window in
front of you, what you just copied. Everything here is deliberately narrow:

* No shell. Each action is one specific OS call.
* Closing an app closes its *window*, the same as clicking X, so the app can
  still ask to save. Nothing is force-killed.
* Windows gets native calls through ctypes (no extra packages). macOS and Linux
  get the nearest standard tool, best effort.
"""

from __future__ import annotations

import base64
import difflib
import io
import logging
import os
import platform
import re
import shutil
import subprocess
import time

log = logging.getLogger("g2-bridge.pc")

SYSTEM = platform.system()
IS_WINDOWS = SYSTEM == "Windows"
IS_MAC = SYSTEM == "Darwin"


# --------------------------------------------------------------------------
# Media and volume
# --------------------------------------------------------------------------

# Windows virtual-key codes for the media keys every keyboard driver honours,
# so this controls whatever is playing -- Spotify, YouTube in a browser, VLC.
_VK = {
    "play_pause": 0xB3,
    "next": 0xB0,
    "previous": 0xB1,
    "stop": 0xB2,
    "volume_up": 0xAF,
    "volume_down": 0xAE,
    "mute": 0xAD,
}
# Each volume key press moves the Windows mixer by 2%.
_VOLUME_STEP = 2


def _press(vk: int, times: int = 1) -> None:
    import ctypes

    user32 = ctypes.windll.user32
    extended, keyup = 0x1, 0x2
    for _ in range(times):
        user32.keybd_event(vk, 0, extended, 0)
        user32.keybd_event(vk, 0, extended | keyup, 0)


def media(action: str, amount: int | None = None) -> str:
    """play_pause | next | previous | stop | volume_up | volume_down | mute | set_volume."""
    action = action.strip().lower().replace(" ", "_").replace("-", "_")
    action = {"play": "play_pause", "pause": "play_pause", "resume": "play_pause", "skip": "next",
              "back": "previous", "prev": "previous", "louder": "volume_up", "quieter": "volume_down",
              "unmute": "mute", "volume": "set_volume"}.get(action, action)

    if IS_WINDOWS:
        if action == "set_volume":
            level = max(0, min(100, int(amount if amount is not None else 50)))
            # No absolute-volume key exists, so bottom out and count up. Crude,
            # but it needs no audio library and works on every Windows box.
            _press(_VK["volume_down"], 50)
            _press(_VK["volume_up"], round(level / _VOLUME_STEP))
            return f"Volume set to about {level}%."
        if action in ("volume_up", "volume_down"):
            step = int(amount) if amount else 10
            _press(_VK[action], max(1, round(step / _VOLUME_STEP)))
            return f"Volume {'up' if action == 'volume_up' else 'down'} {step}%."
        if action not in _VK:
            return f"Unknown media action '{action}'."
        _press(_VK[action])
        return {"play_pause": "Toggled play/pause.", "next": "Skipped to the next track.",
                "previous": "Went back a track.", "stop": "Stopped playback.",
                "mute": "Toggled mute."}[action]

    if IS_MAC:
        if action == "set_volume":
            level = max(0, min(100, int(amount if amount is not None else 50)))
            subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=False)
            return f"Volume set to {level}%."
        if action in ("volume_up", "volume_down"):
            delta = (int(amount) if amount else 10) * (1 if action == "volume_up" else -1)
            script = f"set volume output volume ((output volume of (get volume settings)) + {delta})"
            subprocess.run(["osascript", "-e", script], check=False)
            return "Volume changed."
        if action == "mute":
            subprocess.run(["osascript", "-e", "set volume output muted not (output muted of (get volume settings))"], check=False)
            return "Toggled mute."
        verb = {"play_pause": "playpause", "next": "next track", "previous": "previous track", "stop": "pause"}.get(action)
        if not verb:
            return f"Unknown media action '{action}'."
        for player in ("Spotify", "Music"):
            subprocess.run(["osascript", "-e", f'if application "{player}" is running then tell application "{player}" to {verb}'], check=False)
        return "Done."

    # Linux: playerctl for MPRIS players, pactl for the mixer.
    if action == "set_volume":
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(amount or 50)}%"], check=False)
        return "Volume set."
    if action in ("volume_up", "volume_down"):
        sign = "+" if action == "volume_up" else "-"
        subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{int(amount or 10)}%"], check=False)
        return "Volume changed."
    if action == "mute":
        subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"], check=False)
        return "Toggled mute."
    verb = {"play_pause": "play-pause", "next": "next", "previous": "previous", "stop": "stop"}.get(action)
    if not verb or not shutil.which("playerctl"):
        return "Media control needs playerctl on Linux."
    subprocess.run(["playerctl", verb], check=False)
    return "Done."


# --------------------------------------------------------------------------
# Windows: listing and closing
# --------------------------------------------------------------------------

# Never offered for closing, whatever was misheard.
_PROTECTED = re.compile(r"^(program manager|windows input experience|settings|task manager|nvidia geforce overlay)$", re.I)


def _visible_windows() -> list[tuple[int, str, str]]:
    """(hwnd, title, exe name) for every real top-level window."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    found: list[tuple[int, str, str]] = []
    own_pid = os.getpid()

    def exe_name(pid: int) -> str:
        handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return os.path.basename(buf.value)
            return ""
        finally:
            kernel32.CloseHandle(handle)

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd) or user32.GetWindow(hwnd, 4):  # GW_OWNER: skip dialogs
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        title = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title, length + 1)
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == own_pid or _PROTECTED.match(title.value.strip()):
            return True
        found.append((int(hwnd), title.value.strip(), exe_name(pid.value)))
        return True

    user32.EnumWindows(callback, 0)
    return found


def _fold(text: str) -> str:
    text = text.lower()
    for word, digit in (("two", "2"), ("three", "3"), ("four", "4"), ("one", "1")):
        text = re.sub(rf"\b{word}\b", digit, text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _match_window(name: str, windows: list[tuple[int, str, str]]) -> tuple[int, str] | None:
    wanted = _fold(name)
    if not wanted:
        return None
    best: tuple[float, int, str] | None = None
    for hwnd, title, exe in windows:
        candidates = [_fold(title), _fold(os.path.splitext(exe)[0])]
        # A browser tab title is "Page - YouTube - Google Chrome": the app name
        # is the last segment, and matching that beats matching page text.
        parts = [p for p in re.split(r"\s[-–—|]\s", title) if p]
        if parts:
            candidates.append(_fold(parts[-1]))
        for cand in candidates:
            if not cand:
                continue
            if cand == wanted:
                score = 1.0
            elif wanted in cand or cand in wanted:
                score = 0.9
            else:
                score = difflib.SequenceMatcher(None, wanted, cand).ratio()
            if best is None or score > best[0]:
                best = (score, hwnd, title)
    if best and best[0] >= 0.75:
        return best[1], best[2]
    return None


def open_windows() -> str:
    if not IS_WINDOWS:
        return "Listing windows is only supported on Windows."
    titles = [title for _h, title, _e in _visible_windows()]
    return "Open windows: " + "; ".join(titles[:25]) if titles else "No windows are open."


def foreground_window_title() -> str:
    """Title of the window in front, as context for a screenshot. '' if unknown."""
    if not IS_WINDOWS:
        return ""
    import ctypes

    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    length = user32.GetWindowTextLengthW(hwnd)
    if not hwnd or not length:
        return ""
    title = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, title, length + 1)
    return title.value.strip()


def close_app(name: str) -> str:
    """Closes a window the way its X button would -- the app may still ask to save."""
    if IS_WINDOWS:
        import ctypes

        hit = _match_window(name, _visible_windows())
        if not hit:
            return f"No open window matches '{name}'."
        hwnd, title = hit
        ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)  # WM_CLOSE
        log.info("closed window %r", title)
        return f"Closed {title}."
    if IS_MAC:
        safe = name.replace('"', "")
        subprocess.run(["osascript", "-e", f'tell application "{safe}" to quit'], check=False)
        return f"Asked {safe} to quit."
    if shutil.which("wmctrl"):
        subprocess.run(["wmctrl", "-c", name], check=False)
        return f"Closed {name}."
    return "Closing apps needs wmctrl on Linux."


def lock_computer() -> str:
    if IS_WINDOWS:
        import ctypes

        ctypes.windll.user32.LockWorkStation()
    elif IS_MAC:
        subprocess.run(["pmset", "displaysleepnow"], check=False)
    else:
        subprocess.run(["loginctl", "lock-session"], check=False)
    return "Computer locked."


# --------------------------------------------------------------------------
# Clipboard
# --------------------------------------------------------------------------


def read_clipboard(limit: int = 4000) -> str:
    try:
        if IS_WINDOWS:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Clipboard -Raw"],
                capture_output=True, text=True, encoding="utf-8", timeout=10,
            ).stdout
        elif IS_MAC:
            out = subprocess.run(["pbpaste"], capture_output=True, text=True, timeout=10).stdout
        else:
            out = subprocess.run(["xclip", "-o", "-selection", "clipboard"], capture_output=True, text=True, timeout=10).stdout
    except Exception as exc:
        return f"Could not read the clipboard: {exc}"
    out = (out or "").strip()
    if not out:
        return "The clipboard is empty (or holds an image, which I can't read)."
    return out[:limit] + (" ...(truncated)" if len(out) > limit else "")


def write_clipboard(text: str) -> str:
    try:
        if IS_WINDOWS:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[Console]::InputEncoding=[Text.Encoding]::UTF8; Set-Clipboard -Value ([Console]::In.ReadToEnd())"],
                input=text, text=True, encoding="utf-8", timeout=10, check=False,
            )
        elif IS_MAC:
            subprocess.run(["pbcopy"], input=text, text=True, timeout=10, check=False)
        else:
            subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, timeout=10, check=False)
    except Exception as exc:
        return f"Could not write the clipboard: {exc}"
    return "Copied to the clipboard."


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------


def _gpu_status() -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
    except Exception:
        return ""
    lines = []
    for row in out:
        name, util, used, total, temp = [x.strip() for x in row.split(",")]
        lines.append(f"GPU {name}: {util}% busy, {int(used) / 1024:.1f} of {int(total) / 1024:.1f} GB VRAM, {temp}°C")
    return "; ".join(lines)


def system_status() -> str:
    parts: list[str] = []
    try:
        import psutil

        parts.append(f"CPU {psutil.cpu_percent(interval=0.5):.0f}% busy")
        mem = psutil.virtual_memory()
        parts.append(f"RAM {mem.used / 2**30:.1f} of {mem.total / 2**30:.1f} GB used")
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        parts.append(f"main disk {disk.free / 2**30:.0f} GB free of {disk.total / 2**30:.0f} GB")
        battery = psutil.sensors_battery()
        if battery:
            parts.append(f"battery {battery.percent:.0f}%{' charging' if battery.power_plugged else ''}")
        uptime = time.time() - psutil.boot_time()
        parts.append(f"up {int(uptime // 3600)}h {int(uptime % 3600 // 60)}m")
    except ImportError:
        parts.append("(install psutil for CPU, memory and disk figures)")
    gpu = _gpu_status()
    if gpu:
        parts.append(gpu)
    return "; ".join(parts)


# --------------------------------------------------------------------------
# Seeing the screen
# --------------------------------------------------------------------------


def screenshot_jpeg_b64(max_width: int = 1600) -> str:
    """The primary screen as base64 JPEG, sized for a vision model."""
    from PIL import ImageGrab

    image = ImageGrab.grab()
    if image.width > max_width:
        image = image.resize((max_width, round(image.height * max_width / image.width)))
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")
