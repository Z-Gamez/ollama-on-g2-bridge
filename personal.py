"""Things an assistant on your face should just know: what you told it to
remember, the timers you set, and the weather where you are.

All of it stays on this machine. Memory is a JSON file next to the bridge;
timers live in memory and ring on the computer and on the lens; weather comes
from Open-Meteo, which needs no key and no account.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("g2-bridge.personal")


# --------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------

MAX_MEMORIES = 60


class Memory:
    """Facts the user asked to keep. Injected into every conversation, so
    "where did I park?" works days after "remember I parked on level 3"."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: list[dict] = []
        self._lock = threading.Lock()
        try:
            self.items = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            pass
        except Exception:
            log.exception("could not read %s; starting empty", path)

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.items, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, fact: str) -> str:
        fact = fact.strip().rstrip(".")
        if not fact:
            return "Nothing to remember."
        with self._lock:
            self.items.append({"fact": fact, "saved": datetime.now().strftime("%Y-%m-%d %H:%M")})
            self.items = self.items[-MAX_MEMORIES:]
            self._save()
        return f"Remembered: {fact}."

    def forget(self, about: str) -> str:
        words = [w for w in re.findall(r"[a-z0-9]+", about.lower()) if len(w) > 2]
        with self._lock:
            if about.strip().lower() in ("everything", "all", "all of it"):
                count = len(self.items)
                self.items = []
                self._save()
                return f"Forgot all {count} things."
            keep, dropped = [], []
            for item in self.items:
                text = item["fact"].lower()
                (dropped if words and all(w in text for w in words) else keep).append(item)
            if not dropped:
                return f"I had nothing saved about {about}."
            self.items = keep
            self._save()
        return "Forgot: " + "; ".join(i["fact"] for i in dropped)

    def prompt_block(self) -> str:
        if not self.items:
            return ""
        lines = "\n".join(f"- {i['fact']} (saved {i['saved']})" for i in self.items)
        return (
            "\n\nThings the user asked you to remember. Use them when relevant and "
            f"never claim not to know them:\n{lines}"
        )


# --------------------------------------------------------------------------
# Timers
# --------------------------------------------------------------------------


@dataclass
class Timer:
    id: int
    label: str
    due: float
    fired: bool = False
    created: float = field(default_factory=time.time)


_AT = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", re.I)


def parse_clock(text: str, now: datetime | None = None) -> datetime | None:
    """'5pm', '17:30', '7:05 am' -> the next time it is that o'clock."""
    match = _AT.match(text or "")
    if not match:
        return None
    hour, minute, meridiem = int(match.group(1)), int(match.group(2) or 0), (match.group(3) or "").lower()
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    now = now or datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


class Timers:
    def __init__(self) -> None:
        self.items: list[Timer] = []
        self._next_id = 1
        self._lock = threading.Lock()
        # Fired alerts waiting to be picked up by the glasses app.
        self.inbox: list[dict] = []

    def add(self, seconds: float | None = None, at: str | None = None, label: str = "") -> str:
        if at:
            when = parse_clock(at)
            if not when:
                return f"I couldn't understand the time '{at}'."
            due = when.timestamp()
        elif seconds and seconds > 0:
            due = time.time() + float(seconds)
        else:
            return "A timer needs a duration or a time."
        label = label.strip() or "Timer"
        with self._lock:
            timer = Timer(self._next_id, label, due)
            self._next_id += 1
            self.items.append(timer)
        return f"{label} set for {self._describe(due)}."

    @staticmethod
    def _describe(due: float) -> str:
        left = due - time.time()
        clock = datetime.fromtimestamp(due).strftime("%I:%M %p").lstrip("0")
        if left < 3600:
            mins, secs = divmod(int(round(left)), 60)
            span = f"{mins} min" + (f" {secs} s" if secs and mins < 5 else "") if mins else f"{secs} s"
            return f"{span} from now ({clock})"
        return clock

    def cancel(self, label: str = "") -> str:
        with self._lock:
            active = [t for t in self.items if not t.fired]
            if not active:
                return "There are no timers running."
            if label.strip().lower() in ("", "all", "every", "everything"):
                chosen = active
            else:
                chosen = [t for t in active if label.lower() in t.label.lower()] or []
            if not chosen:
                return f"No timer called {label}."
            for t in chosen:
                t.fired = True
        return "Cancelled " + ", ".join(t.label for t in chosen) + "."

    def describe_all(self) -> str:
        active = self.active()
        if not active:
            return "No timers are running."
        return "; ".join(f"{t.label}: {self._describe(t.due)}" for t in active)

    def active(self) -> list[Timer]:
        with self._lock:
            return sorted((t for t in self.items if not t.fired), key=lambda t: t.due)

    def tick(self) -> list[Timer]:
        """Marks and returns timers that just came due."""
        now = time.time()
        due: list[Timer] = []
        with self._lock:
            for t in self.items:
                if not t.fired and t.due <= now:
                    t.fired = True
                    due.append(t)
                    self.inbox.append({"kind": "timer", "text": f"{t.label} - time's up", "at": now})
            # Drop long-finished timers so the list can't grow forever.
            self.items = [t for t in self.items if not t.fired or now - t.due < 3600]
            self.inbox = [n for n in self.inbox if now - n["at"] < 600]
        return due

    def drain_inbox(self, since: float) -> list[dict]:
        with self._lock:
            return [n for n in self.inbox if n["at"] > since]


def ring(label: str) -> None:
    """Makes a noise on the computer, for a timer that goes off while you're at it."""
    if platform.system() == "Windows":
        import winsound

        for _ in range(3):
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
            time.sleep(0.6)
    log.info("timer done: %s", label)


# --------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------

_WMO = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "ollama-on-g2-bridge"})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def geocode(place: str) -> tuple[float, float, str] | None:
    query = urllib.parse.urlencode({"name": place, "count": 1, "language": "en", "format": "json"})
    results = _get_json(f"https://geocoding-api.open-meteo.com/v1/search?{query}").get("results") or []
    if not results:
        return None
    hit = results[0]
    name = ", ".join(x for x in (hit.get("name"), hit.get("admin1"), hit.get("country_code")) if x)
    return hit["latitude"], hit["longitude"], name


def weather(lat: float, lon: float, place: str, fahrenheit: bool) -> str:
    unit = "fahrenheit" if fahrenheit else "celsius"
    query = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 2,
        "temperature_unit": unit, "wind_speed_unit": "mph" if fahrenheit else "kmh",
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
    })
    data = _get_json(f"https://api.open-meteo.com/v1/forecast?{query}")
    deg = "°F" if fahrenheit else "°C"
    now = data["current"]
    daily = data["daily"]
    lines = [
        f"Weather for {place} right now: {_WMO.get(now['weather_code'], 'unknown')}, "
        f"{now['temperature_2m']:.0f}{deg} (feels {now['apparent_temperature']:.0f}{deg}), "
        f"wind {now['wind_speed_10m']:.0f} {'mph' if fahrenheit else 'km/h'}, humidity {now['relative_humidity_2m']}%."
    ]
    for i, day in enumerate(("Today", "Tomorrow")):
        if i < len(daily["time"]):
            lines.append(
                f"{day}: {_WMO.get(daily['weather_code'][i], 'unknown')}, "
                f"{daily['temperature_2m_min'][i]:.0f} to {daily['temperature_2m_max'][i]:.0f}{deg}, "
                f"{daily['precipitation_probability_max'][i]}% chance of rain."
            )
    return "\n".join(lines)
