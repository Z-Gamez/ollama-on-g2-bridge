"""Wake-word listening: "hey ollama" instead of tapping.

Scope, stated honestly: the Even SDK gives a plugin no hotword hook and no
background service -- `audioControl` is the whole of it -- so this works while
the app is open, not system-wide the way "Hey Even" does. Open it once (or pin
it to the glasses menu) and you never touch the temple again.

Two ideas make continuous listening cheap enough to leave on:

* **An energy gate.** Transcribing every second forever would keep the GPU busy
  all day for nothing. Audio below the noise floor is never decoded, so an idle
  room costs nothing.
* **A rolling window.** Wake detection only ever looks at the last few seconds,
  so the buffer cannot grow without bound while the room is quiet.
"""

from __future__ import annotations

import difflib
import re

import numpy as np

SAMPLE_RATE = 16000

# Only the last few seconds can contain the wake phrase.
WAKE_WINDOW_SECONDS = 5.0
# Silence that ends a question. Long enough to survive a mid-sentence pause.
END_SILENCE_SECONDS = 1.3
# Hard stop, so a noisy room can't record forever.
MAX_UTTERANCE_SECONDS = 25.0
# Below this, treat it as room tone no matter what the noise floor says.
ABSOLUTE_SILENCE_RMS = 120.0


def frame_rms(pcm: bytes) -> float:
    if not pcm:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))


class SpeechGate:
    """Tells speech from room tone, adapting to however loud the room is."""

    def __init__(self) -> None:
        self.noise_floor = 200.0
        self.speaking = False
        # Kept for diagnostics: if the wake phrase never triggers, the question
        # is always "is the gate even seeing speech?", and guessing at mic
        # levels from the other side of a BLE link is hopeless.
        self.peak_level = 0.0
        self.last_level = 0.0

    def update(self, pcm: bytes) -> bool:
        level = frame_rms(pcm)
        self.last_level = level
        self.peak_level = max(self.peak_level, level)
        # Speech has to clear the floor by a good margin; the floor itself only
        # tracks quiet frames, so talking never drags the threshold up with it.
        threshold = max(self.noise_floor * 2.5, ABSOLUTE_SILENCE_RMS)
        self.speaking = level > threshold
        if not self.speaking:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * level
        return self.speaking


_PUNCT = re.compile(r"[^a-z0-9\s]")


def normalize(text: str) -> str:
    return " ".join(_PUNCT.sub(" ", text.lower()).split())


def find_wake(text: str, phrase: str = "hey ollama", cutoff: float = 0.72) -> str | None:
    """Returns whatever was said after the wake phrase, or None.

    Fuzzy on purpose. Whisper renders "hey ollama" as "hey, a llama", "hey oh
    lama", "Hollama" and worse, so exact matching would fail most of the time.
    An empty string is a valid result -- it means the phrase was heard with
    nothing after it yet.
    """
    words = normalize(text).split()
    target = normalize(phrase).split()
    if not words or not target:
        return None

    span = len(target)
    # Try the tightest window first, then one word wider and narrower, since
    # the phrase may be transcribed as more or fewer words than it really is.
    for width in (span, span + 1, max(1, span - 1)):
        for start in range(len(words) - width + 1):
            window = " ".join(words[start : start + width])
            if difflib.SequenceMatcher(None, window, " ".join(target)).ratio() >= cutoff:
                return " ".join(words[start + width :]).strip()
    return None


class WakeSession:
    """Accumulates mic audio and decides when there is something to act on.

    Feed it frames; it answers with what to do next. All the timing lives here
    so the socket handler stays a simple loop.
    """

    def __init__(self, phrase: str = "hey ollama") -> None:
        self.phrase = phrase
        self.gate = SpeechGate()
        self.listening_buffer = bytearray()   # rolling, for spotting the phrase
        self.capture_buffer = bytearray()     # everything since the wake
        self.capturing = False
        self.silence_seconds = 0.0
        self.captured_seconds = 0.0
        self.had_speech = False

    @property
    def _window_bytes(self) -> int:
        return int(WAKE_WINDOW_SECONDS * SAMPLE_RATE) * 2

    def add(self, pcm: bytes) -> dict:
        """Adds one frame. Returns what the caller should do about it.

        {'action': 'none'}                     nothing to do
        {'action': 'check_wake', 'pcm': ...}   decode this to look for the phrase
        {'action': 'progress', 'pcm': ...}     decode for an interim transcript
        {'action': 'finish', 'pcm': ...}       the question ended; decode it
        """
        speaking = self.gate.update(pcm)
        seconds = len(pcm) / 2 / SAMPLE_RATE

        if not self.capturing:
            self.listening_buffer.extend(pcm)
            if len(self.listening_buffer) > self._window_bytes:
                del self.listening_buffer[: len(self.listening_buffer) - self._window_bytes]
            # Only worth decoding once the room has actually made a sound.
            if speaking:
                self.had_speech = True
                return {"action": "check_wake", "pcm": bytes(self.listening_buffer)}
            return {"action": "none"}

        self.capture_buffer.extend(pcm)
        self.captured_seconds += seconds
        self.silence_seconds = 0.0 if speaking else self.silence_seconds + seconds

        ended = self.silence_seconds >= END_SILENCE_SECONDS or self.captured_seconds >= MAX_UTTERANCE_SECONDS
        if ended:
            pcm_out = bytes(self.capture_buffer)
            self.reset()
            return {"action": "finish", "pcm": pcm_out}
        return {"action": "progress", "pcm": bytes(self.capture_buffer)}

    def start_capture(self) -> None:
        self.capturing = True
        self.capture_buffer = bytearray()
        self.silence_seconds = 0.0
        self.captured_seconds = 0.0
        self.listening_buffer = bytearray()

    def reset(self) -> None:
        self.capturing = False
        self.capture_buffer = bytearray()
        self.listening_buffer = bytearray()
        self.silence_seconds = 0.0
        self.captured_seconds = 0.0
        self.had_speech = False
