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
from collections import deque

import numpy as np
import vad

SAMPLE_RATE = 16000

# Only the last few seconds can contain the wake phrase.
WAKE_WINDOW_SECONDS = 5.0
# Silence that ends a question. Long enough to survive a mid-sentence pause.
END_SILENCE_SECONDS = 1.0
# After the phrase, how long to wait for the question to start. People pause
# after "hey ollama" -- ending the capture during that pause lost the question.
NO_SPEECH_SECONDS = 6.0
# Hard stop, so a noisy room can't record forever.
MAX_UTTERANCE_SECONDS = 25.0
# How often to run the voice detector. Frames arrive far faster than this.
VAD_INTERVAL_SECONDS = 0.2
# Keep looking for the phrase this long after speech stops, or "hey ollama"
# followed by a pause would never be decoded at all.
WAKE_TAIL_SECONDS = 1.5
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
    """Loudness, for the diagnostic log line. No longer decides anything.

    It used to gate decoding, and its floor only learned from frames it already
    judged quiet -- so a room steadily louder than the floor read as speech
    forever (the log showed floor=215 against a room at ~1000 for minutes).
    The floor is now a low percentile of recent levels, which follows the room
    up, and the real speech decisions are made by Silero in vad.py.
    """

    def __init__(self) -> None:
        self.levels: deque[float] = deque(maxlen=300)
        self.noise_floor = 200.0
        self.speaking = False
        self.peak_level = 0.0
        self.last_level = 0.0

    def update(self, pcm: bytes) -> bool:
        level = frame_rms(pcm)
        self.last_level = level
        self.peak_level = max(self.peak_level, level)
        self.levels.append(level)
        self.noise_floor = max(float(np.percentile(self.levels, 15)), 1.0)
        self.speaking = level > max(self.noise_floor * 2.5, ABSOLUTE_SILENCE_RMS)
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

    def __init__(self, phrase: str = "hey ollama", end_silence: float = END_SILENCE_SECONDS) -> None:
        self.phrase = phrase
        self.gate = SpeechGate()
        self.end_silence = end_silence
        self.listening_buffer = bytearray()   # rolling, for spotting the phrase
        self.endpointer = self._new_endpointer()
        self.capturing = False
        self.since_vad = 0.0
        self.since_speech = 999.0

    def _new_endpointer(self) -> vad.Endpointer:
        return vad.Endpointer(
            end_silence=self.end_silence,
            no_speech_timeout=NO_SPEECH_SECONDS,
            max_seconds=MAX_UTTERANCE_SECONDS,
        )

    @property
    def capture_buffer(self) -> bytearray:
        return self.endpointer.buffer

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
        self.gate.update(pcm)
        seconds = len(pcm) / 2 / SAMPLE_RATE
        self.since_vad += seconds

        if not self.capturing:
            self.listening_buffer.extend(pcm)
            if len(self.listening_buffer) > self._window_bytes:
                del self.listening_buffer[: len(self.listening_buffer) - self._window_bytes]
            self.since_speech += seconds
            if self.since_vad >= VAD_INTERVAL_SECONDS:
                self.since_vad = 0.0
                # Only worth a Whisper decode when a voice -- not a fan, not a
                # TV hum -- was heard recently.
                if vad.speaking(bytes(self.listening_buffer[-int(0.5 * SAMPLE_RATE) * 2 :])):
                    self.since_speech = 0.0
            if self.since_speech <= WAKE_TAIL_SECONDS:
                return {"action": "check_wake", "pcm": bytes(self.listening_buffer)}
            return {"action": "none"}

        self.endpointer.feed(pcm)
        if self.since_vad >= VAD_INTERVAL_SECONDS:
            self.since_vad = 0.0
            verdict = self.endpointer.check()
            if verdict != "continue":
                pcm_out = bytes(self.endpointer.buffer)
                self.reset()
                return {"action": "finish", "pcm": pcm_out, "reason": verdict}
        return {"action": "progress", "pcm": bytes(self.endpointer.buffer)}

    def start_capture(self) -> None:
        self.capturing = True
        self.endpointer = self._new_endpointer()
        self.listening_buffer = bytearray()
        self.since_vad = 0.0

    def reset(self) -> None:
        self.capturing = False
        self.endpointer = self._new_endpointer()
        self.listening_buffer = bytearray()
        self.since_vad = 0.0
        self.since_speech = 999.0
