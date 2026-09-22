"""Knowing when someone has finished talking.

Loudness alone does not work on the G2. The bridge log from a real pair shows
the room sitting at an RMS of ~1000 while the energy gate's noise floor was
stuck at 215 -- it only learned from frames it already considered quiet, so a
steady background noise locked it into "speaking" for two minutes straight. An
end-of-speech detector built on that would never fire.

So this uses Silero VAD, the small neural voice detector that ships inside
faster-whisper (no extra download, CPU only, a few milliseconds per check). It
tells speech from a fan, a TV or road noise by what the sound *is*, not how loud
it is. If onnxruntime is somehow missing, an energy detector with a floor that
cannot lock up takes over.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

log = logging.getLogger("g2-bridge.vad")

SAMPLE_RATE = 16000
# Silero scores audio in fixed 32 ms frames.
FRAME = 512
FRAME_SECONDS = FRAME / SAMPLE_RATE

# Scores above this are speech; the lower one ends it. The gap stops a word
# that trails off from flickering in and out of "speech".
SPEECH_PROB = 0.5
SILENCE_PROB = 0.35
# This much continuous speech before we believe someone is actually talking,
# so a cough or a door does not count as having started a question.
MIN_SPEECH_SECONDS = 0.25
# Only the recent past matters for "have they stopped?", so each check scores
# a bounded window rather than the whole utterance.
WINDOW_SECONDS = 3.0


def _pcm_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


class _Silero:
    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self.model = get_vad_model()

    def probs(self, audio: np.ndarray) -> np.ndarray:
        usable = len(audio) - len(audio) % FRAME
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(self.model(audio[:usable]), dtype=np.float32).reshape(-1)


class _Energy:
    """Fallback. The floor is a low percentile of recent levels, so it follows a
    noisy room up within seconds instead of freezing the way a floor that only
    learns from 'quiet' frames does."""

    def __init__(self) -> None:
        self.levels: deque[float] = deque(maxlen=int(6 / FRAME_SECONDS))

    def probs(self, audio: np.ndarray) -> np.ndarray:
        usable = len(audio) - len(audio) % FRAME
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        frames = audio[:usable].reshape(-1, FRAME)
        rms = np.sqrt(np.mean(frames * frames, axis=1)) * 32768.0
        self.levels.extend(float(x) for x in rms)
        floor = max(float(np.percentile(self.levels, 15)), 120.0)
        # Map "well above the floor" onto a 0..1 score shaped like Silero's.
        return np.clip((rms / floor - 1.5) / 1.5, 0.0, 1.0).astype(np.float32)


_scorer = None


def scorer():
    """Shared detector. Loaded lazily so importing this module costs nothing."""
    global _scorer
    if _scorer is None:
        try:
            _scorer = _Silero()
            log.info("Voice activity detection: Silero")
        except Exception as exc:
            log.warning("Silero VAD unavailable (%s); using the energy detector", exc)
            _scorer = _Energy()
    return _scorer


def analyze(pcm: bytes) -> tuple[bool, float]:
    """Stateless check of a whole buffer: (heard speech?, trailing silence in s)."""
    probs = scorer().probs(_pcm_to_float(pcm))
    return _had_speech(probs), _trailing_silence(probs)


def speaking(pcm: bytes) -> bool:
    """Is there a voice anywhere in this (short) clip?"""
    probs = scorer().probs(_pcm_to_float(pcm))
    return bool(probs.size) and float(probs.max()) >= SPEECH_PROB


def _had_speech(probs: np.ndarray) -> bool:
    need = max(1, int(MIN_SPEECH_SECONDS / FRAME_SECONDS))
    run = 0
    for p in probs:
        run = run + 1 if p >= SPEECH_PROB else 0
        if run >= need:
            return True
    return False


def _trailing_silence(probs: np.ndarray) -> float:
    count = 0
    for p in probs[::-1]:
        if p >= SILENCE_PROB:
            break
        count += 1
    return count * FRAME_SECONDS


class Endpointer:
    """Feeds on a live stream and says when the question is over.

    check() answers one of:
      'continue'   still talking, or hasn't started yet
      'ended'      spoke, then went quiet for end_silence seconds
      'no_speech'  never said anything within no_speech_timeout
      'max'        hit the hard length limit
    """

    def __init__(
        self,
        end_silence: float = 1.0,
        no_speech_timeout: float = 8.0,
        max_seconds: float = 30.0,
    ) -> None:
        self.end_silence = end_silence
        self.no_speech_timeout = no_speech_timeout
        self.max_seconds = max_seconds
        self.buffer = bytearray()
        self.had_speech = False

    @property
    def seconds(self) -> float:
        return len(self.buffer) / 2 / SAMPLE_RATE

    def feed(self, pcm: bytes) -> None:
        self.buffer.extend(pcm)

    def check(self) -> str:
        if self.seconds >= self.max_seconds:
            return "max"
        window = bytes(self.buffer[-int(WINDOW_SECONDS * SAMPLE_RATE) * 2 :])
        probs = scorer().probs(_pcm_to_float(window))
        # Sticky: once they've spoken, sliding the window past it doesn't
        # un-hear it.
        if not self.had_speech and _had_speech(probs):
            self.had_speech = True
        if self.had_speech:
            return "ended" if _trailing_silence(probs) >= self.end_silence else "continue"
        return "no_speech" if self.seconds >= self.no_speech_timeout else "continue"

    def reset(self) -> None:
        self.buffer = bytearray()
        self.had_speech = False
