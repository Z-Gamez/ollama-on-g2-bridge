"""Live captions: what the people around you are saying, on your lens.

Continuous transcription, split into lines at natural pauses. Each line shows
up as an interim guess while it's being spoken and is replaced by a proper
decode once the speaker pauses -- the same shape as live TV captions.

Everything runs on your own machine, so a conversation you caption is never
uploaded anywhere.
"""

from __future__ import annotations

import vad

SAMPLE_RATE = 16000
# A pause this long ends a caption line. Shorter than a question's end-of-speech
# because captions should keep up with the conversation, not wait for it.
LINE_PAUSE_SECONDS = 0.6
# Long monologues are cut into lines anyway so the lens keeps moving.
MAX_LINE_SECONDS = 9.0
# Silence before anyone speaks is dropped rather than kept for decoding.
IDLE_KEEP_SECONDS = 0.8
VAD_INTERVAL_SECONDS = 0.2


class CaptionSession:
    def __init__(self) -> None:
        self.endpointer = self._fresh()
        self.since_vad = 0.0

    @staticmethod
    def _fresh() -> vad.Endpointer:
        return vad.Endpointer(end_silence=LINE_PAUSE_SECONDS, no_speech_timeout=1e9, max_seconds=MAX_LINE_SECONDS)

    def add(self, pcm: bytes) -> dict:
        """{'action': 'none' | 'progress' | 'line', 'pcm': ...}"""
        ep = self.endpointer
        ep.feed(pcm)
        self.since_vad += len(pcm) / 2 / SAMPLE_RATE
        if self.since_vad < VAD_INTERVAL_SECONDS:
            return {"action": "progress" if ep.had_speech else "none", "pcm": bytes(ep.buffer)}
        self.since_vad = 0.0

        verdict = ep.check()
        if not ep.had_speech:
            # Nobody talking: keep only a short lead-in so the first word of the
            # next line isn't clipped, and never decode room tone.
            keep = int(IDLE_KEEP_SECONDS * SAMPLE_RATE) * 2
            if len(ep.buffer) > keep:
                del ep.buffer[: len(ep.buffer) - keep]
            return {"action": "none"}
        if verdict in ("ended", "max"):
            out = bytes(ep.buffer)
            self.endpointer = self._fresh()
            return {"action": "line", "pcm": out}
        return {"action": "progress", "pcm": bytes(ep.buffer)}
