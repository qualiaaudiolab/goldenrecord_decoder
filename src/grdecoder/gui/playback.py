"""
playback.py - play the recording while the picture builds up on screen.

The audio runs through PortAudio (the `sounddevice` package) at a rate the
sound card accepts (48 kHz preferred; the 384 kHz file is resampled on the
fly). Three parts:

* a feeder thread reads the file ahead in ~50 ms blocks, resamples them and
  puts them in a queue;
* the audio callback (PortAudio's thread) copies queued audio to the
  sound card and counts the frames it has handed over;
* the GUI asks `position()` ~30 times a second: frames played, minus the
  output latency, converted back to a time in the file. That is the audio
  clock the picture follows, so sound and image stay together.

If no sound device can be opened (or `sounddevice` is missing), playback
still runs silently on a wall-clock timer, so the visual part always works.
"""

from __future__ import annotations

import queue
import threading
import time
from fractions import Fraction

import numpy as np
import scipy.signal as ss

from ..audio import AudioSource

try:  # PortAudio ships inside the Windows/macOS wheels of sounddevice
    import sounddevice as sd
except Exception:  # library missing or no PortAudio on this system
    sd = None


class StreamResampler:
    """Polyphase resampling of a stream delivered in blocks.

    `scipy.signal.resample_poly` works on whole arrays; to use it on a stream
    without clicks at block boundaries, every block is processed together
    with `pad` input samples of context on both sides, and only the middle
    part of the output (which no boundary can affect) is kept. The output
    therefore lags the input by `pad` samples - irrelevant here.
    """

    def __init__(self, fs_in: int, fs_out: int, channels: int):
        r = Fraction(fs_out, fs_in).limit_denominator(4000)
        self.up, self.down = r.numerator, r.denominator
        self.pad = self.down * max(1, int(np.ceil(120 / self.down)))  # >= filter half-length
        self.buf = np.zeros((2 * self.pad, channels))

    def process(self, block: np.ndarray) -> np.ndarray:
        """block: (n, channels) with n a multiple of `down`. Returns resampled output."""
        x = np.concatenate([self.buf, block])
        y = ss.resample_poly(x, self.up, self.down, axis=0)
        a = self.pad * self.up // self.down
        b = a + len(block) * self.up // self.down
        self.buf = x[-2 * self.pad :]
        return y[a:b]


class Player:
    """Plays a stretch of the file and reports where playback is."""

    def __init__(self, source: AudioSource):
        self.source = source
        self.fs_in = source.fs
        self.volume = 0.5
        self.audio_ok = False
        self.error = ""
        self._stream = None
        self._thread = None
        self._stop = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=12)
        self._pending = np.zeros((0, 2), dtype=np.float32)
        self._frames_played = 0
        self._t0 = 0.0
        self._wall0 = 0.0
        self.fs_out = 48000

    # ---------------------------------------------------------------- control
    def start(self, t0: float) -> None:
        """Start playing at file time `t0` (seconds)."""
        self.stop()
        self._t0 = max(0.0, t0)
        self._stop.clear()
        self._q = queue.Queue(maxsize=12)
        self._pending = np.zeros((0, 2), dtype=np.float32)
        self._frames_played = 0
        self._wall0 = time.perf_counter()
        self.audio_ok = False
        if sd is not None:
            try:
                self.fs_out = self._pick_rate()
                self._stream = sd.OutputStream(samplerate=self.fs_out, channels=2, dtype="float32",
                                               callback=self._callback, blocksize=0, latency="high")
                self.audio_ok = True
            except Exception as ex:  # no device, unsupported rate, ...
                self.error = str(ex)
                self._stream = None
        else:
            self.error = "sounddevice / PortAudio not available"
        if self.audio_ok:
            self._thread = threading.Thread(target=self._feed, daemon=True)
            self._thread.start()
            self._stream.start()

    def stop(self) -> None:
        """Stop playback and release the sound device."""
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def position(self) -> float:
        """Current playback position in the file, in seconds."""
        if self.audio_ok and self._stream is not None:
            lat = self._stream.latency if np.isscalar(self._stream.latency) else self._stream.latency[0]
            return self._t0 + max(0.0, self._frames_played / self.fs_out - lat)
        return self._t0 + (time.perf_counter() - self._wall0)

    # -------------------------------------------------------------- internals
    def _pick_rate(self) -> int:
        """48 kHz if the default output accepts it, else the device's own rate."""
        for rate in (48000, 44100):
            try:
                sd.check_output_settings(samplerate=rate, channels=2, dtype="float32")
                return rate
            except Exception:
                continue
        return int(sd.query_devices(kind="output")["default_samplerate"])

    def _feed(self) -> None:
        """Feeder thread: read, resample, queue."""
        info = self.source.info
        rs = StreamResampler(self.fs_in, self.fs_out, 2)
        n = rs.down * max(1, int(0.05 * self.fs_in) // rs.down)  # ~50 ms, multiple of `down`
        pos = int(self._t0 * self.fs_in)
        while not self._stop.is_set():
            if pos >= info.frames:
                self._q.put(None)  # end of file
                return
            chans = [self.source.read(c, pos, pos + n) for c in range(min(2, info.channels))]
            block = np.stack(chans if len(chans) == 2 else chans * 2, axis=1)
            out = rs.process(block).astype(np.float32)
            pos += n
            while not self._stop.is_set():
                try:
                    self._q.put(out, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _callback(self, outdata, frames, time_info, status) -> None:
        """PortAudio thread: hand queued audio to the sound card."""
        while len(self._pending) < frames:
            try:
                blk = self._q.get_nowait()
            except queue.Empty:
                break
            if blk is None:
                break
            self._pending = np.concatenate([self._pending, blk])
        k = min(frames, len(self._pending))
        outdata[:k] = self._pending[:k] * self.volume
        outdata[k:] = 0
        self._pending = self._pending[k:]
        self._frames_played += frames
