"""
audio.py - random-access reading of large WAV files.

The master-tape transfer of the Voyager image data is ~1.5 GB (384 kHz,
stereo, 32-bit float). Loading it whole is possible on a well-equipped
machine but unnecessary: every stage of the decoder works on short windows
(a block during analysis, one picture ~4.3 s during rendering). This module
wraps `soundfile` so the rest of the code can ask for "channel c, samples
a..b" without caring about the file format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioInfo:
    """Basic facts about the opened file."""

    path: Path
    samplerate: int
    channels: int
    frames: int
    subtype: str

    @property
    def duration(self) -> float:
        """Length in seconds."""
        return self.frames / self.samplerate


class AudioSource:
    """Random-access reader for one audio file.

    The file handle is opened per call, so an AudioSource can be shared
    between the GUI thread and a worker thread without locking.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with sf.SoundFile(str(self.path)) as f:
            self.info = AudioInfo(
                path=self.path,
                samplerate=f.samplerate,
                channels=f.channels,
                frames=f.frames,
                subtype=f.subtype,
            )

    @property
    def fs(self) -> int:
        """Sample rate of the file in Hz."""
        return self.info.samplerate

    def read(self, channel: int, start: int, stop: int) -> np.ndarray:
        """Return samples [start, stop) of one channel as float64.

        Requests that run past either end of the file are zero-padded, so
        callers can ask for a window around a sync pulse near the file edge
        without special cases.
        """
        start, stop = int(start), int(stop)
        n = stop - start
        out = np.zeros(n, dtype=np.float64)
        a = max(start, 0)
        b = min(stop, self.info.frames)
        if b <= a:
            return out
        with sf.SoundFile(str(self.path)) as f:
            f.seek(a)
            data = f.read(b - a, dtype="float64", always_2d=True)
        out[a - start : a - start + len(data)] = data[:, channel]
        return out

    def blocks(self, channel: int, block: int, overlap: int):
        """Yield (start_index, samples) covering the whole file.

        Consecutive blocks overlap by `overlap` samples so that a pulse
        sitting on a block boundary is always seen whole in one of them.
        """
        step = block - overlap
        with sf.SoundFile(str(self.path)) as f:
            pos = 0
            while pos < self.info.frames:
                f.seek(pos)
                data = f.read(block, dtype="float64", always_2d=True)
                if len(data) == 0:
                    break
                yield pos, data[:, channel]
                pos += step
