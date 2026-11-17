"""
synthetic.py - a fake Voyager picture recording with known faults.

Used by the tests to check that the decoder recovers what we put in. The
signal imitates the structure measured on the master-tape transfer:

* 120 lines per second; each line starts with a sharp sync spike followed by
  a flat pulse top that is short on even lines and long on odd lines
* sync pulses staggered early/late by `stagger` (fraction of a line) while
  the picture content sits on an even grid
* picture lines have dark framing bars at the start and end of the active
  window; lead-in/lead-out lines do not
* faults added: wow and flutter (a known time warp), first-order high-pass
  "droop" with time constant `tau`, and noise
"""

from __future__ import annotations

import numpy as np
import scipy.signal as ss


def make_recording(
    fs: int = 96000,
    lines_before: int = 100,
    lines_picture: int = 512,
    lines_after: int = 40,
    stagger: float = 0.016,
    wow: tuple[tuple[float, float], ...] = ((40e-6, 3.0), (5e-6, 15.0)),
    tau: float | None = 0.6e-3,
    noise: float = 0.002,
    seed: int = 1,
):
    """Return (signal, truth) where truth is a dict of what was put in.

    The picture content is horizontal stripes: a function of the position
    inside the line only, identical in every column. Any timing error in
    the decode therefore shows up as columns that differ from each other.
    """
    rng = np.random.default_rng(seed)
    line_s = 1.0 / 120.0
    n_lines = lines_before + lines_picture + lines_after
    over = 4  # build the ideal signal 4x oversampled, then warp-resample
    fso = fs * over
    P = line_s * fso
    total = int((n_lines + 2) * P)
    u_all = (np.arange(total) / P)  # line coordinate on the even grid
    k = np.floor(u_all).astype(int)
    u = u_all - k
    x = np.zeros(total)

    is_pic = (k >= lines_before) & (k < lines_before + lines_picture)
    stripes = 0.25 * np.sin(2 * np.pi * 6 * (u - 0.115) / 0.8)
    active = (u >= 0.115) & (u < 0.915)
    x[is_pic & active] = stripes[is_pic & active]
    bars = ((u >= 0.10) & (u < 0.115)) | ((u >= 0.915) & (u < 0.93))
    x[is_pic & bars] = -0.4
    x[(u >= 0.93) | (u < 0.10)] = 0.3

    # the recording chain is band-limited: soften every edge (~40 us) before
    # adding the sync spike, which on the real signal is by far the steepest edge
    x = ss.sosfiltfilt(ss.butter(2, 1.0 / (0.005 * P), output="sos"), x)

    # sync pulse: spike + flat top, staggered by +-stagger around the grid
    for line in range(n_lines + 1):
        s = stagger if line % 2 == 0 else -stagger
        t0 = (line + s) * P
        top = 0.047 if line % 2 else 0.014
        i0 = int(t0)
        x[i0 : i0 + int(top * P)] += 0.4
        x[i0 : i0 + int(0.002 * P)] += 0.6

    # wow and flutter: tape position runs ahead/behind real time
    t = np.arange(int(total / over)) / fs
    warp = sum(a * np.sin(2 * np.pi * f * t + 0.7 * i) for i, (a, f) in enumerate(wow))
    pos = (t - warp) * fso
    y = np.interp(pos, np.arange(total), ss.sosfiltfilt(ss.butter(6, 0.8 / over, output="sos"), x))

    if tau:
        a = 1.0 / (1.0 + 1.0 / (tau * fs))  # first-order high-pass, RC = tau
        y = ss.lfilter([a, -a], [1.0, -a], y)
    y = y + noise * rng.standard_normal(len(y))
    truth = dict(fs=fs, lines_before=lines_before, lines_picture=lines_picture, stagger=stagger,
                 tau=tau, warp_t=t, warp=warp, stripes_cycles=6)
    return y.astype(np.float32), truth
