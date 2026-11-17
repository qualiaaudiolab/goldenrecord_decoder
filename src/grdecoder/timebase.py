"""
timebase.py - turning measured sync times into a continuous time axis.

The picture is drawn as 512 vertical lines. To draw line k we need to know,
for every fraction u (0 = top, 1 = bottom of the line), *where in the file*
that point of the picture was recorded. That mapping

        t(q)   with  q = k + u   (a continuous "line coordinate")

is the timebase. A perfect transport would give t(q) = t0 + q * P. A real
tape machine adds wow and flutter, so P wanders; every sync pulse gives us
one measured point of t(q) at integer q, and this module decides how to
connect those points.

Modes
-----
fixed    t(q) = t0 + q * P_fixed. P_fixed is the mean measured period
         (automatic speed calibration) times a manual trim factor. This is
         what straightforward decoders do: any speed error shows up as
         slanted or wavy picture content.
line     Straight segments between consecutive measured syncs. Corrects all
         speed error down to one line, but passes sync-measurement noise
         straight into the picture.
smooth   (default) The measured timing error relative to constant speed is
         low-pass filtered (zero phase) at a chosen bandwidth and joined with
         a cubic spline. Removes wow and slow flutter while ignoring
         line-to-line measurement noise. With the bandwidth at its maximum it
         behaves like `line` but with smooth curvature inside each line.

Stagger
-------
Sync pulses alternate early/late by a fixed amount (see sync.py). Before any
of the above we move every pulse back onto the even grid. The amount and
sign are estimated per picture from the mean of even and odd intervals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal as ss
from scipy.interpolate import CubicSpline


@dataclass
class Destaggered:
    """Sync times moved onto an even grid.

    times     corrected times (samples)
    stagger   half the difference between long and short intervals; even-
              indexed pulses were moved by +stagger, odd ones by -stagger
    """

    times: np.ndarray
    stagger: float


def destagger(times: np.ndarray) -> Destaggered:
    """Remove the alternating early/late offset of the sync pulses.

    With pulses at  t_k = g_k + s * (-1)^k  (g_k the even grid), intervals
    alternate between P - 2s and P + 2s, so  s = (mean even interval -
    mean odd interval) / 4. Using means over the whole picture makes this
    insensitive to wow.
    """
    d = np.diff(times)
    if len(d) < 4:
        return Destaggered(times.copy(), 0.0)
    s = (np.mean(d[0::2]) - np.mean(d[1::2])) / 4.0
    sign = np.where(np.arange(len(times)) % 2 == 0, 1.0, -1.0)
    return Destaggered(times + sign * s, float(s))


@dataclass
class Timebase:
    """Continuous mapping from line coordinate q to file position t(q).

    Attributes useful for diagnostics:
      q_sync      integer line index of each sync (0 = first sync in the run)
      t_sync      destaggered sync times
      period      mean line period from a straight-line fit (samples)
      error_raw   measured timing error vs constant speed (samples)
      error_used  timing error actually applied by the chosen mode
    """

    mode: str
    q_sync: np.ndarray
    t_sync: np.ndarray
    period: float
    t0: float
    error_raw: np.ndarray
    error_used: np.ndarray
    _spline: CubicSpline | None = None
    _fixed_period: float | None = None

    def __call__(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if self.mode == "fixed":
            return self.t0 + q * self._fixed_period
        if self.mode == "line":
            return np.interp(q, self.q_sync, self.t_sync,
                             left=None, right=None) + _extrapolate(q, self.q_sync, self.t_sync)
        # smooth
        return self.t0 + q * self.period + self._spline(q)

    def speed_ppm(self) -> np.ndarray:
        """Instantaneous speed deviation per line, in ppm, from the error used.

        A positive value means the tape ran fast (lines shorter than mean).
        """
        dt = np.diff(self.t0 + self.q_sync * self.period + self.error_used)
        return -(dt / self.period - 1.0) * 1e6


def _extrapolate(q, qs, ts):
    """Linear extension beyond the first/last sync for mode 'line'."""
    out = np.zeros_like(q)
    p0 = (ts[1] - ts[0]) / (qs[1] - qs[0])
    p1 = (ts[-1] - ts[-2]) / (qs[-1] - qs[-2])
    lo = q < qs[0]
    hi = q > qs[-1]
    out[lo] = (q[lo] - qs[0]) * p0
    out[hi] = (q[hi] - qs[-1]) * p1
    return out


def build_timebase(
    t_sync: np.ndarray,
    measured: np.ndarray,
    mode: str = "smooth",
    smooth_hz: float = 30.0,
    fs: float = 384000.0,
    speed_trim: float = 1.0,
) -> Timebase:
    """Create the timebase for one run of destaggered sync times.

    t_sync      destaggered sync positions (samples)
    measured    False where a sync was predicted, not measured (dropout)
    smooth_hz   bandwidth of the timing correction in mode 'smooth', in Hz
                of real time (the line rate is ~120 Hz, so the largest
                meaningful value is ~60 Hz)
    speed_trim  multiplies the playback speed assumed in mode 'fixed'
                (1.001 = file plays 0.1 % fast -> lines 0.1 % shorter)
    """
    q = np.arange(len(t_sync), dtype=np.float64)

    # constant-speed reference: robust straight-line fit through measured syncs
    w = measured.astype(bool)
    slope, icpt = np.polyfit(q[w], t_sync[w], 1)
    for _ in range(3):  # drop gross outliers (mis-detected pulses) and refit
        r = t_sync - (icpt + slope * q)
        mad = 1.4826 * np.median(np.abs(r[w] - np.median(r[w]))) + 1e-9
        w = measured.astype(bool) & (np.abs(r) < 8 * mad + 2.0)
        slope, icpt = np.polyfit(q[w], t_sync[w], 1)
    err = t_sync - (icpt + slope * q)

    if mode == "fixed":
        return Timebase(mode, q, t_sync, slope, icpt, err, np.zeros_like(err),
                        _fixed_period=slope / speed_trim)

    # fill unreliable points (dropouts, outliers) by interpolation of reliable ones
    e = err.copy()
    if (~w).any():
        e[~w] = np.interp(q[~w], q[w], err[w])

    if mode == "line":
        t_used = icpt + slope * q + e
        return Timebase(mode, q, t_used, slope, icpt, err, e)

    # mode 'smooth': zero-phase low-pass of the error sequence (sampled once per line)
    line_rate = fs / slope
    wn = smooth_hz / (line_rate / 2.0)
    if wn < 0.98 and len(e) > 30:
        sos = ss.butter(2, wn, output="sos")
        e_s = ss.sosfiltfilt(sos, e, padtype="odd", padlen=min(len(e) - 1, 60))
    else:
        e_s = e
    spline = CubicSpline(q, e_s, bc_type="natural", extrapolate=True)
    return Timebase(mode, q, icpt + slope * q + e_s, slope, icpt, err, e_s, _spline=spline)
