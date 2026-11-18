"""
sync.py - finding the picture-line sync pulses with sub-sample precision.

Signal facts this module relies on (measured on the 384 kHz master-tape
transfer, see docs/HOW_IT_WORKS.md):

* Every picture line starts with a short positive sync pulse: on the master
  tape it is the steepest rising edge in the signal.
* Lines repeat about every 8.33 ms, but sync pulses are NOT evenly spaced:
  they alternate early/late by about 1.6 % of a line ("staggered
  interlace"). Consecutive intervals therefore alternate between roughly
  P + 2d and P - 2d, while the picture content itself sits on an even grid.
  `timebase.py` removes the stagger; here the tracker only tolerates it.

The detector has to work on any copy of the recording: the 384 kHz master
transfer, a 44.1 kHz file, a band-limited or noisier source. On a
band-limited copy the sync pulse is no longer clearly the steepest edge, so
the method does not rely on that. It works in three steps:

1. `find_candidates` + `estimate_period` - rising edges, and the most common
   line rhythm among them. Only used to get the line period roughly.
2. `aligned_fold` + `template_from_fold` - take a stretch of ~80 lines,
   align consecutive line pairs to each other by cross-correlating whole
   pairs (neighbouring lines of a picture look alike), and average them.
   Picture content averages away, the sync structure that repeats every two
   lines remains, sharp. The template is cut from that average, around its
   steepest rising edge.
3. `detect` - normalised cross-correlation of the whole signal with the
   template (a matched filter). Every local maximum above `min_score` is a
   sync pulse, located to a fraction of a sample by parabolic interpolation.

Then `track_lines` follows the line rhythm through the detections, bridges
dropouts and splits the recording into runs of consecutive lines.

Tuning knobs (SyncSettings in pipeline.py, "Sync detection" in the app):
the template width, the minimum match score and the edge scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal as ss


# --------------------------------------------------------------------------
# 1. rough line period
# --------------------------------------------------------------------------
def rising_edges(x: np.ndarray, fs: float, scale_us: float = 20.0) -> np.ndarray:
    """Rise of the signal over `scale_us` microseconds, negative values set to 0.

    Measuring the slope over a fixed time (not over one sample) makes the
    detector behave the same at any sample rate and on band-limited copies.
    """
    L = max(1, int(round(fs * scale_us * 1e-6)))
    d = np.zeros_like(x)
    d[:-L] = x[L:] - x[:-L]
    return np.maximum(d, 0.0)


def find_candidates(x: np.ndarray, period: float, fs: float, scale_us: float = 20.0,
                    rel_threshold: float = 0.3) -> np.ndarray:
    """Indices of strong rising edges (used only to estimate the line period).

    Accepts every edge rising at least `rel_threshold` times as fast as the
    typical fastest edge of a line-long window.
    """
    d = rising_edges(x, fs, scale_us)
    n_win = len(d) // int(period)
    if n_win < 4:
        return np.empty(0, dtype=np.int64)
    win_max = d[: n_win * int(period)].reshape(n_win, int(period)).max(axis=1)
    height = rel_threshold * np.median(win_max)
    if height <= 0:
        return np.empty(0, dtype=np.int64)
    peaks, _ = ss.find_peaks(d, height=height, distance=max(2, int(0.05 * period)))
    return peaks.astype(np.int64)


def two_line_distances(times: np.ndarray, guess: float) -> np.ndarray:
    """For every candidate, the distance to the candidate closest to two lines later.

    Over two lines the sync stagger cancels, so these distances cluster
    tightly at 2 x the line period, whatever the stagger.
    """
    t = np.sort(np.asarray(times, dtype=float))
    lo = np.searchsorted(t, t + 1.6 * guess)
    hi = np.searchsorted(t, t + 2.4 * guess)
    d = []
    for i in np.flatnonzero(hi > lo):
        cand = t[lo[i] : hi[i]] - t[i]
        d.append(cand[np.argmin(np.abs(cand - 2 * guess))])
    return np.asarray(d)


def estimate_period(times: np.ndarray, guess: float, n_best: int = 1):
    """Line period(s) of the most common line rhythms among the candidates.

    The recording also contains lead-in signals with a slightly different
    rhythm, so we histogram the two-line distances (0.5-sample bins) and
    return the `n_best` strongest, clearly separated peaks, each refined by
    the median of the distances within +-0.2 % of it and halved. With
    n_best=1 a single float is returned.
    """
    d = two_line_distances(times, guess)
    if len(d) < 8:
        raise ValueError("Could not find a regular line rhythm near the expected period.")
    bins = np.arange(d.min(), d.max() + 1.0, 0.5)
    if len(bins) < 3:
        out = [float(np.median(d) / 2)]
    else:
        h, e = np.histogram(d, bins=bins)
        centres = 0.5 * (e[:-1] + e[1:])
        out = []
        for i in np.argsort(h)[::-1]:
            c = centres[i]
            if h[i] < 8 or any(abs(c / (2 * o) - 1) < 0.003 for o in out):
                continue
            near = d[np.abs(d / c - 1) < 0.002]
            out.append(float(np.median(near) / 2))
            if len(out) >= n_best:
                break
    return out[0] if n_best == 1 else out


# --------------------------------------------------------------------------
# 2. template from an aligned fold
# --------------------------------------------------------------------------
def aligned_fold(x: np.ndarray, start: int, period: float, n_pairs: int = 40,
                 search: float = 0.03) -> np.ndarray | None:
    """Average of `n_pairs` consecutive two-line segments, each aligned to the last.

    Every segment is placed where the previous one ended plus two line
    periods, then shifted by up to +-`search` of a pair to best match the
    previous segment (FFT cross-correlation). Because the alignment follows
    the signal, speed drift of the tape does not blur the average.
    Returns the averaged segment (length 2 x period) or None if it does not fit.
    """
    L2 = int(round(2 * period))
    m = int(search * L2)
    if start < 0 or start + n_pairs * (L2 + m) + m >= len(x):
        return None
    ref = x[start : start + L2] - x[start : start + L2].mean()
    acc = ref.copy()
    pos = start
    for _ in range(n_pairs - 1):
        a = pos + L2 - m
        seg = x[a : a + L2 + 2 * m]
        seg = seg - seg.mean()
        c = ss.correlate(seg, ref, mode="valid", method="fft")
        pos = a + int(np.argmax(c))
        cur = x[pos : pos + L2]
        cur = cur - cur.mean()
        acc += cur
        ref = cur
    return acc / n_pairs


@dataclass
class FoldInfo:
    """What an aligned fold reveals about a stretch of signal."""

    profile: np.ndarray
    i1: int            # index of the steepest rising edge in the profile
    i2: int | None     # matching edge of the other line of the pair, if staggered
    sharpness: float   # steepest edge relative to the typical slope


def inspect_fold(profile: np.ndarray, period: float, fs: float, scale_us: float = 20.0) -> FoldInfo:
    """Find the sync edges in an aligned fold and check for the stagger.

    In a picture the two-line profile shows two sync edges about one line
    apart, but 2-5 % more or less than a full line (the stagger measures
    ~3.1 %). Lead-in signals show no such pair.
    """
    d = rising_edges(np.r_[profile, profile[:64]], fs, scale_us)[: len(profile)]
    i1 = int(np.argmax(d))
    sharp = float(d[i1] / (np.mean(d) + 1e-12))
    n = len(profile)
    best, i2 = 0.0, None
    for sign in (+1, -1):
        lo, hi = sorted((i1 + period * (1 + sign * 0.02), i1 + period * (1 + sign * 0.05)))
        idx = np.arange(int(lo), int(hi) + 1) % n
        j = int(idx[np.argmax(d[idx])])
        if d[j] > best:
            best, i2 = d[j], j
    if best < 0.4 * d[i1]:
        i2 = None
    return FoldInfo(profile, i1, i2, sharp)


@dataclass
class SyncTemplate:
    """Sync-pulse waveforms used for matched filtering.

    `shapes` holds one waveform per kind of line (the two kinds of sync pulse
    differ after the first ~1 % of a line), each `pre + post` samples long
    with the reference point - the steepest rise - at index `pre`, so the
    "time of a sync pulse" means the same thing for both kinds, on every
    channel and every copy.
    """

    shapes: list
    pre: int
    post: int

    @property
    def length(self) -> int:
        return self.pre + self.post

    @property
    def shape(self) -> np.ndarray:  # the first waveform, for display
        return self.shapes[0]


def template_from_fold(info: FoldInfo, period: float, width: float = 0.04, fs: float = 384000.0,
                       min_us: float = 60.0) -> SyncTemplate:
    """Cut the template(s) out of the fold around the sync edge(s).

    The window starts 1 % of a line before the rise and runs `width` of a
    line after it, so it includes the flat pulse top whose length tells the
    two kinds of line apart. That makes the template distinctive even on a
    band-limited copy, where a short template would also match ordinary
    picture edges. With a staggered pair, one template per kind of line is
    made; detection uses whichever matches better.
    """
    pre = max(int(round(0.01 * period)), int(round(min_us * 1e-6 * fs)), 4)
    post = max(int(round(width * period)), int(round(min_us * 1e-6 * fs)), 4)
    prof = np.r_[info.profile, info.profile, info.profile]
    n = len(info.profile)
    shapes = []
    for i in ([info.i1, info.i2] if info.i2 is not None else [info.i1]):
        seg = prof[n + i - pre : n + i + post].copy()
        seg -= seg.mean()
        seg /= np.linalg.norm(seg) + 1e-30
        shapes.append(seg)
    return SyncTemplate(shapes=shapes, pre=pre, post=post)


# --------------------------------------------------------------------------
# 3. matched-filter detection
# --------------------------------------------------------------------------
def match(x: np.ndarray, tpl: SyncTemplate) -> np.ndarray:
    """Normalised cross-correlation (-1..1) of `x` with the best-fitting template.

    r[k] compares the template with x[k : k + length]; the reference point
    of a match at k is k + tpl.pre.
    """
    L = tpl.length
    s1 = np.concatenate(([0.0], np.cumsum(x)))
    s2 = np.concatenate(([0.0], np.cumsum(x * x)))
    e = np.sqrt(np.maximum((s2[L:] - s2[:-L]) - (s1[L:] - s1[:-L]) ** 2 / L, 1e-30))
    r = None
    for shape in tpl.shapes:
        ri = ss.oaconvolve(x, shape[::-1], mode="valid") / e
        r = ri if r is None else np.maximum(r, ri)
    return r


def detect(x: np.ndarray, tpl: SyncTemplate, period: float, min_score: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Sync pulses in `x`: (positions, scores).

    Every local maximum of the match above `min_score` is returned (only
    maxima closer than 2 % of a line are merged). Several may fall within one
    line - on a band-limited copy a picture edge can match nearly as well as
    the sync pulse - and the line tracker picks the one that fits the line
    rhythm. Each position is refined with a parabola through the three
    highest points.
    """
    if len(x) < 2 * tpl.length:
        return np.empty(0), np.empty(0)
    r = match(x, tpl)
    k, _ = ss.find_peaks(r, height=min_score, distance=max(1, int(0.02 * period)))
    k = k[(k > 0) & (k < len(r) - 1)]
    # keep at most the 3 best peaks in every half-line stretch
    if len(k):
        bucket = (k // max(1, int(0.5 * period))).astype(np.int64)
        order = np.lexsort((-r[k], bucket))
        rank = np.zeros(len(k), int)
        b_sorted = bucket[order]
        start = np.r_[0, np.flatnonzero(np.diff(b_sorted)) + 1]
        for a, b in zip(start, np.r_[start[1:], len(order)]):
            rank[order[a:b]] = np.arange(b - a)
        k = np.sort(k[rank < 3])
    y0, y1, y2 = r[k - 1], r[k], r[k + 1]
    den = y0 - 2 * y1 + y2
    frac = np.where(den < 0, 0.5 * (y0 - y2) / np.where(den < 0, den, -1), 0.0)
    return k + frac + tpl.pre, y1


# --------------------------------------------------------------------------
# 4. line tracking
# --------------------------------------------------------------------------
@dataclass
class LineTrack:
    """One uninterrupted run of picture lines on one channel.

    times    sub-sample sync positions (absolute sample index in the file)
    measured True where a real pulse was found, False where the position
             was predicted to bridge a dropout
    score    template similarity of each pulse (0 for predicted ones)
    """

    channel: int
    times: np.ndarray
    measured: np.ndarray
    score: np.ndarray
    period: float  # mean line period in samples
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.times)


def track_lines(
    times: np.ndarray,
    scores: np.ndarray,
    period: float,
    min_score: float = 0.6,
    tol: float = 0.012,
    tol_start: float = 0.05,
    max_gap: int = 4,
    min_lines: int = 64,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Group sync pulses into runs of consecutive picture lines.

    A run is grown slot by slot from a seed pulse:

    * Expected position of the next pulse = last pulse + the interval two
      steps back. Because of the sync stagger, intervals alternate long/short,
      and the interval two back has the same parity as the one we predict.
      For the first two steps no such history exists, so the mean period is
      used with a wider search window (`tol_start`, in lines).
    * Inside +-`tol` lines around the prediction, the best-scoring unused
      candidate is taken. If none qualifies the slot is filled with the
      prediction itself and flagged as not measured (tape dropout).
    * A run ends after `max_gap` predicted slots in a row.

    Every candidate seeds a run, so overlapping runs are produced (e.g. one
    seeded by a picture-content transient). They are resolved at the end:
    runs are ranked by how many real pulses they contain and accepted
    greedily if they do not overlap an already accepted run.

    Returns a list of (times, measured, score) arrays, in time order.
    """
    order = np.argsort(times)
    t = times[order]
    s = scores[order]
    good = s >= min_score
    t, s = t[good], s[good]
    used = np.zeros(len(t), bool)
    runs = []

    for i0 in range(len(t)):
        if used[i0]:
            continue
        run_t, run_m, run_s = [t[i0]], [True], [s[i0]]
        taken = [i0]
        misses = 0
        while True:
            if len(run_t) >= 3:
                step = run_t[-2] - run_t[-3]
                win = tol * period
            else:
                step = period
                win = tol_start * period
            pred = run_t[-1] + step
            if pred > t[-1] + period:
                break
            lo = np.searchsorted(t, pred - win)
            hi = np.searchsorted(t, pred + win)
            j = -1
            if hi > lo:
                cand = np.arange(lo, hi)
                cand = cand[~used[cand]]
                if len(cand):
                    j = int(cand[np.argmax(s[cand])])
            if j >= 0:
                run_t.append(t[j]); run_m.append(True); run_s.append(s[j])
                taken.append(j)
                misses = 0
            else:
                run_t.append(pred); run_m.append(False); run_s.append(0.0)
                misses += 1
                if misses > max_gap:
                    break
        if misses:  # trailing predictions belong to no picture
            del run_t[-misses:], run_m[-misses:], run_s[-misses:]
        if len(run_t) >= min_lines and np.mean(run_m) > 0.8:
            runs.append((np.array(run_t), np.array(run_m), np.array(run_s)))
            used[taken] = True

    # resolve overlaps. Picture runs show the sync stagger (intervals
    # alternating long/short by ~3 %); a run that followed a picture edge
    # instead does not, because picture content sits on an even grid. So
    # staggered runs win, then the one with the most measured pulses.
    def staggered(r):
        d = np.diff(r[0])
        return len(d) > 8 and abs(np.mean(d[0::2]) - np.mean(d[1::2])) / 4 > 0.005 * period

    runs.sort(key=lambda r: (not staggered(r), -int(r[1].sum())))
    accepted = []
    for r in runs:
        a, b = r[0][0], r[0][-1]
        clash = False
        for q in accepted:
            ov = min(b, q[0][-1]) - max(a, q[0][0])
            if ov > 0.1 * min(b - a, q[0][-1] - q[0][0]):
                clash = True
                break
        if not clash:
            accepted.append(r)
    accepted.sort(key=lambda r: r[0][0])
    return accepted


def merge_runs(runs: list, period: float, max_gap_lines: int = 60, tol: float = 0.06, max_lines: int = 760) -> list:
    """Join runs that are separated by a short, whole number of missing lines.

    A longer tape dropout can break a picture into two runs. If the second
    run starts an (almost) whole number of line periods after the first
    ends, the missing slots are filled with predicted positions and the two
    runs become one. `runs` are (times, measured, score) tuples in time order.

    The sync stagger must also continue across the gap: consecutive pictures
    are often only ~12 lines apart, but the stagger restarts with each
    picture, so its early/late pattern tells a broken picture from two
    neighbouring ones. As a second guard, the merged run may not be longer
    than `max_lines` (one picture with its framing is ~640-720 lines).
    """
    def stagger(t):
        d = np.diff(t)
        return (np.mean(d[0::2]) - np.mean(d[1::2])) / 4 if len(d) > 4 else 0.0

    if not runs:
        return runs
    out = [runs[0]]
    for t, m, s in runs[1:]:
        t0, m0, s0 = out[-1]
        gap = (t[0] - t0[-1]) / period
        k = int(round(gap))
        offset = len(t0) + k - 1  # index the first pulse of the second run would get
        s_a, s_b = stagger(t0), stagger(t) * (-1) ** offset
        same_pattern = abs(s_a) < 0.005 * period or abs(s_b) < 0.005 * period or np.sign(s_a) == np.sign(s_b)
        fits = len(t0) + k - 1 + len(t) <= max_lines
        if 1 <= k <= max_gap_lines and abs(gap - k) < tol and same_pattern and fits:
            fill = t0[-1] + (t[0] - t0[-1]) * np.arange(1, k) / k
            out[-1] = (np.r_[t0, fill, t], np.r_[m0, np.zeros(k - 1, bool), m], np.r_[s0, np.zeros(k - 1), s])
        else:
            out.append((t, m, s))
    return out
