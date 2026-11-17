"""
sync.py - finding the picture-line sync pulses with sub-sample precision.

Signal facts this module relies on (measured on the 384 kHz master-tape
transfer, see docs/HOW_IT_WORKS.md):

* Every picture line starts with a short, sharp positive pulse. It has the
  steepest slope anywhere in the signal, so the absolute first difference
  |x[n+1] - x[n]| shows it as a tall, narrow spike.
* Lines repeat about every 8.33 ms, but sync pulses are NOT evenly spaced:
  they alternate early/late by about 1.6 % of a line ("staggered interlace"
  on the record cover). Consecutive intervals therefore alternate between
  roughly P + 2d and P - 2d, while the picture content itself sits on an
  even grid. `timebase.py` removes the stagger; here we only have to make
  sure the tracker tolerates it.

Processing chain
----------------
1. `find_candidates`   - every steep transient that could be a sync pulse.
2. `build_template`    - the median shape of the strongest, regularly spaced
                         candidates = what a sync pulse looks like on THIS
                         file and channel (shape differs between channels).
3. `refine`            - cross-correlate each candidate with the template;
                         gives a sub-sample position and a 0..1 similarity
                         score that rejects picture-content transients.
4. `track_lines`       - a flywheel: follow the line rhythm through the
                         candidates, pick the best-scoring candidate in each
                         expected slot, and bridge short gaps (dropouts) with
                         predicted positions that are flagged as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.signal as ss


# --------------------------------------------------------------------------
# 1. candidates
# --------------------------------------------------------------------------
def find_candidates(x: np.ndarray, period: float, rel_threshold: float = 0.3) -> np.ndarray:
    """Return integer indices of steep transients in `x`.

    The threshold adapts to the block: we take the largest slope in every
    line-length window (in a picture that is nearly always the sync pulse),
    use the median of those maxima as "typical sync height", and accept
    everything above `rel_threshold` times that. Candidates only need to be
    a little apart (5 % of a line) so that a large picture-content edge
    cannot hide a nearby sync pulse; the template score sorts them out later.
    """
    slope = np.abs(np.diff(x))
    n_win = len(slope) // int(period)
    if n_win < 4:
        return np.empty(0, dtype=np.int64)
    win_max = slope[: n_win * int(period)].reshape(n_win, int(period)).max(axis=1)
    height = rel_threshold * np.median(win_max)
    peaks, _ = ss.find_peaks(slope, height=height, distance=max(2, int(0.05 * period)))
    return peaks.astype(np.int64)


# --------------------------------------------------------------------------
# 2. template
# --------------------------------------------------------------------------
@dataclass
class SyncTemplate:
    """Median sync-pulse waveform for one channel.

    `pre` samples precede the reference point and `post` follow it; the
    reference point is where the template's own steepest rise is, so the
    "time of a sync pulse" means the same thing on every channel.
    """

    shape: np.ndarray
    pre: int
    post: int

    @property
    def length(self) -> int:
        return self.pre + self.post


def template_extent(period: float) -> tuple[int, int]:
    """Window around a sync pulse used for matching: -1 %..+1 % of a line.

    Kept deliberately short: ~1.2 % of a line after the leading spike the
    two kinds of sync pulse diverge (the blanking plateau that follows is
    short on one line and long on the next), and the template must fit both.
    """
    return int(round(0.01 * period)), int(round(0.01 * period))


def build_template(x: np.ndarray, candidates: np.ndarray, period: float, iterations: int = 3) -> SyncTemplate:
    """Estimate the sync pulse shape from candidates that recur once per line.

    Only candidates followed by another candidate roughly one line later are
    used (picture-content edges rarely are). The shape is the sample-wise
    median, re-aligned a few times by cross-correlation so that small
    alignment errors of the first pass do not blur it.
    """
    pre, post = template_extent(period)
    c = np.sort(np.asarray(candidates))
    # keep only the steepest candidates: sync pulses have the steepest edge
    # of a line, so their slope is close to the typical per-line maximum,
    # while framing and picture edges are usually well below it
    slope = np.abs(np.diff(x))
    c = c[c < len(slope)]
    n_win = len(slope) // int(period)
    typical = np.median(slope[: n_win * int(period)].reshape(n_win, int(period)).max(axis=1))
    c = c[slope[c] >= 0.7 * typical]
    if len(c) > 2:
        d = np.diff(c)
        regular = np.zeros(len(c), bool)
        regular[:-1] |= np.abs(d / period - 1) < 0.08
        c = c[regular]
    c = c[(c - pre - 64 >= 0) & (c + post + 64 < len(x))]
    if len(c) < 8:
        raise ValueError("Too few regular sync candidates to build a template.")
    if len(c) > 2000:  # plenty for a median; keep it fast
        c = c[np.linspace(0, len(c) - 1, 2000).astype(int)]

    pos = c.astype(float)
    margin = 16
    for _ in range(iterations):
        # median shape of the current alignment ...
        segs = np.array([_frac_window(x, p, pre, post) for p in pos])
        segs -= segs.mean(axis=1, keepdims=True)
        shape = np.median(segs, axis=0)
        # ... then align every example to it
        lag = np.array([_xcorr_peak(_frac_window(x, p, pre + margin, post + margin), shape)[0] for p in pos])
        pos = pos + lag - margin
    # final shape; put the reference point on its steepest rise
    segs = np.array([_frac_window(x, p, pre, post) for p in pos])
    segs -= segs.mean(axis=1, keepdims=True)
    shape = np.median(segs, axis=0)
    pos = pos + (int(np.argmax(np.diff(shape))) - pre)
    segs = np.array([_frac_window(x, p, pre, post) for p in pos])
    segs -= segs.mean(axis=1, keepdims=True)
    shape = np.median(segs, axis=0)
    shape /= np.linalg.norm(shape) + 1e-30
    return SyncTemplate(shape=shape, pre=pre, post=post)


# --------------------------------------------------------------------------
# 3. refinement
# --------------------------------------------------------------------------
def refine(x: np.ndarray, candidates: np.ndarray, tpl: SyncTemplate, search: int) -> tuple[np.ndarray, np.ndarray]:
    """Sub-sample position and similarity score for every candidate.

    For each candidate we slide the template over +-`search` samples, take
    the lag with the highest normalised correlation, and refine it with a
    parabola through the three best points (accuracy ~0.05 sample on clean
    pulses). Returns (positions, scores); positions refer to the template
    reference point (steepest rise).
    """
    pos = np.full(len(candidates), np.nan)
    score = np.zeros(len(candidates))
    for i, c in enumerate(candidates):
        a = int(c) - tpl.pre - search
        b = int(c) + tpl.post + search
        if a < 0 or b > len(x):
            continue
        seg = x[a:b]
        lag, r = _xcorr_peak(seg, tpl.shape)
        pos[i] = a + lag + tpl.pre
        score[i] = r
    return pos, score


def _xcorr_peak(seg: np.ndarray, shape: np.ndarray) -> tuple[float, float]:
    """Best fractional lag of `shape` inside `seg` and its normalised correlation."""
    L = len(shape)
    c = np.correlate(seg, shape, mode="valid")
    # normalise by the local energy of seg so the score is a true 0..1 similarity
    seg2 = np.concatenate(([0.0], np.cumsum(seg * seg)))
    seg1 = np.concatenate(([0.0], np.cumsum(seg)))
    e = seg2[L:] - seg2[:-L] - (seg1[L:] - seg1[:-L]) ** 2 / L
    r = c / np.sqrt(np.maximum(e, 1e-30))
    k = int(np.argmax(r))
    frac = 0.0
    if 0 < k < len(r) - 1:
        y0, y1, y2 = r[k - 1], r[k], r[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            frac = 0.5 * (y0 - y2) / den
    return k + frac, float(r[k])


def _frac_window(x: np.ndarray, p: float, pre: int, post: int) -> np.ndarray:
    """Samples x[p-pre : p+post] at fractional start p (linear interpolation)."""
    i = int(np.floor(p))
    f = p - i
    a = x[i - pre : i + post + 1]
    return (1 - f) * a[:-1] + f * a[1:]


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
    rhythm (and more transients per line), so we histogram the two-line
    distances (0.5-sample bins) and return the `n_best` strongest, clearly
    separated peaks, each refined by the median of the distances within
    +-0.2 % of it and halved. With n_best=1 a single float is returned.
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


def staggered(times: np.ndarray, period: float) -> np.ndarray:
    """Mask of candidates that take part in the staggered picture-line rhythm.

    In picture lines consecutive sync intervals alternate around the period
    (long, short, long ... by ~3 % each way) and two of them add up to two
    periods. Lead-in signals and picture-content edges do not do this, so the
    mask picks out genuine picture sync pulses, e.g. for building the template.
    """
    t = np.sort(np.asarray(times, dtype=float))
    m = np.zeros(len(t), bool)
    if len(t) < 3:
        return m
    d1 = t[1:-1] - t[:-2]
    d2 = t[2:] - t[1:-1]
    dev1, dev2 = d1 / period - 1, d2 / period - 1
    ok = ((np.abs(dev1) > 0.01) & (np.abs(dev1) < 0.06) & (np.sign(dev1) == -np.sign(dev2))
          & (np.abs((d1 + d2) / (2 * period) - 1) < 0.003))
    idx = np.flatnonzero(ok)
    m[idx] = m[idx + 1] = m[idx + 2] = True
    return m


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

    # resolve overlaps: best (most measured pulses) first
    runs.sort(key=lambda r: -int(r[1].sum()))
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
