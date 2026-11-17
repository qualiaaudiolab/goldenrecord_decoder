"""
pictures.py - locating the 512-line pictures inside runs of sync pulses.

A run of regular sync pulses (see sync.track_lines) is longer than the
picture it carries: there are ~100 lines of framing before the picture and
blank lines after it. Just before every picture sits a start marker: ~21
lines of high-contrast diagonal hatching. The picture's first line is the
line right after the marker; that is the precise anchor (`find_start_marker`).
If no marker is found (damaged, or cut off by the start of the file), the
picture is placed on the stretch of lines whose blanking structure matches
(`picture_mask`, `find_picture_blocks`, `place_picture`). Picture lines are recognisable because their blanking intervals - the
first ~16 % and last ~10 % of every line - all share the same structure
(sync pulse, black/white framing bars). Lines outside the picture do not.

Method: draw every line of the run at low resolution, take the median line
of the middle of the run as reference, correlate each line's blanking parts
with it, and keep the longest stretch of lines that clearly matches. That
stretch is then centred on the nominal 512 lines.
"""

from __future__ import annotations

import numpy as np

LINES_PER_PICTURE = 512  # from the record cover: "512 vertical lines in a complete picture"


def line_profiles(x: np.ndarray, a: int, t_sync: np.ndarray, period: float, points: int = 512) -> np.ndarray:
    """Every line of a run resampled to `points` samples: shape (lines, points)."""
    u = np.arange(points) * period / points
    pos = (t_sync[:, None] - a) + u[None, :]
    return np.interp(pos, np.arange(len(x)), x)


def hatch_score(profiles: np.ndarray) -> np.ndarray:
    """Number of full-contrast swings inside each line (start-marker detector).

    The start marker's lines cross from black to white ~10-14 times; picture
    lines almost never swing over the full range of the run that often.
    Counted with hysteresis (+-20 % of the run's 2..98 % range around its
    middle) so that noise does not add crossings.
    """
    n, p = profiles.shape
    seg = profiles[:, int(0.15 * p) : int(0.85 * p)]
    lo, hi = np.percentile(profiles, [2, 98])
    mid, h = 0.5 * (lo + hi), 0.2 * (hi - lo)
    state = np.zeros(n)
    count = np.zeros(n, dtype=int)
    for col in seg.T:  # vectorised over lines, stepping along the line
        up = (col > mid + h) & (state <= 0)
        dn = (col < mid - h) & (state >= 0)
        count += up | dn
        state[up] = 1
        state[dn] = -1
    return count


def find_start_marker(hatch: np.ndarray, near: int, search: int = 80, min_count: int = 8,
                      min_len: int = 6) -> int | None:
    """Last line of the start marker closest before line `near`, or None.

    Looks for a run of at least `min_len` consecutive hatched lines (gaps of
    up to 2 lines allowed) ending within `search` lines of `near`.
    """
    lo, hi = max(0, near - search), min(len(hatch), near + search // 2)
    idx = lo + np.flatnonzero(hatch[lo:hi] >= min_count)
    if len(idx) == 0:
        return None
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 2) + 1)
    groups = [g for g in groups if len(g) >= min_len]
    if not groups:
        return None
    # the marker whose end is closest to the expected picture start
    g = min(groups, key=lambda g: abs(g[-1] + 1 - near))
    return int(g[-1])


def picture_mask(profiles: np.ndarray) -> np.ndarray:
    """Per-line similarity (-1..1) of the blanking structure to the run median.

    Two details make this robust:

    * We compare the *first difference* of the profiles, i.e. the positions
      and sizes of edges. Absolute levels in the blanking intervals depend
      on the picture content before them (the recording was AC-coupled), but
      the edges of the framing pulses do not.
    * Even and odd lines have differently shaped sync pulses, so each line is
      compared with the median of its own parity.
    """
    d = np.diff(profiles, axis=1)
    n, p = d.shape
    sel = np.r_[0 : int(0.16 * p), int(0.90 * p) : p]
    sim = np.zeros(n)
    for parity in (0, 1):
        idx = np.arange(parity, n, 2)
        mid = idx[(idx > n // 2 - 150) & (idx < n // 2 + 150)]
        ref = np.median(d[mid][:, sel], axis=0)
        ref = ref - ref.mean()
        b = d[idx][:, sel]
        b = b - b.mean(axis=1, keepdims=True)
        sim[idx] = (b @ ref) / (np.linalg.norm(b, axis=1) * np.linalg.norm(ref) + 1e-30)
    return sim


def find_picture_blocks(similarity: np.ndarray, min_lines: int = 400, max_hole: int = 16) -> list[tuple[int, int]]:
    """Stretches of lines that belong to a picture: list of (first, stop).

    The threshold sits halfway between the typical similarity inside the
    picture (high percentile) and outside it (low percentile). A short
    median filter ignores single odd lines (tape dropouts), and holes of up
    to `max_hole` lines inside a picture (bright content leaking into the
    blanking) are closed.
    """
    from scipy.ndimage import binary_closing, median_filter

    s = median_filter(similarity, size=9, mode="nearest")
    hi = np.percentile(s, 90)
    lo = np.percentile(s, 5)
    if hi - lo < 0.05:
        return []
    good = s > 0.5 * (hi + lo)
    good = binary_closing(np.r_[False, good, False], structure=np.ones(max_hole + 1))[1:-1] | good
    blocks, i = [], 0
    while i < len(good):
        if good[i]:
            j = i
            while j < len(good) and good[j]:
                j += 1
            if j - i >= min_lines:
                blocks.append((i, j))
            i = j
        else:
            i += 1
    return blocks


def place_picture(block: tuple[int, int], similarity: np.ndarray, n: int = LINES_PER_PICTURE) -> tuple[int, int]:
    """Return (first_line, n_lines) of the nominal 512-line picture inside a block.

    * Block close to 512 lines: centre the 512 lines on it.
    * Block clearly longer (e.g. merged with a neighbouring stretch that
      happened to look similar): take the 512-line window with the highest
      mean similarity.
    * Block clearly shorter (picture cut off by the end of the file, or
      damaged): return it unchanged so nothing is lost.
    """
    a, b = block
    length = b - a
    if length < n - 48:
        return a, length
    if length <= n + 48:
        first = int(round((a + b - n) / 2))
        return max(0, min(first, len(similarity) - n)), n
    c = np.concatenate(([0.0], np.cumsum(similarity[a:b])))
    means = (c[n:] - c[:-n]) / n
    return a + int(np.argmax(means)), n
