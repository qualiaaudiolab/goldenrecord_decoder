"""
End-to-end tests on a synthetic recording with known faults (see synthetic.py).

Run with:   python -m pytest -q
"""

from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from grdecoder.pipeline import DecodeSettings, Decoder
from grdecoder.timebase import build_timebase, destagger
from synthetic import make_recording


@pytest.fixture(scope="module")
def decoded(tmp_path_factory):
    y, truth = make_recording()
    path = tmp_path_factory.mktemp("rec") / "synthetic.wav"
    sf.write(path, y, truth["fs"], subtype="FLOAT")
    dec = Decoder(path)
    dec.analyse(channels=(0,))
    return dec, truth


def test_finds_one_picture_of_512_lines(decoded):
    dec, truth = decoded
    assert len(dec.pictures) == 1
    p = dec.pictures[0]
    assert p.n_lines == 512
    # the picture starts where we put it (lines are counted from the run start)
    run = dec.runs[0][p.run_id]
    first_true = truth["lines_before"]
    first_found = int(round((run.times[p.first_line] / truth["fs"]) * 120))
    assert abs(first_found - first_true) <= 12


def test_line_period_and_stagger(decoded):
    dec, truth = decoded
    P = dec.period[0]
    assert abs(P / (truth["fs"] / 120) - 1) < 1e-3
    ds = destagger(dec.runs[0][dec.pictures[0].run_id].times)
    assert abs(abs(ds.stagger) - truth["stagger"] * P) < 1.0  # within one sample


def test_timing_error_matches_injected_wow(decoded):
    dec, truth = decoded
    run = dec.runs[0][dec.pictures[0].run_id]
    tb = build_timebase(destagger(run.times).times, run.measured, "line", fs=truth["fs"])
    t_sync = run.times / truth["fs"]
    injected = np.interp(t_sync, truth["warp_t"], truth["warp"])
    injected = injected - np.polyval(np.polyfit(t_sync, injected, 1), t_sync)
    measured = tb.error_raw / truth["fs"]
    # measured error is tape position behind real time -> same sign as the warp
    r = np.corrcoef(measured, injected)[0, 1]
    assert r > 0.98, r


def test_droop_time_constant_estimate(decoded):
    dec, truth = decoded
    r = dec.render(dec.pictures[0], DecodeSettings(height=256))
    assert r.droop_tau_ms is not None
    assert abs(r.droop_tau_ms / (truth["tau"] * 1e3) - 1) < 0.25


def _column_spread(img):
    """How much the columns differ (truth: all columns identical)."""
    return float(np.std(img - img.mean(axis=1, keepdims=True)) / np.std(img))


def test_sync_locked_beats_fixed_speed(decoded):
    dec, _ = decoded
    p = dec.pictures[0]
    st = DecodeSettings(height=512, margin_lines=0, invert=False)
    smooth = _column_spread(dec.render(p, st).image)
    fixed = _column_spread(dec.render(p, replace(st, mode="fixed")).image)
    assert smooth < 0.5 * fixed, (smooth, fixed)
    assert smooth < 0.15


def test_render_shapes(decoded):
    dec, _ = decoded
    st = DecodeSettings(height=300, margin_lines=4)
    r = dec.render(dec.pictures[0], st)
    assert r.image.shape == (300, 520)
    disp = r.display()
    assert disp.min() >= 0 and disp.max() <= 1
