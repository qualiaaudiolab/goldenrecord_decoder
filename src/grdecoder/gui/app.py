"""
app.py - the graphical interface.

Layout
------
  toolbar        open / export / A-B compare, file information
  left dock      list of pictures found, with thumbnails
  centre         the decoded picture (zoom: wheel, pan: drag, click: pick a line)
  right dock     decoder settings and measured values
  bottom dock    diagnostics: timing error, speed, line scope, picture finder

All heavy work (analysis, rendering, exports) runs in a thread pool so the
window never freezes. Renders are debounced and numbered: only the result of
the most recent request is shown.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..pipeline import (BACK_PORCH, LINES_PER_PICTURE, DecodeSettings, Decoder, Picture, Rendered, SyncSettings,
                        display_width, save_png)
from ..render import auto_levels, combine_colour, measure_circle_aspect, resize_columns, tone_map
from . import theme
from .playback import Player

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True,
                    background=theme.PANEL, foreground=theme.TEXT_DIM)

COLOUR_ORDERS = ["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"]


# ==========================================================================
# background tasks
# ==========================================================================
class _Signals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(float, str)
    item = QtCore.Signal(object)


class Task(QtCore.QRunnable):
    """Run fn(*args, task=self) in the thread pool and report through signals.

    fn may call task.progress(fraction, message) and task.emit_item(obj).
    All connections to these signals are made with Qt.QueuedConnection, so
    the connected functions always run in the GUI thread, never in the
    worker (Qt widgets must only be touched from the GUI thread).
    """

    def __init__(self, fn, *args):
        super().__init__()
        self.fn, self.args = fn, args
        self.signals = _Signals()
        self.cancelled = False

    def progress(self, f: float, msg: str = "") -> None:
        try:
            self.signals.progress.emit(float(f), msg)
        except RuntimeError:  # window closed
            pass

    def emit_item(self, obj) -> None:
        try:
            self.signals.item.emit(obj)
        except RuntimeError:  # window closed
            pass

    def run(self) -> None:
        """Run the job in a worker thread and report the result through signals."""
        try:
            try:
                res = self.fn(*self.args, task=self)
            except Exception:  # report any failure to the GUI instead of dying silently
                self.signals.failed.emit(traceback.format_exc())
            else:
                self.signals.done.emit(res)
        except RuntimeError:
            pass  # the window was closed while this task was still running


# ==========================================================================
# small widget helpers
# ==========================================================================
def readout(text: str = "—") -> QtWidgets.QLabel:
    """Right-aligned, monospaced label for a measured value."""
    lab = QtWidgets.QLabel(text)
    lab.setObjectName("readout")
    lab.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    return lab


def dspin(lo, hi, val, step, dec, suffix="") -> QtWidgets.QDoubleSpinBox:
    """Double spin box with range, default, step, decimals and unit suffix."""
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


def styled_plot(ylabel: str, xlabel: str) -> pg.PlotWidget:
    """pyqtgraph plot widget in the app's scope-like style."""
    p = pg.PlotWidget()
    p.showGrid(x=True, y=True, alpha=0.12)
    p.setLabel("left", ylabel)
    p.setLabel("bottom", xlabel)
    p.getPlotItem().getViewBox().setMouseEnabled(x=True, y=True)
    for ax in ("left", "bottom"):
        p.getAxis(ax).setPen(pg.mkPen(theme.BORDER))
        p.getAxis(ax).setTextPen(pg.mkPen(theme.TEXT_DIM))
    return p


def app_icon() -> QtGui.QIcon:
    """The record cover (icon.png next to this file), or a drawn golden disc."""
    png = Path(__file__).with_name("icon.png")
    if png.exists():
        return QtGui.QIcon(str(png))
    pm = QtGui.QPixmap(64, 64)
    pm.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.setBrush(QtGui.QColor(theme.ACCENT))
    p.setPen(QtCore.Qt.NoPen)
    p.drawEllipse(2, 2, 60, 60)
    p.end()
    return QtGui.QIcon(pm)


def to_qimage(img01: np.ndarray) -> QtGui.QImage:
    """0..1 grey or RGB float array -> QImage (copy)."""
    a = np.ascontiguousarray((np.clip(img01, 0, 1) * 255).astype(np.uint8))
    h, w = a.shape[:2]
    if a.ndim == 2:
        return QtGui.QImage(a.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
    return QtGui.QImage(a.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()


# ==========================================================================
# main window
# ==========================================================================
class MainWindow(QtWidgets.QMainWindow):
    """The decoder window: picture list, picture view, settings and diagnostics."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Golden Record Decoder")
        self.setWindowIcon(app_icon())
        self.resize(1600, 980)
        self.setAcceptDrops(True)

        self.pool = QtCore.QThreadPool.globalInstance()
        self._tasks: set[Task] = set()
        self.decoder: Decoder | None = None
        self.st = DecodeSettings()
        self.current: Picture | None = None
        self.rendered: Rendered | None = None
        self.colour: tuple[Picture, Picture, Picture] | None = None
        self.colour_frames: np.ndarray | None = None  # (rows, lines, 3) registered separations
        self.triples: list[tuple[Picture, Picture, Picture]] = []
        self.selected_line: int | None = None
        self._render_seq = 0
        self._render_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._render_timer.timeout.connect(self._start_render)
        self._qsettings = QtCore.QSettings("QualiaAudioLab", "GoldenRecordDecoder")

        # real-time playback
        self.player: Player | None = None
        self._play_timer = QtCore.QTimer(self, interval=33)
        self._play_timer.timeout.connect(self._play_tick)
        self._reveal: int | None = None          # columns shown while playing (None = all)
        self._col_times: np.ndarray | None = None  # file time (s) of each column's line start
        self._disp_full: np.ndarray | None = None  # the picture as displayed, before reveal
        self._tick = 0

        self._build_toolbar()
        self._build_centre()
        self._build_picture_dock()
        self._build_settings_dock()
        self._build_diagnostics_dock()
        self._build_statusbar()
        # left and right docks run full height; diagnostics sit between them
        self.setCorner(QtCore.Qt.BottomLeftCorner, QtCore.Qt.LeftDockWidgetArea)
        self.setCorner(QtCore.Qt.BottomRightCorner, QtCore.Qt.RightDockWidgetArea)
        self.resizeDocks([self.diag_dock], [250], QtCore.Qt.Vertical)
        self._sync_widgets_from_settings()
        self._set_enabled(False)

    def _run(self, task: Task) -> None:
        """Start a task, keeping a reference until it has reported back.

        Without the reference, Python could garbage-collect the task (and its
        signal object) as soon as the worker finishes, before the queued
        "done" signal reaches the GUI thread.
        """
        self._tasks.add(task)
        forget = lambda *_: self._tasks.discard(task)  # noqa: E731
        task.signals.done.connect(forget, QtCore.Qt.QueuedConnection)
        task.signals.failed.connect(forget, QtCore.Qt.QueuedConnection)
        self.pool.start(task)

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self):
        tb = QtWidgets.QToolBar("main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(16, 16))
        self.addToolBar(tb)
        title = QtWidgets.QLabel("GOLDEN RECORD DECODER")
        title.setObjectName("title")
        tb.addWidget(title)
        self.act_open = tb.addAction("Open…", self.open_dialog)
        self.act_open.setShortcut(QtGui.QKeySequence.Open)
        tb.addSeparator()
        self.act_export = tb.addAction("Export picture", self.export_current)
        self.act_export_all = tb.addAction("Export all…", self.export_all)
        self.act_csv = tb.addAction("Timing CSV", self.export_csv)
        tb.addSeparator()
        self.act_compare = tb.addAction("A/B: uncorrected")
        self.act_compare.setCheckable(True)
        self.act_compare.setToolTip("Show the same picture decoded the plain way:\n"
                                    "fixed speed, no stagger/field/droop/clamp correction.")
        self.act_compare.toggled.connect(lambda _: self.schedule_render())
        tb.addSeparator()
        self.act_play = tb.addAction("▶  Play")
        self.act_play.setCheckable(True)
        self.act_play.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space))
        self.act_play.setToolTip("Play the recording from the selected picture. The picture is drawn\n"
                                 "line by line in step with the sound, and playback moves on to the\n"
                                 "next picture of the channel by itself. (Space)")
        self.act_play.toggled.connect(self._play_toggled)
        self.w_volume = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.w_volume.setRange(0, 100)
        self.w_volume.setValue(int(self._qsettings.value("volume", 40)))
        self.w_volume.setFixedWidth(90)
        self.w_volume.setToolTip("Playback volume")
        self.w_volume.valueChanged.connect(self._volume_changed)
        tb.addWidget(self.w_volume)
        self.file_label = QtWidgets.QLabel("no file — open a recording of the picture section (WAV/FLAC)")
        self.file_label.setObjectName("fileinfo")
        tb.addWidget(self.file_label)

    def _build_centre(self):
        w = QtWidgets.QWidget()
        w.setObjectName("central")
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.caption = QtWidgets.QLabel("")
        self.caption.setObjectName("fileinfo")
        self.caption.setContentsMargins(12, 6, 12, 6)
        lay.addWidget(self.caption)
        self.view = pg.GraphicsLayoutWidget()
        self.view.setBackground(theme.BG)
        self.vb = self.view.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.vb.setMenuEnabled(False)
        self.image_item = pg.ImageItem()
        self.vb.addItem(self.image_item)
        self.line_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen(theme.ACCENT, width=1, style=QtCore.Qt.DashLine))
        self.line_marker.setVisible(False)
        self.vb.addItem(self.line_marker)
        self.vb.scene().sigMouseClicked.connect(self._image_clicked)
        lay.addWidget(self.view, 1)
        self.setCentralWidget(w)

    def _build_picture_dock(self):
        dock = QtWidgets.QDockWidget("PICTURES", self)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        panel = QtWidgets.QWidget()
        panel.setObjectName("panel")
        lay = QtWidgets.QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)
        row = QtWidgets.QHBoxLayout()
        self.chan_filter = QtWidgets.QComboBox()
        self.chan_filter.addItems(["Both channels", "Left only", "Right only"])
        self.chan_filter.currentIndexChanged.connect(self._fill_list)
        row.addWidget(self.chan_filter)
        lay.addLayout(row)
        self.pic_list = QtWidgets.QListWidget()
        self.pic_list.setIconSize(QtCore.QSize(112, 76))
        self.pic_list.setSpacing(2)
        self.pic_list.currentItemChanged.connect(self._picture_selected)
        lay.addWidget(self.pic_list, 1)
        dock.setWidget(panel)
        dock.setMinimumWidth(250)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

    def _build_settings_dock(self):
        dock = QtWidgets.QDockWidget("DECODER", self)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        panel = QtWidgets.QWidget()
        panel.setObjectName("panel")
        v = QtWidgets.QVBoxLayout(panel)
        v.setContentsMargins(12, 0, 12, 12)

        def group(title):
            g = QtWidgets.QGroupBox(title)
            f = QtWidgets.QFormLayout(g)
            f.setLabelAlignment(QtCore.Qt.AlignLeft)
            f.setHorizontalSpacing(12)
            f.setVerticalSpacing(6)
            v.addWidget(g)
            return f

        # --- measured values --------------------------------------------
        f = group("MEASURED")
        self.ro_speed = readout(); f.addRow("Speed vs 120 Hz lines", self.ro_speed)
        self.ro_period = readout(); f.addRow("Line period", self.ro_period)
        self.ro_terr = readout(); f.addRow("Timing error (rms / p-p)", self.ro_terr)
        self.ro_sync = readout(); f.addRow("Sync pulses found", self.ro_sync)
        self.ro_stagger = readout(); f.addRow("Sync stagger", self.ro_stagger)
        self.ro_field = readout(); f.addRow("Field offset", self.ro_field)
        self.ro_tau = readout(); f.addRow("Droop time constant", self.ro_tau)

        # --- sync detection (analysis) -------------------------------------
        f = group("SYNC DETECTION")
        self.w_tplen = dspin(1, 10, 4, 0.5, 1, " % of line")
        self.w_tplen.setToolTip("Length of the sync-pulse template after the rising edge.\n"
                                "Longer = more distinctive on dull or band-limited copies;\n"
                                "shorter = less affected by picture content after the pulse.")
        f.addRow("Template length", self.w_tplen)
        self.w_minscore = dspin(0.3, 0.95, 0.6, 0.05, 2)
        self.w_minscore.setToolTip("Minimum match (0-1) for a pulse to count. Lower it if many pulses\n"
                                   "are shown as predicted (red); raise it if noise is taken for pulses.")
        f.addRow("Minimum match", self.w_minscore)
        self.w_edge = dspin(2, 200, 20, 1, 0, " µs")
        self.w_edge.setToolTip("Time over which a rising edge is measured when looking for the\n"
                               "line rhythm. Raise it for noisy copies, lower it for very sharp ones.")
        f.addRow("Edge scale", self.w_edge)
        self.btn_reanalyse = QtWidgets.QPushButton("Re-analyse with these settings")
        self.btn_reanalyse.clicked.connect(self.reanalyse)
        f.addRow(self.btn_reanalyse)

        # --- timebase -----------------------------------------------------
        f = group("TIMEBASE")
        self.w_mode = QtWidgets.QComboBox()
        self.w_mode.addItems(["Sync-locked, smoothed", "Sync-locked, per line", "Fixed speed"])
        self.w_mode.setToolTip("How the measured sync times are turned into a time axis (see docs/HOW_IT_WORKS.md).")
        f.addRow("Mode", self.w_mode)
        self.w_bw = dspin(1, 60, 40, 1, 0, " Hz")
        self.w_bw.setToolTip("Bandwidth of the wow/flutter correction. The line rate is 120 Hz,\n"
                             "so 60 Hz follows every line; lower values ignore fast jitter.")
        f.addRow("Correction bandwidth", self.w_bw)
        self.w_trim = dspin(-30000, 30000, 0, 50, 0, " ppm")
        self.w_trim.setToolTip("Fixed-speed mode only: fine trim of the playback speed\n"
                               "around the speed measured for this picture.")
        f.addRow("Speed trim (fixed)", self.w_trim)
        self.w_destagger = QtWidgets.QCheckBox("Remove sync stagger")
        f.addRow(self.w_destagger)
        self.w_field = QtWidgets.QCheckBox("Align fields (experimental)")
        self.w_field.setToolTip("Measures a vertical offset between even and odd lines and removes it.\n"
                                "Off by default: on the master tape the fields already line up after\n"
                                "removing the stagger, and the measurement is misled by a fine texture\n"
                                "in the signal. Leave off unless edges show a comb pattern.")
        f.addRow(self.w_field)
        self.w_flevel = QtWidgets.QCheckBox("Equalise field levels")
        f.addRow(self.w_flevel)

        # --- signal -------------------------------------------------------
        f = group("SIGNAL")
        self.w_droop = QtWidgets.QCheckBox("Droop (LF loss) correction")
        f.addRow(self.w_droop)
        self.w_tau = dspin(0, 50, 0, 0.01, 3, " ms")
        self.w_tau.setSpecialValueText("auto")
        self.w_tau.setToolTip("Time constant of the AC coupling to undo. 0 = estimate from the sync pulse tops.")
        f.addRow("Time constant", self.w_tau)
        self.w_clamp = QtWidgets.QCheckBox("Line clamp (back porch)")
        f.addRow(self.w_clamp)
        self.w_invert = QtWidgets.QCheckBox("Invert (high signal = dark)")
        f.addRow(self.w_invert)

        # --- geometry -----------------------------------------------------
        f = group("GEOMETRY")
        self.w_top = dspin(0, 0.5, 0.115, 0.005, 3)
        self.w_bottom = dspin(0.5, 1.0, 0.915, 0.005, 3)
        f.addRow("Active window start", self.w_top)
        f.addRow("Active window end", self.w_bottom)
        self.w_height = QtWidgets.QComboBox()
        self.w_height.addItems(["384", "512", "768", "1024", "1536"])
        f.addRow("Rows", self.w_height)
        self.w_aspect = dspin(0.5, 3, 1.485, 0.005, 3)
        f.addRow("Aspect (W/H)", self.w_aspect)
        self.btn_circle = QtWidgets.QPushButton("Calibrate aspect from circle")
        self.btn_circle.setToolTip("Select picture 1 (the calibration circle) first.\n"
                                   "Sets the aspect so that the circle comes out round.")
        self.btn_circle.clicked.connect(self.calibrate_circle)
        f.addRow(self.btn_circle)
        self.w_margin = QtWidgets.QSpinBox()
        self.w_margin.setRange(0, 64)
        f.addRow("Margin lines", self.w_margin)

        # --- tone ---------------------------------------------------------
        f = group("TONE")
        self.w_black = dspin(0, 20, 0.3, 0.1, 1, " %")
        self.w_white = dspin(80, 100, 99.7, 0.1, 1, " %")
        self.w_gamma = dspin(0.3, 3, 1.0, 0.05, 2)
        f.addRow("Black point (percentile)", self.w_black)
        f.addRow("White point (percentile)", self.w_white)
        f.addRow("Gamma", self.w_gamma)

        # --- colour -------------------------------------------------------
        f = group("COLOUR")
        self.btn_colour = QtWidgets.QPushButton("Combine with next two")
        self.btn_colour.setCheckable(True)
        self.btn_colour.setToolTip("Treat the selected picture and the two that follow on the same channel\n"
                                   "as colour separations and show them combined.")
        self.btn_colour.toggled.connect(self._colour_toggled)
        f.addRow(self.btn_colour)
        self.w_order = QtWidgets.QComboBox()
        self.w_order.addItems(COLOUR_ORDERS)
        self.w_order.setToolTip("Which colour each of the three frames carries (frame 1, 2, 3).\n"
                                "The record uses red, green, blue (RGB), the default.")
        self.w_order.currentIndexChanged.connect(self._redisplay)
        f.addRow("Frame order", self.w_order)
        hint = QtWidgets.QLabel("Pictures marked ◆ look like colour separations.")
        hint.setObjectName("dim")
        hint.setWordWrap(True)
        f.addRow(hint)

        v.addStretch(1)
        self.btn_defaults = QtWidgets.QPushButton("Reset to defaults")
        self.btn_defaults.clicked.connect(self._reset_defaults)
        v.addWidget(self.btn_defaults)

        scroll.setWidget(panel)
        dock.setWidget(scroll)
        dock.setMinimumWidth(330)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)

        # every settings widget -> one handler
        for w in (self.w_mode, self.w_height):
            w.currentIndexChanged.connect(self._settings_changed)
        for w in (self.w_bw, self.w_trim, self.w_tau, self.w_top, self.w_bottom, self.w_aspect):
            w.valueChanged.connect(self._settings_changed)
        self.w_margin.valueChanged.connect(self._settings_changed)
        for w in (self.w_destagger, self.w_field, self.w_flevel, self.w_droop, self.w_clamp):
            w.toggled.connect(self._settings_changed)
        for w in (self.w_black, self.w_white, self.w_gamma):
            w.valueChanged.connect(self._tone_changed)
        self.w_invert.toggled.connect(self._tone_changed)

    def _build_diagnostics_dock(self):
        dock = self.diag_dock = QtWidgets.QDockWidget("DIAGNOSTICS", self)
        dock.setFeatures(QtWidgets.QDockWidget.NoDockWidgetFeatures)
        tabs = QtWidgets.QTabWidget()
        self.p_timing = styled_plot("timing error (µs)", "picture line")
        leg = self.p_timing.addLegend(offset=(-10, 5), labelTextColor=theme.TEXT_DIM)
        leg.setBrush(pg.mkBrush(20, 25, 32, 220))
        self.p_speed = styled_plot("speed deviation (ppm)", "picture line")
        self.p_scope = styled_plot("signal", "fraction of line (0 = sync)")
        self.p_finder = styled_plot("blanking similarity", "line in run")
        self.p_sync = styled_plot("template", "time from the rising edge (µs)")
        tabs.addTab(self.p_timing, "TIMING ERROR")
        tabs.addTab(self.p_speed, "SPEED")
        tabs.addTab(self.p_scope, "LINE SCOPE")
        tabs.addTab(self.p_finder, "PICTURE FINDER")
        tabs.addTab(self.p_sync, "SYNC TEMPLATE")
        dock.setWidget(tabs)
        dock.setMinimumHeight(180)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        sb.addPermanentWidget(self.progress)
        sb.showMessage(f"v{__version__} · ready")

    def _set_enabled(self, on: bool):
        for a in (self.act_export, self.act_export_all, self.act_csv, self.act_compare, self.act_play):
            a.setEnabled(on)
        self.btn_colour.setEnabled(on)
        self.btn_circle.setEnabled(on)
        self.btn_reanalyse.setEnabled(on)

    # ------------------------------------------------------- settings sync
    def _sync_widgets_from_settings(self):
        st = self.st
        widgets = [self.w_mode, self.w_bw, self.w_trim, self.w_destagger, self.w_field, self.w_flevel, self.w_droop, self.w_tau,
                   self.w_clamp, self.w_invert, self.w_top, self.w_bottom, self.w_height, self.w_aspect,
                   self.w_margin, self.w_black, self.w_white, self.w_gamma]
        for w in widgets:
            w.blockSignals(True)
        self.w_mode.setCurrentIndex({"smooth": 0, "line": 1, "fixed": 2}[st.mode])
        self.w_bw.setValue(st.smooth_hz)
        self.w_trim.setValue((st.speed_trim - 1) * 1e6)
        self.w_destagger.setChecked(st.destagger)
        self.w_field.setChecked(st.field_align)
        self.w_flevel.setChecked(st.field_level)
        self.w_droop.setChecked(st.droop)
        self.w_tau.setValue(st.droop_tau_ms or 0)
        self.w_clamp.setChecked(st.clamp)
        self.w_invert.setChecked(st.invert)
        self.w_top.setValue(st.top)
        self.w_bottom.setValue(st.bottom)
        idx = self.w_height.findText(str(st.height))
        self.w_height.setCurrentIndex(idx if idx >= 0 else 2)
        self.w_aspect.setValue(st.aspect)
        self.w_margin.setValue(st.margin_lines)
        self.w_black.setValue(st.black_pct)
        self.w_white.setValue(st.white_pct)
        self.w_gamma.setValue(st.gamma)
        for w in widgets:
            w.blockSignals(False)
        self._update_enabled_controls()

    def _settings_from_widgets(self) -> DecodeSettings:
        return replace(
            self.st,
            mode=["smooth", "line", "fixed"][self.w_mode.currentIndex()],
            smooth_hz=self.w_bw.value(),
            speed_trim=1 + self.w_trim.value() * 1e-6,
            destagger=self.w_destagger.isChecked(),
            field_align=self.w_field.isChecked(),
            field_level=self.w_flevel.isChecked(),
            droop=self.w_droop.isChecked(),
            droop_tau_ms=self.w_tau.value() or None,
            clamp=self.w_clamp.isChecked(),
            invert=self.w_invert.isChecked(),
            top=self.w_top.value(),
            bottom=self.w_bottom.value(),
            height=int(self.w_height.currentText()),
            aspect=self.w_aspect.value(),
            margin_lines=self.w_margin.value(),
            black_pct=self.w_black.value(),
            white_pct=self.w_white.value(),
            gamma=self.w_gamma.value(),
        )

    def _update_enabled_controls(self):
        mode = self.w_mode.currentIndex()
        self.w_bw.setEnabled(mode == 0)
        self.w_trim.setEnabled(mode == 2)
        self.w_tau.setEnabled(self.w_droop.isChecked())

    def _settings_changed(self, *_):
        self.st = self._settings_from_widgets()
        self._update_enabled_controls()
        self.schedule_render()

    def _tone_changed(self, *_):
        self.st = self._settings_from_widgets()
        self._redisplay()

    def _reset_defaults(self):
        self.st = DecodeSettings()
        self._sync_widgets_from_settings()
        self.schedule_render()

    # ------------------------------------------------------------- opening
    def open_dialog(self):
        """Ask for a recording and open it."""
        start = self._qsettings.value("last_dir", str(Path.home()))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open recording", start, "Audio (*.wav *.flac *.aif *.aiff *.w64 *.rf64);;All files (*)")
        if path:
            self.open_file(path)

    def dragEnterEvent(self, e):
        """Accept audio files dragged onto the window."""
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        """Open the first audio file dropped onto the window."""
        urls = e.mimeData().urls()
        if urls:
            self.open_file(urls[0].toLocalFile())

    def sync_settings(self) -> SyncSettings:
        """The sync-detection settings as currently set in the panel."""
        return SyncSettings(template_width=self.w_tplen.value() / 100, min_score=self.w_minscore.value(),
                            edge_scale_us=self.w_edge.value())

    def reanalyse(self):
        """Analyse the open file again with the current sync-detection settings."""
        if self.decoder is not None or getattr(self, "_path", None):
            self.open_file(str(self._path))

    def open_file(self, path: str):
        """Open a recording and analyse it in the background."""
        self._path = Path(path)
        self._qsettings.setValue("last_dir", str(Path(path).parent))
        try:
            dec = Decoder(path, sync_settings=self.sync_settings())
        except Exception as ex:
            QtWidgets.QMessageBox.critical(self, "Cannot open file", str(ex))
            return
        info = dec.source.info
        self.file_label.setText(f"{info.path.name}   ·   {info.samplerate / 1000:g} kHz   ·   "
                                f"{info.channels} ch   ·   {info.duration:.1f} s   ·   {info.subtype}")
        self.act_play.setChecked(False)
        self.player = None
        self.decoder = None
        self.current = None
        self.pic_list.clear()
        self.image_item.clear()
        self._set_enabled(False)
        channels = tuple(range(min(2, info.channels)))
        task = Task(lambda task: dec.analyse(channels, task.progress))
        task.signals.progress.connect(self._show_progress, QtCore.Qt.QueuedConnection)
        task.signals.done.connect(lambda pics, dec=dec: self._analysis_done(dec), QtCore.Qt.QueuedConnection)
        task.signals.failed.connect(self._task_failed, QtCore.Qt.QueuedConnection)
        self.progress.setVisible(True)
        self._run(task)

    def _show_progress(self, f, msg):
        self.progress.setValue(int(f * 1000))
        self.statusBar().showMessage(msg)

    def _task_failed(self, tb: str):
        self.progress.setVisible(False)
        self.statusBar().showMessage("error — see dialog")
        QtWidgets.QMessageBox.critical(self, "Error", tb)

    def _analysis_done(self, dec: Decoder):
        self.progress.setVisible(False)
        self.decoder = dec
        n = len(dec.pictures)
        per = ", ".join(f"{'LR'[c]} {dec.period[c] / dec.fs * 1e3:.4f} ms" for c in dec.period)
        self.statusBar().showMessage(f"{n} pictures found · line period {per}")
        self._set_enabled(n > 0)
        self._fill_list()
        if n:
            self.pic_list.setCurrentRow(0)
        self._start_thumbnails()

    # ----------------------------------------------------------- the list
    def _visible_pictures(self) -> list[Picture]:
        if not self.decoder:
            return []
        f = self.chan_filter.currentIndex()
        return [p for p in self.decoder.pictures if f == 0 or p.channel == f - 1]

    def _fill_list(self):
        self.pic_list.blockSignals(True)
        self.pic_list.clear()
        colour_names = {p.name for t in self.triples for p in t}
        for p in self._visible_pictures():
            mark = "  ◆" if p.name in colour_names else ""
            it = QtWidgets.QListWidgetItem(f"{p.name}{mark}\n{p.t_start:7.2f} s · {p.n_lines} lines")
            it.setData(QtCore.Qt.UserRole, p)
            if hasattr(p, "_thumb"):
                it.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(p._thumb)))
            self.pic_list.addItem(it)
        self.pic_list.blockSignals(False)
        if self.current:
            self._select_in_list(self.current)

    def _select_in_list(self, pic: Picture):
        for i in range(self.pic_list.count()):
            if self.pic_list.item(i).data(QtCore.Qt.UserRole) is pic:
                self.pic_list.blockSignals(True)
                self.pic_list.setCurrentRow(i)
                self.pic_list.blockSignals(False)
                return

    def _start_thumbnails(self):
        dec = self.decoder
        st = replace(DecodeSettings(), height=96, field_align=False, margin_lines=0, invert=self.st.invert)

        def work(task):
            for p in list(dec.pictures):
                r = dec.render(p, st)
                img = resize_columns(r.display(), int(96 * 1.485))
                task.emit_item((p, img))
            return dec.colour_triples()

        task = Task(work)
        task.signals.item.connect(self._thumb_ready, QtCore.Qt.QueuedConnection)
        task.signals.done.connect(self._triples_ready, QtCore.Qt.QueuedConnection)
        task.signals.failed.connect(lambda tb: self.statusBar().showMessage("thumbnail error"), QtCore.Qt.QueuedConnection)
        self._run(task)

    def _thumb_ready(self, obj):
        p, img = obj
        p._thumb = to_qimage(img)
        for i in range(self.pic_list.count()):
            it = self.pic_list.item(i)
            if it.data(QtCore.Qt.UserRole) is p:
                it.setIcon(QtGui.QIcon(QtGui.QPixmap.fromImage(p._thumb)))
                break

    def _triples_ready(self, triples):
        self.triples = triples
        self._fill_list()
        if triples:
            names = ", ".join(f"{a.name}–{c.name}" for a, b, c in triples)
            self.statusBar().showMessage(f"possible colour pictures: {names}")

    def _picture_selected(self, item, _prev=None):
        if item is None:
            return
        self.current = item.data(QtCore.Qt.UserRole)
        self.selected_line = None
        self.line_marker.setVisible(False)
        if self.btn_colour.isChecked():
            self.btn_colour.blockSignals(True)
            self.btn_colour.setChecked(False)
            self.btn_colour.blockSignals(False)
            self.colour = None
        if self.act_play.isChecked() and self.player:
            self.player.start(self.current.t_start - 0.25)
            self._reveal = 0
        self.schedule_render(reset_view=True)

    # ----------------------------------------------------------- rendering
    def schedule_render(self, reset_view: bool = False):
        """Re-render the current picture shortly (debounced while settings change)."""
        self._reset_view = getattr(self, "_reset_view", False) or reset_view
        self._render_timer.start()

    def _effective_settings(self) -> DecodeSettings:
        if self.act_compare.isChecked():
            # the "plain" decode: constant speed, no corrections
            return replace(self.st, mode="fixed", speed_trim=1 + self.w_trim.value() * 1e-6, destagger=False,
                           field_align=False, field_level=False, droop=False, clamp=False)
        return self.st

    def _start_render(self):
        if not (self.decoder and self.current):
            return
        self._render_seq += 1
        seq = self._render_seq
        dec, st = self.decoder, self._effective_settings()
        pics = list(self.colour) if self.colour else [self.current]

        def work(task):
            return seq, [dec.render(p, st) for p in pics]

        task = Task(work)
        task.signals.done.connect(self._render_done, QtCore.Qt.QueuedConnection)
        task.signals.failed.connect(self._task_failed, QtCore.Qt.QueuedConnection)
        self.statusBar().showMessage("rendering…")
        self._run(task)

    def _render_done(self, res):
        seq, rendered = res
        if seq != self._render_seq:
            return  # a newer request is on its way
        self.rendered = rendered[0]
        r0 = self.rendered
        self._col_times = r0.timebase(np.arange(r0.lines.start, r0.lines.stop, dtype=float)) / r0.fs
        # colour: register frames 2 and 3 to frame 1 once per render
        self.colour_frames = combine_colour(*[x.image for x in rendered])[0] if len(rendered) == 3 else None
        self._redisplay()
        self._update_readouts()
        self._update_plots()
        if getattr(self, "_reset_view", False):
            self.vb.autoRange(padding=0.02)
            self._reset_view = False
        self.statusBar().showMessage("ready")

    def display_image(self) -> np.ndarray | None:
        """The picture as shown (0..1 grey or RGB), one column per line."""
        r = self.rendered
        if r is None:
            return None
        st = self.st
        if self.colour_frames is not None:
            # frame i carries colour order[i]; put it into that output channel
            order = COLOUR_ORDERS[self.w_order.currentIndex()]
            out = np.zeros_like(self.colour_frames)
            for i, c in enumerate(order):
                out[:, :, "RGB".index(c)] = self.colour_frames[:, :, i]
            lo, hi = auto_levels(out, st.black_pct, st.white_pct)
            return tone_map(out, lo, hi, st.gamma, st.invert)
        lo, hi = auto_levels(r.image, st.black_pct, st.white_pct)
        return tone_map(r.image, lo, hi, st.gamma, st.invert)

    def _redisplay(self, *_):
        img = self.display_image()
        if img is None:
            return
        self._disp_full = img
        self._apply_reveal()
        h, w = img.shape[:2]
        sx = self.st.height * self.st.aspect / LINES_PER_PICTURE  # make pixels the right shape
        self.image_item.setTransform(QtGui.QTransform.fromScale(sx, self.st.height / h))
        self._sx = sx
        p = self.current
        r = self.rendered
        mode = "UNCORRECTED (fixed speed, no corrections)" if self.act_compare.isChecked() else \
            {"smooth": f"sync-locked {self.st.smooth_hz:g} Hz", "line": "sync-locked per line",
             "fixed": "fixed speed"}[r.settings.mode]
        col = f"   ·   colour {'+'.join(x.name for x in self.colour)} ({self.w_order.currentText()})" if self.colour else ""
        self.caption.setText(f"{p.name}   ·   {'left' if p.channel == 0 else 'right'} channel   ·   "
                             f"{p.t_start:.3f}–{p.t_end:.3f} s   ·   {mode}{col}")

    def _apply_reveal(self):
        """Show the picture, blanking the columns not yet reached by playback."""
        img = self._disp_full
        if img is None:
            return
        if self._reveal is not None:
            img = img.copy()
            img[:, max(0, min(self._reveal, img.shape[1])) :] = 0.04
        self.image_item.setImage(img, levels=(0, 1), autoLevels=False)

    # ----------------------------------------------------------- playback
    def _volume_changed(self, v: int):
        self._qsettings.setValue("volume", v)
        if self.player:
            self.player.volume = v / 100

    def _play_toggled(self, on: bool):
        if on:
            if not (self.decoder and self.current):
                self.act_play.setChecked(False)
                return
            if self.colour:  # playback follows single pictures
                self.btn_colour.setChecked(False)
            if self.player is None:
                self.player = Player(self.decoder.source)
            self.player.volume = self.w_volume.value() / 100
            self.player.start(self.current.t_start - 0.25)
            self._reveal = 0
            self._play_timer.start()
            self.act_play.setText("■  Stop")
            if not self.player.audio_ok:
                self.statusBar().showMessage(f"no sound output ({self.player.error}) — playing silently")
        else:
            self._play_timer.stop()
            if self.player:
                self.player.stop()
            self._reveal = None
            self.act_play.setText("▶  Play")
            self.line_marker.setVisible(False)
            self._apply_reveal()

    def _play_tick(self):
        if not (self.player and self.current and self.decoder):
            return
        t = self.player.position()
        p = self.current
        if t > p.t_end:
            # move on to the next picture of this channel when it begins
            nxt = [q for q in self.decoder.pictures if q.channel == p.channel and q.t_start > p.t_start]
            if not nxt:
                if t > p.t_end + 1.0 or t > self.decoder.source.info.duration:
                    self.act_play.setChecked(False)
                return
            if t >= nxt[0].t_start - 0.2:
                self.current = nxt[0]
                self._select_in_list(nxt[0])
                self.pic_list.scrollToItem(self.pic_list.currentItem())
                self._reveal = 0
                self.schedule_render(reset_view=True)
                return
        r = self.rendered
        if r is None or r.picture is not self.current or self._col_times is None:
            return
        col = int(np.searchsorted(self._col_times, t))
        if col != self._reveal:
            self._reveal = col
            self._apply_reveal()
        n = len(self._col_times)
        if 0 <= col < n:
            self.line_marker.setPos((col + 0.5) * getattr(self, "_sx", 1.0))
            self.line_marker.setVisible(True)
            self._tick += 1
            if self._tick % 4 == 0:  # live line scope, ~8 updates per second
                self.selected_line = r.lines.start + col
                self._update_scope()
        line = col - (r.picture.first_line - r.lines.start)
        self.statusBar().showMessage(f"▶ {t:8.2f} s   ·   {p.name}   ·   line {max(0, min(line, n))}")

    def closeEvent(self, e):
        """Stop playback and background work before closing."""
        if self.player:
            self.player.stop()
        super().closeEvent(e)

    # ----------------------------------------------------------- readouts
    def _update_readouts(self):
        r = self.rendered
        if r is None:
            return
        fs = r.fs
        e = r.timing_error_us("raw")
        self.ro_speed.setText(f"{r.speed_vs_nominal_ppm:+.0f} ppm")
        self.ro_period.setText(f"{r.period_samples / fs * 1e3:.5f} ms")
        self.ro_terr.setText(f"{np.std(e):.1f} / {np.ptp(e):.0f} µs")
        self.ro_sync.setText(f"{r.picture.sync_found * 100:.1f} %")
        self.ro_stagger.setText(f"{r.stagger / fs * 1e6:+.1f} µs")
        self.ro_field.setText(f"{r.field_offset_rows:+.2f} rows" if r.settings.field_align else "not applied")
        self.ro_tau.setText(f"{r.droop_tau_ms:.3f} ms" if r.droop_tau_ms else "off")

    def _update_plots(self):
        r = self.rendered
        if r is None:
            return
        tb = r.timebase
        run = self.decoder.runs[r.picture.channel][r.picture.run_id]
        sl = r.lines
        x = np.arange(sl.stop - sl.start) - (r.picture.first_line - sl.start)
        raw = tb.error_raw[sl] / r.fs * 1e6
        used = tb.error_used[sl] / r.fs * 1e6
        meas = run.measured[sl]

        self.p_timing.clear()
        self.p_timing.plot(x[meas], raw[meas], pen=None, symbol="o", symbolSize=3,
                           symbolBrush=pg.mkBrush(theme.TEXT_DIM), symbolPen=None, name="measured syncs")
        if (~meas).any():
            self.p_timing.plot(x[~meas], raw[~meas], pen=None, symbol="x", symbolSize=7,
                               symbolPen=pg.mkPen(theme.WARN), name="predicted (dropout)")
        if r.settings.mode != "fixed":
            self.p_timing.plot(x, used, pen=pg.mkPen(theme.TRACE, width=1.6), name="correction applied")

        self.p_speed.clear()
        sp = tb.speed_ppm()[sl.start : sl.stop - 1] if r.settings.mode != "fixed" else \
            -(np.diff(tb.t_sync[sl]) / tb.period - 1) * 1e6
        self.p_speed.plot(x[:-1], sp, pen=pg.mkPen(theme.TRACE, width=1.2))
        self.p_speed.addItem(pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen(theme.BORDER)))

        self.p_finder.clear()
        sim = run.meta.get("similarity")
        if sim is not None:
            self.p_finder.plot(sim, pen=pg.mkPen(theme.TRACE, width=1))
            reg = pg.LinearRegionItem((r.picture.first_line, r.picture.first_line + r.picture.n_lines),
                                      movable=False, brush=pg.mkBrush(214, 168, 73, 40),
                                      pen=pg.mkPen(theme.ACCENT_DIM))
            self.p_finder.addItem(reg)
        self._update_scope()

        # the learned sync-pulse templates of this channel (one per kind of line)
        self.p_sync.clear()
        tpl = self.decoder.templates.get(r.picture.channel)
        if tpl is not None:
            rate = self.decoder.template_rate.get(r.picture.channel, r.fs)  # samples/s of the template
            t = (np.arange(tpl.length) - tpl.pre) / rate * 1e6
            for i, shp in enumerate(tpl.shapes):
                self.p_sync.plot(t, shp, pen=pg.mkPen(theme.TRACE if i == 0 else theme.ACCENT, width=1.4),
                                 name=f"line type {'AB'[i]}")
            self.p_sync.addItem(pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(theme.BORDER, style=QtCore.Qt.DashLine)))
            n = len(run.score)
            meas = run.measured
            self.p_sync.setTitle(f"match of the found pulses: median {np.median(run.score[meas]):.3f}, "
                                 f"lowest 5 % below {np.percentile(run.score[meas], 5):.3f}  ·  "
                                 f"{(~meas).sum()} of {n} predicted", color=theme.TEXT_DIM, size="9pt")

    def _image_clicked(self, ev):
        if self.rendered is None or not self.vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        pos = self.vb.mapSceneToView(ev.scenePos())
        col = int(pos.x() / getattr(self, "_sx", 1.0))
        r = self.rendered
        n = r.lines.stop - r.lines.start
        if 0 <= col < n:
            self.selected_line = r.lines.start + col
            self.line_marker.setPos((col + 0.5) * self._sx)
            self.line_marker.setVisible(True)
            self._update_scope()

    def _update_scope(self):
        """Raw signal of the selected line, with the regions the decoder uses."""
        self.p_scope.clear()
        r = self.rendered
        if r is None:
            return
        k = self.selected_line if self.selected_line is not None else (r.lines.start + r.lines.stop) // 2
        tb = r.timebase
        u = np.linspace(-0.03, 1.03, 2400)
        t = tb(k + u)
        a, b = int(t[0]) - 4, int(t[-1]) + 4
        x = self.decoder.source.read(r.picture.channel, a, b)
        y = np.interp(t - a, np.arange(len(x)), x)
        for lo, hi, col in ((r.settings.top, r.settings.bottom, (88, 196, 220, 25)),
                            (BACK_PORCH[0], BACK_PORCH[1], (214, 168, 73, 45))):
            self.p_scope.addItem(pg.LinearRegionItem((lo, hi), movable=False, brush=pg.mkBrush(*col),
                                                     pen=pg.mkPen(None)))
        self.p_scope.plot(u, y, pen=pg.mkPen(theme.TRACE, width=1))
        for v in (0, 1):
            self.p_scope.addItem(pg.InfiniteLine(pos=v, angle=90, pen=pg.mkPen(theme.ACCENT_DIM, style=QtCore.Qt.DashLine)))
        self.p_scope.setTitle(f"line {k - r.picture.first_line} of {r.picture.name}  ·  "
                              f"blue: picture window  ·  gold: clamp reference", color=theme.TEXT_DIM, size="9pt")

    # ------------------------------------------------------------- colour
    def _colour_toggled(self, on: bool):
        if not on:
            self.colour = None
            self.schedule_render()
            return
        if not (self.decoder and self.current):
            self.btn_colour.setChecked(False)
            return
        same = [p for p in self.decoder.pictures if p.channel == self.current.channel]
        i = same.index(self.current)
        if i + 2 >= len(same):
            QtWidgets.QMessageBox.information(self, "Colour", "Need two more pictures after this one on the same channel.")
            self.btn_colour.setChecked(False)
            return
        self.colour = tuple(same[i : i + 3])
        self.schedule_render()

    # ---------------------------------------------------------- calibrate
    def calibrate_circle(self):
        """Set the aspect ratio that makes the calibration circle round."""
        if not self.rendered:
            return
        try:
            aspect, info = measure_circle_aspect(self.rendered.image)
        except ValueError as ex:
            QtWidgets.QMessageBox.warning(self, "Calibration", f"{ex}\nSelect the calibration circle (first picture).")
            return
        self.w_aspect.setValue(round(aspect, 3))
        self.statusBar().showMessage(f"aspect set to {aspect:.3f} (ring rows {info['rows']}, lines {info['lines']})")

    # ------------------------------------------------------------- export
    def _export_image(self) -> np.ndarray:
        img = self.display_image()
        return resize_columns(img, display_width(self.st, img.shape[1]))

    def export_current(self):
        """Save the picture on screen as a PNG with square pixels."""
        if self.rendered is None:
            return
        name = self.current.name + ("_colour" if self.colour else "")
        start = str(Path(self._qsettings.value("last_export", str(Path.home()))) / f"{name}.png")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export picture", start, "PNG (*.png)")
        if path:
            self._qsettings.setValue("last_export", str(Path(path).parent))
            save_png(self._export_image(), path, bits=8 if self.colour else 16)
            self.statusBar().showMessage(f"saved {path}")

    def export_csv(self):
        """Save the per-line timing of the picture on screen as CSV."""
        if self.rendered is None:
            return
        start = str(Path(self._qsettings.value("last_export", str(Path.home()))) / f"{self.current.name}_timing.csv")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export timing", start, "CSV (*.csv)")
        if path:
            self.decoder.export_timing_csv(self.rendered, path)
            self.statusBar().showMessage(f"saved {path}")

    def export_all(self):
        """Decode every picture and save them, with timing CSVs, to a folder."""
        if not self.decoder:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Export all pictures to",
                                                            self._qsettings.value("last_export", str(Path.home())))
        if not folder:
            return
        self._qsettings.setValue("last_export", folder)
        dec, st = self.decoder, self.st

        def work(task):
            pics = dec.pictures
            for i, p in enumerate(pics):
                task.progress(i / len(pics), f"exporting {p.name}")
                r = dec.render(p, st)
                dec.export_png(r, Path(folder) / f"{p.name}.png")
                dec.export_timing_csv(r, Path(folder) / f"{p.name}_timing.csv")
            return len(pics)

        task = Task(work)
        task.signals.progress.connect(self._show_progress, QtCore.Qt.QueuedConnection)
        task.signals.done.connect(lambda n: (self.progress.setVisible(False),
                                             self.statusBar().showMessage(f"exported {n} pictures to {folder}")),
                                  QtCore.Qt.QueuedConnection)
        task.signals.failed.connect(self._task_failed, QtCore.Qt.QueuedConnection)
        self.progress.setVisible(True)
        self._run(task)


def run(argv=None) -> int:
    """Start the app; a recording given on the command line is opened at once."""
    argv = sys.argv if argv is None else argv
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setStyle("Fusion")
    app.setStyleSheet(theme.STYLESHEET)
    win = MainWindow()
    win.show()
    files = [a for a in argv[1:] if Path(a).suffix.lower() in (".wav", ".flac", ".aif", ".aiff", ".w64", ".rf64")]
    if files:
        win.open_file(files[0])
    return app.exec()
