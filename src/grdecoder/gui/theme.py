"""
theme.py - colours, fonts and the Qt stylesheet.

A dark, instrument-like look: near-black panels, thin borders, monospaced
read-outs, and one accent colour (gold, after the record itself) used
sparingly for state and selection. Plots use a cyan trace on the same
background so the diagnostics read like a scope.
"""

BG = "#0d1015"         # window background
PANEL = "#141920"      # panels / docks
PANEL_2 = "#1a2029"    # inputs, list items
BORDER = "#262e39"
TEXT = "#c8d0da"
TEXT_DIM = "#7d8794"
ACCENT = "#d6a849"     # gold
ACCENT_DIM = "#8a6d33"
TRACE = "#58c4dc"      # cyan plot trace
TRACE_2 = "#d6a849"
GRID = "#222a34"
WARN = "#e0705a"

MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, 'DejaVu Sans Mono', monospace"
SANS = "'Segoe UI', 'Inter', 'Helvetica Neue', Arial, sans-serif"

STYLESHEET = f"""
* {{
    font-family: {SANS};
    font-size: 12px;
    color: {TEXT};
}}
QMainWindow, QWidget#central {{ background: {BG}; }}
QDockWidget {{ titlebar-close-icon: none; }}
QDockWidget::title {{
    background: {PANEL}; padding: 6px 8px; border-bottom: 1px solid {BORDER};
    font-size: 10px; letter-spacing: 2px; color: {TEXT_DIM};
}}
QWidget#panel {{ background: {PANEL}; }}
QToolBar {{
    background: {PANEL}; border: none; border-bottom: 1px solid {BORDER};
    spacing: 4px; padding: 4px 8px;
}}
QToolButton, QPushButton {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 3px;
    padding: 5px 12px;
}}
QToolButton:hover, QPushButton:hover {{ border-color: {ACCENT_DIM}; }}
QToolButton:pressed, QPushButton:pressed, QPushButton:checked, QToolButton:checked {{
    background: {ACCENT_DIM}; color: #0d1015; border-color: {ACCENT};
}}
QPushButton:disabled, QToolButton:disabled {{ color: #4a525d; }}
QGroupBox {{
    border: none; border-top: 1px solid {BORDER}; margin-top: 18px; padding-top: 8px;
}}
QGroupBox::title {{
    subcontrol-origin: margin; left: 0px; padding: 0 6px 0 0;
    color: {ACCENT}; font-size: 10px; letter-spacing: 2px;
}}
QLabel#readout {{ font-family: {MONO}; color: {TRACE}; }}
QLabel#title {{ font-size: 13px; letter-spacing: 3px; color: {ACCENT}; padding: 0 12px 0 4px; }}
QLabel#fileinfo {{ font-family: {MONO}; color: {TEXT_DIM}; padding-left: 12px; }}
QLabel#dim {{ color: {TEXT_DIM}; }}
QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 3px;
    padding: 3px 6px; font-family: {MONO}; min-height: 18px;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {ACCENT_DIM}; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; selection-background-color: {ACCENT_DIM}; }}
QCheckBox::indicator {{
    width: 12px; height: 12px; border: 1px solid {BORDER}; background: {PANEL_2}; border-radius: 2px;
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 3px; background: {BORDER}; }}
QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; }}
QSlider::handle:horizontal {{
    width: 10px; height: 14px; margin: -6px 0; background: {ACCENT}; border-radius: 2px;
}}
QListWidget {{ background: {PANEL}; border: none; outline: 0; }}
QListWidget::item {{ color: {TEXT_DIM}; border: 1px solid transparent; border-radius: 3px; padding: 2px; }}
QListWidget::item:selected {{ background: {PANEL_2}; color: {ACCENT}; border-color: {ACCENT_DIM}; }}
QListWidget::item:hover {{ border-color: {BORDER}; }}
QTabWidget::pane {{ border: none; background: {PANEL}; }}
QTabBar::tab {{
    background: {PANEL}; color: {TEXT_DIM}; padding: 6px 14px; border: none;
    font-size: 10px; letter-spacing: 1px;
}}
QTabBar::tab:selected {{ color: {ACCENT}; border-bottom: 2px solid {ACCENT}; }}
QStatusBar {{ background: {PANEL}; border-top: 1px solid {BORDER}; color: {TEXT_DIM}; font-family: {MONO}; }}
QProgressBar {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 2px; height: 8px;
    text-align: center; font-size: 9px;
}}
QProgressBar::chunk {{ background: {ACCENT}; }}
QScrollArea {{ border: none; background: {PANEL}; }}
QScrollBar:vertical {{ background: {PANEL}; width: 8px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 4px; }}
"""
