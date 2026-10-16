import ctypes
import sys
from ctypes import wintypes

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget

_WINDOW = QColor("#202326")
_VIEW = QColor("#141618")
_ALTERNATE_VIEW = QColor("#1d1f22")
_BUTTON = QColor("#292c30")
_TEXT = QColor("#fcfcfc")
_DISABLED_TEXT = QColor("#6e7175")
_HIGHLIGHT = QColor("#3daee9")
_HIGHLIGHTED_TEXT = QColor("#fcfcfc")
_LINK = QColor("#1d99f3")
_LINK_VISITED = QColor("#9b59b6")
_NEGATIVE = QColor("#da4453")
_PLACEHOLDER = QColor("#a1a9b1")
_LIGHT = QColor("#3a3f44")
_MIDLIGHT = QColor("#30343a")
_MID = QColor("#1b1d20")
_DARK = QColor("#0f1012")
_SHADOW = QColor("#000000")

APP_STYLE = "Fusion"
APP_FONT_POINT_SIZE = 10
TITLE_BAR_COLOR = _MID

def _build_palette() -> QPalette:
	palette = QPalette()
	roles = {
		QPalette.ColorRole.Window: _WINDOW,
		QPalette.ColorRole.WindowText: _TEXT,
		QPalette.ColorRole.Base: _VIEW,
		QPalette.ColorRole.AlternateBase: _ALTERNATE_VIEW,
		QPalette.ColorRole.ToolTipBase: _BUTTON,
		QPalette.ColorRole.ToolTipText: _TEXT,
		QPalette.ColorRole.PlaceholderText: _PLACEHOLDER,
		QPalette.ColorRole.Text: _TEXT,
		QPalette.ColorRole.Button: _BUTTON,
		QPalette.ColorRole.ButtonText: _TEXT,
		QPalette.ColorRole.BrightText: _NEGATIVE,
		QPalette.ColorRole.Light: _LIGHT,
		QPalette.ColorRole.Midlight: _MIDLIGHT,
		QPalette.ColorRole.Mid: _MID,
		QPalette.ColorRole.Dark: _DARK,
		QPalette.ColorRole.Shadow: _SHADOW,
		QPalette.ColorRole.Highlight: _HIGHLIGHT,
		QPalette.ColorRole.HighlightedText: _HIGHLIGHTED_TEXT,
		QPalette.ColorRole.Link: _LINK,
		QPalette.ColorRole.LinkVisited: _LINK_VISITED,
	}
	for role, color in roles.items():
		palette.setColor(role, color)

	for role in (
		QPalette.ColorRole.WindowText,
		QPalette.ColorRole.Text,
		QPalette.ColorRole.ButtonText,
		QPalette.ColorRole.HighlightedText,
	):
		palette.setColor(QPalette.ColorGroup.Disabled, role, _DISABLED_TEXT)
	palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Highlight, _MIDLIGHT)

	return palette

def _set_title_bar_color(window: QWidget, color: QColor = TITLE_BAR_COLOR) -> None:
	"""Keep title bar and border dark even with 'accent color on title bars' enabled (Windows 11)"""
	DWMWA_BORDER_COLOR = 34
	DWMWA_CAPTION_COLOR = 35

	hwnd = wintypes.HWND(int(window.winId()))
	colorref = wintypes.DWORD(color.red() | color.green() << 8 | color.blue() << 16)
	for attribute in (DWMWA_BORDER_COLOR, DWMWA_CAPTION_COLOR):
		ctypes.windll.dwmapi.DwmSetWindowAttribute(
			hwnd, attribute, ctypes.byref(colorref), ctypes.sizeof(colorref)
		)

class _TitleBarColorizer(QObject):
	"""Colors the title bar of every window (main window, dialogs, message boxes) when it is shown"""

	def eventFilter(self, obj, event):
		if event.type() == QEvent.Type.Show and isinstance(obj, QWidget) and obj.isWindow():
			_set_title_bar_color(obj)
		return False

def apply_theme(app: QApplication) -> None:
	app.setStyle(APP_STYLE)
	app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
	app.setPalette(_build_palette())

	font = app.font()
	font.setPointSize(APP_FONT_POINT_SIZE)
	app.setFont(font)

	if sys.platform == "win32":
		app.installEventFilter(_TitleBarColorizer(app))