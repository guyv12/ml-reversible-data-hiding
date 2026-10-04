import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
	QFrame, QWidget, QVBoxLayout, QStackedLayout,
	QLabel
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QPixmap, QPainter

from frontend.utils import load_stylesheet
from frontend.config import (
	HISTOGRAM_MARGIN, EMPTY_LAYOUT_SPACING, ICON_SIZE,
	HISTOGRAM_GRID_COLOR, HISTOGRAM_RESIZE_INTERVAL_MS, HISTOGRAM_RESIZE_SETTLE_MS
)

class ThrottledResizeContainer(QWidget):
	def __init__(self, child: QWidget, interval_ms: int, settle_ms: int):
		super().__init__()
		self._child = child
		self._child.setParent(self)
		self._resizing = False
		self._size_pending = False
		self._snapshot: QPixmap | None = None

		self._settle_timer = QTimer(self)
		self._settle_timer.setSingleShot(True)
		self._settle_timer.setInterval(settle_ms)
		self._settle_timer.timeout.connect(self._finish_resizing)

		self._throttle_timer = QTimer(self)
		self._throttle_timer.setSingleShot(True)
		self._throttle_timer.setInterval(interval_ms)
		self._throttle_timer.timeout.connect(self._apply_pending_size)

		

	def resizeEvent(self, event):
		super().resizeEvent(event)

		if not self.isVisible():
			self._child.setGeometry(self.rect())
			return

		if not self._resizing:
			self._resizing = True
			self._child.hide()

		self._settle_timer.start()
		self._size_pending = True

		if not self._throttle_timer.isActive():
			self._apply_pending_size()

	def _apply_pending_size(self):
		if not self._size_pending:
			return

		self._size_pending = False
		self._child.setGeometry(self.rect())

		if self._resizing:
			self._snapshot = self._child.grab()
			self.update()

		self._throttle_timer.start()

	def _finish_resizing(self):
		self._throttle_timer.stop()
		self._resizing = False
		self._apply_pending_size()
		self._snapshot = None
		self._child.show()

	def paintEvent(self, event):
		if self._snapshot is not None:
			painter = QPainter(self)
			painter.drawPixmap(0, 0, self._snapshot)

class Histogram(QFrame):
	"""
	Histogram visualization widget

    Renders a pixel intensity histogram for PGM and DICOM images.
    Responds to image updates and clear events to 
	covert an image from image_path to ndarray and
	update the view dynamically.
    """
	def __init__(
		self,
		icon_name: str | None = None,
		title: str | None = None
	):
		super().__init__()
		self._image_data: np.ndarray | None = None
		
		self.setContentsMargins(*HISTOGRAM_MARGIN)
		self.stacked_layout = QStackedLayout(self)

		self._setup_empty_widget(icon_name, title)
		self._setup_plot_widget()

		self.stacked_layout.addWidget(self.empty_widget)
		self.stacked_layout.addWidget(self.plot_container)

		load_stylesheet(self, "histogram.css")
		self._update_ui()

	@property
	def has_image(self) -> bool:
		return self._image_data is not None

	def _setup_empty_widget(
		self,
		icon_name: str | None = None,
		title: str | None = None
		):
		self.empty_widget = QWidget()
		self.empty_layout = QVBoxLayout(self.empty_widget)
		self.empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.empty_layout.setSpacing(EMPTY_LAYOUT_SPACING)

		self.icon_label: QLabel | None = None
		self.title_label: QLabel | None = None

		if icon_name:
			self._setup_icon(icon_name)
		
		if title:
			self._setup_title(title)
	
	def _setup_icon(self, icon_name: str):
		self.icon_label = QLabel()
		icon = QIcon.fromTheme(icon_name)
		self.icon_label.setPixmap(icon.pixmap(*ICON_SIZE))
		self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		
		self.empty_layout.addWidget(self.icon_label)

	def _setup_title(self, title: str):
		self.title_label = QLabel(title)
		self.title_label.setWordWrap(True)
		self.title_label.setObjectName("titleLabel")
		self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

		self.empty_layout.addWidget(self.title_label)

	def _setup_plot_widget(self):
		self.plot_widget = pg.PlotWidget()
		
		self.plot_widget.setBackground("#323232")
		self.plot_widget.showGrid(x=True, y=True, alpha=0.2)

		for axis_name in ("left", "bottom"):
			axis = self.plot_widget.getAxis(axis_name)
			axis.setStyle(maxTickLevel=1, tickAlpha=255)
			axis.setTickPen(pg.mkPen(HISTOGRAM_GRID_COLOR))
			axis.setZValue(-1)

		self.plot_container = ThrottledResizeContainer(
			self.plot_widget,
			HISTOGRAM_RESIZE_INTERVAL_MS,
			HISTOGRAM_RESIZE_SETTLE_MS
		)

	def _update_ui(self):
		if self.has_image:
			self.stacked_layout.setCurrentWidget(self.plot_container)
		else:
			self.plot_widget.clear()
			self.stacked_layout.setCurrentWidget(self.empty_widget)

	def _set_image(self, image_data: np.ndarray | None):
		self._image_data = image_data

	def plot_histogram(self, image_data: np.ndarray):
		self._set_image(image_data)
		self.plot_widget.clear()

		if image_data.dtype == "uint8":
			_range = (0, 256)
		else:
			_range = (float(image_data.min()), float(image_data.max()))

		counts, bins = np.histogram(
			image_data,
			bins=256,
			range=_range
		)

		histogram_item = pg.PlotCurveItem(
			bins, counts, 
			stepMode="center", 
			fillLevel=0, 
			fillBrush=(74, 144, 226, 100),
			pen=pg.mkPen('#4A90E2', width=1.5)
		)
		self.plot_widget.addItem(histogram_item)
		self._update_ui()

	def clear(self):
		self._image_data = None
		self._update_ui()