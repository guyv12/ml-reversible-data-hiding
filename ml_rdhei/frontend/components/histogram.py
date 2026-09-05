import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
	QFrame, QWidget, QVBoxLayout, QStackedLayout,
	QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

from frontend.utils import load_stylesheet
from frontend.config import (
	HISTOGRAM_MARGIN, EMPTY_LAYOUT_SPACING, ICON_SIZE
)

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
		self.stacked_layout.addWidget(self.plot_widget)

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

	def _update_ui(self):
		if self.has_image:
			self.stacked_layout.setCurrentWidget(self.plot_widget)
		else:
			self.plot_widget.clear()
			self.stacked_layout.setCurrentWidget(self.empty_widget)

	def _set_image(self, image_data: nd.ndarray | None):
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
