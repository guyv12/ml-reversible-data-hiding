from PySide6.QtWidgets import (
	QWidget, QFrame, QHBoxLayout,
	QVBoxLayout, QLabel, QPushButton
)
from PySide6.QtCore import Qt

from frontend.components.image_uploader import ImageUploader
from frontend.components.histogram import Histogram
from frontend.components.preview_manager import PreviewManager
from frontend.config import PREVIEW_MANAGER_SIZE
from frontend.utils import load_stylesheet

class ProcessingView(QWidget):
	"""
	Processing screen view for RDHEI operations.

	Provides workspace and UI controls for RDHEI operations.
	"""

	def __init__(self):
		super().__init__()

		load_stylesheet(self, "panels.css")

		layout = QVBoxLayout(self)
		self.title_label = QLabel("Image Processing View")
		self.title_label.setFixedHeight(30)
		self.return_btn = QPushButton("Return to main window")
		layout.addWidget(self.title_label)

		sections_layout = QHBoxLayout()
		layout.addLayout(sections_layout)

		input_panel = QFrame()
		input_panel.setObjectName("inputPanel")
		input_layout = QVBoxLayout(input_panel)
		input_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		input_subtitle_label = QLabel("Input")
		input_subtitle_label.setFixedHeight(20)
		input_layout.addWidget(input_subtitle_label)

		self.image_uploader = ImageUploader()
		self.image_uploader.setFixedSize(PREVIEW_MANAGER_SIZE)

		self.input_histogram = Histogram(
			"emblem-important",
			"Upload an image to see the histogram"
		)

		input_layout.addWidget(self.image_uploader)
		input_layout.addWidget(self.input_histogram)
		
		metrics_panel = QFrame()
		metrics_panel.setObjectName("metricsPanel")
		metrics_layout = QVBoxLayout(metrics_panel)

		output_panel = QFrame()
		output_panel.setObjectName("outputPanel")
		output_layout = QVBoxLayout(output_panel)
		output_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		output_subtitle_label = QLabel("Output")
		output_subtitle_label.setFixedHeight(20)
		output_layout.addWidget(output_subtitle_label)
		
		self.preview_manager = PreviewManager()
		self.preview_manager.setFixedSize(PREVIEW_MANAGER_SIZE)

		self.output_histogram = Histogram(
			"emblem-important",
			"Upload an image to see the histogram"
		)

		output_layout.addWidget(self.preview_manager)
		output_layout.addWidget(self.output_histogram)
		
		sections_layout.addWidget(input_panel, stretch=1)
		sections_layout.addWidget(metrics_panel, stretch=1)
		sections_layout.addWidget(output_panel, stretch=1)

		layout.addWidget(self.return_btn)

		self.manage_signals()

	def manage_signals(self):
		self.image_uploader.image_uploaded.connect(self.input_histogram.plot_histogram)
		self.image_uploader.image_removed.connect(self.input_histogram.clear)

		self.image_uploader.image_uploaded.connect(self.preview_manager.set_image)

		self.preview_manager.image_loaded.connect(self.output_histogram.plot_histogram)
		self.preview_manager.image_removed.connect(self.output_histogram.clear)