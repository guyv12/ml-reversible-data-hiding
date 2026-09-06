import cv2
from pydicom import dcmread
import torch
import numpy as np
from pathlib import Path
from bitarray import bitarray

from PySide6.QtWidgets import (
	QWidget, QFrame, QHBoxLayout,
	QVBoxLayout, QLabel, QPushButton
)
from PySide6.QtCore import Qt, QSize

from backend.pipeline import predict, hide
from backend.predictor.results import Prediction

from frontend.components.image_uploader import ImageUploader
from frontend.components.histogram import Histogram
from frontend.components.preview_manager import PreviewManager
from frontend.components.preview import (
	EmptyPreview, InputImagePreview, OutputImagePreview
)
from frontend.components.quality_metrics_panel import QualityMetricsPanel
from frontend.components.encryption_panel import EncryptionPanel
from frontend.session import HideSession
from frontend.config import SECTIONS_LABEL_HEIGHT
from frontend.utils import load_stylesheet

class ProcessingView(QWidget):
	"""
	Processing screen view for RDHEI operations.

	Provides workspace and UI controls for RDHEI operations.
	"""

	def __init__(self):
		super().__init__()
		
		# self._prediction: Prediction | None = None
		self._session: HideSession | None = None

		load_stylesheet(self, "sections.css")

		layout = QVBoxLayout(self)
		self.title_label = QLabel("Image Processing View")
		self.title_label.setFixedHeight(30)
		self.return_btn = QPushButton("Return to main window")
		layout.addWidget(self.title_label)

		sections_layout = QHBoxLayout()
		layout.addLayout(sections_layout)

		in_section = QFrame()
		in_section.setObjectName("inputSection")
		in_layout = QVBoxLayout(in_section)
		in_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		in_title_label = QLabel("Input")
		in_title_label.setFixedHeight(SECTIONS_LABEL_HEIGHT)
		in_layout.addWidget(in_title_label)

		self.in_empty_preview = EmptyPreview(
			"Drop Grayscale or DICOM image",
			"system-file-manager",
			"or click to browse",
			[".pgm", ".dcm"]
		)

		self.in_image_preview = InputImagePreview()

		self.in_preview_manager = PreviewManager(self.in_empty_preview, self.in_image_preview)
		self.image_uploader = ImageUploader(self.in_preview_manager)

		self.in_histogram = Histogram(
			"emblem-important",
			"Upload an image to see the histogram"
		)

		in_layout.addWidget(self.image_uploader)
		in_layout.addWidget(self.in_histogram)

		metrics_section = QFrame()
		metrics_section.setObjectName("metricsSection")
		metrics_layout = QVBoxLayout(metrics_section)
		metrics_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		metrics_title_label = QLabel("Metrics")
		metrics_title_label.setFixedHeight(SECTIONS_LABEL_HEIGHT)
		metrics_layout.addWidget(metrics_title_label)

		self.quality_metrics_panel = QualityMetricsPanel()
		self.encryption_panel = EncryptionPanel()
		
		metrics_layout.addWidget(self.quality_metrics_panel)
		metrics_layout.addWidget(self.encryption_panel)

		out_section = QFrame()
		out_section.setObjectName("outputSection")
		out_layout = QVBoxLayout(out_section)
		out_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		out_title_label = QLabel("Output")
		out_title_label.setFixedHeight(SECTIONS_LABEL_HEIGHT)
		out_layout.addWidget(out_title_label)

		self.out_empty_preview = EmptyPreview(
			"Output image",
			"insert-image",
		)
		self.out_image_preview = OutputImagePreview()

		self.out_preview_manager = PreviewManager(self.out_empty_preview, self.out_image_preview)

		self.out_histogram = Histogram(
			"emblem-important",
			"Upload an image to see the histogram"
		)

		out_layout.addWidget(self.out_preview_manager)
		out_layout.addWidget(self.out_histogram)

		sections_layout.addWidget(in_section, stretch=1)
		sections_layout.addWidget(metrics_section, stretch=1)
		sections_layout.addWidget(out_section, stretch=1)

		layout.addWidget(self.return_btn)

		self._manage_signals()

	def _manage_signals(self):
		self.image_uploader.image_uploaded.connect(self._on_image_uploaded)
		self.image_uploader.image_removed.connect(self._on_image_removed)

		self.encryption_panel.hide_request.connect(self._on_hide_request)
		
		self.out_preview_manager.image_loaded.connect(self.out_histogram.plot_histogram)
		self.out_preview_manager.image_removed.connect(self.out_histogram.clear)

	def _on_image_uploaded(self, image_path: str):
		image_data = self._transform_image_to_ndarray(image_path)

		self._session = HideSession(image_path, image_data)
		self._session.prediction = predict(image_data)

		self.quality_metrics_panel.set_metrics(self._session.prediction.metrics)	
		self.encryption_panel.enable_panel(self._session.prediction.metrics.payload_capacity)
		self.in_preview_manager.set_image(image_path, image_data)
		self.in_histogram.plot_histogram(image_data)

	def _on_image_removed(self):
		self._session = None
		self.in_histogram.clear()
		self.quality_metrics_panel.clear()
		self.encryption_panel.clear()

	def _on_hide_request(self, key: str, message: str):
		self.encryption_panel.set_busy(True)
		try:
			self._session.marked_image = hide(self._session.prediction, key, message)
			self.out_preview_manager.set_image(self._session.output_path, self._session.marked_image)
		finally:
			self.encryption_panel.set_busy(False)

	def _transform_image_to_ndarray(self, image_path: str) -> np.ndarray:
		if image_path.lower().endswith(".dcm"):
			try:
				dicom = dcmread(image_path)
				image = dicom.pixel_array
			except Exception as e:
				raise ValueError(f"Failed to decode DICOM file '{path}': {e}") from e
		else:
			image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

		if image is None:
			raise FileNotFoundError(
				f"""Failed to load image: File not found or unreadable at '{image_path}'"""
			)

		return image