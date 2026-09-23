from pathlib import Path
import time

from PySide6.QtWidgets import (
	QWidget, QHBoxLayout, QMessageBox,
	QVBoxLayout, QLabel, QPushButton
)

from backend.pipeline import extract, transform_image_to_ndarray

from frontend.components.section_frame import SectionFrame
from frontend.components.image_uploader import ImageUploader
from frontend.components.histogram import Histogram
from frontend.components.preview_manager import PreviewManager
from frontend.components.preview import (
	EmptyPreview, InputImagePreview, OutputImagePreview
)
from frontend.components.decryption_panel import DecryptionPanel
from frontend.session import Session
from frontend.config import ACCEPTED_FORMATS
from frontend.utils import load_stylesheet

class ExtractionView(QWidget):
	"""
	Extraction screen view for RDHEI operations.

	Provides workspace and UI controls for RDHEI operations.
	"""

	def __init__(self):
		super().__init__()
		
		self._session: Session | None = None

		load_stylesheet(self, "sections.css")

		layout = QVBoxLayout(self)
		self.title_label = QLabel("Extraction View")
		self.title_label.setFixedHeight(30)
		self.return_btn = QPushButton("Return to main window")
		layout.addWidget(self.title_label)

		sections_layout = QHBoxLayout()
		layout.addLayout(sections_layout)

		in_section = SectionFrame("Input", "inputSection")

		self.in_empty_preview = EmptyPreview(
			"Drop Grayscale or DICOM image",
			"system-file-manager",
			"or click to browse",
			ACCEPTED_FORMATS
		)
		self.in_image_preview = InputImagePreview()
		self.in_preview_manager = PreviewManager(self.in_empty_preview, self.in_image_preview)
		self.image_uploader = ImageUploader(self.in_preview_manager)
		self.in_histogram = Histogram(
			"emblem-important",
			"Upload an image to see the histogram"
		)

		in_section.add_widget(self.image_uploader)
		in_section.add_widget(self.in_histogram)

		metrics_section = SectionFrame("Metrics", "metricsSection")

		self.decryption_panel = DecryptionPanel()
	
		metrics_section.add_widget(self.decryption_panel)

		out_section = SectionFrame("Output", "outputSection")

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

		out_section.add_widget(self.out_preview_manager)
		out_section.add_widget(self.out_histogram)

		sections_layout.addWidget(in_section, stretch=1)
		sections_layout.addWidget(metrics_section, stretch=1)
		sections_layout.addWidget(out_section, stretch=1)

		layout.addWidget(self.return_btn)
		self._manage_signals()

	def _manage_signals(self):
		self.image_uploader.image_uploaded.connect(self._on_image_uploaded)
		self.image_uploader.image_removed.connect(self._on_image_removed)

		self.decryption_panel.extraction_request.connect(self._on_extract_request)
		
		self.in_preview_manager.image_loaded.connect(self.in_histogram.plot_histogram)
		self.out_preview_manager.image_loaded.connect(self.out_histogram.plot_histogram)
		self.out_preview_manager.image_removed.connect(self.out_histogram.clear)

	def _on_image_uploaded(self, image_path: Path):
		image_data = transform_image_to_ndarray(image_path)

		self.in_preview_manager.set_image(image_path, image_data)

		self._session = Session(image_path, image_data)

		self.decryption_panel.enable()

	def _on_image_removed(self):
		self._session = None
		self.in_histogram.clear()
		self.decryption_panel.clear()

	def _on_extract_request(self, ad_decryption_key: str, message_decryption_key: str):
		self.decryption_panel.set_busy(True)
		try:
			start = time.perf_counter()
			self._session.marked_image, message = extract(self._session.source_image, ad_decryption_key, message_decryption_key)
			end = time.perf_counter()
			print(end-start)
			self.out_preview_manager.set_image(self._session.output_path, self._session.marked_image, self._session.source_path)
			self.decryption_panel.display_message(message)
		finally:
			self.decryption_panel.set_busy(False)
			