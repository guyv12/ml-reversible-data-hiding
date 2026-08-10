import cv2
from pydicom import dcmread

from PySide6.QtWidgets import (
	QWidget, QFrame, QHBoxLayout,
	QVBoxLayout, QLabel, QPushButton
)
from PySide6.QtCore import Qt, QSize

from frontend.components.image_uploader import ImageUploader
from frontend.components.histogram import Histogram
from frontend.components.preview_manager import PreviewManager
from frontend.components.preview import EmptyPreview, InputImagePreview, OutputImagePreview
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

		in_panel = QFrame()
		in_panel.setObjectName("inputPanel")
		in_layout = QVBoxLayout(in_panel)
		in_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		in_subtitle_label = QLabel("Input")
		in_subtitle_label.setFixedHeight(20)
		in_layout.addWidget(in_subtitle_label)

		self.in_empty_preview = EmptyPreview(
			"Drop Grayscale or DICOM image",
			"system-file-manager",
			"or click to browse",
			[".pgm", ".dcm"]
		)

		self.in_image_preview = InputImagePreview()

		self.in_preview_manager = PreviewManager(self.in_empty_preview, self.in_image_preview)
		self.in_preview_manager.setFixedSize(PREVIEW_MANAGER_SIZE - QSize(5, 5))

		self.image_uploader = ImageUploader(self.in_preview_manager)
		self.image_uploader.setFixedSize(PREVIEW_MANAGER_SIZE)

		self.in_histogram = Histogram(
			"emblem-important",
			"Upload an image to see the histogram"
		)

		in_layout.addWidget(self.image_uploader)
		in_layout.addWidget(self.in_histogram)
		
		metrics_panel = QFrame()
		metrics_panel.setObjectName("metricsPanel")
		metrics_layout = QVBoxLayout(metrics_panel)

		out_panel = QFrame()
		out_panel.setObjectName("outputPanel")
		out_layout = QVBoxLayout(out_panel)
		out_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		out_subtitle_label = QLabel("Output")
		out_subtitle_label.setFixedHeight(20)
		out_layout.addWidget(out_subtitle_label)

		self.out_empty_preview = EmptyPreview(
			"Output image",
			"insert-image",
		)
		self.out_image_preview = OutputImagePreview()
		
		self.out_preview_manager = PreviewManager(self.out_empty_preview, self.out_image_preview)
		self.out_preview_manager.setFixedSize(PREVIEW_MANAGER_SIZE)

		self.out_histogram = Histogram(
			"emblem-important",
			"Upload an image to see the histogram"
		)

		out_layout.addWidget(self.out_preview_manager)
		out_layout.addWidget(self.out_histogram)
		
		sections_layout.addWidget(in_panel, stretch=1)
		sections_layout.addWidget(metrics_panel, stretch=1)
		sections_layout.addWidget(out_panel, stretch=1)

		layout.addWidget(self.return_btn)

		self._manage_signals()

	def _manage_signals(self):
		self.image_uploader.image_uploaded.connect(self._on_image_uploaded)
		self.image_uploader.image_removed.connect(self.in_histogram.clear)

		self.out_preview_manager.image_loaded.connect(self.out_histogram.plot_histogram)
		self.out_preview_manager.image_removed.connect(self.out_histogram.clear)

	def _on_image_uploaded(self, image_path: str):
		image_data = self._transform_image_to_ndarray(image_path)

		self.in_preview_manager.set_image(image_path, image_data)
		self.out_preview_manager.set_image(image_path, image_data)
		self.in_histogram.plot_histogram(image_data)

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