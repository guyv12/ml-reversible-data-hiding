import cv2
from pydicom import dcmread
import torch
from numpy import ndarray
from pathlib import Path

from PySide6.QtWidgets import (
	QWidget, QFrame, QHBoxLayout,
	QVBoxLayout, QLabel, QPushButton
)
from PySide6.QtCore import Qt, QSize

from backend.pipeline import predict

from frontend.components.image_uploader import ImageUploader
from frontend.components.histogram import Histogram
from frontend.components.preview_manager import PreviewManager
from frontend.components.preview import (
	EmptyPreview, InputImagePreview, OutputImagePreview
)
from frontend.components.quality_metrics import QualityMetricsPanel
from frontend.config import SECTIONS_LABEL_HEIGHT
from frontend.utils import load_stylesheet

class ProcessingView(QWidget):
	"""
	Processing screen view for RDHEI operations.

	Provides workspace and UI controls for RDHEI operations.
	"""

	def __init__(self):
		super().__init__()

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
		metrics_layout.addWidget(self.quality_metrics_panel)

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
		self.image_uploader.image_removed.connect(self.in_histogram.clear)
		self.image_uploader.image_removed.connect(self.quality_metrics_panel.clear)

		self.out_preview_manager.image_loaded.connect(self.out_histogram.plot_histogram)
		self.out_preview_manager.image_removed.connect(self.out_histogram.clear)

	def _get_processed_path(self, image_path: str) -> str:
		path = Path(image_path)
		return str(path.parent / f"processed_{path.name}")

	def _on_image_uploaded(self, image_path: str):
		image_data = self._transform_image_to_ndarray(image_path)
		processed_path = self._get_processed_path(image_path)

		prediction = predict(image_data)
		self.quality_metrics_panel.set_metrics(prediction.metrics)

		self.in_preview_manager.set_image(image_path, image_data)
		self.out_preview_manager.set_image(image_path, image_data)
		self.in_histogram.plot_histogram(image_data)

	def _transform_image_to_ndarray(self, image_path: str) -> ndarray:
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

	# def _predict_bytes(self, image: np.ndarray) -> np.ndarray:
	# 	shape = image.shape[:2]
	# 	image_batch = image[np.newaxis, :]

	# 	pixels = shape[0] * shape[1]
	# 	bpp = 8
	# 	data_range = 2 ** bpp -1
	# 	bits_per_image = pixels * bpp
	# 	K_e = "password"
	# 	K_h = "password"

	# 	image_tensor = torch.from_numpy(image_batch).float()
	# 	raw_ad = ppredict.pgm_raw_ad_sklearn(image_tensor)
	# 	kernel_weights, ref_pixels, error_map, original = next(raw_ad)

	# 	mask = ppredict.reference_mask(shape[0], shape[1])
	# 	y = original.flatten()[~mask.flatten()].numpy().astype(np.float32)
	# 	y_pred = y - error_map.numpy().astype(np.float32)

	# 	psnr = peak_signal_noise_ratio(y, y_pred, data_range=data_range)
	# 	ssim = structural_similarity(y, y_pred, data_range=data_range)

	# 	ad = ccompress.compress_pgm_ad(shape, kernel_weights, ref_pixels, error_map)
	# 	ad_enrypted = encryption.encrypt_ad(ad, pixels, bpp, K_e)

	# 	available_bits = bits_per_image - len(ad)
	# 	emb_rate = available_bits / pixels

	# 	self.quality_metrics.update_metrics(psnr, ssim, available_bits, emb_rate)

	# 	print(len(ref_pixels))
	# 	print(mask.sum().item())

	# 	encrypted_image = hider(ad_enrypted, available_bits//8, "bardzo tajna wiadomosc", K_h)
	# 	reconstructed = receive(encrypted_image, K_e, K_h, len(ref_pixels))
	# 	check_images(image, reconstructed)

	# 	return reconstructed

	# def _reconstruct_bytes():
	# 	pass

	# # hider powinien wyladowac w osobnej funkcji, tak samo receive.
	# # Przydałby sie moze jakis compute_manager, który trzyma czesc z wartosci i ma pod soba te funkcje
	# # Tak samo jakas klasa qframe, ktora by trzymala haslo oraz wiadomosc do szyfrowania (moze ten  compute_manager)
	# # Wtedy moglibysmy nałożyć style css na to.