from pathlib import Path
from pydicom import dcmread
import numpy as np

from PySide6.QtWidgets import (
	QWidget, QMessageBox, QVBoxLayout, QHBoxLayout,
	QLabel, QPushButton
) 
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QImage, QFontMetrics

from frontend.config import IMAGE_HEADER_MARGIN, PHOTO_DISPLAY_MARGIN

class ImagePreview(QWidget):
	def __init__(self):
		super().__init__()
		self._image_path: str | None = None
		self._cached_pix: QPixmap | None = None

		self.main_layout = QVBoxLayout(self)
		self.main_layout.setContentsMargins(0, 0, 0, 0) 
		self.main_layout.setSpacing(0)

		self.image_header_layout = QHBoxLayout()
		self.image_header_layout.setContentsMargins(*IMAGE_HEADER_MARGIN)

		self.photo_display = QLabel()
		self.photo_display.setContentsMargins(*PHOTO_DISPLAY_MARGIN)
		self.photo_display.setAlignment(Qt.AlignmentFlag.AlignCenter)

		self.main_layout.addLayout(self.image_header_layout)
		self.main_layout.addWidget(self.photo_display)

	def _update_file_label(self, file_path: str):
		pass

	def set_image(self, file_path: str):
		self._image_path = file_path
		self._update_file_label(file_path)

		if Path(self._image_path).suffix == ".dcm":
			try:
				self._cached_pix = self._convert_dicom_to_pixmap()
			except Exception as e:
				QMessageBox.warning(
					self,
					"DICOM error",
					f"Can't read your dicom file: \n{e}"
				)
				self._cached_pix = None

		else:
			self._cached_pix = QPixmap(self._image_path)

		self._update_photo_display()

	def clear_image(self):
		self._image_path = None
		self._cached_pix = None
		self.file_label.clear()
		self.photo_display.clear()

	def _convert_dicom_to_pixmap(self) -> QPixmap:
		dicom = dcmread(self._image_path)
		raw_pixels = dicom.pixel_array.astype(np.int16)

		pixel_min = raw_pixels.min()
		pixel_max = raw_pixels.max()
	
		if pixel_min != pixel_max:
			normalized = ((raw_pixels - pixel_min) / (pixel_max - pixel_min)) * 65535.0
		else:
			normalized = np.zeros_like(raw_pixels)

		photometric = dicom.get("PhotometricInterpretation", "MONOCHROME2")
		if photometric == "MONOCHROME1":
			normalized = 65535.0 - normalized

		norm_pixels = normalized.astype(np.uint16)

		image = QImage(
			norm_pixels.data,
			raw_pixels.shape[1],
			raw_pixels.shape[0],
			raw_pixels.shape[1] * 2,
			QImage.Format.Format_Grayscale16
		)

		return QPixmap.fromImage(image)

	def _update_photo_display(self):
		if not self._image_path:
			return

		if self._cached_pix is None or self._cached_pix.isNull():
			self.photo_display.clear()
			self.photo_display.setText("[ Photo unavailable ]")
			return

		layout_height = self.image_header_layout.sizeHint().height()

		max_width = self.width() - 16
		max_height = self.height() - layout_height - (5 * 4)

		if max_width <= 0 or max_height <= 0:
			return

		scaled_pix = self._cached_pix.scaled(
			max_width,
			max_height,
			Qt.AspectRatioMode.KeepAspectRatio,
			Qt.TransformationMode.SmoothTransformation
		)

		self.photo_display.setPixmap(scaled_pix)

	def resizeEvent(self, event):
		super().resizeEvent(event)

		# Re-elide file name text to fit current label width (fixes first render issue)
		if self._image_path:
			self._update_file_label(self._image_path)

		# Avoid redundant update on first load; only recalculate image scale on actual resize
		if self._cached_pix and not self._cached_pix.isNull():
			self._update_photo_display()


class InputImagePreview(ImagePreview):
	delete_requested = Signal()
	def __init__(self):
		super().__init__()
		self.file_label = QLabel("")
		self.file_label.setTextFormat(Qt.TextFormat.PlainText)
		self.file_label.setObjectName("fileLabel")
		self.file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

		self.delete_btn = QPushButton("✕")
		self.delete_btn.setObjectName("deleteBtn")
		self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
		self.delete_btn.setFixedSize(20, 20)
		self.delete_btn.clicked.connect(self.delete_requested)

		self.image_header_layout.addWidget(self.file_label, stretch=1)
		self.image_header_layout.addWidget(self.delete_btn)

	def _update_file_label(self, file_path: str):
		file_name = Path(self._image_path).name

		metrics = QFontMetrics(self.file_label.font())
		content = metrics.elidedText(
			file_name,
			Qt.TextElideMode.ElideMiddle,
			self.file_label.width()
		)
		self.file_label.setText(content)

class OutputImagePreview(ImagePreview):
	def __init__(self):
		super().__init__()

		self.download_btn = QPushButton("⭳")
		self.download_btn.setObjectName("downloadBtn")
		self.download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
		self.download_btn.setFixedSize(20, 20)

		self.image_header_layout.addWidget(self.download_btn)
	
	
