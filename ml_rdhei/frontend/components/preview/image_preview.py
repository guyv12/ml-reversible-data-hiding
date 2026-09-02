from pathlib import Path
from pydicom import dcmread
from pydicom.uid import ExplicitVRLittleEndian
import cv2
import numpy as np

from PySide6.QtWidgets import (
	QWidget, QMessageBox, QVBoxLayout, QHBoxLayout,
	QLabel, QPushButton, QFileDialog
) 
from PySide6.QtCore import Qt, Signal, QDir
from PySide6.QtGui import QPixmap, QImage, QFontMetrics

from frontend.config import (
	IMAGE_HEADER_MARGIN, PHOTO_DISPLAY_MARGIN,
	ZERO_MARGINS, LABEL_BUTTON_SIZE
)

class ImagePreview(QWidget):
	delete_requested = Signal()
	
	def __init__(self):
		super().__init__()
		self._image_path: str | None = None
		self._image_data: np.ndarray | None = None
		self._cached_pix: QPixmap | None = None

		self.main_layout = QVBoxLayout(self)
		self.main_layout.setContentsMargins(*ZERO_MARGINS) 
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

	def set_image(self, file_path: str, image_data):
		self._image_path = file_path
		self._image_data = image_data
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
			image = self._convert_ndarray_to_QImage(self._image_data)
			self._cached_pix = QPixmap.fromImage(image)

		self._update_photo_display()

	def clear_image(self):
		self._image_path = None
		self._image_data = None
		self._cached_pix = None
		self.photo_display.clear()

	def _convert_ndarray_to_QImage(self, array: np.ndarray) -> QImage:
		if array is None or array.size == 0:
			return QImage()
		
		bytes_per_line = array.strides[0]
		
		if array.ndim == 2:
			if array.dtype == np.uint8:
				image_format = QImage.Format.Format_Grayscale8

			elif array.dtype == np.uint16:
				image_format = QImage.Format.Format_Grayscale16
				
			else:
				image_format = QImage.Format.Format_Grayscale8

		elif array.ndim == 3:
			channels = array.shape[2]

			if channels == 1:
				array = array.squeeze(axis=2)
				return self._convert_ndarray_to_QImage(array)

			elif channels == 3:
				if array.dtype == np.uint8:
					image_format = QImage.Format.Format_BGR888
				else:
					image_format = QImage.Format.Format_Grayscale8

			elif channels == 4:
				if array.dtype == np.uint8:
					image_format = QImage.Format.Format_ARGB32
				else:
					image_format = QImage.Format.Format_Grayscale8
			else:
				return QImage()

		else:
			return QImage()

		return QImage(
			array.data,
			array.shape[1],
			array.shape[0],
			bytes_per_line,
			image_format
		).copy()

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

		image = self._convert_ndarray_to_QImage(norm_pixels)

		return QPixmap.fromImage(image).copy()

	def _update_photo_display(self):
		if self._cached_pix is None or self._cached_pix.isNull():
			self.photo_display.clear()
			self.photo_display.setText("[ Photo unavailable ]")
			return

		header_height = self.image_header_layout.sizeHint().height()

		margins = self.photo_display.contentsMargins()
		horizontal_margin = margins.left() + margins.right()
		vertical_margin = margins.top() + margins.bottom()

		max_width = self.width() - horizontal_margin
		max_height = self.height() - header_height - vertical_margin

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
	def __init__(self):
		super().__init__()
		self.file_label = QLabel("")
		self.file_label.setTextFormat(Qt.TextFormat.PlainText)
		self.file_label.setObjectName("fileLabel")
		self.file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

		self.delete_btn = QPushButton("✕")
		self.delete_btn.setObjectName("deleteBtn")
		self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
		self.delete_btn.setFixedSize(*LABEL_BUTTON_SIZE)
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

	def clear_image(self):
		self._image_path = None
		self._image_data = None
		self._cached_pix = None
		self.file_label.clear()
		self.photo_display.clear()

class OutputImagePreview(ImagePreview):
	def __init__(self):
		super().__init__()

		self.download_btn = QPushButton("⭳")
		self.download_btn.setObjectName("downloadBtn")
		self.download_btn.setCursor(Qt.CursorShape.PointingHandCursor)
		self.download_btn.setFixedSize(*LABEL_BUTTON_SIZE)
		self.download_btn.clicked.connect(self._save_image)

		self.delete_btn = QPushButton("✕")
		self.delete_btn.setObjectName("deleteBtn")
		self.delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
		self.delete_btn.setFixedSize(*LABEL_BUTTON_SIZE)
		self.delete_btn.clicked.connect(self.delete_requested)

		self.image_header_layout.addWidget(self.download_btn)
		self.image_header_layout.addStretch()
		self.image_header_layout.addWidget(self.delete_btn)

	def _save_pgm(self, file_path: str):
		data_to_save = self._image_data
			
		if data_to_save.ndim == 3:
			channels = data_to_save.shape[2]

			if channels == 1:
				data_to_save = data_to_save.squeeze(axis=2)
			
			elif channels == 3:
				data_to_save = data_to_save[:, :, 0]

			elif channels in (2, 4):
				QMessageBox.critical(
					self,
					"Write Error",
					f"Failed to save the image:\npmg format does not support transparency."
				)
				return

		try:
			success = cv2.imwrite(file_path, data_to_save)
		except Exception as e:
			QMessageBox.critical(self, "Write Error", f"Failed to save the image:\n{e}")
		else:
			QMessageBox.information(self, "Success", f"Saved the image as\n{file_path}")

	def _save_dicom(self, file_path: str):
		data_to_save = self._image_data

		dicom = dcmread(self._image_path)
		array = data_to_save.astype(dicom.pixel_array.dtype)
		dicom.PixelData = array.tobytes()

		dicom.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
		dicom.is_little_endian = True
		dicom.is_implicit_VR = False

		try:
			dicom.save_as(file_path)
		except (OSError, PermissionError, Exception) as e:
			QMessageBox.critical(self, "Write Error", f"Failed to save the image:\n{e}")
		else:
			QMessageBox.information(self, "Success", f"Saved the image as\n{file_path}")
		

	def _save_image(self):
		file_name = Path(self._image_path).name

		file_path, _ = QFileDialog.getSaveFileName(
				self, 
				self.tr("Save Image"), 
				str(Path.home() / str(file_name)), 
				self.tr("Image Files (*.pgm *.dcm)"),
			)

		if not file_path:
			return
		
		if self._image_path.lower().endswith(".pgm"):
			self._save_pgm(file_path)

		elif self._image_path.lower().endswith(".dcm"):
			self._save_dicom(file_path)
