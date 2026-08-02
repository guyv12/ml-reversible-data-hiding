from pathlib import Path

from PySide6.QtWidgets import (
	QFrame, QWidget, QLabel, QFileDialog, QPushButton,
	QVBoxLayout, QHBoxLayout, QStackedLayout
)
from PySide6.QtCore import Qt, Signal, QDir
from PySide6.QtGui import QIcon, QPixmap

from frontend.utils import load_stylesheet
from frontend.components.image_preview import ImagePreview
from frontend.config import STYLES_DIR

class ImageUploader(QFrame):
	"""
	Drag-and-drop image upload widget with file preview.

	Supports selecting or dropping grayscale (PGM) and DICOM images.
	Manages image preview scaling and emits signals when an image is loaded or cleared.
	"""
	image_dropped = Signal(object)
	image_cleared = Signal()

	def __init__(self):
		super().__init__()
		self._image_path: str | None = None
		self.setAcceptDrops(True)
		
		self.main_layout = QVBoxLayout(self)
		self.main_layout.setContentsMargins(0, 0, 0, 0)
		
		self.stacked_layout = QStackedLayout()
		self.main_layout.addLayout(self.stacked_layout)

		self._setup_empty_widget()
		self.image_preview = ImagePreview()

		self.stacked_layout.addWidget(self.empty_widget)
		self.stacked_layout.addWidget(self.image_preview)

		self.image_preview.remove_requested.connect(self._clear_image)

		load_stylesheet(self, STYLES_DIR / "image_uploader.css")
		self._update_ui()

	@property
	def has_image(self) -> bool:
		return self._image_path is not None

	def _set_image(self, file_path: str | None):
		self._image_path = file_path
		self.image_preview.set_image(file_path)
		self._update_ui()

	def _clear_image(self):
		self._image_path = None
		self.image_preview.clear_image()
		self._update_ui()
		self.image_cleared.emit()

	def _setup_empty_widget(self):
		self.empty_widget = QWidget()
		uploader_layout = QVBoxLayout(self.empty_widget)
		uploader_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		uploader_layout.setSpacing(8)

		self.image_label = QLabel()
		icon = QIcon.fromTheme("system-file-manager")
		self.image_label.setPixmap(icon.pixmap(48, 48))
		self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		
		self.title_label = QLabel("Drop Grayscale or DICOM image")
		self.title_label.setWordWrap(True)
		self.title_label.setObjectName("titleLabel")
		self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

		self.subtitle_label = QLabel("or click to browse")
		self.subtitle_label.setWordWrap(True)
		self.subtitle_label.setObjectName("subtitleLabel")
		self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

		chips_layout = QHBoxLayout()
		chips_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		
		formats = [".pgm", ".dcm"]
		for fmt in formats:
			chip = QLabel(fmt)
			chip.setObjectName("chipLabel")
			chips_layout.addWidget(chip)

		uploader_layout.addWidget(self.image_label)
		uploader_layout.addWidget(self.title_label)
		uploader_layout.addWidget(self.subtitle_label)
		uploader_layout.addSpacing(6)
		uploader_layout.addLayout(chips_layout)
		
	def _update_ui(self):
		self.setProperty("has_image", self.has_image)
		self.style().unpolish(self)
		self.style().polish(self)

		if self.has_image:
			self.stacked_layout.setCurrentWidget(self.image_preview)
		else:
			self.stacked_layout.setCurrentWidget(self.empty_widget)

	def _set_drag_active(self, is_active: bool):
		self.setProperty("drag_active", is_active)
		self.style().unpolish(self)
		self.style().polish(self)

	def mousePressEvent(self, event: QMousePressEvent):
		super().mousePressEvent(event)

		if not self.has_image and event.button() == Qt.MouseButton.LeftButton:
			file_name, _ = QFileDialog.getOpenFileName(
				self, 
				self.tr("Select Image"), 
				QDir.homePath(), 
				self.tr("Image Files (*.pgm *.dcm)"),
			)
			if file_name and file_name.lower().endswith(('.pgm', '.dcm')):
				self._set_image(file_name)
				self.image_dropped.emit(file_name)
				print(f"Image selected {self._image_path}")

	def dragEnterEvent(self, event: QDragEnterEvent):
		if not self.has_image and event.mimeData().hasUrls():
			urls = event.mimeData().urls()
			if urls and urls[0].toLocalFile().lower().endswith(('.pgm', '.dcm')):
				event.acceptProposedAction()
				self._set_drag_active(True)
				return
		
		event.ignore()

	def dragLeaveEvent(self, event: QDragLeaveEvent):
		super().dragLeaveEvent(event)
		self._set_drag_active(False)

	def dropEvent(self, event: QDropEvent):
		self._set_drag_active(False)

		if not self.has_image and event.mimeData().hasUrls():
			urls = event.mimeData().urls()
			if urls:
				file_name = urls[0].toLocalFile()
				if file_name.lower().endswith(('.pgm', '.dcm')):
					event.acceptProposedAction()
					self._set_image(file_name)
					self.image_dropped.emit(file_name)
					print(f"Image selected {self._image_path}")

		event.ignore()