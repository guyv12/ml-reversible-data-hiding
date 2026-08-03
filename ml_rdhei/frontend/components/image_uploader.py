from pathlib import Path

from PySide6.QtWidgets import (
	QFrame, QWidget, QLabel, QFileDialog, QPushButton,
	QVBoxLayout, QHBoxLayout, QStackedLayout
)
from PySide6.QtCore import Qt, Signal, QDir
from PySide6.QtGui import QIcon, QPixmap

from frontend.utils import load_stylesheet
from frontend.components.preview_manager import PreviewManager
from frontend.components.preview import InputImagePreview, EmptyPreview
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
		self.setAcceptDrops(True)
		
		self.main_layout = QVBoxLayout(self)
		self.main_layout.setContentsMargins(0, 0, 0, 0)

		self.preview_manager = PreviewManager()

		self.main_layout.addWidget(self.preview_manager)
		
		self._setup_connections()

		load_stylesheet(self, STYLES_DIR / "image_uploader.css")
		self._update_style(has_image=False)

	@property
	def has_image(self) -> bool:
		return self.preview_manager.has_image

	def _setup_connections(self):
		self.preview_manager.image_cleared.connect(self.image_cleared)
		
		self.preview_manager.image_cleared.connect(
			lambda: self._update_style(has_image=False)
		)
	
	def _update_style(self, has_image: bool):
		self.setProperty("has_image", "true" if has_image else "false")
		
		self.style().unpolish(self)
		self.style().polish(self)

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
				self.preview_manager.set_image(file_name)
				self._update_style(has_image=True)
				self.image_dropped.emit(file_name)

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
					self.preview_manager.set_image(file_name)
					self._update_style(has_image=True)
					self.image_dropped.emit(file_name)

		event.ignore()