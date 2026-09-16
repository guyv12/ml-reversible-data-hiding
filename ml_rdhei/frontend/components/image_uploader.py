from pathlib import Path

from PySide6.QtWidgets import (
	QFrame, QWidget, QFileDialog, QVBoxLayout
)
from PySide6.QtCore import Qt, Signal, QDir

from frontend.utils import load_stylesheet
from frontend.config import ZERO_MARGINS, ACCEPTED_FORMATS, IMAGE_FILE_FILTER
from frontend.components.preview_manager import PreviewManager

class ImageUploader(QFrame):
	"""
	Drag-and-drop image upload widget with file preview.

	Supports selecting or dropping grayscale (PGM) and DICOM images.
	Manages image preview scaling and emits signals when an image is loaded or cleared.
	"""
	image_uploaded = Signal(Path)
	image_removed = Signal()

	def __init__(
		self,
		preview_manager: PreviewManager
	):
		super().__init__()
		self.setAcceptDrops(True)
		
		self.main_layout = QVBoxLayout(self)
		self.main_layout.setContentsMargins(*ZERO_MARGINS)

		self.preview_manager = preview_manager

		self.main_layout.addWidget(self.preview_manager)
		
		self._setup_connections()

		load_stylesheet(self,"image_frames.css")
		self._update_style()

	@property
	def has_image(self) -> bool:
		return self.preview_manager.has_image

	def _setup_connections(self):
		self.preview_manager.image_removed.connect(self.image_removed)
		self.preview_manager.image_loaded.connect(self._update_style)
		self.preview_manager.image_removed.connect(self._update_style)
	
	def _update_style(self):
		self.setProperty("has_image", self.has_image)
		self.style().unpolish(self)
		self.style().polish(self)

	def _set_drag_active(self, is_active: bool):
		self.setProperty("drag_active", is_active)
		self.style().unpolish(self)
		self.style().polish(self)

	def mousePressEvent(self, event):
		super().mousePressEvent(event)

		if not self.has_image and event.button() == Qt.MouseButton.LeftButton:
			path, _ = QFileDialog.getOpenFileName(
				self, 
				self.tr("Select Image"), 
				QDir.homePath(), 
				self.tr(IMAGE_FILE_FILTER),
			)
			if not path:
				return

			file_path = Path(path)
			if file_path.suffix.lower() in ACCEPTED_FORMATS:
				# self.preview_manager.set_image(file_path)
				self.image_uploaded.emit(file_path)
				self._update_style()
				
	def dragEnterEvent(self, event):
		if not self.has_image and event.mimeData().hasUrls():
			urls = event.mimeData().urls()
			if urls and Path(urls[0].toLocalFile()).suffix.lower() in ACCEPTED_FORMATS:
				event.acceptProposedAction()
				self._set_drag_active(True)
				return
		
		event.ignore()

	def dragLeaveEvent(self, event):
		super().dragLeaveEvent(event)
		self._set_drag_active(False)

	def dropEvent(self, event):
		self._set_drag_active(False)

		if self.has_image and not event.mimeData().hasUrls():
			event.ignore()
			return
		
		urls = event.mimeData().urls()
		if not urls:
			event.ignore()
			return

		file_path = Path(urls[0].toLocalFile())
		if file_path.suffix.lower() not in ACCEPTED_FORMATS:
			event.ignore()
			return
			
		event.acceptProposedAction()
		self.image_uploaded.emit(file_path)