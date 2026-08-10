from pathlib import Path

from PySide6.QtWidgets import (
	QFrame, QWidget, QFileDialog, QVBoxLayout
)
from PySide6.QtCore import Qt, Signal, QDir

from frontend.utils import load_stylesheet
from frontend.components.preview_manager import PreviewManager

class ImageUploader(QFrame):
	"""
	Drag-and-drop image upload widget with file preview.

	Supports selecting or dropping grayscale (PGM) and DICOM images.
	Manages image preview scaling and emits signals when an image is loaded or cleared.
	"""
	image_uploaded = Signal(object)
	image_removed = Signal()

	def __init__(
		self,
		preview_manager: PreviewManager
	):
		super().__init__()
		self.setAcceptDrops(True)
		
		self.main_layout = QVBoxLayout(self)
		self.main_layout.setContentsMargins(0, 0, 0, 0)

		self.preview_manager = preview_manager

		self.main_layout.addWidget(self.preview_manager)
		
		self._setup_connections()

		load_stylesheet(self,"image_frames.css")
		self._update_style(has_image=False)

	@property
	def has_image(self) -> bool:
		return self.preview_manager.has_image

	def _setup_connections(self):
		self.preview_manager.image_removed.connect(self.image_removed)
		
		self.preview_manager.image_removed.connect(
			lambda: self._update_style(has_image=False)
		)
	
	def _update_style(self, has_image: bool):
		self.setProperty("has_image", has_image)
		
		self.style().unpolish(self)
		self.style().polish(self)

	def _set_drag_active(self, is_active: bool):
		self.setProperty("drag_active", is_active)
		self.style().unpolish(self)
		self.style().polish(self)

	def mousePressEvent(self, event: QMousePressEvent):
		super().mousePressEvent(event)

		if not self.has_image and event.button() == Qt.MouseButton.LeftButton:
			file_path, _ = QFileDialog.getOpenFileName(
				self, 
				self.tr("Select Image"), 
				QDir.homePath(), 
				self.tr("Image Files (*.pgm *.dcm)"),
			)
			if file_path and file_path.lower().endswith(('.pgm', '.dcm')):
				# self.preview_manager.set_image(file_path)
				self.image_uploaded.emit(file_path)
				self._update_style(has_image=True)
				

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
				file_path = urls[0].toLocalFile()
				if file_path.lower().endswith(('.pgm', '.dcm')):
					event.acceptProposedAction()
					# self.preview_manager.set_image(file_path)
					self.image_uploaded.emit(file_path)
					self._update_style(has_image=True)
					

		event.ignore()