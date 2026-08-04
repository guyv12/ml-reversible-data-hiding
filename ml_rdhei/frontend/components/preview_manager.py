from pathlib import Path

from PySide6.QtWidgets import (
	QFrame, QStackedLayout
)
from PySide6.QtCore import Qt, Signal

from frontend.components.preview import InputImagePreview, EmptyPreview
from frontend.utils import load_stylesheet

class PreviewManager(QFrame):
	image_loaded = Signal(object)
	image_removed = Signal()

	def __init__(self, parent=None):
		super().__init__(parent)
		self._image_path: str | None = None
		
		self.stacked_layout = QStackedLayout(self)

		self.empty_preview = EmptyPreview()
		self.image_preview = InputImagePreview()

		self.stacked_layout.addWidget(self.empty_preview)
		self.stacked_layout.addWidget(self.image_preview)

		self.image_preview.delete_requested.connect(self._clear_image)
		
		load_stylesheet(self,"image_frames.css")
		self.update_ui()

	@property
	def has_image(self) -> bool:
		return self._image_path is not None

	def set_image(self, file_path: str | None):
		self._image_path = file_path
		self.image_preview.set_image(file_path)
		self.update_ui()
		self.image_loaded.emit(file_path)
		

	def _clear_image(self):
		self._image_path = None
		self.image_preview.clear_image()
		self.update_ui()
		self.image_removed.emit()

	def update_ui(self):
		self.setProperty("has_image", self.has_image)
		self.style().unpolish(self)
		self.style().polish(self)

		if self.has_image:
			self.stacked_layout.setCurrentWidget(self.image_preview)
		else:
			self.stacked_layout.setCurrentWidget(self.empty_preview)