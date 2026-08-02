from pathlib import Path

from PySide6.QtWidgets import (
	QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
) 
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap, QFontMetrics

from frontend.config import IMAGE_HEADER_MARGIN, PHOTO_DISPLAY_MARGIN

class ImagePreview(QWidget):
	remove_requested = Signal()

	def __init__(self):
		super().__init__()
		self._image_path: str | None = None

		self.main_layout = QVBoxLayout(self)
		self.main_layout.setContentsMargins(0, 0, 0, 0) 
		self.main_layout.setSpacing(0)

		image_header_layout = QHBoxLayout()
		image_header_layout.setContentsMargins(*IMAGE_HEADER_MARGIN)
		self.file_label = QLabel("")
		self.file_label.setTextFormat(Qt.TextFormat.PlainText)
		self.file_label.setObjectName("fileLabel")
		self.file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

		self.remove_btn = QPushButton("✕")
		self.remove_btn.setObjectName("removeBtn")
		self.remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
		self.remove_btn.setFixedSize(20, 20)
		self.remove_btn.clicked.connect(self.remove_requested)

		image_header_layout.addWidget(self.file_label, stretch=1)
		image_header_layout.addWidget(self.remove_btn)

		self.photo_display = QLabel()
		self.photo_display.setContentsMargins(*PHOTO_DISPLAY_MARGIN)
		self.photo_display.setAlignment(Qt.AlignmentFlag.AlignCenter)

		self.main_layout.addLayout(image_header_layout)
		self.main_layout.addWidget(self.photo_display)

	def _update_file_label(self, file_path: str):
		file_name = Path(self._image_path).name

		metrics = QFontMetrics(self.file_label.font())
		content = metrics.elidedText(
			file_name,
			Qt.TextElideMode.ElideMiddle,
			self.file_label.width()
		)
		self.file_label.setText(content)

	def set_image(self, file_path: str):
		self._image_path = file_path
		self._update_file_label(file_path)
		self._update_photo_display()

	def clear_image(self):
		self._image_path = None
		self.file_label.clear()
		self.photo_display.clear()

	def _update_photo_display(self):
		if not self._image_path:
			return

		pix = QPixmap(self._image_path)
		if pix.isNull():
			self.photo_display.setText("[ Photo unavailable ]")
			return

		label_height = self.file_label.fontMetrics().height()

		max_width = self.width() - 16
		max_height = self.height() - label_height - (5 * 4)

		if max_width <= 0 or max_height <= 0:
			return

		scaled_pix = pix.scaled(
			max_width,
			max_height,
			Qt.AspectRatioMode.KeepAspectRatio,
			Qt.TransformationMode.SmoothTransformation
		)
		self.photo_display.setPixmap(scaled_pix)

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self._update_photo_display()
