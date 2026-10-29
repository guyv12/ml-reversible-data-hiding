from pathlib import Path

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from frontend.config import (
	MENU_WIDTH, ZERO_MARGINS,
	HIDING_VIEW_IMAGE_PATH, EXTRACTION_VIEW_IMAGE_PATH,
	ABOUT_DIALOG_IMAGE_PATH, STARTING_IMAGE_PATH
)
from frontend.components.hover_button import HoverButton
	
class MenuView(QWidget):
	"""
	Main application landing view.
	
	Assembles photo display preview on the left and navigation panel on the right.
	Handles hover interaction signals to update image preview dynamically.
	"""

	def __init__(self):
		super().__init__()
		self._pixmap = QPixmap()

		self.photo_display = QLabel()
		self.photo_display.setAlignment(Qt.AlignCenter)
		self.photo_display.setContentsMargins(*ZERO_MARGINS)
		self.photo_display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
		self.update_image(STARTING_IMAGE_PATH)

		self.title = QLabel("RDHEI Studio")
		self.hiding_view_btn = HoverButton("Data Hiding", HIDING_VIEW_IMAGE_PATH)
		self.extraction_view_btn = HoverButton("Data Extraction", EXTRACTION_VIEW_IMAGE_PATH)
		self.about_dialog_btn = HoverButton("About", ABOUT_DIALOG_IMAGE_PATH)

		self.hiding_view_btn.hovered.connect(self.update_image)
		self.extraction_view_btn.hovered.connect(self.update_image)
		self.about_dialog_btn.hovered.connect(self.update_image)

		menu_layout = QVBoxLayout()
		menu_layout.addWidget(self.title, alignment=Qt.AlignmentFlag.AlignHCenter)
		menu_layout.addStretch(1)
		menu_layout.addWidget(self.hiding_view_btn)
		menu_layout.addStretch(1)
		menu_layout.addWidget(self.extraction_view_btn)
		menu_layout.addStretch(1)
		menu_layout.addWidget(self.about_dialog_btn)
		menu_layout.addStretch(1)

		menu_container = QWidget()
		menu_container.setFixedWidth(MENU_WIDTH)
		menu_container.setLayout(menu_layout)

		main_layout = QHBoxLayout(self)
		main_layout.setContentsMargins(*ZERO_MARGINS)
		main_layout.setSpacing(0)
		main_layout.addWidget(self.photo_display)
		main_layout.addWidget(menu_container)

	def update_image(self, path: str):
		pix = QPixmap(path)
		if pix.isNull():
			self._pixmap = QPixmap()
			self.photo_display.setText(f"Error loading image {Path(path).name}")
			return
		self._pixmap = pix
		self._rescale()

	def _rescale(self):
		if self._pixmap.isNull():
			return
		size = self.photo_display.size()
		scaled = self._pixmap.scaled(size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
		y = (scaled.height() - size.height()) // 2
		self.photo_display.setPixmap(scaled.copy(0, y, size.width(), size.height()))

	def resizeEvent(self, event):
		super().resizeEvent(event)
		self._rescale()
			