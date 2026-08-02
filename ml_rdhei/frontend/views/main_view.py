from pathlib import Path

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from frontend.config import (
	MENU_WIDTH, PROCESSING_VIEW_IMAGE_PATH, 
	ABOUT_DIALOG_IMAGE_PATH, STARTING_IMAGE_PATH
)
from frontend.components.hover_button import HoverButton
	
class MainView(QWidget):
	"""
	Main application landing view.
	
	Assembles photo display preview on the left and navigation panel on the right.
	Handles hover interaction signals to update image preview dynamically.
	"""

	def __init__(self):
		super().__init__()
		
		self.photo_display = QLabel()
		self.photo_display.setAlignment(Qt.AlignCenter)
		self.photo_display.setContentsMargins(0, 0, 0, 0)
		self.update_image(STARTING_IMAGE_PATH)

		self.processing_view_btn = HoverButton("Image", PROCESSING_VIEW_IMAGE_PATH)
		self.about_dialog_btn = HoverButton("About", ABOUT_DIALOG_IMAGE_PATH)

		self.processing_view_btn.hovered.connect(self.update_image)
		self.about_dialog_btn.hovered.connect(self.update_image)

		menu_layout = QVBoxLayout(self)
		menu_layout.addWidget(self.processing_view_btn)
		menu_layout.addWidget(self.about_dialog_btn)

		menu_container = QWidget()
		menu_container.setFixedWidth(MENU_WIDTH)
		menu_container.setLayout(menu_layout)

		main_layout = QHBoxLayout(self)
		main_layout.addWidget(self.photo_display)
		main_layout.addWidget(menu_container)

	def update_image(self, path: str):
		pix = QPixmap(path)
		if not pix.isNull():
			self.photo_display.setPixmap(pix)
		else:
			self.photo_display.setText(f"Error loading image {Path(path).name}")