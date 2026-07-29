import os

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

from frontend.config import (
	PHOTO_MARGIN, MENU_WIDTH, 
	processing_view_image_path, about_dialog_image_path, starting_image_path
)
from frontend.components.hover_button import HoverButton
	
class MainView(QWidget):
	def __init__(self):
		super().__init__()
		
		self.photo_display = QLabel()
		self.photo_display.setAlignment(Qt.AlignCenter)
		self.photo_display.setContentsMargins(*PHOTO_MARGIN)
		self.update_image(starting_image_path)

		self.processing_view_btn = HoverButton("Image", processing_view_image_path)
		self.about_dialog_btn = HoverButton("About", about_dialog_image_path)

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

	def update_image(self, path):
		pix = QPixmap(path)
		if not pix.isNull():
			self.photo_display.setPixmap(pix)
		else:
			self.photo_display.setText(f"Error loading image {os.path.basename(path)}")