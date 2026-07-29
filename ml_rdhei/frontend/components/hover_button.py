from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Signal

class HoverButton(QPushButton):

	hovered = Signal(str)
	
	def __init__(self, text, image_path):
		super().__init__(text)
		self.image_path = image_path

	def enterEvent(self, event):
		self.hovered.emit(self.image_path)
		super().enterEvent(event)
