from PySide6.QtWidgets import QPushButton
from PySide6.QtCore import Signal

class HoverButton(QPushButton):
	""" 
	Custom push button widget with image-preview hover signals

	Extends QPushButton behaviour by emitting custom Qt signals
	containing image paths whenever user hovers over or leaves
	the button area.
	"""
	
	hovered = Signal(str)
	
	def __init__(self, text, image_path):
		super().__init__(text)
		self.image_path = image_path

	def enterEvent(self, event):
		self.hovered.emit(self.image_path)
		super().enterEvent(event)
