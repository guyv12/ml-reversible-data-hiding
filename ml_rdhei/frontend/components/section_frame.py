from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt

from frontend.config import SECTIONS_LABEL_HEIGHT

class SectionFrame(QFrame):
	def __init__(
		self,
		title: str,
		object_name: str
	):
		super().__init__()

		self.setObjectName(object_name)
		self.layout = QVBoxLayout(self)
		self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		self.title_label = QLabel(title)
		self.title_label.setFixedHeight(SECTIONS_LABEL_HEIGHT)
		self.layout.addWidget(self.title_label)
