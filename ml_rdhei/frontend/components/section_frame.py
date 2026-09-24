from PySide6.QtWidgets import QWidget, QFrame, QVBoxLayout, QLabel
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
		self._layout = QVBoxLayout(self)
		self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		self.title_label = QLabel(title)
		self.title_label.setFixedHeight(SECTIONS_LABEL_HEIGHT)
		self._layout.addWidget(self.title_label)

	def add_widget(self, widget: QWidget, stretch: int = 0):
		self._layout.addWidget(widget)