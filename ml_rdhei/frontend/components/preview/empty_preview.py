from PySide6.QtWidgets import (
	QWidget, QVBoxLayout, QHBoxLayout, QLabel
) 
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon


class EmptyPreview(QWidget):
	def __init__(self):
		super().__init__()

		self.main_layout = QVBoxLayout(self)
		self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.main_layout.setSpacing(8)

		self.image_label = QLabel()
		icon = QIcon.fromTheme("system-file-manager")
		self.image_label.setPixmap(icon.pixmap(48, 48))
		self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
		
		self.title_label = QLabel("Drop Grayscale or DICOM image")
		self.title_label.setWordWrap(True)
		self.title_label.setObjectName("titleLabel")
		self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

		self.subtitle_label = QLabel("or click to browse")
		self.subtitle_label.setWordWrap(True)
		self.subtitle_label.setObjectName("subtitleLabel")
		self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

		chips_layout = QHBoxLayout()
		chips_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		
		formats = [".pgm", ".dcm"]
		for fmt in formats:
			chip = QLabel(fmt)
			chip.setObjectName("chipLabel")
			chips_layout.addWidget(chip)

		self.main_layout.addWidget(self.image_label)
		self.main_layout.addWidget(self.title_label)
		self.main_layout.addWidget(self.subtitle_label)
		self.main_layout.addSpacing(6)
		self.main_layout.addLayout(chips_layout)