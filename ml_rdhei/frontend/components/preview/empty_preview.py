from PySide6.QtWidgets import (
	QWidget, QVBoxLayout, QHBoxLayout, QLabel
) 
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon


class EmptyPreview(QWidget):
	def __init__(
		self,
		title: str,
		icon_name: str | None = None,
		subtitle: str | None = None,
		chips: list[str] | None = None
		):
		super().__init__()

		self.main_layout = QVBoxLayout(self)
		self.main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		self.main_layout.setSpacing(8)

		self.title_label: QLabel
		self.icon_label: QLabel | None = None
		self.subtitle_label: QLabel | None = None
		self.subtitle_label: QLabel | None = None

		if icon_name:
			self._set_icon(icon_name)

		self._set_title(title)

		if subtitle:
			self._set_subtitle(subtitle)

		if chips:
			self._set_chips(chips)
		
	def _set_icon(self, icon_name: str):
		self.icon_label = QLabel()
		icon = QIcon.fromTheme(icon_name)
		self.icon_label.setPixmap(icon.pixmap(48, 48))
		self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

		self.main_layout.addWidget(self.icon_label)

	def _set_title(self, title: str):
		self.title_label = QLabel(title)
		self.title_label.setWordWrap(True)
		self.title_label.setObjectName("titleLabel")
		self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

		self.main_layout.addWidget(self.title_label)

	def _set_subtitle(self, subtitle: str):
		self.subtitle_label = QLabel(subtitle)
		self.subtitle_label.setWordWrap(True)
		self.subtitle_label.setObjectName("subtitleLabel")
		self.subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

		self.main_layout.addWidget(self.subtitle_label)

	def _set_chips(self, chips: list[str]):
		chips_layout = QHBoxLayout()
		chips_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

		for c in chips:
			chip = QLabel(c)
			chip.setObjectName("chipLabel")
			chips_layout.addWidget(chip)

		self.main_layout.addSpacing(6)
		self.main_layout.addLayout(chips_layout)
