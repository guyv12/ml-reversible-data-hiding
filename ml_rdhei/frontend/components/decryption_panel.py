from PySide6.QtWidgets import (
	QFrame, QVBoxLayout, QLabel, QPushButton,
	QLineEdit, QPlainTextEdit
)
from PySide6.QtCore import Qt, Signal

from frontend.utils import load_stylesheet

class DecryptionPanel(QFrame):
	extraction_request = Signal(str)

	def __init__(self):
		super().__init__()

		layout = QVBoxLayout(self)
		
		self.decryption_key = QLineEdit()
		self.decryption_key.setEnabled(False)
		self.decryption_key.setMaxLength(10)
		self.decryption_key.setPlaceholderText("Enter decryption key")
		self.decryption_key.setObjectName("keyEdit")

		self.message = QPlainTextEdit()
		self.message.setEnabled(False)
		self.message.setReadOnly(True)
		# self.message.setPlaceholderText("")
		self.message.setObjectName("messageEdit")

		self.extraction_button = QPushButton("Extract")
		self.extraction_button.setEnabled(False)

		layout.addWidget(QLabel("Decryption key"))
		layout.addWidget(self.decryption_key)
		layout.addWidget(QLabel("Hidden message"))
		layout.addWidget(self.message)
		layout.addWidget(self.extraction_button)

		self.decryption_key.textChanged.connect(self._update_button)
		self.extraction_button.clicked.connect(self._on_button_pressed)

		load_stylesheet(self, "metrics.css")
		self._update_button()

	def enable(self):
		self.decryption_key.setEnabled(True)

	def disable(self):
		self.decryption_key.setEnabled(False)
		self.extraction_button.setEnabled(False)

	def clear(self):
		self.disable()
		self.message.clear()
		self.decryption_key.clear()

	def set_busy(self, busy: bool):
		self.decryption_key.setEnabled(not busy)
		if busy:
			self.extraction_button.setEnabled(False)
		else:
			self._update_button()

	def _on_button_pressed(self):
		self.extraction_request.emit(self.decryption_key.text())

	def _update_button(self):
		self.extraction_button.setEnabled(bool(self.decryption_key.text()))

