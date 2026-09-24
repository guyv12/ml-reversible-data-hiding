from PySide6.QtWidgets import (
	QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
	QLineEdit, QPlainTextEdit
)
from PySide6.QtCore import Qt, Signal

from frontend.utils import load_stylesheet
from frontend.config import KEYS_LEN_LIMIT

class DecryptionPanel(QFrame):
	extraction_request = Signal(str, str)

	def __init__(self):
		super().__init__()

		layout = QVBoxLayout(self)
		
		self.ad_decryption_key = QLineEdit()
		self.ad_decryption_key.setEnabled(False)
		self.ad_decryption_key.setMaxLength(KEYS_LEN_LIMIT)
		self.ad_decryption_key.setPlaceholderText("Enter key")
		self.ad_decryption_key.setObjectName("adKeyEdit")

		self.message_decryption_key = QLineEdit()
		self.message_decryption_key.setEnabled(False)
		self.message_decryption_key.setMaxLength(KEYS_LEN_LIMIT)
		self.message_decryption_key.setPlaceholderText("Enter key")
		self.message_decryption_key.setObjectName("messageKeyEdit")

		self.message = QPlainTextEdit()
		self.message.setEnabled(False)
		self.message.setReadOnly(True)
		self.message.setPlaceholderText("Hidden message")
		self.message.setObjectName("messageEdit")

		self.extraction_button = QPushButton("Extract")
		self.extraction_button.setEnabled(False)

		image_dec_key_layout = QHBoxLayout()
		image_dec_key_layout.addWidget(QLabel("Image decryption key"))
		image_dec_key_layout.addStretch()
		image_dec_key_layout.addWidget(QLabel(f"max {KEYS_LEN_LIMIT}"))
		layout.addLayout(image_dec_key_layout)
		layout.addWidget(self.ad_decryption_key)

		message_dec_key_layout = QHBoxLayout()
		message_dec_key_layout.addWidget(QLabel("Message decryption key"))
		message_dec_key_layout.addStretch()
		message_dec_key_layout.addWidget(QLabel(f"max {KEYS_LEN_LIMIT}"))
		layout.addLayout(message_dec_key_layout)
		layout.addWidget(self.message_decryption_key)

		layout.addWidget(QLabel("Hidden message"))
		layout.addWidget(self.message)
		layout.addWidget(self.extraction_button)

		self.ad_decryption_key.textChanged.connect(self._update_button)
		self.message_decryption_key.textChanged.connect(self._update_button)
		self.extraction_button.clicked.connect(self._on_button_pressed)

		load_stylesheet(self, "metrics.css")
		self._update_button()

	def enable(self):
		self.ad_decryption_key.setEnabled(True)
		self.message_decryption_key.setEnabled(True)
		self.message.setEnabled(True)

	def disable(self):
		self.ad_decryption_key.setEnabled(False)
		self.message_decryption_key.setEnabled(False)
		self.extraction_button.setEnabled(False)
		self.message.setEnabled(False)

	def clear(self):
		self.disable()
		self.message.clear()
		self.ad_decryption_key.clear()
		self.message_decryption_key.clear()

	def set_busy(self, busy: bool):
		self.ad_decryption_key.setEnabled(not busy)
		self.message_decryption_key.setEnabled(not busy)
		if busy:
			self.extraction_button.setEnabled(False)
		else:
			self._update_button()

	def _on_button_pressed(self):
		self.extraction_request.emit(
			self.ad_decryption_key.text(),
			self.message_decryption_key.text()
		)

	def _update_button(self):
		self.extraction_button.setEnabled(
			bool(self.ad_decryption_key.text())
			and bool(self.message_decryption_key.text())
			)

	def display_message(self, message: str):
		self.message.setPlainText(message)