from PySide6.QtWidgets import (
	QFrame, QVBoxLayout, QLabel, QPushButton,
	QLineEdit, QPlainTextEdit
)
from PySide6.QtCore import Qt, Signal

from frontend.utils import load_stylesheet

class EncryptionPanel(QFrame):
	hide_request = Signal(str, str)

	def __init__(self):
		super().__init__()
		
		self._capacity_bytes: int = 0
		
		self.layout = QVBoxLayout(self)
		
		self.encryption_key = QLineEdit()
		self.encryption_key.setEnabled(False)
		self.encryption_key.setMaxLength(10)
		self.encryption_key.setPlaceholderText("Enter encryption key")
		self.encryption_key.setObjectName("keyEdit")

		self.message = QPlainTextEdit()
		self.message.setEnabled(False)
		self.message.setPlaceholderText("Enter message")
		self.message.setObjectName("messageEdit")

		self.message_count_label = QLabel("-")
		self.message_count_label.setAlignment(Qt.AlignmentFlag.AlignRight)
		self.message_count_label.setObjectName("messageCountLabel")

		self.hide_button = QPushButton("Hide")
		self.hide_button.setEnabled(False)

		self.layout.addWidget(QLabel("Encryption key"))
		self.layout.addWidget(self.encryption_key)
		self.layout.addWidget(QLabel("Message to hide"))
		self.layout.addWidget(self.message)
		self.layout.addWidget(self.message_count_label)
		self.layout.addWidget(self.hide_button)

		self.message.textChanged.connect(self._message_changed)
		self.hide_button.clicked.connect(self._on_button_pressed)
		
		load_stylesheet(self,"metrics.css")

	def _message_changed(self):
		text = self.message.toPlainText()
		bytes_count = len(text.encode("utf-8"))
		exceeded =  bytes_count > self._capacity_bytes
		
		self.message_count_label.setText(f"{bytes_count} / {self._capacity_bytes} B")

		for widget in (self.message, self.message_count_label):
			if widget.property("limit_exceeded") != exceeded:
				widget.setProperty("limit_exceeded", exceeded)
				widget.style().unpolish(widget)
				widget.style().polish(widget)
				widget.update()

		self.hide_button.setEnabled(not exceeded and bool(text))

	def _set_capacity(self, bytes: int):
		self._capacity_bytes = bytes
		self._message_changed()

	def _on_button_pressed(self):
		self.disable_panel()
		self.hide_request.emit(
			self.encryption_key.text(),
			self.message.toPlainText()
		)

	def enable_panel(self, bytes: int):
		self.encryption_key.setEnabled(True)
		self.message.setEnabled(True)
		self._set_capacity(bytes)

	def disable_panel(self):
		self.encryption_key.setEnabled(False)
		self.message.setEnabled(False)
		self.hide_button.setEnabled(False)

	def clear(self):
		self.disable_panel()
		self.message.clear()
		self.encryption_key.clear()

		self._capacity_bytes = 0
		self.message_count_label.setText("-")

