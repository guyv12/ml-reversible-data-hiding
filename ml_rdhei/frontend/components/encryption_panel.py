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
		self._message_bytes: int = 0

		layout = QVBoxLayout(self)
		
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

		layout.addWidget(QLabel("Encryption key"))
		layout.addWidget(self.encryption_key)
		layout.addWidget(QLabel("Message to hide"))
		layout.addWidget(self.message)
		layout.addWidget(self.message_count_label)
		layout.addWidget(self.hide_button)

		self.message.textChanged.connect(self._on_message_changed)
		self.encryption_key.textChanged.connect(self._update_button)
		self.hide_button.clicked.connect(self._on_button_pressed)

		load_stylesheet(self, "metrics.css")
		self._update_button()

	def enable(self, bytes: int):
		self.encryption_key.setEnabled(True)
		self.message.setEnabled(True)
		self._set_capacity(bytes)
		self._on_message_changed()

	def disable(self):
		self.encryption_key.setEnabled(False)
		self.message.setEnabled(False)
		self.hide_button.setEnabled(False)

	def clear(self):
		self.disable()
		self.message.clear()
		self.encryption_key.clear()

		self._capacity_bytes = 0
		self._message_bytes = 0
		self.message_count_label.setText("-")
		self._set_limit_exceeded(False)

	def set_busy(self, busy: bool):
		"""Lock inputs while the owner runs the hide step; unlock afterwards."""
		self.encryption_key.setEnabled(not busy)
		self.message.setEnabled(not busy)
		if busy:
			self.hide_button.setEnabled(False)
		else:
			self._update_button()

	@property
	def _limit_exceeded(self) -> bool:
		return self._message_bytes > self._capacity_bytes

	def _on_message_changed(self):
		text = self.message.toPlainText()
		self._message_bytes = len(text.encode("utf-8"))
		
		self.message_count_label.setText(f"{self._message_bytes} / {self._capacity_bytes} B")

		self._set_limit_exceeded(self._limit_exceeded)
		self._update_button()

	def _set_limit_exceeded(self, exceeded: bool):
		for widget in (self.message, self.message_count_label):
			if widget.property("limit_exceeded") == exceeded:
				continue
			widget.setProperty("limit_exceeded", exceeded)
			widget.style().unpolish(widget)
			widget.style().polish(widget)
			widget.update()

	def _set_capacity(self, bytes_: int):
		self._capacity_bytes = bytes_
		self._on_message_changed()

	def _on_button_pressed(self):
		self.hide_request.emit(
			self.encryption_key.text(),
			self.message.toPlainText()
		)

	def _update_button(self):
		self.hide_button.setEnabled(
			self.message.isEnabled()
			and not self._limit_exceeded
			and self._message_bytes > 0
			and bool(self.encryption_key.text())
		)

