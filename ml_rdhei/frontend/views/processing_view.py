from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton

class ProcessingView(QWidget):
	def __init__(self):
		super().__init__()

		layout = QHBoxLayout(self)
		self.label = QLabel("image processing window")
		self.return_btn = QPushButton("Return to main window")

		layout.addWidget(self.label)
		layout.addWidget(self.return_btn)