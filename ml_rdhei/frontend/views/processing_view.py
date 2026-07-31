from PySide6.QtWidgets import QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton

class ProcessingView(QWidget):
	"""
	Processing screen view for RDHEI operations.

	Provides workspace and UI controls for RDHEI operations.
	"""

	def __init__(self):
		super().__init__()

		layout = QVBoxLayout(self)
		self.label = QLabel("Image Processing View")
		self.label.setFixedHeight(20)
		self.return_btn = QPushButton("Return to main window")
		layout.addWidget(self.label)

		sections_layout = QHBoxLayout()
		layout.addLayout(sections_layout)

		input_layout = QVBoxLayout()
		input_panel = QFrame()
		input_panel.setStyleSheet("""
			QFrame {
				background-color: #2b2b2b;
				border: 2px solid #4A90E2;
				border-radius: 8px;
			}
		""")
		input_layout.addWidget(input_panel)
		sections_layout.addLayout(input_layout)


		metrics_layout = QVBoxLayout()
		metrics_panel = QFrame()
		metrics_panel.setStyleSheet("""
			QFrame {
				background-color: #2b2b2b;
				border: 2px solid #E74C3C;
				border-radius: 8px;
			}
		""")
		metrics_layout.addWidget(metrics_panel)
		sections_layout.addLayout(metrics_layout)


		output_layout = QVBoxLayout()
		output_panel = QFrame()
		output_panel.setStyleSheet("""
			QFrame {
				background-color: #2b2b2b;
				border: 2px solid #2ECC71;
				border-radius: 8px;
			}
		""")
		output_layout.addWidget(output_panel)
		sections_layout.addLayout(output_layout)

		layout.addWidget(self.return_btn)

		