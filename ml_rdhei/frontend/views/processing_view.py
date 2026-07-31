from PySide6.QtWidgets import QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt

from frontend.components.image_uploader import ImageUploader
from frontend.config import IMAGE_UPLOADER_SIZE

class ProcessingView(QWidget):
	"""
	Processing screen view for RDHEI operations.

	Provides workspace and UI controls for RDHEI operations.
	"""

	def __init__(self):
		super().__init__()

		layout = QVBoxLayout(self)
		self.title_label = QLabel("Image Processing View")
		self.title_label.setFixedHeight(30)
		self.return_btn = QPushButton("Return to main window")
		layout.addWidget(self.title_label)

		sections_layout = QHBoxLayout()
		layout.addLayout(sections_layout)

		input_panel = QFrame()
		input_panel.setObjectName("inputPanel")
		input_panel.setStyleSheet("""
			QFrame#inputPanel {
				background-color: #2b2b2b;
				border: 2px solid #4A90E2;
				border-radius: 8px;
			}
		""")
		
		input_layout = QVBoxLayout(input_panel)
		input_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		input_subtitle_label = QLabel("Input")
		input_subtitle_label.setFixedHeight(20)
		input_layout.addWidget(input_subtitle_label)

		image_uploader = ImageUploader()
		image_uploader.setFixedSize(IMAGE_UPLOADER_SIZE)
		input_layout.addWidget(image_uploader)
		
		metrics_panel = QFrame()
		metrics_panel.setObjectName("metricsPanel")
		metrics_panel.setStyleSheet("""
			QFrame#metricsPanel {
				background-color: #2b2b2b;
				border: 2px solid #E74C3C;
				border-radius: 8px;
			}
		""")
		metrics_layout = QVBoxLayout()
		metrics_layout.addWidget(metrics_panel)

		output_panel = QFrame()
		output_panel.setObjectName("outputPanel")
		output_panel.setStyleSheet("""
			QFrame#outputPanel {
				background-color: #2b2b2b;
				border: 2px solid #2ECC71;
				border-radius: 8px;
			}
		""")
		output_layout = QVBoxLayout()
		output_layout.addWidget(output_panel)
		
		sections_layout.addWidget(input_panel, stretch=1)
		sections_layout.addWidget(metrics_panel, stretch=1)
		sections_layout.addWidget(output_panel, stretch=1)

		layout.addWidget(self.return_btn)

		