from PySide6.QtWidgets import QWidget, QFrame, QHBoxLayout, QVBoxLayout, QLabel, QPushButton
from PySide6.QtCore import Qt

from frontend.components.image_uploader import ImageUploader
from frontend.components.histogram import Histogram
from frontend.components.preview_manager import PreviewManager
from frontend.config import PREVIEW_MANAGER_SIZE

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
			QFrame#inputPanel 
			{
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
		image_uploader.setFixedSize(PREVIEW_MANAGER_SIZE)

		input_histogram = Histogram()
		image_uploader.image_uploaded.connect(
			input_histogram.plot_histogram
		)
		image_uploader.image_removed.connect(
			input_histogram.clear
		)

		preview_manager = PreviewManager()
		image_uploader.image_uploaded.connect(
			preview_manager.set_image
		)
		input_layout.addWidget(image_uploader)
		input_layout.addWidget(input_histogram)
		
		metrics_panel = QFrame()
		metrics_panel.setObjectName("metricsPanel")
		metrics_panel.setStyleSheet("""
			QFrame#metricsPanel 
			{
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
			QFrame#outputPanel 
			{
				background-color: #2b2b2b;
				border: 2px solid #2ECC71;
				border-radius: 8px;
			}
		""")
		output_layout = QVBoxLayout(output_panel)
		output_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
		output_subtitle_label = QLabel("Output")
		output_subtitle_label.setFixedHeight(20)
		output_layout.addWidget(output_subtitle_label)

		preview_manager.setFixedSize(PREVIEW_MANAGER_SIZE)

		output_histogram = Histogram()
		preview_manager.image_loaded.connect(
			output_histogram.plot_histogram
		)
		preview_manager.image_removed.connect(
			output_histogram.clear
		)

		output_layout.addWidget(preview_manager)
		output_layout.addWidget(output_histogram)
		
		sections_layout.addWidget(input_panel, stretch=1)
		sections_layout.addWidget(metrics_panel, stretch=1)
		sections_layout.addWidget(output_panel, stretch=1)

		layout.addWidget(self.return_btn)

		