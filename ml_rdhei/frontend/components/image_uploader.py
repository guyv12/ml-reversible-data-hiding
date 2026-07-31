from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon

from frontend.utils import load_stylesheet
from frontend.config import STYLES_DIR


class ImageUploader(QFrame):
	image_dropped = Signal(str)

	def __init__(self):
		super().__init__()
		self.setAcceptDrops(True)
		self._setup_ui()
		load_stylesheet(self, STYLES_DIR / "image_uploader.css")

	def _setup_ui(self):
		uploader_layout = QVBoxLayout(self)
		uploader_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
		uploader_layout.setSpacing(8)

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

		uploader_layout.addWidget(self.image_label)
		uploader_layout.addWidget(self.title_label)
		uploader_layout.addWidget(self.subtitle_label)
		uploader_layout.addSpacing(6)
		uploader_layout.addLayout(chips_layout)

	def _set_drag_active(self, is_active: bool):
		self.setProperty("drag_active", is_active)
		self.style().unpolish(self)
		self.style().polish(self)

	def mousePressEvent(self, event: QMousePressEvent):
		if event.button() == Qt.MouseButton.LeftButton:
			file_name, _ = QFileDialog.getOpenFileName(
				self, 
				self.tr("Select Image"), 
				"/home/", 
				self.tr("Image Files (*.pgm *.dcm)"),
			)
			if file_name and file_name.lower().endswith(('.pgm', '.dcm')):
				self.image_dropped.emit(file_name)
				print("Image selected")

	def dragEnterEvent(self, event: QDragEnterEvent):
		urls = event.mimeData().urls()
		if urls and urls[0].toLocalFile().lower().endswith(('.pgm', '.dcm')):
			event.acceptProposedAction()
			self._set_drag_active(True)

	def dragLeaveEvent(self, event: QDragLeaveEvent):
		self._set_drag_active(False)

	def dropEvent(self, event: QDropEvent):
		self._set_drag_active(False)
		urls = event.mimeData().urls()
		if urls:
			file_name = urls[0].toLocalFile()
			if file_name.lower().endswith(('.pgm', '.dcm')):
				self.image_dropped.emit(file_name)
				print("Image selected")