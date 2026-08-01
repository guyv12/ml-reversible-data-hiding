from pathlib import Path

from PySide6.QtWidgets import (
	QFrame, QWidget, QLabel, QFileDialog, QPushButton,
	QVBoxLayout, QHBoxLayout, QStackedLayout
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap

from frontend.utils import load_stylesheet
from frontend.config import (
	STYLES_DIR, PHOTO_MARGINS,
	IMAGE_UPLOADER_MARGIN
)


class ImageUploader(QFrame):
	image_dropped = Signal(object)
	image_cleared = Signal()

	def __init__(self):
		super().__init__()
		self._image_path = None
		self.setAcceptDrops(True)
		
		self.main_layout = QVBoxLayout(self)
		self.main_layout.setContentsMargins(*PHOTO_MARGINS)
		
		self.stacked_layout = QStackedLayout()
		self.main_layout.addLayout(self.stacked_layout)

		self._setup_empty_view()
		self._setup_preview_view()

		self.stacked_layout.addWidget(self.empty_widget)
		self.stacked_layout.addWidget(self.preview_widget)

		load_stylesheet(self, STYLES_DIR / "image_uploader.css")
		self._update_ui()

	@property
	def has_image(self) -> bool:
		return self._image_path is not None

	def resizeEvent(self, event):
		super().resizeEvent(event)
		if self.has_image:
			self._update_ui()

	def _setup_empty_view(self):
		self.empty_widget = QWidget()
		uploader_layout = QVBoxLayout(self.empty_widget)
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

	def _setup_preview_view(self):
		self.preview_widget = QWidget()
		uploader_layout = QVBoxLayout(self.preview_widget)
		uploader_layout.setContentsMargins(0, 0, 0, 0) 
		uploader_layout.setSpacing(0)

		image_header_layout = QHBoxLayout()
		image_header_layout.setContentsMargins(
			IMAGE_UPLOADER_MARGIN,
			IMAGE_UPLOADER_MARGIN,
			IMAGE_UPLOADER_MARGIN,
			0
		)
		self.file_label = QLabel("")
		self.file_label.setWordWrap(True)
		self.file_label.setObjectName("fileLabel")
		self.file_label.setAlignment(Qt.AlignmentFlag.AlignLeft)

		self.remove_btn = QPushButton("✕")
		self.remove_btn.setObjectName("removeBtn")
		self.remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
		self.remove_btn.setFixedSize(20, 20)
		self.remove_btn.clicked.connect(self._clear_image)

		image_header_layout.addWidget(self.file_label, stretch=1)
		image_header_layout.addWidget(self.remove_btn)

		uploader_layout.addLayout(image_header_layout)

		self.photo_display = QLabel()
		self.photo_display.setContentsMargins(
			0,
			IMAGE_UPLOADER_MARGIN,
			0,
			IMAGE_UPLOADER_MARGIN
		)
		self.photo_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
		uploader_layout.addWidget(self.photo_display)
		
	def _update_ui(self):
		self.setProperty("has_image", self.has_image)
		self.style().unpolish(self)
		self.style().polish(self)

		if self.has_image:
			self.file_label.setText(Path(self._image_path).name)
			self._display_scaled_image()
			self.stacked_layout.setCurrentWidget(self.preview_widget)
		else:
			self.stacked_layout.setCurrentWidget(self.empty_widget)

	def _display_scaled_image(self):
		if not self._image_path:
			return

		pix = QPixmap(self._image_path)
		if pix.isNull():
			self.photo_display.setText("[ Photo unavailable ]")
			return

		label_height = self.file_label.fontMetrics().height()

		max_width = self.width() - 16
		max_height = self.height() - label_height - (IMAGE_UPLOADER_MARGIN*4)

		scaled_pix = pix.scaled(
			max_width,
			max_height,
			Qt.AspectRatioMode.KeepAspectRatio,
			Qt.TransformationMode.SmoothTransformation
		)
		self.photo_display.setPixmap(scaled_pix)

	def _set_image(self, file_path: str | None):
		self._image_path = file_path
		self._update_ui()

	def _clear_image(self):
		self._image_path = None
		self._update_ui()
		self.image_cleared.emit()

	def _set_drag_active(self, is_active: bool):
		self.setProperty("drag_active", is_active)
		self.style().unpolish(self)
		self.style().polish(self)


	def mousePressEvent(self, event: QMousePressEvent):
		if not self.has_image:
			if event.button() == Qt.MouseButton.LeftButton:
				file_name, _ = QFileDialog.getOpenFileName(
					self, 
					self.tr("Select Image"), 
					"/home/", 
					self.tr("Image Files (*.pgm *.dcm)"),
				)
				if file_name and file_name.lower().endswith(('.pgm', '.dcm')):
					self._set_image(file_name)
					self.image_dropped.emit(file_name)
					print(f"Image selected {self._image_path}")

	def dragEnterEvent(self, event: QDragEnterEvent):
		if not self.has_image:
			urls = event.mimeData().urls()
			if urls and urls[0].toLocalFile().lower().endswith(('.pgm', '.dcm')):
				event.acceptProposedAction()
				self._set_drag_active(True)

	def dragLeaveEvent(self, event: QDragLeaveEvent):
		self._set_drag_active(False)

	def dropEvent(self, event: QDropEvent):
		if not self.has_image:
			self._set_drag_active(False)
			urls = event.mimeData().urls()
			if urls:
				file_name = urls[0].toLocalFile()
				if file_name.lower().endswith(('.pgm', '.dcm')):
					self._set_image(file_name)
					self.image_dropped.emit(file_name)
					print(f"Image selected {self._image_path}")