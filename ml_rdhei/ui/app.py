import os
from pathlib import Path

from PySide6.QtWidgets import (
	QApplication, QMainWindow, QDialog,
	QVBoxLayout, QHBoxLayout,
	QWidget, QFrame, QLabel, QPushButton, QDialogButtonBox
	)
	
from PySide6.QtGui import QColor, QPalette, QPixmap
from PySide6.QtCore import QSize, Qt, Signal

current_dir = Path(__file__).parent
processing_window_image_path = os.path.join(current_dir, "assets", "a.webp")
about_window_image_path = os.path.join(current_dir, "assets", "about.webp")
MAIN_SIZE = QSize(720,540)

class HoverButton(QPushButton):

	hovered = Signal(str)
	
	def __init__(self, text, image_path):
		super().__init__(text)
		self.image_path = image_path

	def enterEvent(self, event):
		self.hovered.emit(self.image_path)
		super().enterEvent(event)

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		
		self.setWindowTitle("RDHEI Application")
		self.setFixedSize(MAIN_SIZE)

		self.sidebar = Sidebar(self)
		self.display = PhotoDisplay(processing_window_image_path)

		central_widget = QWidget()
		main_layout = QHBoxLayout(central_widget)
		main_layout.addWidget(self.display)
		main_layout.addWidget(self.sidebar)

		self.setCentralWidget(central_widget)

	def update_image(self, path):
		self.display.update_image(path)

	@property
	def processing_window_btn(self): return self.sidebar.processing_window_btn

	@property
	def about_window_btn(self): return self.sidebar.about_window_btn

class Sidebar(QWidget):
		def __init__(self, parent_window):
			super().__init__()
			self.setFixedWidth(320)
			layout = QVBoxLayout(self)

			self.processing_window_btn = HoverButton("Image", processing_window_image_path)
			self.about_window_btn = HoverButton("About", about_window_image_path)
		
			layout.addWidget(self.processing_window_btn)
			layout.addWidget(self.about_window_btn)

class PhotoDisplay(QFrame):
	def __init__(self, default_path):
		super().__init__()
		
		self.setFrameShape(QFrame.StyledPanel)
		layout = QVBoxLayout(self)
		layout.setContentsMargins(0, 0, 0, 0)

		self.label = QLabel()
		self.label.setAlignment(Qt.AlignCenter)
		self.update_image(default_path)

		layout.addWidget(self.label)

	def update_image(self, path):
		pix = QPixmap(path)
		if not pix.isNull():
			self.label.setPixmap(pix)
		else:
			self.label.setText("Error loading image")

class ProcessingWindow(QWidget):
	def __init__(self):
		super().__init__()

		layout = QHBoxLayout()

		self.label = QLabel("image processing window")
		layout.addWidget(self.label)

		self.return_btn = QPushButton("Return to main window")
		layout.addWidget(self.return_btn)
		self.setLayout(layout)

class AboutWindow(QDialog):
	def __init__(self):
		super().__init__()
		self.setWindowTitle("About")
		self.setFixedSize(300, 200)

		layout = QVBoxLayout(self)

		layout.addWidget(QLabel("<b>RDHEI Version 1.0</b>"))
		layout.addWidget(QLabel("Igor Sitko-Bajorski: DICOM Processing"))
		layout.addWidget(QLabel("Jakub Wiśniewski: Image Processing"))
		layout.addWidget(QLabel("Konrad Machura: User Interface"))

		self.close_btn = QDialogButtonBox(QDialogButtonBox.Close)
		self.close_btn.rejected.connect(self.reject)
		layout.addWidget(self.close_btn)

		self.setLayout(layout)

class AppController:
	def __init__(self):
		self.main_window = MainWindow()
		self.processing_window = ProcessingWindow()
		self.about_window = AboutWindow()

		self.all_windows = [self.main_window, self.processing_window, self.about_window]

		self.main_window.processing_window_btn.clicked.connect(self.show_processing_window)
		self.main_window.about_window_btn.clicked.connect(self.show_about_window)

		self.processing_window.return_btn.clicked.connect(self.show_main_window)

	def show_about_window(self):	
		self.about_window.exec()

	def show_processing_window(self):
		for window in self.all_windows:
				window.hide()
		self.processing_window.show()

	def show_main_window(self):
		for window in self.all_windows:
				window.hide()
		self.main_window.show()

if __name__ == "__main__":
	app = QApplication([])
	controller = AppController()
	controller.main_window.show()
	app.exec()