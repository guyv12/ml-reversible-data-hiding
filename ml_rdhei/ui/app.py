import os
from pathlib import Path

from PySide6.QtWidgets import (
	QApplication, QMainWindow, QDialog,
	QVBoxLayout, QHBoxLayout,
	QWidget, QFrame, QLabel, QPushButton, QDialogButtonBox
	)
	
from PySide6.QtGui import QColor, QPalette, QPixmap
from PySide6.QtCore import QSize, Qt

current_dir = Path(__file__).parent
dicom_image_path = os.path.join(current_dir, "assets", "dicom.png")
grayscale_image_path = os.path.join(current_dir, "assets", "a.webp")
about_image_path = os.path.join(current_dir, "assets", "about.webp")
MAIN_SIZE = QSize(720,540)
DISPLAY_SIZE = QSize(620,540)

class HoverButton(QPushButton):
    def __init__(self, text, image_path, parent_window):
        super().__init__(text)
        self.image_path = image_path
        self.parent_window = parent_window

    def enterEvent(self, event):
        self.parent_window.update_image(self.image_path)
        super().enterEvent(event)

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		
		self.setWindowTitle("RDHEI Application")
		self.setFixedSize(MAIN_SIZE)

		self.sidebar = Sidebar(self)
		self.display = PhotoDisplay(grayscale_image_path)

		central_widget = QWidget()
		main_layout = QHBoxLayout(central_widget)
		main_layout.addWidget(self.display)
		main_layout.addWidget(self.sidebar)

		self.setCentralWidget(central_widget)

	def update_image(self, path):
		self.display.update_image(path)

	@property
	def dicom_btn(self): return self.sidebar.dicom_btn

	@property
	def grayscale_btn(self): return self.sidebar.grayscale_btn

	@property
	def about_btn(self): return self.sidebar.about_btn

class Sidebar(QWidget):
		def __init__(self, parent_window):
			super().__init__()
			self.setFixedWidth(320)
			layout = QVBoxLayout(self)

			self.dicom_btn= HoverButton("DICOM", dicom_image_path, parent_window)
			self.grayscale_btn = HoverButton("Grayscale", grayscale_image_path, parent_window)
			self.about_btn = HoverButton("About", about_image_path, parent_window)
			
			layout.addWidget(self.dicom_btn)
			layout.addWidget(self.grayscale_btn)
			layout.addWidget(self.about_btn)

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

class GrayscaleWindow(QWidget):
	def __init__(self):
		super().__init__()

		layout = QHBoxLayout()
		self.label = QLabel("grayscale processing window")
		self.return_btn = QPushButton("Return to main window")
		layout.addWidget(self.label)
		layout.addWidget(self.return_btn)
		self.setLayout(layout)

class DicomWindow(QWidget):
	def __init__(self):
		super().__init__()

		layout = QHBoxLayout()
		self.label = QLabel("dicom processing window")
		self.return_btn = QPushButton("Return to main window")
		layout.addWidget(self.label)
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
		layout.addWidget(QLabel("Kuba Wiśniewski: Grayscale Processing"))
		layout.addWidget(QLabel("Konrad Machura: User Interface"))

		self.return_btn = QDialogButtonBox(QDialogButtonBox.Close)
		layout.addWidget(self.return_btn)

		self.setLayout(layout)

class AppController:
	def __init__(self):
		self.main_window = MainWindow()
		self.dicom_window = DicomWindow()
		self.grayscale_window = GrayscaleWindow()
		self.about_window = AboutWindow()

		self.all_windows = [self.main_window, self.dicom_window, self.grayscale_window, self.about_window]

		self.main_window.dicom_btn.clicked.connect(lambda: self.switch_to(self.dicom_window))
		self.main_window.grayscale_btn.clicked.connect(lambda: self.switch_to(self.grayscale_window))
		self.main_window.about_btn.clicked.connect(lambda: self.switch_to(self.about_window))

		self.dicom_window.return_btn.clicked.connect(lambda: self.switch_to(self.main_window))
		self.grayscale_window.return_btn.clicked.connect(lambda: self.switch_to(self.main_window))

		self.about_window.return_btn.rejected.connect(lambda: self.switch_to(self.main_window))

	def switch_to(self, target_window):
			for window in self.all_windows:
				window.hide()
			target_window.show()

if __name__ == "__main__":
	app = QApplication([])
	controller = AppController()
	controller.main_window.show()
	app.exec()