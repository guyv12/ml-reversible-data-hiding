import os
from pathlib import Path

from PySide6.QtWidgets import (
	QApplication, QMainWindow, QDialog, QStackedWidget,
	QVBoxLayout, QHBoxLayout,
	QWidget, QFrame, QLabel, QPushButton, QDialogButtonBox
	)
	
from PySide6.QtGui import QColor, QPalette, QPixmap
from PySide6.QtCore import QSize, Qt, Signal

current_dir = Path(__file__).parent
processing_view_image_path = os.path.join(current_dir, "assets", "a.webp")
about_view_image_path = os.path.join(current_dir, "assets", "about.webp")
MAIN_SIZE = QSize(720,540)

class HoverButton(QPushButton):

	hovered = Signal(str)
	
	def __init__(self, text, image_path):
		super().__init__(text)
		self.image_path = image_path

	def enterEvent(self, event):
		self.hovered.emit(self.image_path)
		super().enterEvent(event)

class MainView(QWidget):
	def __init__(self):
		super().__init__()
		
		self.sidebar = Sidebar()
		self.display = PhotoDisplay(processing_view_image_path)

		self.sidebar.processing_view_btn.hovered.connect(self.update_image)
		self.sidebar.about_view_btn.hovered.connect(self.update_image)

		main_layout = QHBoxLayout(self)
		main_layout.addWidget(self.display)
		main_layout.addWidget(self.sidebar)

	def update_image(self, path):
		self.display.update_image(path)

	@property
	def processing_view_btn(self): return self.sidebar.processing_view_btn

	@property
	def about_view_btn(self): return self.sidebar.about_view_btn

class Sidebar(QWidget):
		def __init__(self):
			super().__init__()
			self.setFixedWidth(320)
			layout = QVBoxLayout(self)

			self.processing_view_btn = HoverButton("Image", processing_view_image_path)
			self.about_view_btn = HoverButton("About", about_view_image_path)
		
			layout.addWidget(self.processing_view_btn)
			layout.addWidget(self.about_view_btn)

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

class ProcessingView(QWidget):
	def __init__(self):
		super().__init__()

		layout = QHBoxLayout(self)
		self.label = QLabel("image processing window")
		self.return_btn = QPushButton("Return to main window")

		layout.addWidget(self.label)
		layout.addWidget(self.return_btn)

class AboutDialog(QDialog):
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


class AppController:
	def __init__(self):
		self.app_shell = QMainWindow()
		self.app_shell.setWindowTitle("RDHEI Application")
		self.app_shell.setFixedSize(MAIN_SIZE)

		self.stack = QStackedWidget()
		self.app_shell.setCentralWidget(self.stack)

		self.main_view = MainView()
		self.processing_view = ProcessingView()
		self.about_dialog = AboutDialog()

		self.stack.addWidget(self.main_view)
		self.stack.addWidget(self.processing_view)
		
		self.main_view.processing_view_btn.clicked.connect(self.show_processing)
		self.main_view.about_view_btn.clicked.connect(self.about_dialog.exec)

		self.processing_view.return_btn.clicked.connect(self.show_main)

	def show_processing(self):
		self.stack.setCurrentWidget(self.processing_view)

	def show_main(self):
		self.stack.setCurrentWidget(self.main_view)

	def run(self):
		self.app_shell.show()

if __name__ == "__main__":
	app = QApplication([])
	controller = AppController()
	controller.run()
	app.exec()