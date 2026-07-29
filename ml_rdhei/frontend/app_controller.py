from PySide6.QtWidgets import QMainWindow, QStackedWidget

from frontend.config import APP_SHELL_SIZE
from frontend.views import MainView, ProcessingView, AboutDialog

class AppController:
	def __init__(self):
		self.app_shell = QMainWindow()
		self.app_shell.setWindowTitle("RDHEI Application")
		self.app_shell.setFixedSize(APP_SHELL_SIZE)

		self.stack = QStackedWidget()
		self.app_shell.setCentralWidget(self.stack)

		self.main_view = MainView()
		self.processing_view = ProcessingView()
		self.about_dialog = AboutDialog()

		self.stack.addWidget(self.main_view)
		self.stack.addWidget(self.processing_view)
		
		self.main_view.processing_view_btn.clicked.connect(
			lambda: stack.setCurrentWidget(self.processing_view)
		)
		
		self.main_view.about_dialog_btn.clicked.connect(self.about_dialog.exec)

		self.processing_view.return_btn.clicked.connect(
			lambda: self.stack.setCurrentWidget(self.main_view)
		)
		
	def run(self):
		self.app_shell.show()