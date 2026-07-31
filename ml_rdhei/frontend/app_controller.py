from PySide6.QtWidgets import QMainWindow, QStackedWidget, QSizePolicy

from frontend.config import MAIN_VIEW_SIZE, MIN_PROCESSING_VIEW_SIZE, MAX_PROCESSING_VIEW_SIZE
from frontend.views import MainView, ProcessingView, AboutDialog

class AppController:
	"""
	Central UI Controler 
	
	Manages application navigation inside QStackedWidget
	"""
	
	def __init__(self):
		"Creating and configuring app container that display views"
		self.app_shell = QMainWindow()
		self.app_shell.setWindowTitle("RDHEI Application")
		self.app_shell.setFixedSize(MAIN_VIEW_SIZE)

		self.stack = QStackedWidget()
		self.app_shell.setCentralWidget(self.stack)

		self.main_view = MainView()
		self.processing_view = ProcessingView()
		self.about_dialog = AboutDialog()

		self.stack.addWidget(self.main_view)
		self.stack.addWidget(self.processing_view)
		
		self.main_view.processing_view_btn.clicked.connect(self.show_processing_view)
		self.main_view.about_dialog_btn.clicked.connect(self.about_dialog.exec)
		self.processing_view.return_btn.clicked.connect(self.show_main_view)

		self.show_main_view()

	def show_main_view(self):
		self.app_shell.showNormal()
		self.app_shell.setFixedSize(MAIN_VIEW_SIZE)
		self.stack.setCurrentWidget(self.main_view)

	def show_processing_view(self):
		self.app_shell.setMinimumSize(MIN_PROCESSING_VIEW_SIZE)
		self.app_shell.setMaximumSize(MAX_PROCESSING_VIEW_SIZE)

		self.stack.setCurrentWidget(self.processing_view)
		self.app_shell.showMaximized()

	def run(self):
		self.app_shell.show()