from PySide6.QtWidgets import QMainWindow, QStackedWidget, QSizePolicy

from frontend.config import(
	APP_SHELL_SIZE, MAIN_VIEW_SIZE,
	MIN_PROCESSING_VIEW_SIZE, MAX_PROCESSING_VIEW_SIZE
) 
from frontend.views import (
	MenuView, HidingView, 
	ExtractionView, AboutDialog
)

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

		self.menu_view = MenuView()
		self.hiding_view = HidingView()
		self.extraction_view = ExtractionView()
		self.about_dialog = AboutDialog()

		self.stack.addWidget(self.menu_view)
		self.stack.addWidget(self.hiding_view)
		self.stack.addWidget(self.extraction_view)
		
		self.menu_view.hiding_view_btn.clicked.connect(self.show_hiding_view)
		self.menu_view.extraction_view_btn.clicked.connect(self.show_extraction_view)
		self.menu_view.about_dialog_btn.clicked.connect(self.about_dialog.exec)
		self.hiding_view.return_btn.clicked.connect(self.show_main_view)
		self.extraction_view.return_btn.clicked.connect(self.show_main_view)
		
		self.main_view.processing_view_btn.clicked.connect(self.show_processing_view)
		self.main_view.about_dialog_btn.clicked.connect(self.about_dialog.exec)
		self.processing_view.return_btn.clicked.connect(self.show_main_view)

		self.show_main_view()

	def show_main_view(self):
		self.app_shell.showNormal()
		self.app_shell.setFixedSize(MAIN_VIEW_SIZE)
		self.stack.setCurrentWidget(self.main_view)

	def show_hiding_view(self):
		self.app_shell.setMinimumSize(MIN_PROCESSING_VIEW_SIZE)
		self.app_shell.setMaximumSize(MAX_PROCESSING_VIEW_SIZE)

		self.stack.setCurrentWidget(self.hiding_view)
		self.app_shell.showMaximized()

	def show_extracting_view(self):
		self.app_shell.setMinimumSize(MIN_PROCESSING_VIEW_SIZE)
		self.app_shell.setMaximumSize(MAX_PROCESSING_VIEW_SIZE)

		self.stack.setCurrentWidget(self.extracting_view)
		self.app_shell.showMaximized()

	def run(self):
		self.app_shell.show()