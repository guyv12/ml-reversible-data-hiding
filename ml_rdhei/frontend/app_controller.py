from PySide6.QtWidgets import QMainWindow, QStackedWidget, QSizePolicy

from frontend.config import(
	MENU_VIEW_SIZE, MIN_PROCESSING_VIEW_SIZE, MAX_PROCESSING_VIEW_SIZE
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
		self.app_shell.setFixedSize(MENU_VIEW_SIZE)

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
		self.hiding_view.return_btn.clicked.connect(self.show_menu_view)
		self.extraction_view.return_btn.clicked.connect(self.show_menu_view)

		self.show_menu_view()

	def show_menu_view(self):
		self.app_shell.showNormal()
		self.app_shell.setFixedSize(MENU_VIEW_SIZE)
		self.stack.setCurrentWidget(self.menu_view)

	def show_hiding_view(self):
		self.app_shell.setMinimumSize(MIN_PROCESSING_VIEW_SIZE)
		self.app_shell.setMaximumSize(MAX_PROCESSING_VIEW_SIZE)

		self.stack.setCurrentWidget(self.hiding_view)
		self.app_shell.showMaximized()

	def show_extraction_view(self):
		self.app_shell.setMinimumSize(MIN_PROCESSING_VIEW_SIZE)
		self.app_shell.setMaximumSize(MAX_PROCESSING_VIEW_SIZE)

		self.stack.setCurrentWidget(self.extraction_view)
		self.app_shell.showMaximized()

	def run(self):
		self.app_shell.show()