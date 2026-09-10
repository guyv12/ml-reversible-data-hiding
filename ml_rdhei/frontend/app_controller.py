from PySide6.QtWidgets import QMainWindow, QStackedWidget

from frontend.config import APP_SHELL_SIZE
from frontend.views import MenuView, HidingView, AboutDialog

class AppController:
	"""
	Central UI Controler 
	
	Manages application navigation inside QStackedWidget
	"""
	
	def __init__(self):
		"Creating and configuring app container that display views"
		self.app_shell = QMainWindow()
		self.app_shell.setWindowTitle("RDHEI Application")
		self.app_shell.setFixedSize(APP_SHELL_SIZE)

		self.stack = QStackedWidget()
		self.app_shell.setCentralWidget(self.stack)

		self.menu_view = MenuView()
		self.hiding_view = HidingView()
		self.about_dialog = AboutDialog()

		self.stack.addWidget(self.menu_view)
		self.stack.addWidget(self.hiding_view)
		
		self.menu_view.hiding_view_btn.clicked.connect(
			lambda: self.stack.setCurrentWidget(self.hiding_view)
		)
		
		self.menu_view.about_dialog_btn.clicked.connect(self.about_dialog.exec)

		self.hiding_view.return_btn.clicked.connect(
			lambda: self.stack.setCurrentWidget(self.menu_view)
		)
		
	def run(self):
		self.app_shell.show()