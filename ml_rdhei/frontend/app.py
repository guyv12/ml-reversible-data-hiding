import sys

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QIcon
from PySide6.QtCore import QDir

from frontend.theme import apply_theme
from frontend.app_controller import AppController
from frontend.config import ICONS_DIR, ICON_SET

def _excepthook(exc_type, exc, tb):
    sys.__excepthook__(exc_type, exc, tb)  
    QMessageBox.critical(None, "Unexpected error", "Something went wrong.")

try:
    from ctypes import windll
    myappid = 'RDHEI.Studio'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

def main():
	app = QApplication(sys.argv)
	QDir.addSearchPath("icons", str(ICONS_DIR / ICON_SET)) 
	QDir.addSearchPath("buttons", str(ICONS_DIR / "buttons"))
	apply_theme(app)
	app.setWindowIcon(QIcon(str(ICONS_DIR / "app.svg")))
	sys.excepthook = _excepthook
	controller = AppController()
	controller.run()
	app.exec()

if __name__ == "__main__":
	main()