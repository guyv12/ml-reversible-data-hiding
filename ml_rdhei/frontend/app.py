import sys

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QIcon

from frontend.app_controller import AppController
from frontend.config import ASSETS_DIR

def _excepthook(exc_type, exc, tb):
    sys.__excepthook__(exc_type, exc, tb)  
    QMessageBox.critical(None, "Unexpected error", "Something went wrong.")

try:
    from ctypes import windll
    myappid = 'RDHEI Studio'
    windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass

def main():
	app = QApplication(sys.argv)
	app.setWindowIcon(QIcon(str(ASSETS_DIR / "app.svg")))
	sys.excepthook = _excepthook
	controller = AppController()
	controller.run()
	app.exec()


if __name__ == "__main__":
	main()