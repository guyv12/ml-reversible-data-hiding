import sys

from PySide6.QtWidgets import QApplication, QMessageBox

from frontend.app_controller import AppController

def _excepthook(exc_type, exc, tb):
    sys.__excepthook__(exc_type, exc, tb)  
    QMessageBox.critical(None, "Unexpected error", "Something went wrong.")

def main():
	app = QApplication(sys.argv)
	sys.excepthook = _excepthook
	controller = AppController()
	controller.run()
	app.exec()


if __name__ == "__main__":
	main()