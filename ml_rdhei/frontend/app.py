import sys

from PySide6.QtWidgets import QApplication

from frontend.app_controller import AppController

def main():
	app = QApplication(sys.argv)
	controller = AppController()
	controller.run()
	app.exec()


if __name__ == "__main__":
	main()