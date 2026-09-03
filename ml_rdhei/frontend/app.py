from PySide6.QtWidgets import QApplication

from frontend.app_controller import AppController

if __name__ == "__main__":
	app = QApplication([])
	controller = AppController()
	controller.run()
	app.exec()