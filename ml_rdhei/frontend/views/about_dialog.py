from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox

from frontend.config import ABOUT_DIALOG_SIZE, APP_VERSION

class AboutDialog(QDialog):
	"""
	Custom QDialog that present project team members and project version
	"""
	
	def __init__(self):
		super().__init__()
		self.setWindowTitle("About")
		self.setFixedSize(ABOUT_DIALOG_SIZE)

		layout = QVBoxLayout(self)
		layout.addWidget(QLabel(f"<b>RDHEI Version {APP_VERSION}</b>"))
		layout.addWidget(QLabel("Igor Sitko-Bajorski: DICOM Processing"))
		layout.addWidget(QLabel("Jakub Wiśniewski: Image Processing"))
		layout.addWidget(QLabel("Konrad Machura: User Interface"))

		self.close_btn = QDialogButtonBox(QDialogButtonBox.Close)
		self.close_btn.rejected.connect(self.reject)
		layout.addWidget(self.close_btn)