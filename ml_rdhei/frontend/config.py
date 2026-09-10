import os
from pathlib import Path
from typing import Final
from PySide6.QtCore import QSize

APP_VERSION: Final[str] = "1.0"
APP_SHELL_SIZE: Final[QSize] = QSize(720, 540)
ABOUT_DIALOG_SIZE: Final[QSize] = QSize(300, 200)
MENU_WIDTH: Final[int] = 320
PHOTO_MARGIN: Final[tuple[int, int, int, int]] = (0, 0, 0, 0)

current_dir = Path(__file__).parent
starting_image_path = os.path.join(current_dir, "assets", "start.webp")
processing_view_image_path = os.path.join(current_dir, "assets", "processing.webp")
about_dialog_image_path = os.path.join(current_dir, "assets", "about.webp")