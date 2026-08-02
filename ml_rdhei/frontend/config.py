from pathlib import Path
from typing import Final
from PySide6.QtCore import QSize

APP_VERSION: Final[str] = "1.0"
APP_SHELL_SIZE: Final[QSize] = QSize(720, 540)
ABOUT_DIALOG_SIZE: Final[QSize] = QSize(300, 200)
MENU_WIDTH: Final[int] = 320
IMAGE_UPLOADER_SIZE: Final[QSize] = QSize(300, 200)
IMAGE_HEADER_MARGIN: Final[tuple[int, int, int, int]] = (5, 5, 5, 0)
PHOTO_DISPLAY_MARGIN: Final[tuple[int, int, int, int]] = (0, 5, 0, 5)

FRONTEND_DIR = Path(__file__).resolve().parent

ASSETS_DIR = FRONTEND_DIR / "assets"
STYLES_DIR = ASSETS_DIR / "styles"

STARTING_IMAGE_PATH = str(ASSETS_DIR / "start.webp")
PROCESSING_VIEW_IMAGE_PATH = str(ASSETS_DIR / "processing.webp")
ABOUT_DIALOG_IMAGE_PATH = str(ASSETS_DIR / "about.webp")
